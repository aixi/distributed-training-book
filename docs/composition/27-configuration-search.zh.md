---
title: "配置搜索"
subtitle: "在指数空间中寻找最优并行方案"
---

<div class="chapter-opener" markdown>
在 5 个并行维度和数十个超参数之下，大规模训练的配置空间呈指数增长。一台 10000 GPU 的集群，若 DP、PP、TP、CP、EP 各在 1 到 16 之间取值，就存在超过 10 万种可能配置。找到最优配置不是试错——而是应用数学。
</div>

<div class="investigation-question" markdown>
**问题**：给定模型结构、目标批大小和集群规格，你如何系统地找到既满足内存约束、又使训练时间最短的配置？我们能比穷举做得更好吗？
</div>

!!! abstract "本章导览"
    **承上**：[第 23 章](23-device-mesh.md)（设备网格抽象）、[第 24 章](24-3d-parallelism.md)（三维并行组合）、[第 25 章](25-4d-5d-parallelism.md)（四维／五维并行）、[第 26 章](26-failures-checkpoints.md)（容错约束）

    **核心洞察**：配置空间随并行维度指数增长，但内存约束、通信成本和硬件拓扑会剪掉大部分候选。从启发式到贝叶斯优化的系统化搜索，能在不穷举的情况下找到近优配置。

!!! note "代码风格"
    本章的代码示例展示的是搜索*算法与数据结构*。为清晰起见它们做了简化——生产级配置搜索工具（如 Alpa、Megatron 的自动并行）要处理更多附加约束。请关注算法思想，而非实现细节。

!!! tip "推荐阅读路径"
    如果你想先抓住核心思想，先读每节的问题陈述、公式和决策规则，再深入实现代码。
    把长代码清单当作可选的深潜，在构建优化器时再回来细读。

## 配置空间

### 配置的维度

一个完整的分布式训练配置规定：

```python
@dataclass
class TrainingConfig:
    """Complete specification of distributed training setup."""

    # Parallelism dimensions
    dp_size: int          # Data parallelism degree
    pp_size: int          # Pipeline parallelism stages
    tp_size: int          # Tensor parallelism degree
    cp_size: int          # Context parallelism (for long sequences)
    ep_size: int          # Expert parallelism (for MoE)

    # Memory optimizations
    zero_stage: int       # ZeRO optimization level (0, 1, 2, 3)
    activation_checkpointing: bool
    offload_optimizer: bool
    offload_params: bool

    # Micro-batching
    micro_batch_size: int
    gradient_accumulation_steps: int

    # Model shape
    seq_length: int
    hidden_dim: int

    # Pipeline schedule
    pipeline_schedule: str  # "1F1B", "interleaved", "zero-bubble"
    num_interleaved_stages: int

    # Communication
    overlap_comm_compute: bool
    bucket_size_mb: int

    # Precision
    precision: str        # "fp32", "fp16", "bf16", "fp8"

    @property
    def global_batch_size(self) -> int:
        return (self.dp_size * self.micro_batch_size *
                self.gradient_accumulation_steps)

    @property
    def total_gpus(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size * self.cp_size * self.ep_size
```

### 组合爆炸

对一台 1024 GPU 的集群，取：

- DP ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
- PP ∈ {1, 2, 4, 8, 16, 32}
- TP ∈ {1, 2, 4, 8}
- ZeRO ∈ {0, 1, 2, 3}
- micro_batch_size ∈ {1, 2, 4, 8, 16, 32}
- gradient_accumulation ∈ {1, 2, 4, 8, 16, 32, 64}

满足 DP × PP × TP = 1024 的合法组合：

| 配置 | 数量 |
|------|------|
| (1024, 1, 1) | 1 |
| (512, 2, 1), (512, 1, 2) | 2 |
| (256, 4, 1), (256, 2, 2), (256, 1, 4) | 3 |
| (128, 8, 1), (128, 4, 2), (128, 2, 4), (128, 1, 8) | 4 |
| ... | ... |

并行组合总数：**约 50**

再乘上其他超参数：**50 × 4 × 6 × 7 = 8,400** 种配置。

再加上精度、检查点（checkpoint）、卸载（offloading）：**50,000+** 种配置。

**穷举搜索不可行。**

## 约束满足

在搜索最优之前，必须先筛出可行。

### 内存约束

每块 GPU 有固定的 HBM 容量 $M_{\text{GPU}}$：

$$M_{\text{model}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{gradients}} \leq M_{\text{GPU}}$$

```python
class MemoryConstraint:
    """Check if configuration fits in GPU memory."""

    def __init__(self,
                 model_params: int,
                 hidden_dim: int,
                 num_layers: int,
                 seq_length: int,
                 vocab_size: int,
                 gpu_memory_gb: float = 80.0):

        self.model_params = model_params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.gpu_memory = gpu_memory_gb * (1 << 30)  # Convert to bytes

    def is_feasible(self, config: TrainingConfig) -> Tuple[bool, Dict[str, float]]:
        """Check memory feasibility and return breakdown."""

        bytes_per_param = self._bytes_per_param(config.precision)

        # Model memory (sharded by TP and PP, then by ZeRO-3)
        params_per_gpu = self.model_params / (config.tp_size * config.pp_size)
        if config.zero_stage >= 3:
            params_per_gpu /= config.dp_size

        model_mem = params_per_gpu * bytes_per_param

        # Optimizer memory (sharded by ZeRO)
        optimizer_states = 2 if config.precision == "fp32" else 3  # m, v, [master]
        optimizer_bytes = 4 * optimizer_states  # Always FP32

        optimizer_mem = params_per_gpu * optimizer_bytes
        if config.zero_stage >= 1:
            optimizer_mem /= config.dp_size

        # Gradient memory (sharded by ZeRO-2+)
        gradient_mem = params_per_gpu * bytes_per_param
        if config.zero_stage >= 2:
            gradient_mem /= config.dp_size

        # Activation memory
        layers_per_stage = self.num_layers // config.pp_size
        activation_mem = self._activation_memory(
            config, layers_per_stage
        )

        # Apply offloading
        if config.offload_optimizer:
            optimizer_mem = 0
        if config.offload_params:
            model_mem *= 0.1  # Keep only active layer

        total = model_mem + optimizer_mem + gradient_mem + activation_mem

        breakdown = {
            'model_gb': model_mem / (1 << 30),
            'optimizer_gb': optimizer_mem / (1 << 30),
            'gradient_gb': gradient_mem / (1 << 30),
            'activation_gb': activation_mem / (1 << 30),
            'total_gb': total / (1 << 30),
            'available_gb': self.gpu_memory / (1 << 30),
        }

        return total <= self.gpu_memory, breakdown

    def _activation_memory(self,
                           config: TrainingConfig,
                           layers: int) -> float:
        """Compute activation memory per GPU."""

        batch = config.micro_batch_size
        seq = self.seq_length // config.cp_size
        hidden = self.hidden_dim // config.tp_size

        bytes_per_element = self._bytes_per_param(config.precision)

        # Activations per layer
        # Input: batch × seq × hidden
        # Attention: batch × heads × seq × seq
        # FFN: batch × seq × 4*hidden

        # Tensor counts (bytes applied below)
        per_layer = batch * seq * hidden * 2  # Input + output
        per_layer += batch * seq * seq * 2    # Attention scores + probs (approx)
        per_layer += batch * seq * hidden * 4  # FFN intermediate

        total = per_layer * layers * bytes_per_element

        # Activation checkpointing reduces by ~layers factor
        if config.activation_checkpointing:
            total /= layers

        return total

    def _bytes_per_param(self, precision: str) -> int:
        return {'fp32': 4, 'fp16': 2, 'bf16': 2, 'fp8': 1}[precision]
```

### 带宽约束

通信时间不能压过计算：

$$T_{\text{comm}} \leq \alpha \cdot T_{\text{compute}}$$

通常 $\alpha \leq 0.2$（通信应不超过计算的 20%）。

```python
class BandwidthConstraint:
    """Check if configuration has acceptable communication overhead."""

    def __init__(self,
                 model_params: int,
                 flops_per_token: int,
                 hidden_dim: int,
                 seq_length: int,
                 num_layers: int,
                 intra_node_bw: float = 300e9,   # NVLink: 300 GB/s
                 inter_node_bw: float = 100e9,   # IB: 100 GB/s
                 gpu_flops: float = 989e12,     # H100: 989 TFLOPs FP16/BF16 dense
                 max_comm_ratio: float = 0.2):

        self.model_params = model_params
        self.flops_per_token = flops_per_token
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.intra_bw = intra_node_bw
        self.inter_bw = inter_node_bw
        self.gpu_flops = gpu_flops
        self.max_comm_ratio = max_comm_ratio

    def is_feasible(self, config: TrainingConfig) -> Tuple[bool, Dict[str, float]]:
        """Check bandwidth feasibility."""

        bytes_per_param = 2 if config.precision in ['fp16', 'bf16'] else 4

        # TP communication (AllReduce per layer, high frequency)
        # Happens within node (NVLink)
        if config.tp_size > 1:
            layers_per_stage = self.num_layers // config.pp_size
            activation_size = (config.micro_batch_size *
                               self.seq_length *
                               self.hidden_dim *
                               bytes_per_param)
            per_allreduce = 2 * (config.tp_size - 1) / config.tp_size * activation_size
            tp_volume = 2 * layers_per_stage * per_allreduce  # ~2 AllReduces per layer
            tp_time = tp_volume / self.intra_bw
        else:
            tp_time = 0

        # DP communication (AllReduce gradients, once per step)
        # May cross nodes (IB)
        if config.dp_size > 1:
            params_per_rank = self.model_params / (config.tp_size * config.pp_size)
            grad_volume = params_per_rank * bytes_per_param

            # ZeRO reduces communication
            if config.zero_stage >= 1:
                # ReduceScatter + AllGather instead of AllReduce
                dp_volume = grad_volume * 2 * (config.dp_size - 1) / config.dp_size
            else:
                dp_volume = grad_volume * 2 * (config.dp_size - 1) / config.dp_size

            # Determine if inter-node
            gpus_per_node = 8  # Typical
            if config.dp_size <= gpus_per_node // config.tp_size:
                dp_bw = self.intra_bw
            else:
                dp_bw = self.inter_bw

            dp_time = dp_volume / dp_bw
        else:
            dp_time = 0

        # PP communication (point-to-point, per micro-batch)
        if config.pp_size > 1:
            activation_size = (config.micro_batch_size *
                             self.seq_length *
                             self.hidden_dim *
                             bytes_per_param)
            pp_volume = activation_size * 2  # Forward + backward
            pp_time = pp_volume / self.inter_bw
        else:
            pp_time = 0

        total_comm_time = tp_time + dp_time + pp_time

        # Compute time
        tokens = config.micro_batch_size * config.seq_length
        flops = self.flops_per_token * tokens
        compute_time = flops / self.gpu_flops

        comm_ratio = total_comm_time / (compute_time + total_comm_time)

        breakdown = {
            'tp_time_ms': tp_time * 1000,
            'dp_time_ms': dp_time * 1000,
            'pp_time_ms': pp_time * 1000,
            'compute_time_ms': compute_time * 1000,
            'comm_ratio': comm_ratio,
        }

        return comm_ratio <= self.max_comm_ratio, breakdown
```

### 整除约束

配置必须在数学上自洽：

```python
def divisibility_constraints(config: TrainingConfig,
                             num_layers: int,
                             num_attention_heads: int,
                             vocab_size: int,
                             target_gpus: int) -> List[str]:
    """Check all divisibility requirements."""

    violations = []

    # GPU count constraint
    actual_gpus = config.dp_size * config.pp_size * config.tp_size
    if actual_gpus != target_gpus:
        violations.append(
            f"GPU count: {actual_gpus} != target {target_gpus}"
        )

    # Layers divisible by PP stages
    if num_layers % config.pp_size != 0:
        violations.append(
            f"Layers {num_layers} not divisible by PP {config.pp_size}"
        )

    # Attention heads divisible by TP
    if num_attention_heads % config.tp_size != 0:
        violations.append(
            f"Heads {num_attention_heads} not divisible by TP {config.tp_size}"
        )

    # Vocab size divisible by TP (for embedding sharding)
    if vocab_size % config.tp_size != 0:
        violations.append(
            f"Vocab {vocab_size} not divisible by TP {config.tp_size}"
        )

    # Global batch size constraint
    if config.global_batch_size < config.dp_size:
        violations.append(
            f"Global batch {config.global_batch_size} < DP {config.dp_size}"
        )

    return violations
```

## 成本模型

与其逐个运行配置，我们解析式地预测性能。

### 性能模型

```python
class PerformanceModel:
    """Predict training throughput without execution."""

    def __init__(self,
                 model_params: int,
                 hidden_dim: int,
                 num_layers: int,
                 seq_length: int,
                 hardware: HardwareSpec):

        self.model_params = model_params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.hw = hardware

    def predict_step_time(self, config: TrainingConfig) -> StepTimePrediction:
        """Predict time for one training step."""

        # Compute time
        t_compute = self._compute_time(config)

        # Communication times
        t_tp = self._tp_comm_time(config)
        t_dp = self._dp_comm_time(config)
        t_pp = self._pp_comm_time(config)

        # Pipeline bubble
        t_bubble = self._pipeline_bubble(config)

        # Memory operations
        t_memory = self._memory_time(config)

        # Overlap adjustments
        if config.overlap_comm_compute:
            # DP comm overlaps with compute
            t_dp_exposed = max(0, t_dp - t_compute * 0.8)
        else:
            t_dp_exposed = t_dp

        # Total time
        # TP comm is in critical path (per-layer)
        # PP comm happens at stage boundaries
        t_total = t_compute + t_tp + t_dp_exposed + t_bubble + t_memory

        return StepTimePrediction(
            compute=t_compute,
            tp_comm=t_tp,
            dp_comm=t_dp,
            dp_exposed=t_dp_exposed,
            pp_comm=t_pp,
            bubble=t_bubble,
            memory=t_memory,
            total=t_total,
        )

    def _compute_time(self, config: TrainingConfig) -> float:
        """Time for forward + backward compute."""

        tokens_per_step = (config.micro_batch_size *
                          config.seq_length *
                          config.gradient_accumulation_steps)

        # FLOPs for transformer (forward only)
        # Approximately 2 * params * tokens for attention + FFN
        flops_forward = 2 * self.model_params * tokens_per_step

        # Backward is ~2x forward
        flops_total = flops_forward * 3  # Forward + backward

        # Divide by parallelism
        flops_per_gpu = flops_total / (config.tp_size * config.pp_size)

        # Account for efficiency loss from small batches
        efficiency = self._compute_efficiency(config)

        return flops_per_gpu / (self.hw.flops * efficiency)

    def _compute_efficiency(self, config: TrainingConfig) -> float:
        """GPU compute efficiency based on batch size."""

        # Small batches underutilize GPU
        tokens = config.micro_batch_size * config.seq_length
        min_efficient_tokens = 2048  # Empirical

        if tokens >= min_efficient_tokens:
            return 0.5  # Typical for well-optimized kernels
        else:
            return 0.5 * (tokens / min_efficient_tokens) ** 0.5

    def _tp_comm_time(self, config: TrainingConfig) -> float:
        """Tensor parallel AllReduce time."""

        if config.tp_size == 1:
            return 0

        # Two AllReduces per layer: after attention, after FFN
        # Volume per AllReduce: batch × seq × hidden / tp × bytes × 2
        bytes_per = 2 if config.precision in ['fp16', 'bf16'] else 4

        volume_per_layer = (config.micro_batch_size *
                          config.seq_length *
                          self.hidden_dim / config.tp_size *
                          bytes_per * 2)  # AllReduce factor

        layers_per_stage = self.num_layers // config.pp_size
        total_volume = volume_per_layer * 2 * layers_per_stage

        # TP is within node (NVLink)
        return total_volume / self.hw.intra_node_bw

    def _dp_comm_time(self, config: TrainingConfig) -> float:
        """Data parallel gradient synchronization time."""

        if config.dp_size == 1:
            return 0

        bytes_per = 2 if config.precision in ['fp16', 'bf16'] else 4

        # Gradient volume
        params_per_gpu = self.model_params / config.tp_size / config.pp_size
        grad_volume = params_per_gpu * bytes_per

        # AllReduce volume
        allreduce_volume = grad_volume * 2 * (config.dp_size - 1) / config.dp_size

        # Determine bandwidth (may cross nodes)
        if config.dp_size * config.tp_size <= 8:  # Within node
            bw = self.hw.intra_node_bw
        else:
            bw = self.hw.inter_node_bw

        return allreduce_volume / bw

    def _pp_comm_time(self, config: TrainingConfig) -> float:
        """Pipeline parallel P2P communication time."""

        if config.pp_size == 1:
            return 0

        bytes_per = 2 if config.precision in ['fp16', 'bf16'] else 4

        # Activation size at stage boundary
        activation_size = (config.micro_batch_size *
                          config.seq_length *
                          self.hidden_dim *
                          bytes_per)

        # Forward + backward per micro-batch
        volume_per_mb = activation_size * 2

        # Total micro-batches
        num_mb = config.gradient_accumulation_steps

        # P2P is usually across nodes
        return volume_per_mb * num_mb / self.hw.inter_node_bw

    def _pipeline_bubble(self, config: TrainingConfig) -> float:
        """Pipeline bubble overhead."""

        if config.pp_size == 1:
            return 0

        # Bubble fraction
        num_mb = config.gradient_accumulation_steps

        if config.pipeline_schedule == "1F1B":
            # Bubble = (p-1) / (m + p - 1)
            bubble_frac = (config.pp_size - 1) / (num_mb + config.pp_size - 1)
        elif config.pipeline_schedule == "interleaved":
            # Bubble = (p-1) / (m * v + p - 1) where v = virtual stages
            v = config.num_interleaved_stages
            bubble_frac = (config.pp_size - 1) / (num_mb * v + config.pp_size - 1)
        else:  # zero-bubble
            bubble_frac = 0.05  # Small overhead for synchronization

        # Bubble adds this fraction to total time
        compute_time = self._compute_time(config)
        return compute_time * bubble_frac / (1 - bubble_frac)

    def _memory_time(self, config: TrainingConfig) -> float:
        """Overhead from memory operations (offloading, etc.)."""

        overhead = 0

        if config.offload_optimizer:
            # PCIe transfer for optimizer states
            optimizer_size = self.model_params * 12  # FP32 m, v, master
            overhead += optimizer_size / self.hw.pcie_bw

        if config.offload_params:
            # PCIe transfer for parameters
            params_size = self.model_params * 2  # FP16
            overhead += params_size / self.hw.pcie_bw * 2  # To and from

        return overhead

@dataclass
class StepTimePrediction:
    """Breakdown of predicted step time."""

    compute: float
    tp_comm: float
    dp_comm: float
    dp_exposed: float
    pp_comm: float
    bubble: float
    memory: float
    total: float

    def throughput_tokens_per_sec(self,
                                   batch_size: int,
                                   seq_length: int) -> float:
        return batch_size * seq_length / self.total

@dataclass
class HardwareSpec:
    """Hardware specifications for a single GPU."""

    flops: float = 989e12         # Peak TFLOPs (H100 FP16/BF16 dense)
    memory: float = 80e9           # HBM capacity
    intra_node_bw: float = 300e9   # NVLink bandwidth
    inter_node_bw: float = 100e9   # InfiniBand bandwidth
    pcie_bw: float = 32e9          # PCIe bandwidth
```

### 模型校验

成本模型必须对照实测进行校验：

```python
class ModelValidator:
    """Validate cost model against real measurements."""

    def __init__(self, performance_model: PerformanceModel):
        self.model = performance_model
        self.measurements: List[Tuple[TrainingConfig, float]] = []

    def add_measurement(self,
                        config: TrainingConfig,
                        actual_step_time: float) -> None:
        """Add an actual measurement for calibration."""
        self.measurements.append((config, actual_step_time))

    def compute_error(self) -> Dict[str, float]:
        """Compute prediction error statistics."""

        errors = []

        for config, actual in self.measurements:
            predicted = self.model.predict_step_time(config).total
            relative_error = (predicted - actual) / actual
            errors.append(relative_error)

        return {
            'mean_error': np.mean(errors),
            'abs_mean_error': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'std_error': np.std(errors),
        }

    def calibrate(self) -> Dict[str, float]:
        """Fit correction factors to minimize prediction error."""

        from scipy.optimize import minimize

        def objective(params):
            compute_scale, comm_scale = params

            total_error = 0
            for config, actual in self.measurements:
                pred = self.model.predict_step_time(config)
                adjusted = (pred.compute * compute_scale +
                           (pred.tp_comm + pred.dp_exposed + pred.pp_comm) * comm_scale +
                           pred.bubble + pred.memory)
                total_error += (adjusted - actual) ** 2

            return total_error

        result = minimize(objective, [1.0, 1.0], method='Nelder-Mead')

        return {
            'compute_scale': result.x[0],
            'comm_scale': result.x[1],
        }
```

## 搜索算法

### 带剪枝的网格搜索

```python
class PrunedGridSearch:
    """Grid search with early constraint-based pruning."""

    def __init__(self,
                 memory_constraint: MemoryConstraint,
                 bandwidth_constraint: BandwidthConstraint,
                 performance_model: PerformanceModel,
                 target_gpus: int,
                 num_layers: int,
                 num_heads: int):

        self.memory = memory_constraint
        self.bandwidth = bandwidth_constraint
        self.perf_model = performance_model
        self.target_gpus = target_gpus
        self.num_layers = num_layers
        self.num_heads = num_heads

    def search(self,
               dp_choices: List[int],
               pp_choices: List[int],
               tp_choices: List[int],
               zero_choices: List[int],
               mbs_choices: List[int],
               ga_choices: List[int]) -> List[Tuple[TrainingConfig, float]]:
        """Search configuration space with pruning."""

        valid_configs = []
        pruned_counts = {'gpu': 0, 'divisibility': 0, 'memory': 0, 'bandwidth': 0}

        for dp in dp_choices:
            for pp in pp_choices:
                for tp in tp_choices:
                    # Early pruning: GPU count
                    if dp * pp * tp != self.target_gpus:
                        pruned_counts['gpu'] += 1
                        continue

                    # Early pruning: divisibility
                    if self.num_layers % pp != 0:
                        pruned_counts['divisibility'] += 1
                        continue
                    if self.num_heads % tp != 0:
                        pruned_counts['divisibility'] += 1
                        continue

                    for zero in zero_choices:
                        for mbs in mbs_choices:
                            for ga in ga_choices:
                                config = TrainingConfig(
                                    dp_size=dp,
                                    pp_size=pp,
                                    tp_size=tp,
                                    cp_size=1,
                                    ep_size=1,
                                    zero_stage=zero,
                                    activation_checkpointing=True,
                                    offload_optimizer=False,
                                    offload_params=False,
                                    micro_batch_size=mbs,
                                    gradient_accumulation_steps=ga,
                                    seq_length=self.seq_length,
                                    hidden_dim=self.hidden_dim,
                                    pipeline_schedule="1F1B",
                                    num_interleaved_stages=1,
                                    overlap_comm_compute=True,
                                    bucket_size_mb=25,
                                    precision="bf16",
                                )

                                # Memory constraint
                                mem_ok, _ = self.memory.is_feasible(config)
                                if not mem_ok:
                                    pruned_counts['memory'] += 1
                                    continue

                                # Bandwidth constraint
                                bw_ok, _ = self.bandwidth.is_feasible(config)
                                if not bw_ok:
                                    pruned_counts['bandwidth'] += 1
                                    continue

                                # Predict performance
                                pred = self.perf_model.predict_step_time(config)
                                valid_configs.append((config, pred.total))

        print(f"Pruning statistics: {pruned_counts}")
        print(f"Valid configurations: {len(valid_configs)}")

        # Sort by predicted step time
        valid_configs.sort(key=lambda x: x[1])

        return valid_configs
```

### 贝叶斯优化

对代价高昂的评估（真实运行），贝叶斯优化更高效：

```python
class BayesianConfigSearch:
    """Bayesian optimization for configuration search."""

    def __init__(self,
                 objective_fn: Callable[[TrainingConfig], float],
                 constraints: List[Callable[[TrainingConfig], bool]],
                 param_space: Dict[str, List[Any]]):

        self.objective = objective_fn
        self.constraints = constraints
        self.param_space = param_space

        # Gaussian Process surrogate
        self.observations: List[Tuple[TrainingConfig, float]] = []

    def suggest_next(self) -> TrainingConfig:
        """Suggest next configuration to evaluate."""

        if len(self.observations) < 10:
            # Initial random exploration
            return self._random_sample()

        # Fit GP to observations
        X, y = self._prepare_data()
        gp = self._fit_gaussian_process(X, y)

        # Maximize acquisition function
        best_config = None
        best_acquisition = float('-inf')

        for _ in range(1000):  # Random search for acquisition max
            config = self._random_sample()

            if not all(c(config) for c in self.constraints):
                continue

            x = self._config_to_vector(config)
            acq = self._expected_improvement(x, gp, np.min(y))

            if acq > best_acquisition:
                best_acquisition = acq
                best_config = config

        return best_config

    def observe(self, config: TrainingConfig, value: float) -> None:
        """Record observation."""
        self.observations.append((config, value))

    def _expected_improvement(self,
                               x: np.ndarray,
                               gp,
                               y_best: float) -> float:
        """Expected improvement acquisition function."""

        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)

        if sigma < 1e-6:
            return 0.0

        z = (y_best - mu) / sigma
        ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        return ei[0]

    def _random_sample(self) -> TrainingConfig:
        """Generate random valid configuration."""

        while True:
            dp = random.choice(self.param_space['dp_size'])
            pp = random.choice(self.param_space['pp_size'])
            tp = random.choice(self.param_space['tp_size'])

            if dp * pp * tp != self.param_space['target_gpus']:
                continue

            config = TrainingConfig(
                dp_size=dp,
                pp_size=pp,
                tp_size=tp,
                cp_size=1,
                ep_size=1,
                zero_stage=random.choice(self.param_space['zero_stage']),
                activation_checkpointing=True,
                offload_optimizer=False,
                offload_params=False,
                micro_batch_size=random.choice(self.param_space['micro_batch_size']),
                gradient_accumulation_steps=random.choice(
                    self.param_space['gradient_accumulation']
                ),
                seq_length=self.param_space['seq_length'],
                hidden_dim=self.param_space['hidden_dim'],
                pipeline_schedule="1F1B",
                num_interleaved_stages=1,
                overlap_comm_compute=True,
                bucket_size_mb=25,
                precision="bf16",
            )

            return config

    def _config_to_vector(self, config: TrainingConfig) -> np.ndarray:
        """Convert configuration to numerical vector."""

        return np.array([
            np.log2(config.dp_size),
            np.log2(config.pp_size),
            np.log2(config.tp_size),
            config.zero_stage,
            np.log2(config.micro_batch_size),
            np.log2(config.gradient_accumulation_steps),
        ])

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for GP."""

        X = np.array([self._config_to_vector(c) for c, _ in self.observations])
        y = np.array([v for _, v in self.observations])

        return X, y

    def _fit_gaussian_process(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Process to data."""

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(X, y)

        return gp
```

### 进化搜索

对存在许多局部最优的复杂配置空间：

```python
class EvolutionarySearch:
    """Evolutionary algorithm for configuration optimization."""

    def __init__(self,
                 fitness_fn: Callable[[TrainingConfig], float],
                 constraints: List[Callable[[TrainingConfig], bool]],
                 param_space: Dict[str, List[Any]],
                 population_size: int = 50,
                 generations: int = 100):

        self.fitness = fitness_fn
        self.constraints = constraints
        self.param_space = param_space
        self.pop_size = population_size
        self.generations = generations

    def search(self) -> Tuple[TrainingConfig, float]:
        """Run evolutionary search."""

        # Initialize population
        population = self._initialize_population()

        best_config = None
        best_fitness = float('inf')

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for config in population:
                if all(c(config) for c in self.constraints):
                    score = self.fitness(config)
                else:
                    score = float('inf')  # Infeasible

                fitness_scores.append(score)

                if score < best_fitness:
                    best_fitness = score
                    best_config = config

            print(f"Generation {gen}: best fitness = {best_fitness:.4f}")

            # Selection (tournament)
            parents = self._tournament_selection(population, fitness_scores)

            # Crossover
            offspring = self._crossover(parents)

            # Mutation
            offspring = [self._mutate(c) for c in offspring]

            # Replace population
            population = offspring

        return best_config, best_fitness

    def _initialize_population(self) -> List[TrainingConfig]:
        """Create initial random population."""

        population = []

        while len(population) < self.pop_size:
            config = self._random_config()
            if all(c(config) for c in self.constraints):
                population.append(config)

        return population

    def _tournament_selection(self,
                               population: List[TrainingConfig],
                               fitness: List[float],
                               tournament_size: int = 3) -> List[TrainingConfig]:
        """Select parents via tournament selection."""

        parents = []

        for _ in range(self.pop_size):
            tournament_idx = random.sample(range(len(population)), tournament_size)
            winner_idx = min(tournament_idx, key=lambda i: fitness[i])
            parents.append(population[winner_idx])

        return parents

    def _crossover(self, parents: List[TrainingConfig]) -> List[TrainingConfig]:
        """Uniform crossover between pairs of parents."""

        offspring = []
        random.shuffle(parents)

        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                continue

            p1, p2 = parents[i], parents[i + 1]

            # Create children by mixing parent attributes
            child1_attrs = {}
            child2_attrs = {}

            for field in ['dp_size', 'pp_size', 'tp_size', 'zero_stage',
                         'micro_batch_size', 'gradient_accumulation_steps']:
                if random.random() < 0.5:
                    child1_attrs[field] = getattr(p1, field)
                    child2_attrs[field] = getattr(p2, field)
                else:
                    child1_attrs[field] = getattr(p2, field)
                    child2_attrs[field] = getattr(p1, field)

            # Repair GPU constraint
            child1_attrs = self._repair_gpu_constraint(child1_attrs)
            child2_attrs = self._repair_gpu_constraint(child2_attrs)

            offspring.append(self._attrs_to_config(child1_attrs))
            offspring.append(self._attrs_to_config(child2_attrs))

        return offspring[:self.pop_size]

    def _mutate(self, config: TrainingConfig, mutation_rate: float = 0.1) -> TrainingConfig:
        """Mutate configuration with given probability."""

        attrs = {
            'dp_size': config.dp_size,
            'pp_size': config.pp_size,
            'tp_size': config.tp_size,
            'zero_stage': config.zero_stage,
            'micro_batch_size': config.micro_batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
        }

        for field in attrs:
            if random.random() < mutation_rate:
                attrs[field] = random.choice(self.param_space[field])

        # Repair constraints
        attrs = self._repair_gpu_constraint(attrs)

        return self._attrs_to_config(attrs)

    def _repair_gpu_constraint(self, attrs: Dict) -> Dict:
        """Repair configuration to satisfy GPU count constraint."""

        target = self.param_space['target_gpus']

        # Adjust DP to satisfy constraint
        current = attrs['pp_size'] * attrs['tp_size']
        if target % current == 0:
            attrs['dp_size'] = target // current
        else:
            # Find valid combination
            for pp in self.param_space['pp_size']:
                for tp in self.param_space['tp_size']:
                    if target % (pp * tp) == 0:
                        attrs['pp_size'] = pp
                        attrs['tp_size'] = tp
                        attrs['dp_size'] = target // (pp * tp)
                        return attrs

        return attrs

    def _attrs_to_config(self, attrs: Dict) -> TrainingConfig:
        """Convert attributes dict to TrainingConfig."""

        return TrainingConfig(
            dp_size=attrs['dp_size'],
            pp_size=attrs['pp_size'],
            tp_size=attrs['tp_size'],
            cp_size=1,
            ep_size=1,
            zero_stage=attrs['zero_stage'],
            activation_checkpointing=True,
            offload_optimizer=False,
            offload_params=False,
            micro_batch_size=attrs['micro_batch_size'],
            gradient_accumulation_steps=attrs['gradient_accumulation_steps'],
            seq_length=self.param_space['seq_length'],
            hidden_dim=self.param_space['hidden_dim'],
            pipeline_schedule="1F1B",
            num_interleaved_stages=1,
            overlap_comm_compute=True,
            bucket_size_mb=25,
            precision="bf16",
        )

    def _random_config(self) -> TrainingConfig:
        """Generate random configuration."""

        while True:
            dp = random.choice(self.param_space['dp_size'])
            pp = random.choice(self.param_space['pp_size'])
            tp = random.choice(self.param_space['tp_size'])

            if dp * pp * tp == self.param_space['target_gpus']:
                break

        return TrainingConfig(
            dp_size=dp,
            pp_size=pp,
            tp_size=tp,
            cp_size=1,
            ep_size=1,
            zero_stage=random.choice(self.param_space['zero_stage']),
            activation_checkpointing=True,
            offload_optimizer=False,
            offload_params=False,
            micro_batch_size=random.choice(self.param_space['micro_batch_size']),
            gradient_accumulation_steps=random.choice(
                self.param_space['gradient_accumulation']
            ),
            seq_length=self.param_space['seq_length'],
            hidden_dim=self.param_space['hidden_dim'],
            pipeline_schedule="1F1B",
            num_interleaved_stages=1,
            overlap_comm_compute=True,
            bucket_size_mb=25,
            precision="bf16",
        )
```

## 分层搜索

大配置空间适合分层分解：

```python
class HierarchicalSearch:
    """Search parallelism first, then memory optimizations, then micro-batching."""

    def __init__(self,
                 model_spec: ModelSpec,
                 hardware_spec: HardwareSpec,
                 target_gpus: int,
                 target_batch_size: int):

        self.model = model_spec
        self.hardware = hardware_spec
        self.target_gpus = target_gpus
        self.target_batch = target_batch_size

    def search(self) -> TrainingConfig:
        """Hierarchical configuration search."""

        # Level 1: Find valid parallelism dimensions
        parallelism_configs = self._search_parallelism()
        print(f"Level 1: {len(parallelism_configs)} parallelism configs")

        # Level 2: For each parallelism, find memory optimization
        memory_configs = []
        for p_config in parallelism_configs:
            m_configs = self._search_memory_optimizations(p_config)
            memory_configs.extend(m_configs)
        print(f"Level 2: {len(memory_configs)} memory configs")

        # Level 3: For each memory config, optimize micro-batching
        final_configs = []
        for m_config in memory_configs:
            f_config = self._search_micro_batching(m_config)
            if f_config:
                final_configs.append(f_config)
        print(f"Level 3: {len(final_configs)} final configs")

        # Rank by predicted throughput
        final_configs.sort(
            key=lambda c: self._predict_throughput(c),
            reverse=True
        )

        return final_configs[0] if final_configs else None

    def _search_parallelism(self) -> List[TrainingConfig]:
        """Find valid parallelism dimension combinations."""

        valid = []

        for pp in [1, 2, 4, 8, 16, 32]:
            for tp in [1, 2, 4, 8]:
                if self.target_gpus % (pp * tp) != 0:
                    continue

                dp = self.target_gpus // (pp * tp)

                # Check divisibility
                if self.model.num_layers % pp != 0:
                    continue
                if self.model.num_heads % tp != 0:
                    continue

                config = TrainingConfig(
                    dp_size=dp,
                    pp_size=pp,
                    tp_size=tp,
                    cp_size=1,
                    ep_size=1,
                    zero_stage=0,  # Placeholder
                    activation_checkpointing=False,
                    offload_optimizer=False,
                    offload_params=False,
                    micro_batch_size=1,  # Placeholder
                    gradient_accumulation_steps=1,  # Placeholder
                    seq_length=self.model.seq_length,
                    hidden_dim=self.model.hidden_dim,
                    pipeline_schedule="1F1B",
                    num_interleaved_stages=1,
                    overlap_comm_compute=True,
                    bucket_size_mb=25,
                    precision="bf16",
                )

                valid.append(config)

        return valid

    def _search_memory_optimizations(self,
                                      base_config: TrainingConfig) -> List[TrainingConfig]:
        """Find memory optimizations that fit in GPU."""

        valid = []

        for zero in [0, 1, 2, 3]:
            for ckpt in [False, True]:
                for offload_opt in [False, True]:
                    # Skip invalid combinations
                    if offload_opt and zero < 2:
                        continue

                    config = TrainingConfig(
                        dp_size=base_config.dp_size,
                        pp_size=base_config.pp_size,
                        tp_size=base_config.tp_size,
                        cp_size=1,
                        ep_size=1,
                        zero_stage=zero,
                        activation_checkpointing=ckpt,
                        offload_optimizer=offload_opt,
                        offload_params=False,
                        micro_batch_size=1,
                        gradient_accumulation_steps=1,
                        seq_length=base_config.seq_length,
                        hidden_dim=base_config.hidden_dim,
                        pipeline_schedule="1F1B",
                        num_interleaved_stages=1,
                        overlap_comm_compute=True,
                        bucket_size_mb=25,
                        precision="bf16",
                    )

                    # Check if base memory fits
                    if self._check_memory_feasible(config, micro_batch=1):
                        valid.append(config)

        return valid

    def _search_micro_batching(self,
                                base_config: TrainingConfig) -> Optional[TrainingConfig]:
        """Find optimal micro-batch size and gradient accumulation."""

        best_config = None
        best_throughput = 0

        for mbs in [1, 2, 4, 8, 16, 32, 64]:
            if not self._check_memory_feasible(base_config, micro_batch=mbs):
                break  # Larger won't fit either

            # Calculate gradient accumulation to achieve target batch
            per_dp_batch = self.target_batch // base_config.dp_size
            ga = per_dp_batch // mbs

            if ga < 1:
                continue

            config = TrainingConfig(
                dp_size=base_config.dp_size,
                pp_size=base_config.pp_size,
                tp_size=base_config.tp_size,
                cp_size=1,
                ep_size=1,
                zero_stage=base_config.zero_stage,
                activation_checkpointing=base_config.activation_checkpointing,
                offload_optimizer=base_config.offload_optimizer,
                offload_params=base_config.offload_params,
                micro_batch_size=mbs,
                gradient_accumulation_steps=ga,
                seq_length=base_config.seq_length,
                hidden_dim=base_config.hidden_dim,
                pipeline_schedule="1F1B",
                num_interleaved_stages=1,
                overlap_comm_compute=True,
                bucket_size_mb=25,
                precision="bf16",
            )

            throughput = self._predict_throughput(config)

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config

        return best_config

    def _check_memory_feasible(self,
                                config: TrainingConfig,
                                micro_batch: int) -> bool:
        """Check if configuration fits in GPU memory."""

        # Simplified memory check
        bytes_per_param = 2  # BF16

        # Model params per GPU
        params = self.model.params / config.tp_size / config.pp_size

        # With ZeRO-3, further divide by DP
        if config.zero_stage >= 3:
            params /= config.dp_size

        model_mem = params * bytes_per_param

        # Optimizer (12 bytes per param for Adam)
        opt_mem = params * 12
        if config.zero_stage >= 1:
            opt_mem /= config.dp_size
        if config.offload_optimizer:
            opt_mem = 0

        # Activations
        layers_per_stage = self.model.num_layers // config.pp_size
        act_per_layer = micro_batch * self.model.seq_length * self.model.hidden_dim * 2
        act_mem = act_per_layer * layers_per_stage

        if config.activation_checkpointing:
            act_mem /= layers_per_stage

        total = model_mem + opt_mem + act_mem

        return total < self.hardware.memory * 0.9  # 90% threshold

    def _predict_throughput(self, config: TrainingConfig) -> float:
        """Predict tokens per second."""

        perf_model = PerformanceModel(
            self.model.params,
            self.model.hidden_dim,
            self.model.num_layers,
            self.model.seq_length,
            self.hardware
        )

        step_time = perf_model.predict_step_time(config).total
        tokens_per_step = config.global_batch_size * self.model.seq_length

        return tokens_per_step / step_time
```

## 自动调优集成

### DeepSpeed 自动调优器风格

```python
class AutoTuner:
    """Automatic configuration tuning with profiling."""

    def __init__(self,
                 model_fn: Callable[[], nn.Module],
                 dataset: Dataset,
                 target_gpus: int,
                 profile_steps: int = 10):

        self.model_fn = model_fn
        self.dataset = dataset
        self.target_gpus = target_gpus
        self.profile_steps = profile_steps

        self.results: Dict[str, Tuple[TrainingConfig, float]] = {}

    def run_profiling(self, config: TrainingConfig) -> float:
        """Run actual profiling with given configuration."""

        try:
            # Initialize model with config
            model = self._setup_distributed_model(config)
            optimizer = self._setup_optimizer(model, config)
            dataloader = self._setup_dataloader(config)

            # Warmup
            for _ in range(3):
                batch = next(iter(dataloader))
                self._train_step(model, optimizer, batch)

            # Profile
            start_time = time.time()
            for i, batch in enumerate(dataloader):
                if i >= self.profile_steps:
                    break
                self._train_step(model, optimizer, batch)

            elapsed = time.time() - start_time
            step_time = elapsed / self.profile_steps

            # Cleanup
            del model, optimizer
            torch.cuda.empty_cache()

            return step_time

        except RuntimeError as e:
            if "out of memory" in str(e):
                return float('inf')  # OOM = infeasible
            raise

    def auto_tune(self,
                  max_trials: int = 50,
                  strategy: str = "bayesian") -> TrainingConfig:
        """Run automatic tuning."""

        param_space = {
            'dp_size': [2**i for i in range(int(np.log2(self.target_gpus)) + 1)],
            'pp_size': [1, 2, 4, 8],
            'tp_size': [1, 2, 4, 8],
            'zero_stage': [0, 1, 2, 3],
            'micro_batch_size': [1, 2, 4, 8, 16],
            'gradient_accumulation': [1, 2, 4, 8, 16, 32],
            'target_gpus': self.target_gpus,
        }

        if strategy == "bayesian":
            searcher = BayesianConfigSearch(
                objective_fn=self.run_profiling,
                constraints=[self._basic_constraints],
                param_space=param_space
            )

            for trial in range(max_trials):
                config = searcher.suggest_next()
                step_time = self.run_profiling(config)
                searcher.observe(config, step_time)

                print(f"Trial {trial}: step_time = {step_time:.4f}s")

                self.results[self._config_key(config)] = (config, step_time)

        # Return best
        best_key = min(self.results, key=lambda k: self.results[k][1])
        return self.results[best_key][0]

    def _basic_constraints(self, config: TrainingConfig) -> bool:
        """Basic constraint check."""

        if config.dp_size * config.pp_size * config.tp_size != self.target_gpus:
            return False

        return True

    def _config_key(self, config: TrainingConfig) -> str:
        """Create unique key for configuration."""

        return (f"dp{config.dp_size}_pp{config.pp_size}_tp{config.tp_size}_"
                f"z{config.zero_stage}_mbs{config.micro_batch_size}_"
                f"ga{config.gradient_accumulation_steps}")

    def _setup_distributed_model(self, config: TrainingConfig) -> nn.Module:
        """Set up model with given parallelism configuration."""
        # Implementation depends on framework (DeepSpeed, Megatron, etc.)
        raise NotImplementedError

    def _setup_optimizer(self, model: nn.Module, config: TrainingConfig):
        """Set up optimizer with proper ZeRO configuration."""
        raise NotImplementedError

    def _setup_dataloader(self, config: TrainingConfig):
        """Set up dataloader with proper batch sizing."""
        raise NotImplementedError

    def _train_step(self, model, optimizer, batch):
        """Execute one training step."""
        raise NotImplementedError
```

## 配置推荐

### 基于规则的初始猜测

在昂贵的搜索之前，先用启发式给出一个好的起点：

```python
class ConfigurationAdvisor:
    """Rule-based configuration recommendations."""

    def recommend(self,
                  model_params: int,
                  hidden_dim: int,
                  num_layers: int,
                  num_heads: int,
                  seq_length: int,
                  target_gpus: int,
                  target_batch_size: int,
                  gpu_memory_gb: float = 80.0) -> TrainingConfig:
        """Recommend initial configuration based on heuristics."""

        # Rule 1: TP size based on model width
        # Use TP if model is too wide for single GPU
        bytes_per_param = 16  # fp16 params+grads + fp32 master/m/v (AdamW)
        params_per_gpu = model_params * bytes_per_param
        if params_per_gpu > gpu_memory_gb * 1e9:
            tp_size = min(8, self._ceil_power_of_2(
                params_per_gpu / (gpu_memory_gb * 1e9 * 0.5)
            ))
        else:
            tp_size = 1

        # Ensure divisibility
        while num_heads % tp_size != 0 and tp_size > 1:
            tp_size //= 2

        # Rule 2: PP size based on layers and remaining GPUs
        remaining_gpus = target_gpus // tp_size

        if num_layers >= 64 and remaining_gpus >= 8:
            pp_size = min(8, num_layers // 8)
        elif num_layers >= 32 and remaining_gpus >= 4:
            pp_size = 4
        else:
            pp_size = 1

        # Ensure divisibility
        while num_layers % pp_size != 0 and pp_size > 1:
            pp_size //= 2

        # Rule 3: DP fills the rest
        dp_size = target_gpus // (tp_size * pp_size)

        # Rule 4: ZeRO based on model size and DP
        bytes_per_param = 2
        params_per_dp = model_params / tp_size / pp_size
        mem_per_dp = params_per_dp * (bytes_per_param + 12)  # Model + optimizer

        if mem_per_dp > gpu_memory_gb * 1e9 * 0.8:
            if dp_size >= 8:
                zero_stage = 3
            elif dp_size >= 4:
                zero_stage = 2
            else:
                zero_stage = 1
        else:
            zero_stage = 1  # ZeRO-1 is usually beneficial

        # Rule 5: Micro-batch size
        # Start small, increase until memory limit
        micro_batch_size = 1
        activation_mem = micro_batch_size * seq_length * hidden_dim * 2
        layers_per_stage = num_layers // pp_size

        while (activation_mem * layers_per_stage <
               gpu_memory_gb * 1e9 * 0.2 and micro_batch_size < 32):
            micro_batch_size *= 2
            activation_mem = micro_batch_size * seq_length * hidden_dim * 2

        micro_batch_size //= 2  # Back off one step

        # Rule 6: Gradient accumulation
        per_dp_batch = target_batch_size // dp_size
        gradient_accumulation = max(1, per_dp_batch // micro_batch_size)

        # Rule 7: Activation checkpointing if large model
        activation_checkpointing = model_params > 1e9

        return TrainingConfig(
            dp_size=dp_size,
            pp_size=pp_size,
            tp_size=tp_size,
            cp_size=1,
            ep_size=1,
            zero_stage=zero_stage,
            activation_checkpointing=activation_checkpointing,
            offload_optimizer=False,
            offload_params=False,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            pipeline_schedule="interleaved" if pp_size >= 4 else "1F1B",
            num_interleaved_stages=2 if pp_size >= 4 else 1,
            overlap_comm_compute=True,
            bucket_size_mb=25,
            precision="bf16",
        )

    def _ceil_power_of_2(self, x: float) -> int:
        """Round up to nearest power of 2."""
        return 2 ** int(np.ceil(np.log2(max(1, x))))
```

### 案例研究

#### GPT-3 175B 跑在 1024 块 A100 上

```python
# Model specification
model_params = 175e9
hidden_dim = 12288
num_layers = 96
num_heads = 96
seq_length = 2048
vocab_size = 50257

# Hardware
target_gpus = 1024
gpu_memory = 80  # GB

# Recommended configuration
config = TrainingConfig(
    dp_size=64,
    pp_size=8,
    tp_size=2,
    cp_size=1,
    ep_size=1,
    zero_stage=1,
    activation_checkpointing=True,
    offload_optimizer=False,
    offload_params=False,
    micro_batch_size=2,
    gradient_accumulation_steps=8,
    seq_length=seq_length,
    hidden_dim=hidden_dim,
    pipeline_schedule="interleaved",
    num_interleaved_stages=2,
    overlap_comm_compute=True,
    bucket_size_mb=25,
    precision="bf16",
)

# Global batch size: 64 * 2 * 8 = 1024 samples
# Per-GPU memory: ~60GB (fits in 80GB)
# Predicted efficiency: ~50% MFU
```

#### LLaMA 70B 跑在 128 块 H100 上

```python
# Model specification
model_params = 70e9
hidden_dim = 8192
num_layers = 80
num_heads = 64
seq_length = 8192  # Long context
vocab_size = 32000

# Hardware
target_gpus = 128
gpu_memory = 80

# Recommended configuration
config = TrainingConfig(
    dp_size=32,
    pp_size=4,
    tp_size=1,  # GQA reduces TP need
    cp_size=1,
    ep_size=1,
    zero_stage=1,
    activation_checkpointing=True,
    offload_optimizer=False,
    offload_params=False,
    micro_batch_size=1,
    gradient_accumulation_steps=4,
    seq_length=seq_length,
    hidden_dim=hidden_dim,
    pipeline_schedule="1F1B",
    num_interleaved_stages=1,
    overlap_comm_compute=True,
    bucket_size_mb=25,
    precision="bf16",
)

# Global batch size: 32 * 1 * 4 = 128 samples
# Tokens per batch: 128 * 8192 = 1M tokens
```

## 练习题

1. **配置计数**：对 512 块 GPU，DP ∈ {1, 2, 4, ..., 512}、PP ∈ {1, 2, 4, 8}、TP ∈ {1, 2, 4, 8}，有多少种合法的 (DP, PP, TP) 组合？

??? success "解答"
    **约束：**

    $$DP \times PP \times TP = 512$$

    **可取的值：**

    - DP ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}（10 个 2 的幂）
    - PP ∈ {1, 2, 4, 8}
    - TP ∈ {1, 2, 4, 8}

    **枚举合法组合：**

    对每个 (PP, TP) 对，求 DP = 512 / (PP × TP)：

    | PP | TP | PP × TP | DP = 512/(PP×TP) | 合法？ |
    |----|----|---------|-----------------:|--------|
    | 1 | 1 | 1 | 512 | ✓ |
    | 1 | 2 | 2 | 256 | ✓ |
    | 1 | 4 | 4 | 128 | ✓ |
    | 1 | 8 | 8 | 64 | ✓ |
    | 2 | 1 | 2 | 256 | ✓ |
    | 2 | 2 | 4 | 128 | ✓ |
    | 2 | 4 | 8 | 64 | ✓ |
    | 2 | 8 | 16 | 32 | ✓ |
    | 4 | 1 | 4 | 128 | ✓ |
    | 4 | 2 | 8 | 64 | ✓ |
    | 4 | 4 | 16 | 32 | ✓ |
    | 4 | 8 | 32 | 16 | ✓ |
    | 8 | 1 | 8 | 64 | ✓ |
    | 8 | 2 | 16 | 32 | ✓ |
    | 8 | 4 | 32 | 16 | ✓ |
    | 8 | 8 | 64 | 8 | ✓ |

    全部 16 种组合都得到合法的 DP 值（都是 ≤ 512 的 2 的幂）。

    **答案：**

    $$\boxed{16 \text{ valid configurations}}$$

    **验证：** $|PP| \times |TP| = 4 \times 4 = 16$

    所有组合都合法，因为：
    - PP × TP ∈ {1, 2, 4, 8, 16, 32, 64} ⊆ 512 的约数
    - 所有得到的 DP 值都是 ≤ 512 的 2 的幂

2. **内存约束**：一个 30B 参数模型，40 层，隐层维 7168。使用 TP=4、PP=4、ZeRO-1，在 DP=16 的集群上。计算每 GPU 内存（假设 BF16 + FP32 优化器）。

??? success "解答"
    **已知：**

    - Ψ = 30B 参数
    - L = 40 层
    - H = 7168 隐层维
    - TP = 4, PP = 4, DP = 16
    - GPU 总数 = 4 × 4 × 16 = 256

    **参数分布：**

    $$\Psi_{GPU} = \frac{\Psi}{TP \times PP} = \frac{30B}{4 \times 4} = 1.875B \text{ parameters/GPU}$$

    **静态内存组成：**

    | 组成 | 公式 | 每 GPU 内存 |
    |------|------|-------------|
    | 参数（BF16） | $2 \times \Psi_{GPU}$ | 3.75 GB |
    | 梯度（BF16） | $2 \times \Psi_{GPU}$ | 3.75 GB |
    | 优化器（FP32） | $12 \times \Psi_{GPU} / DP$（ZeRO-1） | 1.41 GB |

    **ZeRO-1 下的优化器内存：**

    ZeRO-1 在 DP 维度上分片优化器状态（optimizer states）：

    $$M_{opt}^{ZeRO1} = \frac{12 \times 1.875B}{16} = 1.41 \text{ GB}$$

    **激活内存：**

    每层激活（TP=4）：

    $$M_{act}^{layer} = \frac{BSH \times 34}{TP}$$

    假设 B=4、S=4096：

    $$M_{act}^{layer} = \frac{4 \times 4096 \times 7168 \times 34}{4} = 997 \text{ MB/layer}$$

    每个 PP 阶段的层数：$40/4 = 10$ 层

    使用激活检查点（每层都检查）：

    $$M_{act}^{total} \approx 2 \times 997 = 1.99 \text{ GB}$$

    不使用检查点：

    $$M_{act}^{total} = 10 \times 997 = 9.97 \text{ GB}$$

    **每 GPU 总内存：**

    | 组成 | 内存 |
    |------|------|
    | 参数（BF16） | 3.75 GB |
    | 梯度（BF16） | 3.75 GB |
    | 优化器（ZeRO-1） | 1.41 GB |
    | 激活（检查点） | ~2 GB |
    | 工作缓冲区 | ~2 GB |
    | **合计** | $\boxed{\sim 13 \text{ GB}}$ |

    不使用检查点：

    | 组成 | 内存 |
    |------|------|
    | 参数（BF16） | 3.75 GB |
    | 梯度（BF16） | 3.75 GB |
    | 优化器（ZeRO-1） | 1.41 GB |
    | 激活 | ~10 GB |
    | 工作缓冲区 | ~2 GB |
    | **合计** | $\boxed{\sim 21 \text{ GB}}$ |

    两者都能轻松放进 80GB 的 H100。

3. **搜索空间缩减**：你有 10,000 种配置。应用 GPU 数量约束去掉 80%，整除约束去掉剩余部分的 50%，内存约束去掉剩余部分的 30%。还剩多少种？

??? success "解答"
    **初始配置数：** 10,000

    **第 1 阶段：GPU 数量约束**

    去掉 80%：

    $$N_1 = 10000 \times (1 - 0.80) = 10000 \times 0.20 = 2000$$

    **第 2 阶段：整除约束**

    去掉剩余部分的 50%：

    $$N_2 = 2000 \times (1 - 0.50) = 2000 \times 0.50 = 1000$$

    **第 3 阶段：内存约束**

    去掉剩余部分的 30%：

    $$N_3 = 1000 \times (1 - 0.30) = 1000 \times 0.70 = 700$$

    **答案：**

    $$\boxed{700 \text{ configurations remain}}$$

    **累计缩减：**

    $$\text{Survival rate} = 0.20 \times 0.50 \times 0.70 = 0.07 = 7\%$$

    **可视化：**

    | 阶段 | 输入配置 | 去掉 | 剩余配置 |
    |------|----------|------|----------|
    | 初始 | - | - | 10,000 |
    | GPU 数量 | 10,000 | 8,000 (80%) | 2,000 |
    | 整除 | 2,000 | 1,000 (50%) | 1,000 |
    | 内存 | 1,000 | 300 (30%) | 700 |

    这说明约束剪枝为何如此有效——即便中等的淘汰率，复合起来也能消除 93% 的搜索空间。

4. **贝叶斯 vs 网格**：网格搜索在 100 小时内评估 1000 种配置。贝叶斯优化用 50 次评估就达到同样的最优结果。若每次评估耗时 6 分钟，计算节省的时间。

??? success "解答"
    **网格搜索：**

    - 评估次数：1,000
    - 总时间：100 小时

    每次评估时间：$\frac{100 \times 60}{1000} = 6$ 分钟 ✓（与题设一致）

    **贝叶斯优化：**

    - 所需评估次数：50
    - 每次评估时间：6 分钟

    总时间：

    $$T_{bayes} = 50 \times 6 = 300 \text{ minutes} = \boxed{5 \text{ hours}}$$

    **节省时间：**

    $$\text{Savings} = 100 - 5 = \boxed{95 \text{ hours}}$$

    $$\text{Speedup} = \frac{100}{5} = \boxed{20\times}$$

    **效率对比：**

    | 方法 | 评估次数 | 时间 | 结果 |
    |------|----------|------|------|
    | 网格搜索 | 1,000 | 100 小时 | 最优配置 |
    | 贝叶斯优化 | 50 | 5 小时 | 同样的最优配置 |
    | **节省** | **95%** | **95%** | - |

    **贝叶斯为何如此有效：**

    1. **利用（exploitation）**：聚焦于有希望的区域
    2. **探索（exploration）**：在不确定区域采样
    3. **模型引导**：从每次评估中学习

    **按 4 美元/GPU 小时的成本分析：**

    假设每次评估用 64 块 GPU：

    | 方法 | GPU 小时 | 成本 |
    |------|----------|------|
    | 网格 | 64 × 100 = 6,400 | 25,600 美元 |
    | 贝叶斯 | 64 × 5 = 320 | 1,280 美元 |
    | **节省** | 6,080 | **24,320 美元** |

5. **启发式验证**：用 ConfigurationAdvisor 为一个 13B 参数模型（40 层、h=5120、40 头）在 64 块 GPU 上生成配置，然后用 MemoryConstraint 验证。

??? success "解答"
    **模型规格：**

    - Ψ = 13B 参数
    - L = 40 层
    - H = 5120 隐层维
    - A = 40 个注意力头
    - GPU 数 = 64

    **ConfigurationAdvisor 的启发式：**

    **第 1 步：选择 TP**

    模型规模 → TP 启发式：
    - 13B 属于中等规模
    - 隐层维 5120 能被 8 整除
    - 优先 TP ≤ 8（节点内）

    $$\boxed{TP = 4}$$（让层保持合理大小）

    **第 2 步：选择 PP**

    TP 之后，剩余 GPU：64/4 = 16

    不同 PP 下的每阶段层数：
    - PP=1：每阶段 40 层（无流水线）
    - PP=2：每阶段 20 层
    - PP=4：每阶段 10 层
    - PP=8：每阶段 5 层

    均衡：40 必须被整除 → PP ∈ {1, 2, 4, 5, 8, 10, 20, 40}

    受 64/TP 整除性约束：PP ∈ {1, 2, 4, 8, 16}

    $$\boxed{PP = 4}$$（每阶段 10 层，均衡良好）

    **第 3 步：选择 DP**

    $$DP = \frac{64}{TP \times PP} = \frac{64}{4 \times 4} = \boxed{4}$$

    **生成的配置：**

    | 维度 | 取值 |
    |------|------|
    | TP | 4 |
    | PP | 4 |
    | DP | 4 |
    | 合计 | 64 ✓ |

    **内存验证：**

    每 GPU 参数：

    $$\Psi_{GPU} = \frac{13B}{TP \times PP} = \frac{13B}{16} = 812.5M$$

    **内存分解：**

    | 组成 | 公式 | 内存 |
    |------|------|------|
    | 参数（BF16） | $2 \times 812.5M$ | 1.63 GB |
    | 梯度（BF16） | $2 \times 812.5M$ | 1.63 GB |
    | 优化器（FP32） | $12 \times 812.5M$ | 9.75 GB |
    | **静态合计** | | **13.0 GB** |

    使用 ZeRO-1（DP=4）：

    $$M_{opt}^{ZeRO1} = 9.75 / 4 = 2.44 \text{ GB}$$
    ZeRO-1 下的静态内存：5.7 GB

    **激活内存（B=8，S=4096）：**

    $$M_{act}^{layer} = \frac{8 \times 4096 \times 5120 \times 34}{4} = 1.43 \text{ GB/layer}$$

    每阶段层数：10

    使用检查点：

    $$M_{act} \approx 2 \times 1.43 = 2.86 \text{ GB}$$

    **每 GPU 总量：**

    | 组成 | 内存 |
    |------|------|
    | 静态（ZeRO-1） | 5.7 GB |
    | 激活（检查点） | ~3 GB |
    | 工作缓冲区 | ~2 GB |
    | **合计** | **~11 GB** |

    **验证结果：**

    $$11 \text{ GB} < 80 \text{ GB (H100)} \quad \boxed{\checkmark \text{ PASS}}$$

    还有充裕的余量来提高批大小。

6. **流水线气泡（pipeline bubble）权衡**：比较配置 (PP=4, GA=16) 与 (PP=8, GA=32) 的气泡比例。GA 达到多少时，更高的 PP 才变得有利？

??? success "解答"
    **气泡比例公式：**

    $$\text{Bubble} = \frac{PP - 1}{GA + PP - 1}$$

    **配置 1：PP=4，GA=16**

    $$\text{Bubble}_1 = \frac{4 - 1}{16 + 4 - 1} = \frac{3}{19} = 15.8\%$$

    **配置 2：PP=8，GA=32**

    $$\text{Bubble}_2 = \frac{8 - 1}{32 + 8 - 1} = \frac{7}{39} = 17.9\%$$

    **对比：**

    | 配置 | PP | GA | 气泡比例 |
    |------|----|----|----------|
    | 1 | 4 | 16 | 15.8% |
    | 2 | 8 | 32 | 17.9% |

    配置 1（PP=4，GA=16）的气泡比例更低。

    **PP=8 何时变得有利？**

    要让 PP=8 的气泡低于 PP=4（GA=16）：

    $$\frac{7}{GA + 7} < \frac{3}{19}$$

    求解：

    $$7 \times 19 < 3 \times (GA + 7)$$

    $$133 < 3 \cdot GA + 21$$

    $$112 < 3 \cdot GA$$

    $$GA > 37.3$$

    $$\boxed{GA \geq 38}$$ 时 PP=8 的气泡才低于 PP=4/GA=16

    **在 GA=38 处验证：**

    $$\text{Bubble}_{PP=8,GA=38} = \frac{7}{38 + 7} = \frac{7}{45} = 15.6\%$$

    对比 PP=4、GA=16 的 15.8% ✓

    **盈亏平衡的通用公式：**

    要使 PP₂ 的气泡与 PP₁（GA₁）相同：

    $$\frac{PP_2 - 1}{GA_2 + PP_2 - 1} = \frac{PP_1 - 1}{GA_1 + PP_1 - 1}$$

    解出 GA₂：

    $$GA_2 = \frac{(PP_2 - 1)(GA_1 + PP_1 - 1)}{PP_1 - 1} - (PP_2 - 1)$$

    **权衡分析：**

    | PP | 10% 气泡所需 GA | 5% 气泡所需 GA |
    |----|-------------------|------------------|
    | 4 | 27 | 57 |
    | 8 | 63 | 133 |
    | 16 | 135 | 285 |

    **何时偏好更高的 PP：**

    - 更高 PP 能降低每 GPU 内存（更多分片）
    - 但需要更多梯度累积（Gradient Accumulation）来维持效率
    - 更高 PP 的阶段间通信开销也更大

    **小结：**

    $$\boxed{GA \geq 38 \text{ for PP=8 to beat PP=4/GA=16}}$$

## 关键要点

1. **配置空间是指数级的**：需要系统搜索，而非试错。

2. **约束剪枝非常强势**：内存、带宽和整除约束会在早期消除大部分配置。

3. **成本模型支持快速搜索**：解析式预测性能比逐个运行配置便宜得多。

4. **分层搜索很有效**：先优化并行维度，再优化内存优化项，最后优化微批（micro-batch）。

5. **贝叶斯优化胜过低网格搜索**：对昂贵的评估，自适应方法用更少的试验找到最优。

6. **启发式给出好的起点**：基于规则的推荐能很快接近最优的 20% 以内。

7. **校验必不可少**：成本模型必须对照实测进行校准。

8. **搜索是值得的**：2 倍的吞吐提升，在大规模下能省下数月的训练时间。