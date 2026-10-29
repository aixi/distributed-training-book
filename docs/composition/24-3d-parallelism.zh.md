---
title: "三维并行"
subtitle: "经典组合：DP × TP × PP"
---

<div class="chapter-opener" markdown>
没有任何单一并行策略能扩展到数千块 GPU。数据并行（Data Parallelism）浪费内存。张量并行（Tensor Parallelism）淹没在通信里。流水线并行（Pipeline Parallelism）产生气泡（pipeline bubble）。解决方案：把三者组合起来，让各自在最优的规模上工作。
</div>

<div class="investigation-question" markdown>
**问题**：你有 1024 块 GPU 和一个 175B 参数的模型。只用 DP：每块 GPU 都需要完整模型——不可能。只用 TP：1024 路切分会给每层带来 1023 道通信屏障。只用 PP：1023/1024 = 99.9% 的气泡开销。你如何组合它们以实现高效训练？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 14–16 章](../parallelism/14-data-parallelism-associativity.md)（DP、TP、PP 基础；另见[第 15 章](../parallelism/15-tensor-parallelism-linearity.md)与[第 16 章](../parallelism/16-pipeline-parallelism-separability.md)）、[第 23 章](23-device-mesh.md)（设备网格抽象）

    **核心洞见**：每种并行策略都有一个最优规模。TP 在节点内效果最好（快速 NVLink）。PP 跨节点、通信量极小。DP 扩展最外层维度。经典的三维组合：节点内 TP、跨节点 PP、跨副本 DP。

## 单一策略的局限

每种并行策略都有一个能发挥其优势的自然规模：

### 数据并行

**优势**：计算的完美扩展。

**劣势**：内存冗余——每块 GPU 都存储完整模型。

$$\text{Memory per GPU} = M_{\text{model}} + M_{\text{optimizer}} + M_{\text{activation}}$$

对于混合精度（mixed precision）下使用 Adam 优化器的 175B 模型：

$$M_{\text{model}} + M_{\text{optimizer}} = 175\text{B} \times (2 + 2 + 8 + 4) = 2.80\text{ TB}$$

这里：2 字节（fp16 参数）+ 2 字节（fp16 梯度）+ 8 字节（fp32 的 Adam $m$ + $v$）+ 4 字节（fp32 主权重）= 每参数 16 字节，与第 19 章的 $16\Psi$ 规则一致。

没有哪块 GPU 放得下。

### 张量并行

**优势**：线性地切分模型内存。

**劣势**：通信量随隐藏维度增长。

对于 $P$ 块 GPU，每个线性层都需要：

$$\text{Communication} = O(P \times H \times \text{batch}) \text{ per forward and backward}$$

当 $P = 1024$ 时，AllReduce 延迟占据主导：

$$T_{\text{allreduce}} = 2(P-1) \alpha + \frac{2(P-1)}{P\beta} \cdot n$$

$2046\alpha$ 的延迟项会摧毁吞吐。

### 流水线并行

**优势**：通信量极小（只在阶段边界）。

**劣势**：流水线气泡。

有 $P$ 个阶段、微批数为 $M$ 时：

$$\text{Bubble fraction} = \frac{P - 1}{P + M - 1}$$

当 $P = 1024$、$M = 4$ 时：

$$\text{Bubble fraction} = \frac{1023}{1027} = 99.6\%$$

## 三维组合

洞见是：不同策略有不同的最优规模。把它们分层组合起来。

### 三个维度

```
                        Global (1024 GPUs)
                              │
              ┌───────────────┴───────────────┐
              │           DP = 32             │
              │    (replication groups)       │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │           PP = 8              │
              │     (pipeline stages)         │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │           TP = 4              │
              │    (tensor sharding)          │
              └───────────────┘
```

**GPU 总数**：$N = \text{DP} \times \text{PP} \times \text{TP} = 32 \times 8 \times 4 = 1024$

### 为什么采用这个顺序？

**TP 在最内层**：需要最低延迟（节点内 NVLink）。

**PP 在中间**：中等带宽，可以跨节点。

**DP 在最外层**：最能容忍带宽（大梯度张量）。

```
Node 0 (8 GPUs):                Node 1 (8 GPUs):
┌─────────┬─────────┐           ┌─────────┬─────────┐
│ Stage 0 │ Stage 1 │           │ Stage 0 │ Stage 1 │
│ TP=0-3  │ TP=0-3  │           │ TP=0-3  │ TP=0-3  │
└─────────┴─────────┘           └─────────┴─────────┘
     │         │                     │         │
     └─────────┼─────────────────────┘         │
               │        DP sync                │
               └───────────────────────────────┘
```

## 映射到设备网格

借助第 23 章的设备网格（Device Mesh）抽象，三维并行就变成了一个网格配置：

```python
from typing import Tuple, Optional
import torch
import torch.distributed as dist
import numpy as np

class ThreeDMesh:
    """Device mesh configured for 3D parallelism."""

    def __init__(
        self,
        dp_size: int,
        pp_size: int,
        tp_size: int,
        device_ids: Optional[np.ndarray] = None
    ):
        self.dp_size = dp_size
        self.pp_size = pp_size
        self.tp_size = tp_size

        world_size = dp_size * pp_size * tp_size

        if device_ids is None:
            device_ids = np.arange(world_size)

        # Shape: (DP, PP, TP)
        self.mesh = device_ids.reshape(dp_size, pp_size, tp_size)

        # Create process groups
        self._create_groups()

    def _create_groups(self):
        """Create DP, PP, and TP process groups."""
        rank = dist.get_rank()

        # Find my coordinates
        coords = np.argwhere(self.mesh == rank)[0]
        self.dp_rank = coords[0]
        self.pp_rank = coords[1]
        self.tp_rank = coords[2]

        # TP group: same DP, same PP, vary TP
        tp_ranks = self.mesh[self.dp_rank, self.pp_rank, :].tolist()
        self.tp_group = dist.new_group(tp_ranks)

        # PP group: same DP, vary PP, same TP
        pp_ranks = self.mesh[self.dp_rank, :, self.tp_rank].tolist()
        self.pp_group = dist.new_group(pp_ranks)

        # DP group: vary DP, same PP, same TP
        dp_ranks = self.mesh[:, self.pp_rank, self.tp_rank].tolist()
        self.dp_group = dist.new_group(dp_ranks)

    def get_coordinates(self) -> Tuple[int, int, int]:
        """Return (dp_rank, pp_rank, tp_rank)."""
        return (self.dp_rank, self.pp_rank, self.tp_rank)

    def get_pipeline_neighbors(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (prev_rank, next_rank) in pipeline."""
        prev_rank = None
        next_rank = None

        if self.pp_rank > 0:
            prev_rank = self.mesh[self.dp_rank, self.pp_rank - 1, self.tp_rank]

        if self.pp_rank < self.pp_size - 1:
            next_rank = self.mesh[self.dp_rank, self.pp_rank + 1, self.tp_rank]

        return (prev_rank, next_rank)
```

### 进程组对应定理（三维情形）

**定理**：在形状为 $(D, P, T)$ 的三维网格中，进程组满足：

1. 每个 TP 组有 $T$ 个成员
2. 每个 PP 组有 $P$ 个成员
3. 每个 DP 组有 $D$ 个成员
4. 组总数：$D \cdot P$ 个 TP 组、$D \cdot T$ 个 PP 组、$P \cdot T$ 个 DP 组
5. 每个设备在每种类型中恰好属于一个组

**证明**：每个 TP 组由固定 $(d, p)$ 确定，共有 $D \cdot P$ 个。PP 与 DP 同理。位于 $(d, p, t)$ 的设备属于：

- TP 组 $(d, p, *)$
- PP 组 $(d, *, t)$
- DP 组 $(*, p, t)$

由于坐标唯一，每个设备在每种类型中都恰好属于一个组。$\square$

## 通信模式

### 张量并行的通信

在每个 TP 组内部，每一层都需要 AllReduce 或 ReduceScatter/AllGather：

```
TP Group (ranks 0-3):
┌─────┬─────┬─────┬─────┐
│ G0  │ G1  │ G2  │ G3  │  Partial activations
└──┬──┴──┬──┴──┬──┴──┬──┘
   └─────┴─────┴─────┘
         AllReduce           High bandwidth via NVLink
   ┌─────┬─────┬─────┐
┌──┴──┬──┴──┬──┴──┬──┴──┐
│ Sum │ Sum │ Sum │ Sum │  Full activations
└─────┴─────┴─────┴─────┘
```

**带宽**：每层 $\frac{2(T-1)}{T} \cdot H \cdot B$。

### 流水线并行的通信

在 PP 阶段之间，是点对点发送：

```
PP Group (ranks 0-7):
Stage 0 → Stage 1 → Stage 2 → ... → Stage 7
         ↑           ↑           ↑
       Send        Send        Send
    activations  activations  activations
```

**带宽**：每个微批（micro-batch）边界 $H \cdot B$（远小于 TP）。

### 数据并行的通信

反向传播之后，进行梯度同步：

```
DP Group (ranks across nodes):
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Grads 0 │     │ Grads 1 │ ... │ Grads 31│
└────┬────┘     └────┬────┘     └────┬────┘
     └───────────────┴───────────────┘
              AllReduce
     ┌───────────────┬───────────────┐
┌────┴────┐     ┌────┴────┐     ┌────┴────┐
│ Avg Grad│     │ Avg Grad│ ... │ Avg Grad│
└─────────┘     └─────────┘     └─────────┘
```

**带宽**：每步 $\frac{2(D-1)}{D} \cdot \frac{\text{Parameters}}{T \cdot P}$。

## 内存分析

### 每块 GPU 的内存分解

在三维并行下，内存被分布开来：

**模型参数**（由 TP 和 PP 分片）：

$$M_{\text{params}} = \frac{\Psi \times 2}{T \times P}$$

**优化器状态**（由 TP 和 PP 分片）：

$$M_{\text{optimizer}} = \frac{\Psi \times 8}{T \times P} \quad (\text{m} + \text{v only; add } \frac{4\Psi}{T P} \text{ for FP32 master})$$

**激活值**（由 TP 分片，并乘以流水线深度）：

$$M_{\text{activations}} = \frac{B \times L_{\text{stage}} \times H}{T} \times k_{\text{buffer}}$$

其中 $k_{\text{buffer}}$ 计入在途的微批。

### 示例：1024 块 GPU 上的 175B 模型

配置：DP=32、PP=8、TP=4。

**每块 GPU 的参数**：

$$\frac{175\text{B} \times 2}{4 \times 8} = \frac{350\text{GB}}{32} = 10.9\text{ GB}$$

**每块 GPU 的优化器**：

$$\frac{175\text{B} \times 8}{4 \times 8} = \frac{1.4\text{TB}}{32} = 43.8\text{ GB}$$

**每块 GPU 的激活值**（在途微批 $M=32$、微批大小 $B_{\mu}=32$，为简单起见取 $S=1$）：

$$\frac{M \times L_{\text{stage}} \times B_{\mu} \times H \times 2}{T} = \frac{32 \times 12 \times 32 \times 12288 \times 2}{4} \approx 24 \text{ GB}$$

**总计**：10.9 + 43.8 + 24 ≈ **79 GB**（能放进 80GB 的 A100）。

## 性能模型

### 计算时间

前向与反向传播时间（每个微批）：

$$T_{\text{compute}} = \frac{6 \times \Psi \times B_{\mu} \times S}{P \times \text{FLOPs}_{\text{GPU}}}$$

其中 $B_{\mu}$ 是微批大小。

### 通信时间

**TP 通信**（每层，前向和反向各一次）：

$$T_{\text{TP}} = 4 \times L_{\text{stage}} \times \left(\alpha + \frac{2(T-1)}{T} \times \frac{B_{\mu} \times S \times H \times \text{bytes}}{\beta_{\text{NVLink}}}\right)$$

其中 FP16/BF16 的 $\text{bytes} = 2$。

**PP 通信**（每个微批）：

$$T_{\text{PP}} = 2 \times \left(\alpha + \frac{H \times B_{\mu}}{\beta_{\text{IB}}}\right)$$

**DP 通信**（每步一次）：

$$T_{\text{DP}} = \frac{2(D-1)}{D} \times \frac{\Psi/(T \times P)}{\beta_{\text{IB}}}$$

### 流水线效率

采用 1F1B 调度和 $M$ 个微批时：

$$\eta_{\text{pipeline}} = \frac{M}{M + P - 1}$$

### 总步时间

$$T_{\text{step}} = \frac{M \times T_{\text{compute}} + M \times T_{\text{TP}}}{\eta_{\text{pipeline}}} + T_{\text{DP}}$$

在 1F1B 中，PP 通信与计算重叠，因此不进入关键路径。

## 实现

### 三维并行训练器

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist

@dataclass
class ParallelConfig:
    """Configuration for 3D parallelism."""
    dp_size: int
    pp_size: int
    tp_size: int
    num_microbatches: int
    gradient_accumulation_steps: int = 1

    @property
    def world_size(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size

class ThreeDParallelTrainer:
    """Trainer implementing 3D parallelism."""

    def __init__(
        self,
        model_fn,  # Function to create model
        config: ParallelConfig,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = None
    ):
        self.config = config

        # Initialize mesh
        self.mesh = ThreeDMesh(
            config.dp_size,
            config.pp_size,
            config.tp_size
        )

        # Create model shard for this rank
        self.model = self._create_model_shard(model_fn)

        # Create optimizer
        optimizer_kwargs = optimizer_kwargs or {'lr': 1e-4}
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_kwargs
        )

        # Setup communication buffers
        self._setup_buffers()

    def _create_model_shard(self, model_fn):
        """Create the local model shard based on PP and TP rank."""
        # Get which layers this rank handles
        pp_rank = self.mesh.pp_rank
        tp_rank = self.mesh.tp_rank

        # model_fn should return the appropriate shard
        model = model_fn(
            stage_id=pp_rank,
            num_stages=self.config.pp_size,
            tp_rank=tp_rank,
            tp_size=self.config.tp_size
        )

        return model.cuda()

    def _setup_buffers(self):
        """Allocate communication buffers."""
        # Activation buffers for pipeline (double-buffered)
        self.activation_buffers = [None, None]

        # Gradient buffers for DP sync
        self.gradient_buffer = None

    def train_step(self, data_iterator) -> float:
        """Execute one training step with 3D parallelism."""
        self.optimizer.zero_grad()

        num_microbatches = self.config.num_microbatches
        losses = []

        # 1F1B Schedule
        # Warmup: forward passes only
        for i in range(min(self.mesh.pp_size - 1, num_microbatches)):
            loss = self._forward_step(data_iterator, i)
            losses.append(loss)

        # Steady state: 1F1B
        for i in range(num_microbatches - self.mesh.pp_size + 1):
            # Forward
            loss = self._forward_step(
                data_iterator,
                i + self.mesh.pp_size - 1
            )
            losses.append(loss)

            # Backward for earlier microbatch
            self._backward_step(i)

        # Cooldown: backward passes only
        for i in range(self.mesh.pp_size - 1):
            self._backward_step(
                num_microbatches - self.mesh.pp_size + 1 + i
            )

        # DP gradient sync
        self._sync_gradients()

        # Optimizer step
        self.optimizer.step()

        return sum(losses) / len(losses)

    def _forward_step(self, data_iterator, micro_idx: int) -> float:
        """Forward pass for one microbatch."""
        # Get input
        if self.mesh.pp_rank == 0:
            # First stage: get from data
            batch = next(data_iterator)
            input_tensor = batch['input'].cuda()
        else:
            # Receive from previous stage
            input_tensor = self._recv_forward()

        # Forward through local layers (with TP)
        output = self.model(input_tensor)

        # Save for backward
        self._save_activation(micro_idx, input_tensor, output)

        if self.mesh.pp_rank == self.mesh.pp_size - 1:
            # Last stage: compute loss
            target = next(data_iterator)['target'].cuda()
            loss = self._compute_loss(output, target)
            return loss.item()
        else:
            # Send to next stage
            self._send_forward(output)
            return 0.0

    def _backward_step(self, micro_idx: int):
        """Backward pass for one microbatch."""
        # Get saved activation
        input_tensor, output = self._get_saved_activation(micro_idx)

        if self.mesh.pp_rank == self.mesh.pp_size - 1:
            # Last stage: gradient from loss
            output.backward(self._saved_loss_grad[micro_idx])
        else:
            # Receive gradient from next stage
            grad = self._recv_backward()
            output.backward(grad)

        if self.mesh.pp_rank > 0:
            # Send gradient to previous stage
            self._send_backward(input_tensor.grad)

    def _sync_gradients(self):
        """AllReduce gradients across DP group."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    group=self.mesh.dp_group
                )
                param.grad /= self.mesh.dp_size

    def _send_forward(self, tensor: torch.Tensor):
        """Send activation to next pipeline stage."""
        _, next_rank = self.mesh.get_pipeline_neighbors()
        dist.send(tensor, dst=next_rank, group=self.mesh.pp_group)

    def _recv_forward(self) -> torch.Tensor:
        """Receive activation from previous pipeline stage."""
        prev_rank, _ = self.mesh.get_pipeline_neighbors()
        tensor = torch.empty(self._get_activation_shape()).cuda()
        dist.recv(tensor, src=prev_rank, group=self.mesh.pp_group)
        return tensor

    def _send_backward(self, tensor: torch.Tensor):
        """Send gradient to previous pipeline stage."""
        prev_rank, _ = self.mesh.get_pipeline_neighbors()
        dist.send(tensor, dst=prev_rank, group=self.mesh.pp_group)

    def _recv_backward(self) -> torch.Tensor:
        """Receive gradient from next pipeline stage."""
        _, next_rank = self.mesh.get_pipeline_neighbors()
        tensor = torch.empty(self._get_activation_shape()).cuda()
        dist.recv(tensor, src=next_rank, group=self.mesh.pp_group)
        return tensor
```

### 集成张量并行层

每一层都必须实现 TP 感知的前向传播：

```python
class TPLinear(nn.Module):
    """Linear layer with tensor parallelism."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: dist.ProcessGroup,
        tp_size: int,
        split_input: bool = False  # Column vs row parallel
    ):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.split_input = split_input

        if split_input:
            # Row parallel: split input dimension
            assert in_features % tp_size == 0
            local_in = in_features // tp_size
            local_out = out_features
        else:
            # Column parallel: split output dimension
            assert out_features % tp_size == 0
            local_in = in_features
            local_out = out_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(local_out, local_in)
        )
        self.bias = nn.Parameter(torch.empty(local_out))

        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.split_input:
            # Row parallel: AllReduce output
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)

        return output
```

## 选择各维度

三维并行的艺术在于：给定以下条件选择 $(D, P, T)$：

- GPU 总数 $N$
- 模型规模 $M$
- 每 GPU 内存 $G$
- 网络拓扑

### 维度选择算法

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareSpec:
    """Hardware specifications."""
    total_gpus: int
    gpus_per_node: int
    gpu_memory_gb: float
    nvlink_bandwidth_gbps: float
    ib_bandwidth_gbps: float
    gpu_flops: float  # TFLOPs

@dataclass
class ModelSpec:
    """Model specifications."""
    num_params: int  # in billions
    hidden_dim: int
    num_layers: int
    vocab_size: int

def choose_3d_config(
    hw: HardwareSpec,
    model: ModelSpec,
    target_batch_size: int
) -> ParallelConfig:
    """Choose optimal 3D parallelism configuration."""

    # Calculate memory requirements
    param_memory_gb = model.num_params * 2  # FP16
    optimizer_memory_gb = model.num_params * 8  # Adam FP32

    # TP should fit in a node for NVLink
    max_tp = hw.gpus_per_node

    # Start with minimum TP that allows model to fit
    for tp in [1, 2, 4, 8]:
        if tp > max_tp:
            break

        # Try PP sizes
        for pp in [1, 2, 4, 8, 16, 32]:
            model_mem = (param_memory_gb + optimizer_memory_gb) / (tp * pp)

            # Leave room for activations (~40% of memory)
            if model_mem < hw.gpu_memory_gb * 0.6:
                # This (tp, pp) might work
                dp = hw.total_gpus // (tp * pp)

                if dp >= 1 and tp * pp * dp == hw.total_gpus:
                    # Validate activation memory
                    activation_mem = estimate_activation_memory(
                        model, target_batch_size // dp, tp, pp
                    )

                    total_mem = model_mem + activation_mem
                    if total_mem < hw.gpu_memory_gb * 0.9:
                        # Calculate efficiency
                        efficiency = calculate_efficiency(
                            hw, model, dp, pp, tp, target_batch_size
                        )

                        return ParallelConfig(
                            dp_size=dp,
                            pp_size=pp,
                            tp_size=tp,
                            num_microbatches=max(pp * 2, 8)
                        )

    raise ValueError("Cannot fit model with available resources")

def estimate_activation_memory(
    model: ModelSpec,
    batch_size: int,
    tp: int,
    pp: int
) -> float:
    """Estimate activation memory in GB."""
    layers_per_stage = model.num_layers // pp
    hidden = model.hidden_dim // tp

    # Per layer: hidden * batch * 2 (input + output) * 2 (bytes)
    per_layer = hidden * batch_size * 4 / 1e9

    # Pipeline buffers
    pipeline_factor = min(pp, 8)  # Microbatches in flight

    return per_layer * layers_per_stage * pipeline_factor

def calculate_efficiency(
    hw: HardwareSpec,
    model: ModelSpec,
    dp: int, pp: int, tp: int,
    batch_size: int
) -> float:
    """Estimate training efficiency (0-1)."""
    # Pipeline efficiency
    microbatches = max(pp * 2, 8)
    pipeline_eff = microbatches / (microbatches + pp - 1)

    # TP overhead (communication / compute ratio)
    layers_per_stage = model.num_layers // pp
    compute_per_layer = 6 * model.num_params * 1e9 / model.num_layers * batch_size / dp
    comm_per_layer = 4 * model.hidden_dim * batch_size / dp * 2  # AllReduce

    if tp > 1:
        # NVLink communication
        tp_time = comm_per_layer / (hw.nvlink_bandwidth_gbps * 1e9)
        compute_time = compute_per_layer / (hw.gpu_flops * 1e12)
        tp_eff = compute_time / (compute_time + tp_time)
    else:
        tp_eff = 1.0

    # DP overhead (gradient sync / step time)
    grad_sync = 2 * model.num_params * 1e9 / (tp * pp) / (hw.ib_bandwidth_gbps * 1e9)
    step_time = compute_time * layers_per_stage * microbatches / pipeline_eff
    dp_eff = step_time / (step_time + grad_sync)

    return pipeline_eff * tp_eff * dp_eff
```

### 经验法则

1. **TP ≤ 8**：保持在 NVLink 域内
2. **PP = 层数 / 每阶段最少层数**：通常是 4–16
3. **DP = 剩余 GPU**：为吞吐最大化
4. **微批数 ≥ 2 × PP**：减少气泡开销

## 案例研究：Megatron-LM 配置

Megatron-LM 用三维并行训练大模型：

### 175B GPT-3 规模

| 配置 | 值 |
|--------------|-------|
| GPU 总数 | 1024 |
| TP | 8 |
| PP | 16 |
| DP | 8 |
| 微批数 | 32 |

**每块 GPU 的内存**：

- 参数：175B × 2 / (8 × 16) = 2.7 GB
- 优化器：175B × 8 / (8 × 16) = 10.9 GB
- 激活值：约 40 GB
- 总计：约 54 GB（能放进 80GB A100）

**流水线效率**：

$$\eta = \frac{32}{32 + 16 - 1} = \frac{32}{47} = 68\%$$

**TP 通信**（每层）：

$$T_{\text{TP}} = 2 \times \left(\alpha + \frac{7}{8} \times \frac{12288 \times 2048 \times 2}{600 \times 10^9}\right) \approx 0.07\text{ ms}$$

**总体 MFU**：在 A100 集群上约 45–50%。

### 530B MT-NLG 配置

| 配置 | 值 |
|--------------|-------|
| GPU 总数 | 2240 |
| TP | 8 |
| PP | 35 |
| DP | 8 |
| 微批数 | 70 |

**流水线效率**：

$$\eta = \frac{70}{70 + 35 - 1} = \frac{70}{104} = 67\%$$

## 交错式流水线阶段

进阶优化：每个 PP rank 处理多个不连续的阶段。

### 标准与交错

**标准**（虚拟阶段数 = 1）：
```
Rank 0: [Layer 0-11]
Rank 1: [Layer 12-23]
Rank 2: [Layer 24-35]
Rank 3: [Layer 36-47]
```

**交错**（虚拟阶段数 = 2）：
```
Rank 0: [Layer 0-5] + [Layer 24-29]
Rank 1: [Layer 6-11] + [Layer 30-35]
Rank 2: [Layer 12-17] + [Layer 36-41]
Rank 3: [Layer 18-23] + [Layer 42-47]
```

### 交错的好处

**减少气泡**：

$$\text{Bubble} = \frac{P - 1}{P + M - 1} \to \frac{P - 1}{M \cdot v + P - 1}$$

其中 $v$ 是每个 rank 的虚拟阶段数（Narayanan 等，2021）。每个微批要经过 $P \cdot v$ 个虚拟阶段，但预热的代价仍然是 $P - 1$ 个时间槽。

**示例**：PP=8，M=16
- 标准：气泡 = $\frac{7}{23} = 30\%$
- 交错（v=4）：气泡 = $\frac{7}{71} \approx 10\%$

**代价**：更多通信（但与计算重叠）。

```python
class InterleavedPipelineSchedule:
    """Interleaved 1F1B schedule for reduced bubble."""

    def __init__(
        self,
        num_stages: int,
        virtual_stages: int,
        num_microbatches: int
    ):
        self.num_stages = num_stages
        self.virtual_stages = virtual_stages
        self.num_microbatches = num_microbatches

        # Each rank has multiple virtual stages
        self.stages_per_rank = virtual_stages
        self.total_virtual_stages = num_stages * virtual_stages

    def get_schedule(self, rank: int) -> List[Tuple[str, int, int]]:
        """
        Return schedule as list of (op_type, microbatch_id, virtual_stage_id).
        """
        schedule = []

        # Warmup phases
        warmup_steps = (self.num_stages - rank - 1) * self.virtual_stages

        for step in range(warmup_steps):
            virtual_stage = step % self.virtual_stages
            microbatch = step // self.virtual_stages
            schedule.append(('forward', microbatch, virtual_stage))

        # Steady state
        forward_mb = warmup_steps // self.virtual_stages
        backward_mb = 0

        steady_steps = self.num_microbatches - warmup_steps // self.virtual_stages

        for _ in range(steady_steps * self.virtual_stages):
            # Alternate forward and backward across virtual stages
            schedule.append(('forward', forward_mb, forward_mb % self.virtual_stages))
            forward_mb += 1

            schedule.append(('backward', backward_mb, backward_mb % self.virtual_stages))
            backward_mb += 1

        # Cooldown
        for step in range(warmup_steps):
            virtual_stage = step % self.virtual_stages
            schedule.append(('backward', backward_mb, virtual_stage))
            if (step + 1) % self.virtual_stages == 0:
                backward_mb += 1

        return schedule
```

## 与 ZeRO 集成

把三维并行与 ZeRO 结合，以进一步降低内存：

### ZeRO-1 + 三维并行

沿 DP 维度分片优化器状态（optimizer states）：

```
Before ZeRO-1:
Each DP replica: full optimizer states (8 bytes/param)

After ZeRO-1:
Each DP replica: 1/D of optimizer states

Memory savings: (D-1)/D of optimizer memory
```

**通信变化**：优化器步需要 AllGather 参数。

### 内存栈

```
┌────────────────────────────────────────┐
│           Original Model               │
│          175B params × 10B             │
│              = 1.75 TB                 │
├────────────────────────────────────────┤
│     After TP/PP (8×16 = 128-way)       │
│           1.75TB / 128                 │
│              = 13.7 GB                 │
├────────────────────────────────────────┤
│     After ZeRO-1 (DP=8)                │
│      Params: 2.7 GB (unchanged)        │
│      Optim: 10.9GB / 8 = 1.4 GB        │
│           Total: 4.1 GB                │
└────────────────────────────────────────┘
```

## 练习题

1. **配置设计**：你有 256 块 A100 GPU（32 节点 × 8 GPU），想训练一个 40B 参数的模型。设计一个三维并行配置。计算每块 GPU 的内存与预期流水线效率。

??? success "解答"
    **模型特征（40B）：**

    $$M_{total} = 16\Psi = 16 \times 40 \times 10^9 = 640 \text{ GB}$$

    **配置设计：**

    | 策略 | 值 | 理由 |
    |----------|-------|-----------|
    | TP | 8 | 占满节点内（NVLink） |
    | PP | 8 | 每条流水线 4 个节点 |
    | DP | 4 | 256 / (8 × 8) = 4 个副本 |

    **验证：** $8 \times 8 \times 4 = 256$ ✓

    **每块 GPU 内存：**

    | 组件 | 每 GPU 内存 |
    |-----------|----------------|
    | 参数 | $\frac{2 \times 40B}{TP \times PP} = \frac{80}{64} = 1.25$ GB |
    | 梯度 | 1.25 GB |
    | 优化器 | $\frac{12 \times 40B}{TP \times PP} = 7.5$ GB |
    | **静态小计** | **10 GB** |

    在 DP=4 上使用 ZeRO-1：

    | 组件 | 每 GPU 内存 |
    |-----------|----------------|
    | 参数 | 1.25 GB |
    | 梯度 | 1.25 GB |
    | 优化器 | $\frac{7.5}{4} = 1.875$ GB |
    | **静态小计** | **4.375 GB** |

    **激活内存估算：**

    假设每个 DP 上 B=4、S=4096、H=8192、每阶段 40 层：

    $$M_{act} \approx \frac{40}{8} \times B \times S \times H \times 34 / TP$$

    $$M_{act} \approx 5 \times 4 \times 4096 \times 8192 \times 34 / 8 \approx 35 \text{ GB (with checkpointing)}$$

    **每块 GPU 总计：** 约 40 GB —— 能放进 A100 80GB ✓

    **流水线效率：**

    在 32 个微批（M=32）和 PP=8 下：

    $$\eta_{PP} = \frac{M}{M + P - 1} = \frac{32}{32 + 8 - 1} = \frac{32}{39} = \boxed{82\%}$$

    | 指标 | 值 |
    |--------|-------|
    | 气泡比例 | 18% |
    | 预期 MFU | 约 40–45%（计入所有开销） |

2. **通信分析**：对于 DP=8、PP=4、TP=4 的配置，训练一个 hidden_dim=8192、batch_size=512 的模型：

   - 计算每层的 TP 通信量
   - 计算每个微批的 PP 通信量
   - 计算每步的 DP 通信量
   - 哪个是瓶颈？

??? success "解答"
    **配置：** DP=8、PP=4、TP=4 → GPU 总数 = 128

    **给定：** H=8192、batch_size=512，假设 S=2048

    **1. TP 通信（每层）：**

    每层有 2 次 AllReduce（列并行之后和行并行之后各一次）：

    $$V_{TP}^{layer} = 2 \times B_{micro} \times S \times H \times 2 \text{ bytes}$$

    微批大小：$B_{micro} = \frac{512}{DP \times M} = \frac{512}{8 \times 32} = 2$（假设 M=32）

    $$V_{TP}^{layer} = 2 \times 2 \times 2048 \times 8192 \times 2 = 134 \text{ MB}$$

    有效通信量（环形 AllReduce）：$\frac{TP-1}{TP} \times 134 = 100.5$ MB

    **2. PP 通信（每个微批）：**

    阶段之间传输激活值：

    $$V_{PP}^{micro} = B_{micro} \times S \times H \times 2 = 2 \times 2048 \times 8192 \times 2 = 67 \text{ MB}$$

    点对点通信，因此按完整量计算。

    **3. DP 通信（每步）：**

    跨 DP 副本对梯度做 AllReduce：

    假设 30B 模型被 PP×TP 切分：

    $$V_{DP}^{step} = 2\Psi = 2 \times 30 \times 10^9 / (PP \times TP) = \frac{60 \text{ GB}}{16} = 3.75 \text{ GB}$$

    有效量：$\frac{DP-1}{DP} \times 3.75 = 3.28$ GB

    **4. 时间分析（假设带宽）：**

    | 通信 | 量 | 带宽 | 时间 |
    |---------------|--------|-----------|------|
    | TP（每层） | 100 MB | 450 GB/s（NVLink） | 0.22 ms |
    | TP（总计，80 层） | 8 GB | 450 GB/s | 18 ms |
    | PP（总计，32 个微批） | 2.1 GB | 50 GB/s（IB） | 43 ms |
    | DP（每步） | 3.28 GB | 50 GB/s | 66 ms |

    $$\boxed{\text{DP AllReduce is the bottleneck (66 ms)}}$$

    **缓解措施：** 让 DP AllReduce 与反向传播计算重叠。

3. **交错权衡**：比较 PP=16、M=32 时以下情形的气泡比例：

   - 不交错（v=1）
   - v=2 交错
   - v=4 交错
   每种情形的通信开销各是多少？

??? success "解答"
    **给定：** PP=16 个阶段，M=32 个微批

    **气泡比例公式：**

    不交错（1F1B）：

    $$\text{Bubble} = \frac{P - 1}{M + P - 1}$$

    交错因子为 $v$ 时：

    $$\text{Bubble}_{interleaved} = \frac{P - 1}{v \times M + P - 1}$$

    **计算：**

    | 交错 | 气泡公式 | 气泡比例 |
    |--------------|----------------|-----------------|
    | v=1 | $\frac{15}{32+15}$ | $\frac{15}{47} = 31.9\%$ |
    | v=2 | $\frac{15}{64+15}$ | $\frac{15}{79} = 19.0\%$ |
    | v=4 | $\frac{15}{128+15}$ | $\frac{15}{143} = \boxed{10.5\%}$ |

    **通信开销：**

    交错因子为 $v$ 时，每个微批跨越阶段边界的次数变成 $v$ 倍：

    $$\text{Comm}_{interleaved} = v \times \text{Comm}_{base}$$

    | 交错 | 通信倍数 | 气泡减少 | 净收益？ |
    |--------------|-----------------|------------------|--------------|
    | v=1 | 1× | 基线 | 基线 |
    | v=2 | 2× | 气泡少 12.9% | 是，若通信 < 气泡 |
    | v=4 | 4× | 气泡少 21.4% | 收益递减 |

    **盈亏平衡分析：**

    设 $T_c$ = 计算时间，$T_{comm}$ = 基础 PP 通信时间。

    交错有益的条件：

    $$\Delta\text{Bubble} \times T_c > (v-1) \times T_{comm}$$

    对于 v=2：

    $$0.129 \times T_c > 1 \times T_{comm}$$

    如果 $T_c = 1000$ ms 且 $T_{comm} = 50$ ms：

    $$129 \text{ ms} > 50 \text{ ms} \checkmark$$

    **总结：**

    | v | 气泡 | 通信开销 | 推荐场景 |
    |---|--------|---------------|------------------|
    | 1 | 31.9% | 1× | 带宽受限 |
    | 2 | 19.0% | 2× | 典型（均衡） |
    | 4 | 10.5% | 4× | 高带宽链路 |

4. **扩展效率**：一个三维并行配置在 512 块 GPU 上达到 50% MFU。扩展到 2048 块 GPU（4× DP）时，预测新的 MFU。瓶颈是什么？

??? success "解答"
    **基线：** 512 块 GPU 上 50% MFU

    **扩展场景：** 只把 DP 从 D 增加到 4D（TP、PP 保持不变）

    **开销分析：**

    1. **计算效率**（不变）：
       - 每块 GPU 的批相同 → 计算密度相同
       - 核函数效率不变

    2. **TP 通信**（不变）：
       - TP 组不变
       - 每次前向/反向开销相同

    3. **PP 效率**（不变）：
       - 流水线深度相同
       - 气泡比例相同

    4. **DP 通信**（增加）：
       - AllReduce 现在跨 4 倍的 GPU
       - 环形 AllReduce：$T_{DP} = \frac{D-1}{D} \times \frac{2\Psi}{B}$

    **DP 扩展的影响：**

    512 块 GPU、DP=D 时：

    $$T_{DP}^{512} = \frac{D-1}{D} \times \frac{2\Psi}{B}$$

    2048 块 GPU、DP=4D 时：

    $$T_{DP}^{2048} = \frac{4D-1}{4D} \times \frac{2\Psi}{B}$$

    比值：$\frac{T_{DP}^{2048}}{T_{DP}^{512}} = \frac{(4D-1)/4D}{(D-1)/D} = \frac{4D-1}{4(D-1)} \approx 1.0$（D 较大时）

    **但是**，每块 GPU 的 DP AllReduce 时间不变，而每块 GPU 的计算量也不变。

    **真正的瓶颈：批大小的扩展**

    为保持效率，全局批必须扩展 4 倍：

    $$B_{global}^{new} = 4 \times B_{global}^{old}$$

    如果我们**不**扩展批：
    - 每块 GPU 只做 1/4 的工作
    - 通信量不变
    - MFU 显著下降

    **预测 MFU：**

    | 场景 | 全局批 | 每 GPU 工作量 | MFU |
    |----------|--------------|--------------|-----|
    | 批保持不变 | 1× | 0.25× | ~20% |
    | 批扩展 2× | 2× | 0.5× | ~35% |
    | 批扩展 4× | 4× | 1× | $\boxed{\sim 48\%}$ |

    **2048 块 GPU 上的瓶颈：**

    1. **DP AllReduce 延迟** —— 更大的环意味着更多跳
    2. **网络拥塞** —— 4 倍的跨节点流量
    3. **批大小限制** —— 可能触及学习率稳定性问题
    4. **内存带宽** —— 更大规模下激活值重载

5. **ZeRO 集成**：对于 175B 参数、DP=32、PP=8、TP=4：

   - 计算不使用 ZeRO 时每块 GPU 的优化器内存
   - 计算使用 ZeRO-1 时每块 GPU 的优化器内存
   - 释放出多少激活内存预算？

??? success "解答"
    **配置：** DP=32、PP=8、TP=4 → GPU 总数 = 1024

    **模型切分：**

    每块 GPU 上的参数（ZeRO 之前）：

    $$\Psi_{GPU} = \frac{175B}{PP \times TP} = \frac{175B}{32} = 5.47B \text{ params}$$

    **不使用 ZeRO：**

    | 组件 | 大小 | 内存 |
    |-----------|------|--------|
    | 参数（fp16） | $5.47B \times 2$ | 10.9 GB |
    | 梯度（fp16） | $5.47B \times 2$ | 10.9 GB |
    | 优化器状态（fp32） | $5.47B \times 12$ | 65.6 GB |
    | **总计** | | **87.4 GB** |

    $$M_{opt}^{no\_ZeRO} = \boxed{65.6 \text{ GB}}$$

    **使用 ZeRO-1（跨 DP 分片优化器状态）：**

    ZeRO-1 在 DP=32 上分片优化器状态：

    $$M_{opt}^{ZeRO1} = \frac{65.6 \text{ GB}}{32} = \boxed{2.05 \text{ GB}}$$

    | 组件 | 不使用 ZeRO | 使用 ZeRO-1 |
    |-----------|--------------|-------------|
    | 参数 | 10.9 GB | 10.9 GB |
    | 梯度 | 10.9 GB | 10.9 GB |
    | 优化器 | 65.6 GB | 2.05 GB |
    | **静态总计** | **87.4 GB** | **23.85 GB** |

    **释放出的激活内存预算：**

    节省的内存：

    $$\Delta M = 87.4 - 23.85 = 63.55 \text{ GB}$$

    在 80GB A100 上：

    | 指标 | 不使用 ZeRO | 使用 ZeRO-1 |
    |--------|--------------|-------------|
    | 静态内存 | 87.4 GB | 23.85 GB |
    | 可用于激活 | -7.4 GB（OOM！） | 56.15 GB |
    | **可行？** | 否 | 是 |

    $$\boxed{\text{ZeRO-1 frees } \sim 64 \text{ GB for activations}}$$

    **注：** ZeRO-1 在更新步骤中为优化器状态增加了 AllGather，但这是每步一次的开销，且可以重叠。

6. **另一种排序**：标准排序是 (DP, PP, TP)。如果改用 (DP, TP, PP) 会怎样？分析通信模式的变化。

??? success "解答"
    **标准排序 (DP, PP, TP)：**

    ```
    Mesh shape: (DP, PP, TP)
    Innermost (TP): Consecutive ranks → intra-node (NVLink)
    Middle (PP): Strides by TP → may cross nodes
    Outermost (DP): Strides by PP×TP → crosses nodes
    ```

    **另一种排序 (DP, TP, PP)：**

    ```
    Mesh shape: (DP, TP, PP)
    Innermost (PP): Consecutive ranks
    Middle (TP): Strides by PP
    Outermost (DP): Strides by TP×PP
    ```

    **64 块 GPU 的示例（DP=2、TP=4、PP=8）：**

    | 排序 | TP 组（rank 0） | PP 链（TP=0、DP=0） |
    |----------|-------------------|------------------------|
    | (DP, PP, TP) | {0,1,2,3} | {0,4,8,12,16,20,24,28} |
    | (DP, TP, PP) | {0,8,16,24} | {0,1,2,3,4,5,6,7} |

    **通信影响：**

    | 集合通信（collective communication） | 标准 (DP,PP,TP) | 另一种 (DP,TP,PP) |
    |------------|---------------------|------------------------|
    | TP AllReduce | 连续 → NVLink | 跨步 → 可能跨节点 |
    | PP Send/Recv | 跨步 → 跨节点 | 连续 → 可能走 NVLink |
    | DP AllReduce | 大跨步 → 跨节点 | 大跨步 → 跨节点 |

    **分析：**

    **标准排序更受青睐，因为：**

    1. **TP 频率**：每层 2× AllReduce × 80+ 层 = 每步 160+ 次集合通信
    2. **PP 频率**：每个微批 1× Send/Recv × 32 = 每步 32 次传输
    3. **DP 频率**：每步 1× AllReduce

    TP 需要最高带宽 → 必须在最内层（NVLink）。

    **另一种排序的后果：**

    | 问题 | 影响 |
    |-------|--------|
    | TP 跨节点 | 带宽降至 1/18（900 → 50 GB/s） |
    | TP 延迟 | 增加网络跳数 |
    | PP 连续 | PP 略有改善 |

    **定量示例：**

    TP AllReduce 时间（每次操作 100MB）：

    | 排序 | 带宽 | 每次 AllReduce 时间 |
    |----------|-----------|-------------------|
    | 标准（NVLink） | 900 GB/s | 0.11 ms |
    | 另一种（IB） | 50 GB/s | 2.0 ms |

    每步（160 次 AllReduce）：

    | 排序 | TP 时间 |
    |----------|---------|
    | 标准 | 18 ms |
    | 另一种 | 320 ms |

    $$\boxed{\text{Alternative ordering: 18× slower TP communication}}$$

    **另一种排序何时可行？**

    - PP 占主导的极深流水线
    - 极小的 TP 组
    - NVLink 覆盖方式不同的定制拓扑

## 关键要点

1. **没有单一策略能独自扩展**：DP 浪费内存，TP 有通信开销，PP 有气泡。

2. **三维 = DP × PP × TP**：让每种策略在各自的最优规模上工作。

3. **TP 在最内层**：利用节点内的 NVLink 带宽。

4. **流水线效率**：使用大量微批与交错。

5. **配置是一个优化问题**：在内存、计算与通信之间取得平衡。

6. **ZeRO 能叠加收益**：跨 DP 分片优化器状态，进一步节省内存。

7. **Megatron-LM 的模式有效**：TP=8、PP=8–16、DP 用于剩余 GPU，在 100B+ 规模上已获验证。