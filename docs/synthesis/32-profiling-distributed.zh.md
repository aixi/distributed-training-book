---
title: "分布式训练的剖析"
subtitle: "并行系统的插桩、测量与优化"
---

<div class="chapter-opener" markdown>
一次分布式训练会消耗数千 GPU 小时。然而大多数实践者并不清楚这些时间花在了哪里。剖析（profiling）把直觉转化为数据，揭示你究竟是受计算限制、受通信限制，还是单纯在等待。
</div>

<div class="investigation-question" markdown>
**问题**：你的 64 GPU 训练任务达到 35% MFU（Model FLOP Utilization，模型浮点利用率）。另外的 65% 在哪里？是通信？内存带宽？kernel 启动开销？空闲时间？没有测量，优化就只能靠猜。
</div>

<div class="notation-banner" markdown>
**本章记号：** MFU = Model FLOP Utilization，$T_{\text{step}}$ = 步时间。见[记号表](../appendices/notation.md)。
</div>

!!! abstract "承接：前面所有部分"
    最后的这一部分综合了一切。你已经理解了**屋顶线与估算**（[第一部分](../foundations/01-scale-imperative.md)）、**最优资源分配**（[第二部分](../scaling-laws/07-compute-loss-surface.md)）、**集合通信成本**（[第三部分](../collectives/11-primitives-properties.md)）、**并行策略**（[第四部分](../parallelism/14-data-parallelism-associativity.md)）、**内存管理**（[第五部分](../memory/19-memory-equation.md)）、**组合**（[第六部分](../composition/23-device-mesh.md)）与**效率技术**（[第七部分](../efficiency/28-gradient-compression.md)）。现在我们应用这些知识：诊断真实系统、调查瓶颈，并分析最先进的训练任务。

!!! note "第八部分的代码风格"
    第八部分的代码示例是**教学骨架**——它们展示核心逻辑与接口，但省略了错误处理、边界情况与生产环境加固。它们的目的是说明剖析*策略*，而不是作为开箱即用的工具。生产环境剖析请直接使用 PyTorch Profiler、Nsight Systems 与 NCCL 调试日志。

## 剖析的必要性

在规模之上，低效会复利累积。单块 GPU 上 10% 的低效，在 64 块 GPU 上就变成每天约 154 GPU 小时的浪费。理解时间*究竟*去了哪里至关重要。

### 我们测量什么

分布式训练剖析考察四个领域：

**1. 计算**
- kernel 执行时间
- Tensor Core 利用率
- 内存带宽利用率
- 达到的 FLOPs 与理论峰值之比

**2. 通信**
- 集合通信操作的持续时间
- 网络带宽利用率
- 消息大小与频率
- 与计算的重叠程度

**3. 内存**
- 峰值分配
- 碎片化
- 数据搬运（HBM ↔ 主机，主机 ↔ 设备）
- 缓存命中率

**4. 编排**
- kernel 启动开销
- Python／框架开销
- 同步等待
- 各 worker 之间的负载不均衡

### 时间预算

每个训练步都有固定的时间预算：

$$T_{\text{step}} = T_{\text{forward}} + T_{\text{backward}} + T_{\text{comm}} + T_{\text{optimizer}} + T_{\text{other}}$$

目标：在保持模型质量的同时最小化 $T_{\text{step}}$。这里 $T_{\text{compute}}$ 通常指前向 + 反向 + 优化器的 kernel 时间。

完美重叠时：

$$T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{comm}})$$

无重叠时：

$$T_{\text{step}} = T_{\text{compute}} + T_{\text{comm}}$$

剖析会告诉你处于这一谱系上的哪一点。

## 剖析工具

### NVIDIA Nsight Systems

GPU 剖析的黄金标准。可捕获：

- CUDA kernel 执行
- 内存操作
- NCCL 集合通信调用
- CPU 活动
- GPU 间通信

**基本用法**：
```bash
nsys profile -o trace \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    python train.py
```

**用于分布式训练**：
```bash
# On each rank
nsys profile -o trace_rank${RANK} \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    torchrun --nproc_per_node=8 train.py
```

**关键命令选项**：
```bash
# Capture NCCL operations explicitly
nsys profile --trace=cuda,nvtx,osrt,nccl \
    --capture-range=cudaProfilerApi \
    python train.py

# Limit trace duration (avoid huge files)
nsys profile --duration=60 -o trace python train.py

# Export to multiple formats
nsys export -o trace.json --type=json trace.nsys-rep
```

### PyTorch Profiler

内置剖析，与 TensorBoard 集成：

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(
        wait=1,      # Skip first step
        warmup=1,    # Warmup step
        active=3,    # Profile 3 steps
        repeat=1
    ),
    on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        train_step(batch)
        prof.step()
```

**分布式感知剖析**：
```python
import torch.distributed as dist

def profile_distributed_training(model, dataloader, num_steps=5):
    """Profile distributed training with rank-aware output."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    profiler_config = profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(
            f'./logs/rank_{rank}'
        ),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,  # Estimate FLOPs
    )

    with profiler_config as prof:
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            # Add NVTX markers for visibility
            with torch.cuda.nvtx.range(f"step_{step}"):
                with torch.cuda.nvtx.range("forward"):
                    output = model(batch)

                with torch.cuda.nvtx.range("backward"):
                    loss = compute_loss(output)
                    loss.backward()

                with torch.cuda.nvtx.range("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()

            prof.step()

    # Print summary for rank 0
    if rank == 0:
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20
        ))
```

### NVTX 标注

NVIDIA Tools Extension 提供手工标注：

```python
import torch.cuda.nvtx as nvtx

class NVTXAnnotatedModule(nn.Module):
    """Module with NVTX range annotations."""

    def __init__(self, module, name):
        super().__init__()
        self.module = module
        self.name = name

    def forward(self, x):
        with nvtx.range(self.name):
            return self.module(x)

# Annotate model layers
def annotate_model(model):
    """Add NVTX annotations to all layers."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            annotate_model(module)
        else:
            setattr(model, name, NVTXAnnotatedModule(module, name))
```

**细粒度集合通信标注**：
```python
import torch.distributed as dist

class ProfiledAllReduce:
    """AllReduce with detailed profiling."""

    def __init__(self, process_group=None):
        self.process_group = process_group
        self.call_count = 0
        self.total_bytes = 0
        self.total_time = 0

    def __call__(self, tensor):
        size_bytes = tensor.numel() * tensor.element_size()
        self.total_bytes += size_bytes

        with nvtx.range(f"AllReduce_{self.call_count}_{size_bytes/1e6:.1f}MB"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            dist.all_reduce(tensor, group=self.process_group)
            end.record()

            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            self.total_time += elapsed

        self.call_count += 1
        return tensor

    def summary(self):
        return {
            'calls': self.call_count,
            'total_bytes_GB': self.total_bytes / 1e9,
            'total_time_ms': self.total_time,
            'avg_bandwidth_GBps': (self.total_bytes / 1e9) / (self.total_time / 1e3)
        }
```

### NCCL 调试输出

启用详细的 NCCL 日志：

```bash
# Basic info
export NCCL_DEBUG=INFO

# Detailed subsystem logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Trace all operations
export NCCL_DEBUG=TRACE

# Log to file per rank
export NCCL_DEBUG_FILE=nccl_log_%h_%p.txt
```

**解读 NCCL 日志**：
```
# Topology detection
NCCL INFO Ring 00 : 0 -> 1 -> 2 -> 3 -> 0

# Algorithm selection
NCCL INFO Channel 00/08 : 0[0] -> 1[1] via P2P/CUMEM

# Operation timing (with TRACE)
NCCL INFO AllReduce: opCount 42 bytes 104857600 datatype 7 op 0
```

### 内存剖析

跟踪内存分配与碎片化：

```python
import torch

class MemoryProfiler:
    """Track GPU memory usage during training."""

    def __init__(self, device=None):
        self.device = device or torch.cuda.current_device()
        self.snapshots = []

    def snapshot(self, label=""):
        """Capture current memory state."""
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)

        self.snapshots.append({
            'label': label,
            'allocated_GB': allocated / 1e9,
            'reserved_GB': reserved / 1e9,
            'max_allocated_GB': max_allocated / 1e9,
            'fragmentation': 1 - (allocated / reserved) if reserved > 0 else 0
        })

    def reset_peak(self):
        """Reset peak memory tracking."""
        torch.cuda.reset_peak_memory_stats(self.device)

    def report(self):
        """Print memory timeline."""
        print("\n=== Memory Profile ===")
        print(f"{'Label':<30} {'Allocated':>12} {'Reserved':>12} {'Peak':>12} {'Frag':>8}")
        print("-" * 74)
        for snap in self.snapshots:
            print(f"{snap['label']:<30} "
                  f"{snap['allocated_GB']:>10.2f}GB "
                  f"{snap['reserved_GB']:>10.2f}GB "
                  f"{snap['max_allocated_GB']:>10.2f}GB "
                  f"{snap['fragmentation']:>7.1%}")

# Usage during training
mem_profiler = MemoryProfiler()
mem_profiler.reset_peak()

mem_profiler.snapshot("after_model_init")
# ... training ...
mem_profiler.snapshot("after_forward")
mem_profiler.snapshot("after_backward")
mem_profiler.snapshot("after_optimizer_step")

mem_profiler.report()
```

**详细内存分解**：
```python
def memory_breakdown():
    """Get detailed memory breakdown by category."""
    snapshot = torch.cuda.memory_snapshot()

    categories = {}
    for block in snapshot:
        # Categorize by allocation context
        if 'frames' in block and block['frames']:
            context = block['frames'][0]['filename']
        else:
            context = 'unknown'

        if context not in categories:
            categories[context] = 0
        categories[context] += block['total_size']

    # Sort by size
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1])

    print("\n=== Memory by Source ===")
    total = sum(categories.values())
    for source, size in sorted_cats[:10]:
        print(f"{source}: {size/1e9:.2f}GB ({size/total*100:.1f}%)")
```

## 解读轨迹

### Nsight Systems 时间线

典型的分布式训练轨迹显示为：

```
Time →
|--Forward--|--Backward----|--AllReduce--|--Optimizer--|
       |                   |<--Overlap-->|
       |--Compute Stream---|
       |---Comm Stream-----|
```

**要寻找的关键模式**：

1. **kernel 之间的空隙**：说明存在 CPU 开销或 kernel 启动延迟
2. **计算与通信串行**：没有重叠，存在优化空间
3. **集合通信的长尾**：掉队 worker 或网络争用
4. **内存拷贝操作**：可考虑预取或固定内存（pinned memory）

### 识别瓶颈

```python
class BottleneckAnalyzer:
    """Analyze profiling data to identify bottlenecks."""

    def __init__(self):
        self.timings = {
            'forward': [],
            'backward': [],
            'allreduce': [],
            'optimizer': [],
            'data_loading': [],
        }

    def record(self, phase, duration_ms):
        self.timings[phase].append(duration_ms)

    def analyze(self):
        """Identify the primary bottleneck."""
        results = {}

        for phase, times in self.timings.items():
            if times:
                results[phase] = {
                    'mean_ms': sum(times) / len(times),
                    'max_ms': max(times),
                    'min_ms': min(times),
                    'variance': self._variance(times),
                }

        total = sum(r['mean_ms'] for r in results.values())

        print("\n=== Bottleneck Analysis ===")
        print(f"{'Phase':<15} {'Mean':>10} {'Max':>10} {'Variance':>10} {'%Total':>10}")
        print("-" * 55)

        for phase, stats in sorted(results.items(), key=lambda x: -x[1]['mean_ms']):
            pct = stats['mean_ms'] / total * 100 if total > 0 else 0
            print(f"{phase:<15} {stats['mean_ms']:>8.2f}ms {stats['max_ms']:>8.2f}ms "
                  f"{stats['variance']:>8.2f}ms {pct:>9.1f}%")

        # Identify bottleneck type
        compute_time = results.get('forward', {}).get('mean_ms', 0) + \
                      results.get('backward', {}).get('mean_ms', 0)
        comm_time = results.get('allreduce', {}).get('mean_ms', 0)

        print("\n=== Diagnosis ===")
        if comm_time > compute_time * 1.2:
            print("COMMUNICATION BOUND: Collective operations dominate.")
            print("  Recommendations:")
            print("  - Increase batch size to amortize communication")
            print("  - Enable gradient bucketing if not already")
            print("  - Check for network bottlenecks")
        elif compute_time > comm_time * 1.2:
            print("COMPUTE BOUND: Forward/backward computation dominates.")
            print("  Recommendations:")
            print("  - Good! System is well-utilized")
            print("  - Consider mixed precision if not already using")
            print("  - May be able to overlap more communication")
        else:
            print("BALANCED: Compute and communication roughly equal.")
            print("  Recommendations:")
            print("  - Verify overlap is enabled and effective")
            print("  - This is often the optimal regime")

    def _variance(self, values):
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
```

### 阅读集合通信时间线

```
AllReduce Timeline (ideal overlap):
GPU 0: |--Compute--|--AllReduce--|        |--Compute--|
GPU 1: |--Compute--|--AllReduce--|        |--Compute--|
                   |<--overlap-->|

AllReduce Timeline (straggler):
GPU 0: |--Compute--|--AllReduce--|---wait---|--Compute--|
GPU 1: |--Compute--|------AllReduce--------|--Compute--|
                              ^
                        Slow GPU 1 affects all
```

**检测掉队者**：
```python
import torch.distributed as dist

def detect_stragglers(num_iterations=10):
    """Measure timing variance across ranks."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_times = []

    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()

        # Simulate work with some variation
        dummy_work()

        torch.cuda.synchronize()
        local_time = time.time() - start
        local_times.append(local_time)

        # Gather all times to rank 0
        all_times = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(all_times, torch.tensor([local_time]))

        if rank == 0:
            times = [t.item() for t in all_times]
            mean_time = sum(times) / len(times)
            max_time = max(times)
            straggler_rank = times.index(max_time)

            if max_time > mean_time * 1.1:  # 10% slower
                print(f"Straggler detected: rank {straggler_rank} "
                      f"({max_time:.3f}s vs mean {mean_time:.3f}s)")
```

## MFU 与效率指标

### 模型浮点利用率

MFU 衡量实际计算效率：

$$\text{MFU} = \frac{\text{Achieved FLOPs}}{\text{Peak FLOPs}}$$

**计算 MFU**：
```python
def calculate_mfu(
    model_flops_per_sample: int,
    batch_size: int,
    step_time_seconds: float,
    num_gpus: int,
    peak_flops_per_gpu: float
) -> float:
    """
    Calculate Model FLOP Utilization.

    Args:
        model_flops_per_sample: Forward + backward FLOPs per sample
        batch_size: Global batch size
        step_time_seconds: Time for one training step
        num_gpus: Number of GPUs used
        peak_flops_per_gpu: Theoretical peak FLOPs (e.g., 312 TFLOPs for A100)

    Returns:
        MFU as a fraction (0-1)
    """
    # Total FLOPs for this step
    total_flops = model_flops_per_sample * batch_size

    # Achieved FLOPs
    achieved_flops = total_flops / step_time_seconds

    # Peak system FLOPs
    peak_flops = peak_flops_per_gpu * num_gpus

    return achieved_flops / peak_flops

# Example: GPT-3 175B on 1024 A100s
model_flops = 6 * 175e9 * 2048  # ~2.15e15 FLOPs per sample (6 * Psi * seq_len)
batch_size = 1024
step_time = 60.0  # seconds
num_gpus = 1024
peak_per_gpu = 312e12  # A100 FP16 peak

mfu = calculate_mfu(model_flops, batch_size, step_time, num_gpus, peak_per_gpu)
print(f"MFU: {mfu:.1%}")  # ~11% for these numbers; higher MFU needs shorter step time
```

### 硬件浮点利用率（HFU）

HFU 把重算（recomputation）也计入：

$$\text{HFU} = \frac{\text{Achieved FLOPs including recomputation}}{\text{Peak FLOPs}}$$

```python
def calculate_hfu(
    model_flops_per_sample: int,
    batch_size: int,
    step_time_seconds: float,
    num_gpus: int,
    peak_flops_per_gpu: float,
    recomputation_ratio: float = 1.0  # 1.0 = no recomputation, 2.0 = full recomputation
) -> float:
    """
    Calculate Hardware FLOP Utilization (includes recomputation).

    The recomputation_ratio accounts for activation checkpointing.
    """
    # Effective FLOPs including recomputation
    effective_flops = model_flops_per_sample * batch_size * recomputation_ratio

    achieved_flops = effective_flops / step_time_seconds
    peak_flops = peak_flops_per_gpu * num_gpus

    return achieved_flops / peak_flops
```

### 通信效率

```python
def calculate_comm_efficiency(
    bytes_communicated: int,
    comm_time_seconds: float,
    network_bandwidth_bytes_per_sec: float
) -> float:
    """Calculate communication bandwidth efficiency."""
    achieved_bandwidth = bytes_communicated / comm_time_seconds
    return achieved_bandwidth / network_bandwidth_bytes_per_sec

# Example: AllReduce of 1GB gradients over 100Gbps InfiniBand
bytes_comm = 1e9
comm_time = 0.1  # 100ms
network_bw = 12.5e9  # 100Gbps = 12.5 GB/s

efficiency = calculate_comm_efficiency(bytes_comm, comm_time, network_bw)
print(f"Communication efficiency: {efficiency:.1%}")
```

### 重叠效率

衡量计算与通信的重叠程度：

```python
class OverlapEfficiencyTracker:
    """Track overlap between compute and communication."""

    def __init__(self):
        self.compute_time = 0
        self.comm_time = 0
        self.total_time = 0

    def record_step(self, compute_ms, comm_ms, step_ms):
        self.compute_time += compute_ms
        self.comm_time += comm_ms
        self.total_time += step_ms

    def overlap_efficiency(self):
        """
        Calculate overlap efficiency.

        Perfect overlap: total = max(compute, comm)
        No overlap: total = compute + comm

        Efficiency = 1 - (actual - theoretical_min) / (theoretical_max - theoretical_min)
        """
        theoretical_min = max(self.compute_time, self.comm_time)
        theoretical_max = self.compute_time + self.comm_time

        if theoretical_max == theoretical_min:
            return 1.0

        return 1 - (self.total_time - theoretical_min) / (theoretical_max - theoretical_min)

    def report(self):
        print(f"\n=== Overlap Efficiency ===")
        print(f"Total compute time: {self.compute_time:.2f}ms")
        print(f"Total comm time: {self.comm_time:.2f}ms")
        print(f"Total wall time: {self.total_time:.2f}ms")
        print(f"Theoretical min (perfect overlap): {max(self.compute_time, self.comm_time):.2f}ms")
        print(f"Theoretical max (no overlap): {self.compute_time + self.comm_time:.2f}ms")
        print(f"Overlap efficiency: {self.overlap_efficiency():.1%}")
```

## 实践中的 Alpha-Beta 模型

### 测量 α 与 β

alpha-beta 模型预测集合通信时间：

$$T = \alpha + \frac{n}{\beta}$$

**在你的硬件上测量**：
```python
import time

def measure_alpha_beta(
    process_group,
    sizes_bytes: list,
    num_warmup: int = 5,
    num_measure: int = 20
) -> tuple:
    """
    Measure alpha (latency) and beta (bandwidth) for a process group.

    Returns:
        (alpha_seconds, beta_bytes_per_second)
    """
    rank = dist.get_rank()
    times = []

    for size in sizes_bytes:
        tensor = torch.zeros(size // 4, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            dist.all_reduce(tensor, group=process_group)
            torch.cuda.synchronize()

        # Measure
        elapsed = []
        for _ in range(num_measure):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(tensor, group=process_group)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)

        avg_time = sum(elapsed) / len(elapsed)
        times.append((size, avg_time))

    if rank == 0:
        # Linear regression: T = alpha + size/beta
        # Solve for alpha and beta using least squares
        import numpy as np
        sizes = np.array([s for s, _ in times])
        measured = np.array([t for _, t in times])

        # Design matrix: [1, size]
        X = np.column_stack([np.ones_like(sizes), sizes])
        # Solve: [alpha, 1/beta] = (X^T X)^{-1} X^T y
        coeffs = np.linalg.lstsq(X, measured, rcond=None)[0]

        alpha = coeffs[0]
        beta = 1 / coeffs[1]

        print(f"\n=== Alpha-Beta Measurement ===")
        print(f"Alpha (latency): {alpha*1e6:.2f} μs")
        print(f"Beta (bandwidth): {beta/1e9:.2f} GB/s")
        print(f"\nPredicted times:")
        for size, actual in times:
            predicted = alpha + size / beta
            error = abs(predicted - actual) / actual * 100
            print(f"  {size/1e6:.1f}MB: predicted={predicted*1e3:.2f}ms, "
                  f"actual={actual*1e3:.2f}ms, error={error:.1f}%")

        return alpha, beta

    return None, None

# Measure with various sizes
sizes = [1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024]
alpha, beta = measure_alpha_beta(dist.group.WORLD, sizes)
```

### 用模型做预测

```python
class CollectiveTimePredictor:
    """Predict collective operation times using alpha-beta model."""

    def __init__(self, alpha_seconds: float, beta_bytes_per_sec: float, world_size: int):
        self.alpha = alpha_seconds
        self.beta = beta_bytes_per_sec
        self.world_size = world_size

    def allreduce_ring(self, size_bytes: int) -> float:
        """Predict ring AllReduce time."""
        # Ring: 2(P-1) messages, each of size n/P
        # Total: 2(P-1)/P * n bytes sent
        effective_size = 2 * (self.world_size - 1) / self.world_size * size_bytes
        num_steps = 2 * (self.world_size - 1)
        return num_steps * self.alpha + effective_size / self.beta

    def allgather(self, size_bytes: int) -> float:
        """Predict AllGather time (size_bytes is per-rank input)."""
        # Each rank receives (P-1) chunks of size size_bytes
        effective_size = (self.world_size - 1) * size_bytes
        num_steps = self.world_size - 1
        return num_steps * self.alpha + effective_size / self.beta

    def reduce_scatter(self, size_bytes: int) -> float:
        """Predict ReduceScatter time (size_bytes is per-rank input)."""
        # Each rank sends/receives (P-1)/P of its input
        effective_size = (self.world_size - 1) / self.world_size * size_bytes
        num_steps = self.world_size - 1
        return num_steps * self.alpha + effective_size / self.beta

    def alltoall(self, size_bytes: int) -> float:
        """Predict AlltoAll time (size_bytes is per-rank total input)."""
        # Each rank sends (P-1)/P of its input
        effective_size = (self.world_size - 1) / self.world_size * size_bytes
        num_steps = self.world_size - 1
        return num_steps * self.alpha + effective_size / self.beta

    def compare_algorithms(self, size_bytes: int):
        """Compare different algorithms for a given size."""
        ring = self.allreduce_ring(size_bytes)

        # Tree algorithm (better for small messages)
        tree_steps = 2 * math.ceil(math.log2(self.world_size))
        tree_time = tree_steps * self.alpha + size_bytes / self.beta * 2

        print(f"\n=== AllReduce Comparison for {size_bytes/1e6:.1f}MB ===")
        print(f"Ring: {ring*1e3:.2f}ms")
        print(f"Tree: {tree_time*1e3:.2f}ms")
        print(f"Recommended: {'Ring' if ring < tree_time else 'Tree'}")
```

## 剖析不同的并行策略

### 数据并行剖析

需要跟踪的关键指标：

```python
class DDPProfiler:
    """Profiler specialized for DistributedDataParallel."""

    def __init__(self, model):
        self.model = model
        self.bucket_times = []
        self.hook_overheads = []
        self.gradient_sizes = []

        # Instrument buckets
        self._instrument_buckets()

    def _instrument_buckets(self):
        """Add timing instrumentation to DDP buckets."""
        # Access internal bucket info (PyTorch internals)
        if hasattr(self.model, '_module_copies'):
            # Extract bucket information
            pass

    def profile_step(self, batch):
        """Profile a single training step."""
        timings = {}

        # Forward pass
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        output = self.model(batch)
        torch.cuda.synchronize()
        timings['forward'] = time.perf_counter() - forward_start

        # Backward pass (triggers AllReduce)
        loss = output.sum()  # Dummy loss
        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        timings['backward_with_allreduce'] = time.perf_counter() - backward_start

        return timings

    def analyze_bucket_efficiency(self):
        """Analyze if bucket sizes are optimal."""
        if not self.bucket_times:
            print("No bucket timing data collected")
            return

        avg_bucket_time = sum(self.bucket_times) / len(self.bucket_times)

        print(f"\n=== Bucket Analysis ===")
        print(f"Number of buckets: {len(self.bucket_times)}")
        print(f"Average bucket AllReduce: {avg_bucket_time*1e3:.2f}ms")
        print(f"Total bucket AllReduce: {sum(self.bucket_times)*1e3:.2f}ms")
```

### 张量并行剖析

```python
class TPProfiler:
    """Profiler for tensor parallelism."""

    def __init__(self, tp_degree: int):
        self.tp_degree = tp_degree
        self.allreduce_times = []
        self.allgather_times = []
        self.split_times = []

    def profile_layer(self, layer_fn, input_tensor):
        """Profile a tensor-parallel layer."""
        results = {}

        torch.cuda.synchronize()
        start = time.perf_counter()

        output = layer_fn(input_tensor)

        torch.cuda.synchronize()
        results['total_time'] = time.perf_counter() - start

        # Estimate communication fraction
        # For column-parallel linear: AllReduce after
        # For row-parallel linear: no communication

        return results

    def analyze_communication_fraction(self):
        """Analyze fraction of time spent in TP communication."""
        total_ar = sum(self.allreduce_times)
        total_ag = sum(self.allgather_times)

        print(f"\n=== Tensor Parallelism Communication ===")
        print(f"Total AllReduce time: {total_ar*1e3:.2f}ms ({len(self.allreduce_times)} calls)")
        print(f"Total AllGather time: {total_ag*1e3:.2f}ms ({len(self.allgather_times)} calls)")
```

### 流水线并行剖析

```python
class PPProfiler:
    """Profiler for pipeline parallelism."""

    def __init__(self, num_stages: int, num_microbatches: int):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.stage_times = [[] for _ in range(num_stages)]
        self.bubble_time = 0

    def profile_schedule(self, schedule_fn):
        """Profile a pipeline schedule execution."""
        rank = dist.get_rank()
        stage = rank  # Assuming 1 stage per rank

        results = {
            'forward_times': [],
            'backward_times': [],
            'send_times': [],
            'recv_times': [],
            'idle_time': 0,
        }

        # Execute schedule with timing
        # ... implementation depends on pipeline framework

        return results

    def calculate_bubble_fraction(self):
        """Calculate pipeline bubble fraction."""
        # Bubble = (P - 1) * microbatch_time / total_time
        p = self.num_stages
        m = self.num_microbatches

        # For 1F1B schedule
        bubble_fraction = (p - 1) / (m + p - 1)

        print(f"\n=== Pipeline Bubble Analysis ===")
        print(f"Stages: {p}, Microbatches: {m}")
        print(f"Theoretical bubble fraction: {bubble_fraction:.1%}")
        print(f"Theoretical efficiency: {1 - bubble_fraction:.1%}")
```

### ZeRO 剖析

```python
class ZeROProfiler:
    """Profiler for ZeRO-style sharding."""

    def __init__(self, stage: int, world_size: int):
        self.stage = stage  # 1, 2, or 3
        self.world_size = world_size
        self.gather_times = []
        self.scatter_times = []
        self.optimizer_times = []

    def profile_forward(self, model, input_batch):
        """Profile forward pass with parameter gathering."""
        results = {'gather_time': 0, 'compute_time': 0}

        if self.stage == 3:
            # Track AllGather for each layer
            for name, param in model.named_parameters():
                gather_start = time.perf_counter()
                # AllGather param
                torch.cuda.synchronize()
                results['gather_time'] += time.perf_counter() - gather_start

        compute_start = time.perf_counter()
        output = model(input_batch)
        torch.cuda.synchronize()
        results['compute_time'] = time.perf_counter() - compute_start

        return results

    def analyze_memory_comm_tradeoff(self):
        """Analyze memory savings vs communication overhead."""
        print(f"\n=== ZeRO Stage {self.stage} Analysis ===")

        # Memory reduction factors (for total model state: params + grads + optimizer)
        P = self.world_size
        mem_factors = {
            1: 16 / (4 + 12 / P),
            2: 16 / (2 + 14 / P),
            3: P,
        }

        # Communication overhead factors (rough bandwidth relative to DP)
        comm_factors = {1: 1.0, 2: 2.0, 3: 3.0}

        print(f"Memory reduction: {mem_factors[self.stage]:.1f}x (approx)")
        print(f"Communication overhead: {comm_factors[self.stage]:.1f}x vs DP (approx)")
```

## 分布式剖析的协调

### 跨 rank 的同步剖析

```python
class DistributedProfiler:
    """Coordinate profiling across all ranks."""

    def __init__(self, output_dir: str):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.output_dir = Path(output_dir) / f"rank_{self.rank}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile_synchronized(self, train_fn, num_steps: int = 5):
        """
        Profile training with synchronized start across ranks.

        Ensures all ranks start profiling at the same time for
        aligned timelines.
        """
        # Barrier to synchronize start
        dist.barrier()

        # Start profiling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(num_steps):
                with torch.cuda.nvtx.range(f"step_{step}"):
                    train_fn()
                prof.step()

        # Barrier to synchronize end
        dist.barrier()

        if self.rank == 0:
            print(f"Profiling complete. Traces saved to {self.output_dir.parent}")

    def aggregate_statistics(self, local_stats: dict) -> dict:
        """Aggregate statistics across all ranks."""
        # Gather all local stats to rank 0
        all_stats = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_stats, local_stats)

        if self.rank == 0:
            # Aggregate
            aggregated = {}
            for key in local_stats.keys():
                values = [s[key] for s in all_stats if key in s]
                aggregated[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5
                }
            return aggregated
        return None
```

### 跨 rank 时间线对齐

```python
def align_timelines(traces_dir: Path):
    """
    Align profiler traces from multiple ranks using barrier timestamps.

    This helps identify relative timing across ranks.
    """
    import json

    traces = []
    for trace_file in traces_dir.glob("rank_*/trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            rank = int(trace_file.parent.name.split('_')[1])
            traces.append((rank, trace))

    # Find barrier events in each trace
    barrier_times = {}
    for rank, trace in traces:
        for event in trace['traceEvents']:
            if 'name' in event and 'barrier' in event['name'].lower():
                if rank not in barrier_times:
                    barrier_times[rank] = []
                barrier_times[rank].append(event['ts'])

    # Calculate offsets relative to rank 0
    offsets = {0: 0}
    if 0 in barrier_times:
        ref_time = barrier_times[0][0]
        for rank in barrier_times:
            if rank != 0:
                offsets[rank] = barrier_times[rank][0] - ref_time

    print("Timeline offsets (μs):", offsets)
    return offsets
```

## 实用的剖析工作流

### 调查协议

```
Step 1: Quick Baseline
├── Single step timing
├── GPU utilization (nvidia-smi)
└── Basic MFU estimate

Step 2: Identify Bottleneck Category
├── Compute vs Communication ratio
├── Memory pressure indicators
└── Straggler detection

Step 3: Detailed Analysis
├── Full Nsight trace (2-3 steps)
├── Memory breakdown
└── Collective timing breakdown

Step 4: Root Cause
├── Kernel-level analysis
├── Algorithm selection validation
└── Overlap efficiency measurement

Step 5: Validate Fix
├── A/B comparison
├── Confirm improvement
└── Check for regressions
```

### 快速健康检查

```python
def quick_health_check(model, dataloader, num_steps=3):
    """Fast profiling to identify obvious issues."""
    rank = dist.get_rank()

    timings = []
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Training step
        with torch.cuda.nvtx.range(f"step_{i}"):
            output = model(batch)
            loss = output.sum()
            loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    # Gather from all ranks
    all_timings = [None] * dist.get_world_size()
    dist.all_gather_object(all_timings, timings)

    if rank == 0:
        print("\n=== Quick Health Check ===")
        for r, times in enumerate(all_timings):
            avg = sum(times) / len(times)
            print(f"Rank {r}: avg={avg*1e3:.1f}ms, times={[f'{t*1e3:.1f}' for t in times]}")

        # Check for stragglers
        avg_times = [sum(t)/len(t) for t in all_timings]
        mean = sum(avg_times) / len(avg_times)
        max_time = max(avg_times)

        if max_time > mean * 1.1:
            slowest = avg_times.index(max_time)
            print(f"\n⚠️  Straggler detected: Rank {slowest} is {max_time/mean:.0%} of mean")
        else:
            print("\n✓ No obvious stragglers")
```

### 对比剖析

```python
class ABProfiler:
    """Compare two configurations."""

    def __init__(self):
        self.results_a = []
        self.results_b = []

    def profile_config(self, name: str, setup_fn, train_fn, num_steps: int = 10):
        """Profile a configuration."""
        setup_fn()  # Apply configuration

        timings = []
        for _ in range(num_steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            train_fn()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

        if name == 'A':
            self.results_a = timings
        else:
            self.results_b = timings

    def compare(self):
        """Compare A and B configurations."""
        if not self.results_a or not self.results_b:
            print("Need both A and B results")
            return

        avg_a = sum(self.results_a) / len(self.results_a)
        avg_b = sum(self.results_b) / len(self.results_b)

        print("\n=== A/B Comparison ===")
        print(f"Config A: {avg_a*1e3:.2f}ms avg")
        print(f"Config B: {avg_b*1e3:.2f}ms avg")
        print(f"Difference: {(avg_b - avg_a)/avg_a*100:+.1f}%")

        # Statistical significance (simple t-test)
        if len(self.results_a) >= 5 and len(self.results_b) >= 5:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
            print(f"p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
```

## 常见问题与诊断

### 问题：硬件强大但 MFU 很低

```python
def diagnose_low_mfu():
    """Diagnostic checklist for low MFU."""
    checks = [
        ("Mixed precision enabled?", "torch.cuda.amp.autocast"),
        ("Tensor Cores used?", "Check for TF32/FP16 kernels in trace"),
        ("Batch size sufficient?", "Small batches → kernel launch overhead dominates"),
        ("Memory bandwidth limited?", "Check memory throughput in Nsight"),
        ("Python overhead?", "Check CPU utilization, GIL contention"),
        ("Data loading bottleneck?", "Profile DataLoader separately"),
    ]

    print("\n=== Low MFU Diagnostic Checklist ===")
    for check, how in checks:
        print(f"[ ] {check}")
        print(f"    → {how}")
```

### 问题：通信耗时过长

```python
def diagnose_slow_communication():
    """Diagnostic for communication bottlenecks."""
    checks = [
        "Measure actual vs theoretical bandwidth",
        "Check for network congestion (multiple jobs)",
        "Verify NCCL algorithm selection (NCCL_DEBUG=INFO)",
        "Check for imbalanced work (stragglers)",
        "Verify overlap is working (Nsight timeline)",
        "Check bucket sizes for DDP",
    ]

    print("\n=== Slow Communication Diagnostic ===")
    for i, check in enumerate(checks, 1):
        print(f"{i}. {check}")
```

### 问题：内存压力

```python
def diagnose_memory_pressure():
    """Diagnostic for memory issues."""

    # Check current state
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n=== Memory Diagnostic ===")
    print(f"Currently allocated: {allocated:.2f}GB")
    print(f"Reserved by PyTorch: {reserved:.2f}GB")
    print(f"Peak allocation: {max_alloc:.2f}GB")
    print(f"Fragmentation: {1 - allocated/reserved:.1%}")

    if max_alloc > reserved * 0.95:
        print("\n⚠️  Near memory limit - consider:")
        print("   - Gradient checkpointing")
        print("   - Smaller batch size")
        print("   - ZeRO stage 2 or 3")
        print("   - Offloading")
```

## 练习题

1. **MFU 测量**：为你的模型实现一套完整的 MFU 测量。把理论 FLOPs（来自模型架构）与达到的 FLOPs（来自步时间）作比较。你能达到多高的效率？

??? success "解答"
    **MFU 测量实现：**

    ```python
    import torch
    import time
    from dataclasses import dataclass

    @dataclass
    class ModelConfig:
        hidden_dim: int
        num_layers: int
        num_heads: int
        vocab_size: int
        seq_len: int
        batch_size: int

    def count_flops_per_token(config: ModelConfig) -> int:
        """Count FLOPs per token for a transformer."""
        H = config.hidden_dim
        L = config.num_layers
        V = config.vocab_size
        S = config.seq_len

        # Per-layer FLOPs
        # Attention: 4H² (QKV proj) + 2S·H (attn scores) + 2S·H (attn output) + H² (output proj)
        # Approximation: 4H² + 2H² = 6H² for projections, plus O(S·H) for attention itself
        attn_flops = 4 * H * H + 2 * S * H + 2 * S * H + H * H  # ~5H² + 4SH

        # MLP: 2 * (H * 4H) + 2 * (4H * H) = 16H²
        mlp_flops = 8 * H * H + 8 * H * H  # 16H²

        # Per-layer total
        layer_flops = attn_flops + mlp_flops

        # All layers
        total_flops = L * layer_flops

        # Embedding and output projection: 2 * V * H
        embedding_flops = 2 * V * H

        # Total per token
        return total_flops + embedding_flops

    def measure_mfu(model, config: ModelConfig, num_warmup=5, num_measure=20):
        """Measure Model FLOP Utilization."""
        device = next(model.parameters()).device

        # Theoretical FLOPs per step
        flops_per_token = count_flops_per_token(config)
        tokens_per_step = config.batch_size * config.seq_len

        # Forward: F, Backward: 2F (gradient for weights and activations)
        flops_per_step = 6 * flops_per_token * tokens_per_step  # 6 = 1 fwd + 2 bwd

        # Peak FLOPs for GPU
        # H100: 989 TFLOP/s (dense FP16/BF16)
        # A100: 312 TFLOP/s (FP16 Tensor Core)
        gpu_name = torch.cuda.get_device_name(device)
        if 'H100' in gpu_name:
            peak_flops = 989e12
        elif 'A100' in gpu_name:
            peak_flops = 312e12
        else:
            peak_flops = 100e12  # Conservative estimate

        # Create dummy input
        input_ids = torch.randint(0, config.vocab_size,
                                  (config.batch_size, config.seq_len),
                                  device=device)

        # Warmup
        for _ in range(num_warmup):
            output = model(input_ids)
            loss = output.sum()
            loss.backward()
            model.zero_grad()

        torch.cuda.synchronize()

        # Measure
        start = time.time()
        for _ in range(num_measure):
            output = model(input_ids)
            loss = output.sum()
            loss.backward()
            model.zero_grad()
            torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate MFU
        time_per_step = elapsed / num_measure
        achieved_flops = flops_per_step / time_per_step
        mfu = achieved_flops / peak_flops

        return {
            'time_per_step_ms': time_per_step * 1000,
            'flops_per_step': flops_per_step,
            'achieved_tflops': achieved_flops / 1e12,
            'peak_tflops': peak_flops / 1e12,
            'mfu': mfu
        }

    # Example usage
    def run_mfu_measurement():
        from transformers import AutoModelForCausalLM, AutoConfig

        config = ModelConfig(
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            vocab_size=32000,
            seq_len=2048,
            batch_size=4
        )

        model_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_config(model_config).cuda().half()

        results = measure_mfu(model, config)

        print(f"Time per step: {results['time_per_step_ms']:.2f} ms")
        print(f"Achieved: {results['achieved_tflops']:.1f} TFLOP/s")
        print(f"Peak: {results['peak_tflops']:.1f} TFLOP/s")
        print(f"MFU: {results['mfu']:.1%}")

    # run_mfu_measurement()
    ```

    **预期结果（7B 模型，H100）：**

    | 指标 | 数值 |
    |--------|-------|
    | 每步时间 | ~180 ms |
    | 达到 | ~450 TFLOP/s |
    | 峰值 | 989 TFLOP/s |
    | **MFU** | **~23%** |

    注意：单 GPU MFU 通常低于多 GPU 张量并行（20–30% 对 40–50%），原因是内存带宽限制。

    $$\boxed{\text{Typical single-GPU MFU: 20-30\%; optimized multi-GPU: 40-50\%}}$$

2. **Alpha-Beta 校准**：为你的集群测量 α 与 β。使用多种消息大小（1KB 到 1GB）。绘制预测时间与实际时间的对比图。线性模型有多准确？

??? success "解答"
    **alpha-beta 校准实现：**

    ```python
    import torch
    import torch.distributed as dist
    import numpy as np
    import time
    from scipy import stats

    def calibrate_alpha_beta(process_group=None, sizes=None, num_trials=50):
        """Calibrate alpha-beta model for collective communication."""

        if sizes is None:
            # Log-spaced sizes from 1KB to 1GB
            sizes = [int(s) for s in np.logspace(10, 30, 20, base=2)]  # 1KB to 1GB

        device = torch.device('cuda')
        results = []

        for size in sizes:
            # Create buffer
            tensor = torch.ones(size // 4, dtype=torch.float32, device=device)

            # Warmup
            for _ in range(5):
                dist.all_reduce(tensor, group=process_group)
            torch.cuda.synchronize()

            # Measure
            times = []
            for _ in range(num_trials):
                torch.cuda.synchronize()
                start = time.time()
                dist.all_reduce(tensor, group=process_group)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)

            median_time = np.median(times)
            results.append({
                'size': size,
                'time': median_time,
                'time_std': np.std(times)
            })

        # Fit linear model: T = alpha + size / beta
        sizes_arr = np.array([r['size'] for r in results])
        times_arr = np.array([r['time'] for r in results])

        # Linear regression: T = alpha + size/beta
        # Rewrite as: T = alpha + (1/beta) * size
        slope, intercept, r_value, _, _ = stats.linregress(sizes_arr, times_arr)

        alpha = intercept  # Latency (seconds)
        beta = 1 / slope    # Bandwidth (bytes/second)

        # Compute R² and prediction accuracy
        predicted = alpha + sizes_arr / beta
        relative_errors = np.abs(predicted - times_arr) / times_arr

        return {
            'alpha_us': alpha * 1e6,  # Convert to microseconds
            'beta_gbps': beta * 8 / 1e9,  # Convert to Gbps
            'r_squared': r_value ** 2,
            'mean_relative_error': np.mean(relative_errors),
            'max_relative_error': np.max(relative_errors),
            'measurements': results
        }

    def plot_calibration(results):
        """Plot predicted vs actual times."""
        import matplotlib.pyplot as plt

        sizes = [r['size'] for r in results['measurements']]
        times = [r['time'] * 1000 for r in results['measurements']]  # ms

        alpha = results['alpha_us'] / 1000  # ms
        beta = results['beta_gbps'] * 1e9 / 8  # bytes/s
        predicted = [alpha + s / beta * 1000 for s in sizes]

        plt.figure(figsize=(10, 6))
        plt.loglog(sizes, times, 'o-', label='Measured', markersize=8)
        plt.loglog(sizes, predicted, '--', label=f'Predicted (α={results["alpha_us"]:.1f}μs, β={results["beta_gbps"]:.1f}Gbps)')
        plt.xlabel('Message Size (bytes)')
        plt.ylabel('Time (ms)')
        plt.title(f'AllReduce Alpha-Beta Calibration (R²={results["r_squared"]:.4f})')
        plt.legend()
        plt.grid(True, which='both', ls='-', alpha=0.2)
        plt.savefig('alpha_beta_calibration.png', dpi=150)
        print("Saved plot to alpha_beta_calibration.png")

    def run_calibration():
        dist.init_process_group('nccl')
        rank = dist.get_rank()

        results = calibrate_alpha_beta()

        if rank == 0:
            print(f"\nAlpha-Beta Calibration Results:")
            print(f"  α (latency): {results['alpha_us']:.1f} μs")
            print(f"  β (bandwidth): {results['beta_gbps']:.1f} Gbps")
            print(f"  R²: {results['r_squared']:.4f}")
            print(f"  Mean error: {results['mean_relative_error']:.1%}")
            print(f"  Max error: {results['max_relative_error']:.1%}")

            # plot_calibration(results)

    # run_calibration()
    ```

    **预期结果（8×H100，NVLink）：**

    | 参数 | 数值 |
    |-----------|-------|
    | α（延迟） | 5–15 μs |
    | β（带宽） | 400–900 Gbps |
    | R² | 0.97–0.99 |
    | 平均误差 | 5–15% |

    **线性模型的准确度：**

    α+size/β 模型对大消息效果很好，但对小消息可能有 20–50% 的误差，原因是：

    - kernel 启动开销占主导
    - Ring 算法的启动成本显著
    - NCCL 分块行为导致非线性扩展

    $$\boxed{\alpha \approx 10\mu\text{s}, \beta \approx 500\text{ Gbps（NVLink）；R}^2 > 0.95}$$

3. **重叠效率**：用 NVTX 标注剖析你的 DDP 训练。计算重叠效率。有多大比例的通信被计算隐藏了？

??? success "解答"
    **NVTX 标注的重叠剖析：**

    ```python
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import time

    # Try to import NVTX
    try:
        import nvtx
        HAS_NVTX = True
    except ImportError:
        HAS_NVTX = False
        class nvtx:
            @staticmethod
            def range(name):
                return contextlib.nullcontext()

    class OverlapProfiler:
        def __init__(self, model):
            self.model = model
            self.compute_times = []
            self.comm_times = []
            self.total_times = []

        def profile_step(self, input_data, num_trials=10):
            """Profile a training step with NVTX markers."""

            # Measure compute-only time (no DDP)
            self.model.module.zero_grad()
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_trials):
                with nvtx.range("forward"):
                    output = self.model.module(input_data)
                    loss = output.sum()
                with nvtx.range("backward_compute"):
                    loss.backward()
                self.model.module.zero_grad()
            torch.cuda.synchronize()
            compute_only = (time.time() - start) / num_trials

            # Measure comm-only time
            # Simulate by doing AllReduce on gradient-sized tensors
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_trials):
                with nvtx.range("allreduce"):
                    for p in self.model.parameters():
                        if p.requires_grad:
                            fake_grad = torch.ones_like(p)
                            dist.all_reduce(fake_grad)
            torch.cuda.synchronize()
            comm_only = (time.time() - start) / num_trials

            # Measure total time with DDP overlap
            self.model.zero_grad()
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_trials):
                with nvtx.range("ddp_step"):
                    with nvtx.range("forward"):
                        output = self.model(input_data)
                        loss = output.sum()
                    with nvtx.range("backward_with_comm"):
                        loss.backward()
                self.model.zero_grad()
            torch.cuda.synchronize()
            total_time = (time.time() - start) / num_trials

            return {
                'compute_only_ms': compute_only * 1000,
                'comm_only_ms': comm_only * 1000,
                'total_time_ms': total_time * 1000,
                'sequential_ms': (compute_only + comm_only) * 1000,
                'overlap_efficiency': self._calc_overlap_efficiency(
                    compute_only, comm_only, total_time
                )
            }

        def _calc_overlap_efficiency(self, compute, comm, total):
            """
            Overlap efficiency = time saved / potential savings

            If total = max(compute, comm): perfect overlap (100%)
            If total = compute + comm: no overlap (0%)
            """
            sequential = compute + comm
            best_case = max(compute, comm)
            potential_savings = sequential - best_case
            actual_savings = sequential - total

            if potential_savings <= 0:
                return 1.0  # Already at best case

            return actual_savings / potential_savings

    def run_overlap_profiling():
        dist.init_process_group('nccl')
        rank = dist.get_rank()

        # Create model
        model = nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(20)]).cuda()
        model = DDP(model)

        profiler = OverlapProfiler(model)
        input_data = torch.randn(32, 4096).cuda()

        results = profiler.profile_step(input_data)

        if rank == 0:
            print(f"\nOverlap Profiling Results:")
            print(f"  Compute only: {results['compute_only_ms']:.2f} ms")
            print(f"  Comm only: {results['comm_only_ms']:.2f} ms")
            print(f"  Sequential (no overlap): {results['sequential_ms']:.2f} ms")
            print(f"  Actual total: {results['total_time_ms']:.2f} ms")
            print(f"  Overlap efficiency: {results['overlap_efficiency']:.1%}")

            hidden_comm = results['comm_only_ms'] - (results['total_time_ms'] - results['compute_only_ms'])
            print(f"  Communication hidden: {hidden_comm:.2f} ms ({hidden_comm/results['comm_only_ms']:.1%})")

    # run_overlap_profiling()
    ```

    **预期结果：**

    | 指标 | 数值 |
    |--------|-------|
    | 仅计算 | ~15 ms |
    | 仅通信 | ~20 ms |
    | 串行 | ~35 ms |
    | 实际合计 | ~24 ms |
    | **重叠效率** | **~79%** |
    | 通信被隐藏 | ~11 ms（55%） |

    **解读：**

    - 79% 重叠效率意味着我们节省了 79% 的潜在节省量
    - 55% 的通信时间被计算隐藏
    - 剩余 45% 是暴露的通信（关键路径）

    $$\boxed{\text{Typical DDP overlap efficiency: 50-80\%; 40-70\% of comm hidden}}$$

4. **掉队者检测**：在 8 块 GPU 上运行训练。人为拖慢一块 GPU（插入 sleep）。测量对整体吞吐的影响。实现自动掉队检测。

??? success "解答"
    **掉队者检测实现：**

    ```python
    import torch
    import torch.distributed as dist
    import time
    import numpy as np

    class StragglerDetector:
        def __init__(self, window_size=100, threshold_std=2.0):
            self.window_size = window_size
            self.threshold_std = threshold_std
            self.step_times = []
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        def record_step(self, step_time):
            """Record step time and check for stragglers."""
            self.step_times.append(step_time)
            if len(self.step_times) > self.window_size:
                self.step_times.pop(0)

        def gather_times(self, local_time):
            """Gather step times from all ranks."""
            times_tensor = torch.tensor([local_time], device='cuda')
            all_times = [torch.zeros(1, device='cuda') for _ in range(self.world_size)]
            dist.all_gather(all_times, times_tensor)
            return [t.item() for t in all_times]

        def detect_straggler(self, all_times):
            """Detect if any rank is a straggler."""
            times = np.array(all_times)
            mean_time = np.mean(times)
            std_time = np.std(times)

            if std_time < 1e-6:  # All times identical
                return None, {}

            stragglers = []
            for rank, t in enumerate(times):
                z_score = (t - mean_time) / std_time
                if z_score > self.threshold_std:
                    stragglers.append({
                        'rank': rank,
                        'time': t,
                        'z_score': z_score,
                        'slowdown': t / mean_time
                    })

            return stragglers, {
                'mean': mean_time,
                'std': std_time,
                'max': np.max(times),
                'min': np.min(times),
                'spread': np.max(times) / np.min(times)
            }

    def simulate_straggler(model, straggler_rank=3, slowdown_ms=50):
        """Simulate a straggler by adding artificial delay."""
        rank = dist.get_rank()

        detector = StragglerDetector()
        input_data = torch.randn(32, 4096).cuda()

        throughputs = {'normal': [], 'straggler': []}

        for phase in ['normal', 'straggler']:
            for step in range(50):
                torch.cuda.synchronize()
                start = time.time()

                # Forward + backward
                output = model(input_data)
                loss = output.sum()
                loss.backward()

                # Simulate straggler
                if phase == 'straggler' and rank == straggler_rank:
                    time.sleep(slowdown_ms / 1000)

                # AllReduce (synchronization point)
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)

                torch.cuda.synchronize()
                step_time = time.time() - start

                # Gather and analyze
                all_times = detector.gather_times(step_time)
                stragglers, stats = detector.detect_straggler(all_times)

                if rank == 0:
                    throughputs[phase].append(1000 / stats['max'])  # samples/sec

                    if stragglers and step % 10 == 0:
                        print(f"Step {step}: Straggler detected!")
                        for s in stragglers:
                            print(f"  Rank {s['rank']}: {s['time']*1000:.1f}ms "
                                  f"(z={s['z_score']:.1f}, {s['slowdown']:.2f}x slower)")

                model.zero_grad()

        if rank == 0:
            normal_throughput = np.mean(throughputs['normal'])
            straggler_throughput = np.mean(throughputs['straggler'])
            impact = (normal_throughput - straggler_throughput) / normal_throughput

            print(f"\n=== Straggler Impact Analysis ===")
            print(f"Normal throughput: {normal_throughput:.1f} samples/sec")
            print(f"With straggler: {straggler_throughput:.1f} samples/sec")
            print(f"Throughput loss: {impact:.1%}")
            print(f"Slowdown factor: {normal_throughput/straggler_throughput:.2f}x")

    def run_straggler_detection():
        dist.init_process_group('nccl')

        model = torch.nn.Sequential(
            *[torch.nn.Linear(4096, 4096) for _ in range(10)]
        ).cuda()

        simulate_straggler(model, straggler_rank=3, slowdown_ms=50)

    # run_straggler_detection()
    ```

    **预期结果（8 块 GPU，在 rank 3 上人为增加 50ms 延迟）：**

    | 指标 | 正常 | 有掉队者 |
    |--------|--------|----------------|
    | 平均步时间 | 15 ms | 65 ms |
    | 吞吐 | 66.7 samples/s | 15.4 samples/s |
    | **吞吐损失** | - | **77%** |

    **关键洞察：** 单块慢 GPU 会拖慢整个训练，因为：

    - AllReduce 需要所有参与者
    - 集合通信操作是同步的
    - 最慢的 rank 决定了步时间

    **缓解策略：**
    1. 跨节点负载均衡
    2. 排除持续掉队的节点
    3. 异步 SGD（需权衡收敛性）

    $$\boxed{\text{50ms straggler on 1/8 GPUs causes } \sim77\% \text{ throughput loss}}$$

5. **桶优化**：用不同桶大小（1MB、25MB、100MB、250MB）剖析 DDP。测量每种配置的步时间。你的模型的最优桶大小是多少？

??? success "解答"
    **桶大小优化：**

    ```python
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import time

    def benchmark_bucket_size(model_fn, bucket_sizes_mb, input_shape, num_warmup=10, num_measure=50):
        """Benchmark DDP with different bucket sizes."""

        results = []
        device = torch.device('cuda')

        for bucket_mb in bucket_sizes_mb:
            bucket_bytes = bucket_mb * 1024 * 1024

            # Create fresh model for each test
            model = model_fn().to(device)
            model = DDP(model, bucket_cap_mb=bucket_mb)

            input_data = torch.randn(*input_shape, device=device)

            # Warmup
            for _ in range(num_warmup):
                output = model(input_data)
                loss = output.sum()
                loss.backward()
                model.zero_grad()

            torch.cuda.synchronize()

            # Measure
            times = []
            for _ in range(num_measure):
                torch.cuda.synchronize()
                start = time.time()
                output = model(input_data)
                loss = output.sum()
                loss.backward()
                torch.cuda.synchronize()
                times.append(time.time() - start)
                model.zero_grad()

            median_time = torch.tensor(times).median().item()
            std_time = torch.tensor(times).std().item()

            results.append({
                'bucket_mb': bucket_mb,
                'median_ms': median_time * 1000,
                'std_ms': std_time * 1000
            })

            # Clean up
            del model
            torch.cuda.empty_cache()

        return results

    def run_bucket_optimization():
        dist.init_process_group('nccl')
        rank = dist.get_rank()

        def create_model():
            return nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(20)])

        bucket_sizes = [1, 5, 10, 25, 50, 100, 250]
        input_shape = (32, 4096)

        results = benchmark_bucket_size(create_model, bucket_sizes, input_shape)

        if rank == 0:
            print("\nBucket Size Optimization Results:")
            print("-" * 45)
            print(f"{'Bucket Size':>12} {'Median Time':>12} {'Std Dev':>10}")
            print("-" * 45)

            best_result = min(results, key=lambda x: x['median_ms'])

            for r in results:
                marker = " *" if r['bucket_mb'] == best_result['bucket_mb'] else ""
                print(f"{r['bucket_mb']:>10} MB {r['median_ms']:>10.2f} ms {r['std_ms']:>8.2f} ms{marker}")

            print("-" * 45)
            print(f"Optimal bucket size: {best_result['bucket_mb']} MB")

    # run_bucket_optimization()
    ```

    **预期结果（20 层 MLP，8 块 GPU）：**

    | 桶大小 | 步时间 | 备注 |
    |-------------|-----------|-------|
    | 1 MB | 28.5 ms | 延迟开销高 |
    | 5 MB | 22.1 ms | |
    | 10 MB | 19.8 ms | |
    | **25 MB** | **18.2 ms** | **最优** |
    | 50 MB | 18.5 ms | |
    | 100 MB | 19.1 ms | 重叠较少 |
    | 250 MB | 21.3 ms | 重叠差 |

    **分析：**

    - **过小（1–5 MB）**：每个桶的延迟开销高
    - **最优（10–50 MB）**：延迟摊销与重叠机会的良好平衡
    - **过大（100+ MB）**：重叠粒度不足，桶在计算结束后才完成

    **权衡：**
    - 更小的桶 → 更多 AllReduce 调用 → 更高的延迟开销
    - 更大的桶 → 与计算的重叠更少

    $$\boxed{\text{Optimal bucket size typically 10-50 MB; default 25 MB is reasonable}}$$

6. **内存剖析**：跟踪一个训练步中的内存使用。找出峰值分配点。什么占用内存最多：参数、梯度、激活值，还是优化器状态？

??? success "解答"
    **内存剖析实现：**

    ```python
    import torch
    import torch.nn as nn
    from torch.cuda import memory_allocated, max_memory_allocated, reset_peak_memory_stats

    class MemoryProfiler:
        def __init__(self):
            self.checkpoints = []

        def checkpoint(self, name):
            """Record current memory usage."""
            torch.cuda.synchronize()
            self.checkpoints.append({
                'name': name,
                'allocated_mb': memory_allocated() / 1e6,
                'peak_mb': max_memory_allocated() / 1e6
            })

        def reset(self):
            self.checkpoints = []
            reset_peak_memory_stats()

        def report(self):
            """Print memory usage report."""
            print("\n=== Memory Profile ===")
            print(f"{'Checkpoint':<30} {'Allocated':>12} {'Peak':>12} {'Delta':>12}")
            print("-" * 70)

            prev_alloc = 0
            for cp in self.checkpoints:
                delta = cp['allocated_mb'] - prev_alloc
                print(f"{cp['name']:<30} {cp['allocated_mb']:>10.1f} MB {cp['peak_mb']:>10.1f} MB {delta:>+10.1f} MB")
                prev_alloc = cp['allocated_mb']

            print("-" * 70)
            print(f"{'Peak memory':>30} {max(c['peak_mb'] for c in self.checkpoints):>10.1f} MB")

    def profile_training_step(model, optimizer, input_data, target):
        """Profile memory through a complete training step."""
        profiler = MemoryProfiler()
        profiler.reset()

        profiler.checkpoint("Initial")

        # Model parameters
        model = model.cuda()
        profiler.checkpoint("After model.cuda()")

        # Optimizer state (allocates momentum, variance buffers on first step)
        # We'll trigger this by doing one dummy step
        dummy_input = torch.randn_like(input_data).cuda()
        dummy_output = model(dummy_input)
        dummy_output.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        profiler.checkpoint("After optimizer init")

        # Fresh start for actual measurement
        torch.cuda.empty_cache()
        profiler.checkpoint("After cache clear")

        # Forward pass - activations allocated
        input_data = input_data.cuda()
        profiler.checkpoint("Input on GPU")

        output = model(input_data)
        profiler.checkpoint("After forward (activations)")

        # Loss computation
        loss = output.sum()
        profiler.checkpoint("After loss")

        # Backward pass - gradients allocated
        loss.backward()
        profiler.checkpoint("After backward (gradients)")

        # Optimizer step
        optimizer.step()
        profiler.checkpoint("After optimizer step")

        # Zero gradients
        optimizer.zero_grad()
        profiler.checkpoint("After zero_grad")

        profiler.report()

        return profiler.checkpoints

    def analyze_memory_breakdown(model_params, batch_size, seq_len, hidden_dim, num_layers):
        """Theoretical memory breakdown."""
        bytes_per_param = 4  # FP32

        # Parameters
        param_memory = model_params * bytes_per_param

        # Gradients (same size as parameters)
        gradient_memory = model_params * bytes_per_param

        # Optimizer state (Adam: 2 states per parameter)
        optimizer_memory = model_params * bytes_per_param * 2

        # Activations (rough estimate for transformer)
        # Per layer: ~34 * B * S * H bytes
        activation_memory = num_layers * 34 * batch_size * seq_len * hidden_dim

        total = param_memory + gradient_memory + optimizer_memory + activation_memory

        breakdown = {
            'parameters': param_memory / 1e9,
            'gradients': gradient_memory / 1e9,
            'optimizer': optimizer_memory / 1e9,
            'activations': activation_memory / 1e9,
            'total': total / 1e9
        }

        print("\n=== Theoretical Memory Breakdown ===")
        print(f"{'Component':<20} {'Memory (GB)':>12} {'Percentage':>12}")
        print("-" * 50)
        for name, mem in breakdown.items():
            if name != 'total':
                pct = mem / breakdown['total'] * 100
                print(f"{name:<20} {mem:>10.2f} GB {pct:>10.1f}%")
        print("-" * 50)
        print(f"{'TOTAL':<20} {breakdown['total']:>10.2f} GB")

        return breakdown

    def run_memory_profiling():
        # 7B parameter model approximation
        model = nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(32)])

        optimizer = torch.optim.Adam(model.parameters())
        input_data = torch.randn(32, 4096)
        target = torch.randn(32, 4096)

        profile_training_step(model, optimizer, input_data, target)

        # Theoretical breakdown for comparison
        num_params = sum(p.numel() for p in model.parameters())
        analyze_memory_breakdown(
            model_params=num_params,
            batch_size=32,
            seq_len=2048,
            hidden_dim=4096,
            num_layers=32
        )

    # run_memory_profiling()
    ```

    **预期内存分解（7B 模型，batch=4，seq=2048）：**

    | 组成 | 内存 | 百分比 |
    |-----------|--------|------------|
    | 参数 | 28 GB | 16% |
    | 梯度 | 28 GB | 16% |
    | 优化器（Adam） | 56 GB | 32% |
    | **激活值** | **64 GB** | **36%** |
    | **合计** | **176 GB** | 100% |

    **峰值出现在反向传播期间**，此时同时满足：
    - 所有激活值仍被梯度计算所需要
    - 梯度正在被分配

    **关键洞察：**
    1. 优化器状态主导静态内存（非激活部分的 48%）
    2. 激活值主导动态内存，并随批大小扩展
    3. 峰值内存 ≈ 参数 + 梯度 + 优化器 + 激活值

    $$\boxed{\text{Activations: 36\%, Optimizer: 32\%, Params+Grads: 32\%}}$$

## 关键要点

1. **测量，而非猜测**：剖析把直觉转化为可操作的数据。

2. **多种工具对应多种用途**：Nsight 用于详细轨迹，PyTorch Profiler 用于快速检查，NCCL 调试用于通信分析。

3. **MFU 是终极指标**：它刻画了你对硬件的利用程度。

4. **alpha-beta 模型能预测通信**：为你的集群校准它，以理解瓶颈。

5. **重叠效率很重要**：计算+通信与 max(计算, 通信) 之间的差距就是你的机会。

6. **掉队者扼杀扩展性**：在同步训练中，一块慢 GPU 会影响所有 GPU。

7. **迭代地剖析**：先广泛扫描，再聚焦到瓶颈上。