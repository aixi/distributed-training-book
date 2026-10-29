---
title: "通信与计算重叠"
subtitle: "通过并发执行隐藏延迟"
---

<div class="chapter-opener" markdown>
最快的通信，是你无需等待的通信。通过把数据传输与计算重叠，我们可以逼近通信成本趋近于零的理论极限。
</div>

<div class="investigation-question" markdown>
**问题**：一个训练步有 100ms 计算和 50ms 通信。朴素做法需要 150ms。完美的重叠只需 100ms。是什么决定了你落在这两个极端之间的哪一点？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 4 章](../foundations/04-alpha-beta-model.md)（α-β 模型）、[第 14 章](../parallelism/14-data-parallelism-associativity.md)（数据并行与梯度同步）

    **关键洞察**：反向传播中，靠后层的梯度最先就绪。把 AllReduce 与仍在进行的反向计算重叠，可以隐藏大部分通信延迟——目标是 max(计算, 通信)，而不是两者之和。

## 重叠的机会

考虑一个典型的训练迭代：

```
Naive execution (sequential):

[Forward Pass] → [Backward Pass] → [AllReduce] → [Update]
     40ms            50ms             50ms         10ms
                                                        Total: 150ms
```

但计算和通信使用不同的资源：

- **计算**：GPU 的 SM（流式多处理器）
- **通信**：NVLink/InfiniBand + 网络接口

它们可以并发执行：

```
Overlapped execution:

[Forward Pass] → [Backward Pass + AllReduce overlap] → [Update]
     40ms                      60ms                       10ms
                                                              Total: 110ms
```

关键洞察：**靠前层的梯度在反向传播的后期才被算出**。我们可以在仍在计算靠前层梯度的同时，把靠后层的梯度发送出去。

### 理论加速比

定义：

- $T_c$：计算时间（前向 + 反向）
- $T_m$：通信时间（AllReduce）
- $\alpha$：重叠比例（0 = 不重叠，1 = 完全重叠）

有重叠的总时间（以计算／通信中较长者为上限）：

$$T_{\text{total}} = T_c + T_m - \alpha \cdot \min(T_c, T_m)$$

**加速比**：

$$\text{Speedup} = \frac{T_c + T_m}{T_c + T_m - \alpha \cdot \min(T_c, T_m)}$$

对完全重叠（$\alpha = 1$）：加速比 $= \frac{T_c + T_m}{\max(T_c, T_m)}$

| $T_m/T_c$ | 不重叠 | 50% 重叠 | 完全重叠 |
|-----------|--------|----------|----------|
| 0.25 | 1.0× | 1.11× | 1.25× |
| 0.50 | 1.0× | 1.20× | 1.50× |
| 1.00 | 1.0× | 1.33× | 2.00× |
| 2.00 | 1.0× | 1.20× | 1.50× |

当通信占主导（$T_m \gg T_c$）时，重叠变得至关重要。

## 反向传播的结构

理解反向传播的结构是重叠的关键。

### 逐层反向传播

```python
def backward_pass(model, loss):
    """Standard backward pass - layer by layer."""
    gradients = {}

    # Start from loss, work backward through layers
    grad_output = loss.backward()  # dL/d(output)

    # Layer N (last layer)
    grad_output, grad_params_N = layer_N.backward(grad_output)
    gradients['layer_N'] = grad_params_N
    # At this point, layer_N gradients are COMPLETE and can be communicated

    # Layer N-1
    grad_output, grad_params_N1 = layer_N1.backward(grad_output)
    gradients['layer_N-1'] = grad_params_N1
    # layer_N-1 gradients now complete

    # ... continue to layer 0

    return gradients
```

**关键观察**：第 $k$ 层的梯度在第 $k-1$ 层开始计算之前就已经完整。

### 重叠窗口

```
Layer computation during backward:

Layer 6: [=====]                              ← Gradients ready first
Layer 5:       [=====]
Layer 4:             [=====]
Layer 3:                   [=====]
Layer 2:                         [=====]
Layer 1:                               [=====]
Layer 0:                                     [=====]  ← Gradients ready last

Communication can start here:
         ↓
         [Comm L6][Comm L5][Comm L4][Comm L3][Comm L2][Comm L1][Comm L0]
```

重叠窗口等于反向计算时间减去一层。

## 梯度分桶

### 为何分桶？

小消息开销高：

- 每条消息的延迟 $\alpha$ 在小传输中占主导
- 网络利用率低
- NCCL kernel 启动开销

**解法**：在通信前把梯度聚合进桶（bucket）。

### 桶的形成

```python
class GradientBucketer:
    """Accumulate gradients into fixed-size buckets."""

    def __init__(self, bucket_size_mb: float = 25.0):
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets = []
        self.current_bucket = []
        self.current_size = 0

    def add_gradient(self, param_name: str, gradient: torch.Tensor):
        """Add a gradient to the current bucket."""
        grad_size = gradient.numel() * gradient.element_size()

        if self.current_size + grad_size > self.bucket_size_bytes:
            # Bucket full, finalize and start new bucket
            if self.current_bucket:
                self.buckets.append(self._finalize_bucket())
            self.current_bucket = []
            self.current_size = 0

        self.current_bucket.append((param_name, gradient))
        self.current_size += grad_size

    def _finalize_bucket(self) -> torch.Tensor:
        """Flatten bucket into contiguous tensor."""
        flat_grads = []
        for _, grad in self.current_bucket:
            flat_grads.append(grad.view(-1))
        return torch.cat(flat_grads)

    def flush(self):
        """Finalize any remaining gradients."""
        if self.current_bucket:
            self.buckets.append(self._finalize_bucket())
            self.current_bucket = []
            self.current_size = 0
```

### 最优桶大小

桶大小影响重叠质量：

**太小**：

- 每桶延迟开销高
- 许多小的 AllReduce 操作
- 通信跟不上计算

**太大**：

- 通信启动延迟
- 重叠机会减少
- 缓冲带来内存压力

**寻找最优大小**：

$$B^* = \underset{B}{\text{argmax}} \left[ \alpha(B) - \frac{\alpha_{\text{comm}}}{B} \right]$$

其中：

- $\alpha(B)$：桶大小为 $B$ 时的重叠比例
- $\alpha_{\text{comm}}$：每桶的延迟开销

经验上，25–50 MB 的桶对多数配置都有效。

### PyTorch DDP 分桶

```python
# PyTorch DDP bucket configuration
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,  # Bucket size in MB
    gradient_as_bucket_view=True,  # Avoid extra memory copies
    find_unused_parameters=False,  # Faster if no unused params
)
```

## 用于重叠的 CUDA 流

### 流的基础

CUDA 流使并发执行成为可能：

```python
# Create separate streams
compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

# Operations on different streams can overlap
with torch.cuda.stream(compute_stream):
    # Compute operations
    y = torch.matmul(A, B)

with torch.cuda.stream(comm_stream):
    # Communication operations
    dist.all_reduce(tensor, async_op=True)
```

### 用事件同步

```python
class StreamOverlap:
    """Manage compute-communication overlap with CUDA streams."""

    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
        self.comm_stream = torch.cuda.Stream()
        self.comm_handles = []

    def backward_with_overlap(self, model, loss):
        """Backward pass with overlapped communication."""
        # Backward happens on compute stream
        with torch.cuda.stream(self.compute_stream):
            loss.backward()

        # As gradients become ready, queue communication
        for bucket in self.get_ready_buckets():
            # Record event when bucket's compute is done
            event = torch.cuda.Event()
            event.record(self.compute_stream)

            # Comm stream waits for compute
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(event)
                handle = dist.all_reduce(bucket, async_op=True)
                self.comm_handles.append(handle)

    def synchronize(self):
        """Wait for all communication to complete."""
        for handle in self.comm_handles:
            handle.wait()
        self.comm_stream.synchronize()
        self.comm_handles.clear()
```

### 流优先级

```python
# High priority stream for communication
# (ensures comm doesn't starve)
comm_stream = torch.cuda.Stream(priority=-1)  # Higher priority

# Normal priority for compute
compute_stream = torch.cuda.Stream(priority=0)
```

## 反向钩子机制

PyTorch 用钩子在恰当时机触发通信。

### 梯度钩子

```python
class OverlappedAllReduce:
    """Use hooks to trigger AllReduce as gradients become ready."""

    def __init__(self, model, process_group):
        self.model = model
        self.process_group = process_group
        self.handles = []
        self.comm_stream = torch.cuda.Stream()

        # Register backward hooks
        for param in model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param)
                )

    def _make_hook(self, param):
        def hook(grad):
            # Record when gradient is ready
            event = torch.cuda.Event()
            event.record()

            # Launch AllReduce on comm stream
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(event)
                handle = dist.all_reduce(
                    param.grad,
                    group=self.process_group,
                    async_op=True
                )
                self.handles.append(handle)

            return grad
        return hook

    def finish_step(self):
        """Wait for all AllReduce operations."""
        self.comm_stream.synchronize()
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
```

### 感知分桶的钩子

```python
class BucketedOverlapAllReduce:
    """Bucket gradients before triggering AllReduce."""

    def __init__(self, model, process_group, bucket_size_mb=25):
        self.process_group = process_group
        self.bucket_size = int(bucket_size_mb * 1024 * 1024)

        # Group parameters into buckets (reverse order for backward)
        self.buckets = self._create_buckets(model)
        self.pending_grads = {}
        self.comm_stream = torch.cuda.Stream()
        self.handles = []

        # Register hooks
        for bucket_id, (params, _) in enumerate(self.buckets):
            for param in params:
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param, bucket_id)
                )

    def _create_buckets(self, model):
        """Create buckets in reverse parameter order."""
        buckets = []
        current_params = []
        current_size = 0

        # Reverse order: last params first (computed first in backward)
        for param in reversed(list(model.parameters())):
            if not param.requires_grad:
                continue

            param_size = param.numel() * param.element_size()

            if current_size + param_size > self.bucket_size:
                if current_params:
                    flat_buffer = self._allocate_flat_buffer(current_params)
                    buckets.append((current_params.copy(), flat_buffer))
                current_params = []
                current_size = 0

            current_params.append(param)
            current_size += param_size

        if current_params:
            flat_buffer = self._allocate_flat_buffer(current_params)
            buckets.append((current_params, flat_buffer))

        return buckets

    def _make_hook(self, param, bucket_id):
        def hook(grad):
            self.pending_grads[param] = True
            self._check_bucket_ready(bucket_id)
            return grad
        return hook

    def _check_bucket_ready(self, bucket_id):
        """Launch AllReduce if all gradients in bucket are ready."""
        params, flat_buffer = self.buckets[bucket_id]

        if all(p in self.pending_grads for p in params):
            # Copy gradients to flat buffer
            offset = 0
            for param in params:
                numel = param.numel()
                flat_buffer[offset:offset+numel].copy_(param.grad.view(-1))
                offset += numel

            # Launch AllReduce
            with torch.cuda.stream(self.comm_stream):
                handle = dist.all_reduce(
                    flat_buffer,
                    group=self.process_group,
                    async_op=True
                )
                self.handles.append((handle, params, flat_buffer))
```

## 重叠分析

### 测量重叠效率

```python
class OverlapProfiler:
    """Profile compute-communication overlap."""

    def __init__(self):
        self.compute_events = []
        self.comm_events = []

    def profile_step(self, model, data, target, criterion):
        """Profile one training step."""
        # Mark compute regions
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)

        # Forward pass
        compute_start.record()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        compute_end.record()

        # Communication (if using custom overlap, track separately)
        comm_start.record()
        # ... AllReduce ...
        comm_end.record()

        torch.cuda.synchronize()

        compute_time = compute_start.elapsed_time(compute_end)
        comm_time = comm_start.elapsed_time(comm_end)
        total_time = compute_start.elapsed_time(comm_end)

        # Calculate overlap
        sequential_time = compute_time + comm_time
        overlap_time = sequential_time - total_time
        overlap_fraction = overlap_time / comm_time if comm_time > 0 else 0

        return {
            'compute_ms': compute_time,
            'comm_ms': comm_time,
            'total_ms': total_time,
            'overlap_fraction': overlap_fraction,
            'speedup': sequential_time / total_time if total_time > 0 else 1
        }
```

### 重叠可视化

```
Timeline visualization:

Time →  0ms    20ms    40ms    60ms    80ms   100ms
        |       |       |       |       |       |
Compute:[============================]
                  ↑ gradient ready
                  |
Comm:             [================]
                  ← overlap →

Overlap region: 60ms - 40ms = 20ms
Overlap fraction: 20ms / 40ms = 50%
```

### 识别重叠瓶颈

```python
def analyze_overlap_bottleneck(model, bucket_size_mb=25):
    """Identify what limits overlap."""
    params = list(model.parameters())

    # Time to compute all gradients
    total_param_bytes = sum(p.numel() * p.element_size() for p in params)

    # Number of buckets
    bucket_bytes = bucket_size_mb * 1024 * 1024
    num_buckets = (total_param_bytes + bucket_bytes - 1) // bucket_bytes

    # Compute time per bucket (assume uniform)
    compute_time_per_bucket = total_backward_time / num_buckets

    # Communication time per bucket
    # Using α-β model: T = α + n/β
    comm_time_per_bucket = alpha + bucket_bytes / bandwidth

    # Overlap is limited by slower of compute or comm per bucket
    if compute_time_per_bucket > comm_time_per_bucket:
        bottleneck = "compute"
        # Comm can keep up, limited by when gradients are ready
    else:
        bottleneck = "communication"
        # Comm slower than gradient production, will queue up

    return {
        'bottleneck': bottleneck,
        'compute_per_bucket': compute_time_per_bucket,
        'comm_per_bucket': comm_time_per_bucket,
        'num_buckets': num_buckets
    }
```

## 双缓冲

要在迭代之间持续重叠，使用双缓冲。

### 通信期间更新权重

```python
class DoubleBufferedOptimizer:
    """
    Use two sets of weights to overlap update with next iteration.
    """

    def __init__(self, model, base_optimizer, lr: float):
        self.model = model
        self.base_optimizer = base_optimizer
        self.lr = lr

        # Two sets of weights
        self.weights_a = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.weights_b = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.current = 'a'

        # Streams
        self.compute_stream = torch.cuda.Stream()
        self.update_stream = torch.cuda.Stream()

    def step_async(self, gradients):
        """
        Apply gradients to inactive buffer while compute uses active buffer.
        """
        inactive = self.weights_b if self.current == 'a' else self.weights_a

        with torch.cuda.stream(self.update_stream):
            for name, grad in gradients.items():
                # Apply update to inactive weights
                inactive[name].add_(grad, alpha=-self.lr)

    def swap_buffers(self):
        """Swap active and inactive buffers."""
        # Wait for update to complete
        self.update_stream.synchronize()

        # Swap
        self.current = 'b' if self.current == 'a' else 'a'

        # Copy active weights to model
        active = self.weights_a if self.current == 'a' else self.weights_b
        for name, param in self.model.named_parameters():
            param.data.copy_(active[name])
```

### 流水线双缓冲

```python
class PipelinedDataLoader:
    """
    Prefetch next batch while current batch is processing.
    """

    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self._prefetch()

    def _prefetch(self):
        """Load and transfer next batch asynchronously."""
        try:
            batch = next(self.dataloader)
            with torch.cuda.stream(self.stream):
                self.next_batch = tuple(
                    t.cuda(non_blocking=True) if isinstance(t, torch.Tensor)
                    else t for t in batch
                )
        except StopIteration:
            self.next_batch = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        # Wait for prefetch to complete
        torch.cuda.current_stream().wait_stream(self.stream)

        batch = self.next_batch
        self._prefetch()  # Start next prefetch
        return batch
```

## ZeRO 与重叠

ZeRO 优化需要谨慎的重叠策略。

### ZeRO 第 1 阶段：优化器状态重叠

```python
class ZeRO1WithOverlap:
    """
    ZeRO-1 with overlapped gradient reduction and optimizer step.
    """

    def __init__(self, model, optimizer, process_group):
        self.model = model
        self.optimizer = optimizer
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)

        # Partition optimizer states
        self.param_to_partition = self._partition_params()

        # Streams
        self.reduce_stream = torch.cuda.Stream()
        self.update_stream = torch.cuda.Stream()

    def backward_and_step(self, loss):
        """Overlapped backward, ReduceScatter, and optimizer step."""
        handles = []

        # Backward pass
        loss.backward()

        # For each bucket of gradients
        for bucket_params in self.buckets:
            # Flatten gradients
            flat_grad = self._flatten_grads(bucket_params)

            # ReduceScatter on reduce stream
            with torch.cuda.stream(self.reduce_stream):
                output = torch.zeros(
                    flat_grad.size(0) // self.world_size,
                    device=flat_grad.device
                )
                handle = dist.reduce_scatter(
                    output, list(flat_grad.chunk(self.world_size)),
                    group=self.process_group,
                    async_op=True
                )
                handles.append((handle, bucket_params, output))

        # As ReduceScatters complete, apply optimizer on update stream
        for handle, bucket_params, reduced_grad in handles:
            handle.wait()

            with torch.cuda.stream(self.update_stream):
                self._apply_optimizer_step(bucket_params, reduced_grad)
```

### ZeRO 第 3 阶段：预取参数

```python
class ZeRO3PrefetchScheduler:
    """
    Schedule parameter gathering to overlap with compute.
    """

    def __init__(self, model, process_group, prefetch_count=2):
        self.model = model
        self.process_group = process_group
        self.prefetch_count = prefetch_count

        # Parameter sharding info
        self.param_shards = self._shard_parameters()

        # Prefetch streams
        self.prefetch_streams = [
            torch.cuda.Stream() for _ in range(prefetch_count)
        ]
        self.prefetch_buffers = {}

    def forward_with_prefetch(self, input_tensor):
        """Forward pass with parameter prefetching."""
        layers = list(self.model.children())

        # Start prefetching first layers
        for i in range(min(self.prefetch_count, len(layers))):
            self._start_prefetch(layers[i], stream_idx=i)

        # Execute layers with rolling prefetch
        x = input_tensor
        for i, layer in enumerate(layers):
            # Wait for current layer's parameters
            self._wait_prefetch(layer)

            # Execute layer
            x = layer(x)

            # Release parameters (for memory efficiency)
            self._release_params(layer)

            # Start prefetching future layer
            next_prefetch = i + self.prefetch_count
            if next_prefetch < len(layers):
                stream_idx = next_prefetch % self.prefetch_count
                self._start_prefetch(layers[next_prefetch], stream_idx)

        return x

    def _start_prefetch(self, layer, stream_idx):
        """Begin AllGather for layer's parameters."""
        stream = self.prefetch_streams[stream_idx]

        with torch.cuda.stream(stream):
            for param in layer.parameters():
                shard = self.param_shards[param]
                full_param = torch.empty(
                    param.numel() * self.world_size,
                    device=param.device
                )
                dist.all_gather_into_tensor(
                    full_param, shard, group=self.process_group
                )
                self.prefetch_buffers[param] = full_param

    def _wait_prefetch(self, layer):
        """Wait for layer's parameters to be gathered."""
        for param in layer.parameters():
            full_param = self.prefetch_buffers[param]
            param.data = full_param.view(param.shape)
```

## 通信-计算平衡

### 重叠收益最大的情形

以下情形重叠收益最大：

1. **计算／通信比例均衡**：任何一方都不应严重占主导
2. **许多小操作**：有更多交错的机会
3. **高带宽网络**：通信能跟得上计算

### 重叠的极限

即便完全重叠，你仍受限于：

$$T_{\text{min}} = \max(T_c, T_m)$$

如果通信比计算慢（$T_m > T_c$），最终会排队：

```
Compute-bound (Tc > Tm): Perfect overlap possible
Time:   [=========Compute==========]
             [===Comm===]
                    Total = Tc ✓

Communication-bound (Tm > Tc): Overlap limited
Time:   [===Compute===]
        [========Comm========]
                    Total = Tm ✗
```

### 再平衡策略

当通信受限时：

```python
def rebalance_for_overlap(model_config, comm_config):
    """Adjust configuration to improve overlap."""
    tc = estimate_compute_time(model_config)
    tm = estimate_comm_time(model_config, comm_config)

    if tm > tc:
        # Communication bound - reduce communication volume
        options = [
            ("gradient_compression", "Reduce gradient bits"),
            ("increase_batch", "Larger batches, fewer steps"),
            ("tensor_parallelism", "Split model, reduce AllReduce size"),
        ]
    else:
        # Compute bound - communication easily hidden
        options = [
            ("smaller_buckets", "More granular overlap"),
            ("reduce_prefetch", "Save memory"),
        ]

    return options
```

## NCCL 重叠模式

### 分组操作

```python
def overlapped_allreduce_with_compute(tensors, compute_fn):
    """
    Overlap multiple AllReduce operations with compute.
    """
    # Start all AllReduce operations
    handles = []
    for tensor in tensors:
        handle = dist.all_reduce(tensor, async_op=True)
        handles.append(handle)

    # Do compute while communication proceeds
    result = compute_fn()

    # Wait for all communication
    for handle in handles:
        handle.wait()

    return result, tensors
```

### 用于批处理 NCCL 组

```python
def batched_collectives():
    """Batch multiple collectives for efficiency."""
    with dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send_tensor, dst_rank),
        dist.P2POp(dist.irecv, recv_tensor, src_rank),
    ]) as handles:
        # All operations launched together
        pass

    # Wait for completion
    for handle in handles:
        handle.wait()
```

## 流水线并行中的重叠

流水线并行天然制造重叠机会。

### 交错的 1F1B 调度

```python
class InterleavedPipelineOverlap:
    """
    Pipeline schedule that overlaps send/recv with compute.
    """

    def __init__(self, num_stages, num_microbatches):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

        # Separate streams for compute and communication
        self.compute_stream = torch.cuda.Stream()
        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

    def execute_stage(self, stage_fn, input_tensor, is_forward):
        """Execute one pipeline stage with overlapped communication."""
        # Receive input (if not first stage)
        if self.stage_id > 0:
            with torch.cuda.stream(self.recv_stream):
                recv_handle = self._recv_activation(input_tensor)
            torch.cuda.current_stream().wait_stream(self.recv_stream)

        # Compute
        with torch.cuda.stream(self.compute_stream):
            output = stage_fn(input_tensor)

        # Send output (if not last stage) - overlapped with next recv
        if self.stage_id < self.num_stages - 1:
            # Wait for compute
            self.send_stream.wait_stream(self.compute_stream)
            with torch.cuda.stream(self.send_stream):
                self._send_activation(output)

        return output
```

## 实践实现

### PyTorch DDP 配置

```python
def configure_ddp_for_overlap(model, local_rank):
    """Configure DDP for optimal overlap."""
    return DistributedDataParallel(
        model.cuda(local_rank),
        device_ids=[local_rank],
        output_device=local_rank,

        # Bucketing
        bucket_cap_mb=25,

        # Avoid extra copies
        gradient_as_bucket_view=True,

        # Static graph for optimization (if model is fixed)
        static_graph=True,

        # Don't find unused (faster hooks)
        find_unused_parameters=False,
    )
```

### DeepSpeed 重叠设置

```python
deepspeed_config = {
    "train_batch_size": 1024,
    "gradient_accumulation_steps": 4,

    "zero_optimization": {
        "stage": 2,

        # Overlap settings
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,  # 50MB

        # Prefetch
        "prefetch_bucket_size": 5e7,
    },

    "communication_data_type": "fp16",
}
```

### FSDP 重叠配置

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    ShardingStrategy,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,

    # Prefetch during backward
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,

    # Forward prefetch
    forward_prefetch=True,

    # Limit concurrent AllGather
    limit_all_gathers=True,
)
```

## 重叠调试

### 常见问题

**1. 观察不到重叠**

```python
# Check if operations are on same stream (bad)
print(f"Compute stream: {torch.cuda.current_stream()}")
print(f"Comm stream: {dist.get_default_stream()}")

# Should be different for overlap
```

**2. 重叠中的死锁**

```python
# Ensure proper synchronization order
event = torch.cuda.Event()
event.record(compute_stream)
comm_stream.wait_event(event)  # Comm waits for compute
# Don't do: compute_stream.wait_stream(comm_stream) here
```

**3. 重叠导致内存爆炸**

```python
# Too many in-flight operations
# Limit concurrent buckets
max_concurrent_buckets = 2
if len(active_handles) >= max_concurrent_buckets:
    active_handles[0].wait()
    active_handles.pop(0)
```

### 对重叠做剖析

```python
# Use PyTorch profiler with CUDA events
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    train_step()

# Look for overlapping CUDA kernels in trace
prof.export_chrome_trace("overlap_trace.json")
```

## 练习题

1. **重叠计算**：一个模型有 10 层，每层反向需要 5ms。每层梯度的通信需要 3ms。使用分桶（每桶 2 层），最大理论重叠是多少？总时间是多少？

??? success "解答"
    **配置：**

    - 10 层，每层反向 5ms → 反向计算共 50ms
    - 每层通信 3ms
    - 分桶：每桶 2 层 → 5 个桶
    - 每桶：10ms 计算，6ms 通信

    **不重叠（顺序执行）：**

    $$T_{\text{sequential}} = 50\text{ms（计算）} + 30\text{ms（通信）} = 80\text{ms}$$

    **有重叠（流水化）：**

    ```
    Time:   0   10   20   30   40   50   56ms
            |----|----|----|----|----|----|

    Compute: [B1  ][B2  ][B3  ][B4  ][B5  ]
    Comm:        [C1  ][C2  ][C3  ][C4  ][C5 ]
                      ↑ overlap region ↑
    ```

    由于计算（10ms）> 通信（6ms），除最后一个桶外，通信被完全重叠。

    **重叠分析：**

    - 桶 1-4：通信与下一桶的计算完全重叠
    - 桶 5：通信无法重叠（后面没有计算了）

    $$T_{\text{overlapped}} = 50\text{ms（计算）} + 6\text{ms（最后一桶通信）} = 56\text{ms}$$

    **最大理论重叠：**

    $$\text{Overlap} = \frac{T_{\text{sequential}} - T_{\text{overlapped}}}{T_{\text{comm}}} = \frac{80 - 56}{30} = \frac{24}{30} = 80\%$$

    换言之，5 次桶通信中有 4 次被完全重叠。

    | 指标 | 数值 |
    |------|------|
    | 顺序时间 | 80ms |
    | 重叠时间 | **56ms** |
    | 加速比 | 1.43× |
    | 重叠比例 | **80%** |

    $$\boxed{T_{\text{total}} = 56\text{ms}；\text{ 重叠} = 80\%}$$

2. **桶大小优化**：你有 1GB 梯度和 100 Gbps 带宽。通信延迟为 100μs。找出使每桶通信时间最小的桶大小。最优的桶数是多少？

??? success "解答"
    **参数：**

    - 总梯度：$G = 1\text{ GB} = 10^9$ 字节
    - 带宽：$\beta = 100\text{ Gbps} = 12.5\text{ GB/s}$
    - 延迟：$L = 100\mu\text{s} = 10^{-4}\text{s}$

    **大小为 $B$ 的桶的通信时间：**

    $$T(B) = L + \frac{B}{\beta}$$

    **$n = G/B$ 个桶的总时间：**

    $$T_{\text{total}} = n \cdot T(B) = \frac{G}{B}\left(L + \frac{B}{\beta}\right) = \frac{GL}{B} + \frac{G}{\beta}$$

    第二项是固定的。最小化桶开销意味着平衡延迟成本与吞吐。

    **对重叠而言，我们希望桶小到足以与计算重叠，又大到足以摊薄延迟。**

    **每桶时间最小化：**

    我们想在有效吞吐的约束下最小化每桶时间 $T(B) = L + B/\beta$。

    有效吞吐：

    $$\text{Throughput} = \frac{B}{T(B)} = \frac{B}{L + B/\beta}$$

    要达峰值吞吐的 90%：

    $$\frac{B}{L + B/\beta} \geq 0.9\beta$$

    解得：

    $$B \geq 0.9\beta L + 0.9B$$

    $$0.1B \geq 0.9\beta L$$

    $$B \geq 9\beta L = 9 \times 12.5 \times 10^9 \times 10^{-4} = 11.25\text{ MB}$$

    **90% 效率下的最优桶大小：**

    $$B_{\text{opt}} \approx 9\beta L \approx 11.25\text{ MB}$$

    **桶数：**

    $$n = \frac{G}{B_{\text{opt}}} = \frac{1000\text{ MB}}{11.25\text{ MB}} \approx 89 \text{ 个桶}$$

    **验证：**

    ```python
    import numpy as np

    G = 1e9  # 1 GB
    beta = 12.5e9  # 12.5 GB/s
    L = 100e-6  # 100 μs

    def total_time(bucket_size):
        n_buckets = G / bucket_size
        time_per_bucket = L + bucket_size / beta
        return n_buckets * time_per_bucket

    def efficiency(bucket_size):
        return bucket_size / (L * beta + bucket_size)

    # Test various bucket sizes
    bucket_sizes = np.logspace(6, 9, 20)  # 1MB to 1GB
    for b in bucket_sizes:
        n = G / b
        t = total_time(b)
        eff = efficiency(b)
        print(f"Bucket: {b/1e6:.1f} MB, n={n:.0f}, time={t*1000:.2f} ms, eff={eff:.1%}")
    ```

    **结果：**

    | 桶大小 | 桶数 | 总时间 | 效率 |
    |--------|------|--------|------|
    | 1 MB | 1000 | 180 ms | 44% |
    | 10 MB | 100 | 90 ms | 89% |
    | **11.25 MB** | **89** | **88.9 ms** | **90%** |
    | 25 MB | 40 | 84 ms | 95% |
    | 100 MB | 10 | 81 ms | 99% |

    $$\boxed{B_{\text{opt}} \approx 11.25\text{ MB}；\text{ 90% 效率约需 } n \approx 89 \text{ 个桶}}$$

3. **流调度**：实现一个简单的双流调度器，让反向传播在一个流上运行，同时另一个流执行 AllReduce。测量达到的重叠比例。

??? success "解答"
    **双流调度器实现：**

    ```python
    import torch
    import torch.distributed as dist
    import time

    class TwoStreamScheduler:
        def __init__(self, model):
            self.model = model
            self.compute_stream = torch.cuda.Stream()
            self.comm_stream = torch.cuda.Stream()

            # Gradient buckets
            self.buckets = []
            self.bucket_params = []
            self._create_buckets()

            # Timing
            self.compute_time = 0
            self.comm_time = 0
            self.total_time = 0

        def _create_buckets(self, bucket_size_mb=25):
            """Group parameters into buckets."""
            bucket_size_bytes = bucket_size_mb * 1024 * 1024
            current_bucket = []
            current_size = 0

            for param in self.model.parameters():
                if param.requires_grad:
                    param_size = param.numel() * param.element_size()
                    if current_size + param_size > bucket_size_bytes and current_bucket:
                        self.bucket_params.append(current_bucket)
                        current_bucket = []
                        current_size = 0
                    current_bucket.append(param)
                    current_size += param_size

            if current_bucket:
                self.bucket_params.append(current_bucket)

        def _allreduce_bucket(self, bucket_idx):
            """AllReduce gradients for a bucket on comm stream."""
            with torch.cuda.stream(self.comm_stream):
                grads = [p.grad for p in self.bucket_params[bucket_idx] if p.grad is not None]
                if grads:
                    flat = torch.cat([g.view(-1) for g in grads])
                    dist.all_reduce(flat)

                    # Unflatten back
                    offset = 0
                    for g in grads:
                        numel = g.numel()
                        g.copy_(flat[offset:offset+numel].view_as(g))
                        offset += numel

        def backward_with_overlap(self, loss):
            """Run backward with overlapped communication."""
            start = time.time()

            # Register hooks to trigger AllReduce when gradients are ready
            bucket_ready = [False] * len(self.bucket_params)
            grad_counts = [0] * len(self.bucket_params)
            param_to_bucket = {}

            for bucket_idx, params in enumerate(self.bucket_params):
                for param in params:
                    param_to_bucket[param] = bucket_idx

            def make_hook(param):
                def hook(grad):
                    bucket_idx = param_to_bucket[param]
                    grad_counts[bucket_idx] += 1

                    # When all grads in bucket are ready, launch AllReduce
                    if grad_counts[bucket_idx] == len(self.bucket_params[bucket_idx]):
                        if not bucket_ready[bucket_idx]:
                            bucket_ready[bucket_idx] = True
                            # Record event on compute stream
                            event = torch.cuda.Event()
                            event.record(self.compute_stream)
                            # Wait for event on comm stream, then AllReduce
                            self.comm_stream.wait_event(event)
                            self._allreduce_bucket(bucket_idx)
                    return grad

                return hook

            # Register hooks
            handles = []
            for param in self.model.parameters():
                if param.requires_grad:
                    h = param.register_hook(make_hook(param))
                    handles.append(h)

            # Run backward on compute stream
            with torch.cuda.stream(self.compute_stream):
                loss.backward()

            # Wait for all communication to complete
            self.comm_stream.synchronize()
            self.compute_stream.synchronize()

            # Clean up hooks
            for h in handles:
                h.remove()

            self.total_time = time.time() - start

        def measure_overlap(self, loss, num_trials=10):
            """Measure overlap fraction."""
            # Measure sequential time
            torch.cuda.synchronize()
            start = time.time()
            loss.backward(retain_graph=True)
            torch.cuda.synchronize()
            compute_only = time.time() - start

            # Measure comm-only time
            torch.cuda.synchronize()
            start = time.time()
            for bucket_idx in range(len(self.bucket_params)):
                self._allreduce_bucket(bucket_idx)
            torch.cuda.synchronize()
            comm_only = time.time() - start

            sequential_time = compute_only + comm_only

            # Measure overlapped time
            overlapped_times = []
            for _ in range(num_trials):
                self.model.zero_grad()
                torch.cuda.synchronize()
                start = time.time()
                self.backward_with_overlap(loss)
                torch.cuda.synchronize()
                overlapped_times.append(time.time() - start)

            overlapped_time = sum(overlapped_times) / len(overlapped_times)

            overlap_fraction = (sequential_time - overlapped_time) / comm_only

            return {
                'compute_time': compute_only,
                'comm_time': comm_only,
                'sequential_time': sequential_time,
                'overlapped_time': overlapped_time,
                'overlap_fraction': overlap_fraction,
                'speedup': sequential_time / overlapped_time
            }

    # Usage
    def test_overlap():
        import torch.nn as nn

        # Initialize distributed
        dist.init_process_group('nccl')

        model = nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(20)]).cuda()
        scheduler = TwoStreamScheduler(model)

        x = torch.randn(32, 4096).cuda()
        loss = model(x).sum()

        results = scheduler.measure_overlap(loss)

        print(f"Compute time: {results['compute_time']*1000:.2f} ms")
        print(f"Comm time: {results['comm_time']*1000:.2f} ms")
        print(f"Sequential: {results['sequential_time']*1000:.2f} ms")
        print(f"Overlapped: {results['overlapped_time']*1000:.2f} ms")
        print(f"Overlap fraction: {results['overlap_fraction']:.1%}")
        print(f"Speedup: {results['speedup']:.2f}x")

    # test_overlap()
    ```

    **预期结果：**

    | 指标 | 数值 |
    |------|------|
    | 计算时间 | ~15 ms |
    | 通信时间 | ~20 ms |
    | 顺序 | ~35 ms |
    | 重叠 | ~22 ms |
    | **重叠比例** | **~65%** |
    | 加速比 | ~1.6× |

    **限制重叠的因素：**

    1. 第一个桶必须等到第一层反向完成
    2. 最后一个桶的通信延伸到计算之后
    3. CUDA 流调度开销
    4. 计算与 NCCL 之间的内存带宽争用

    $$\boxed{\text{典型重叠比例：50-80\%，取决于计算／通信比}}$$

4. **预取深度**：对 24 层的 ZeRO-3，若每次 AllGather 需 2ms、每层计算需 8ms，应预取多少层才能隐藏 AllGather 延迟？

??? success "解答"
    **配置：**

    - 层数：$L = 24$
    - 每层 AllGather 时间：$T_{\text{gather}} = 2\text{ms}$
    - 每层计算时间：$T_{\text{compute}} = 8\text{ms}$

    **预取分析：**

    要完全隐藏 AllGather 延迟，第 $i+k$ 层的 AllGather 必须在第 $i+k$ 层开始计算之前完成。

    **时序约束：**

    若提前预取 $k$ 层，我们就有 $k \times T_{\text{compute}}$ 的时间来完成 AllGather。

    对完全重叠：

    $$k \times T_{\text{compute}} \geq T_{\text{gather}}$$

    $$k \geq \frac{T_{\text{gather}}}{T_{\text{compute}}} = \frac{2\text{ms}}{8\text{ms}} = 0.25$$

    所以 $k = 1$ 层的预取就足够了！

    **可视化：**

    ```
    Time(ms):    0    8    16   24   32   40
                 |----|----|----|----|----| ...

    Layer 0:     [=compute=]
    Layer 1:          [=compute=]
    Layer 2:               [=compute=]

    AllGather 0: [AG]      (2ms, done before L0 compute)
    AllGather 1: [AG]      (overlaps with L0, ready for L1)
    AllGather 2:      [AG] (overlaps with L1, ready for L2)
    ```

    **当 prefetch_depth = 1 时：**

    - 计算第 $i$ 层时，预取第 $i+1$ 层
    - $T_{\text{compute}} = 8\text{ms} > T_{\text{gather}} = 2\text{ms}$ → 完全隐藏！

    **内存开销：**

    预取 $k$ 层需要的内存：

    $$M_{\text{prefetch}} = k \times \frac{\text{Model size}}{L} = 1 \times \frac{\text{Model size}}{24}$$

    对一个 70B 模型：

    $$M_{\text{prefetch}} = \frac{70\text{B} \times 2\text{ bytes}}{24} \approx 5.8\text{ GB}$$

    **如果计算更快呢？**

    | $T_{\text{compute}}$ | $T_{\text{gather}}$ | 最小预取 $k$ |
    |----------------------|---------------------|--------------|
    | 8 ms | 2 ms | 1 |
    | 4 ms | 2 ms | 1 |
    | 2 ms | 2 ms | 1 |
    | 1 ms | 2 ms | 2 |
    | 0.5 ms | 2 ms | 4 |

    **DeepSpeed ZeRO-3 预取设置：**

    ```python
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "prefetch_bucket_size": 50_000_000,  # ~50M params
            "param_persistence_threshold": 100_000,
        }
    }
    ```

    $$\boxed{k = 1 \text{ 层预取已足够（因 } T_{\text{compute}} > T_{\text{gather}}）}$$

5. **通信受限分析**：你的训练步显示 80ms 计算、120ms 通信，但总时间为 140ms。计算重叠比例。哪些技术可以改善它？

??? success "解答"
    **已知：**

    - 计算时间：$T_c = 80\text{ms}$
    - 通信时间：$T_m = 120\text{ms}$
    - 总时间：$T_{\text{total}} = 140\text{ms}$

    **重叠计算：**

    不重叠：$T_{\text{sequential}} = T_c + T_m = 80 + 120 = 200\text{ms}$

    重叠节省的时间：$T_{\text{saved}} = T_{\text{sequential}} - T_{\text{total}} = 200 - 140 = 60\text{ms}$

    重叠比例：

    $$\text{Overlap} = \frac{T_{\text{saved}}}{\min(T_c, T_m)} = \frac{60}{80} = 75\%$$

    或者，用被隐藏的通信比例表示：

    $$\text{Comm hidden} = \frac{T_{\text{saved}}}{T_m} = \frac{60}{120} = 50\%$$

    **分析：**

    系统是**通信受限**的，因为 $T_m > T_c$。

    可见的通信时间：$T_{\text{total}} - T_c = 140 - 80 = 60\text{ms}$

    隐藏的通信时间：$T_m - 60 = 120 - 60 = 60\text{ms}$

    **屋顶线视角：**

    ```
    Time:  0        80       140      200ms
           |--------|--------|--------|

    Compute: [=======] 80ms

    Comm:    [60ms hidden][60ms visible]
             └── overlapped ──┘└── exposed ──┘
    ```

    **可改善的技术：**

    | 技术 | 如何帮助 | 预期改善 |
    |------|----------|----------|
    | **梯度压缩** | 把 $T_m$ 减少 10-100× | 显著 |
    | **更大的桶** | 更好的带宽利用 | 轻微 |
    | **每步更多计算** | 更大的批 → 更多 $T_c$ 来隐藏 $T_m$ | 中等 |
    | **张量并行** | 减少每 GPU 的通信量 | 中等 |
    | **更快的互连** | NVLink vs PCIe | 显著 |
    | **流水线并行** | 把通信分散到各阶段 | 中等 |

    **定量改善估计：**

    1. **梯度压缩（10× 压缩）：**
       $$T_m' = 12\text{ms}, \quad T_{\text{total}}' = \max(80, 12) = 80\text{ms}$$
       加速比：$140/80 = 1.75\times$

    2. **批大小增 2×（计算翻倍）：**
       $$T_c' = 160\text{ms}, \quad T_{\text{total}}' = \max(160, 120) = 160\text{ms}$$
       吞吐：$2\times / (160/140) = 1.75\times$

    3. **切换到 NVLink（快 3×）：**
       $$T_m' = 40\text{ms}, \quad T_{\text{total}}' = \max(80, 40) = 80\text{ms}$$
       加速比：$140/80 = 1.75\times$

    **推荐策略：**

    ```python
    # Profiling to identify bottleneck
    def diagnose_overlap(compute_ms, comm_ms, total_ms):
        overlap_frac = (compute_ms + comm_ms - total_ms) / min(compute_ms, comm_ms)
        exposed_comm = total_ms - compute_ms

        if comm_ms > compute_ms:
            print(f"Communication-bound: {exposed_comm:.0f}ms exposed")
            print("Recommendations:")
            print("  1. Use gradient compression")
            print("  2. Increase batch size")
            print("  3. Use faster interconnect")
        else:
            print(f"Compute-bound: good overlap achievable")
            print("  Focus on compute optimization")

        return overlap_frac

    overlap = diagnose_overlap(80, 120, 140)
    print(f"Current overlap: {overlap:.1%}")
    ```

    $$\boxed{\text{重叠} = 75\% \text{ 的计算，或 } 50\% \text{ 的通信被隐藏}}$$

6. **双缓冲**：实现双缓冲梯度 AllReduce。测量内存开销与重叠改善。

??? success "解答"
    **双缓冲概念：**

    使用两块交替角色的梯度缓冲区：
    - 缓冲区 A：被当前反向写入
    - 缓冲区 B：正在被上一步 AllReduce

    这使得完全重叠成为可能，因为 AllReduce 永远不会阻塞反向。

    **实现：**

    ```python
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    class DoubleBufferedDDP:
        """Double-buffered gradient synchronization."""

        def __init__(self, model, process_group=None):
            self.model = model
            self.pg = process_group

            # Create double buffers for each parameter
            self.buffer_a = {}  # Current step gradients
            self.buffer_b = {}  # Previous step gradients (being AllReduced)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.buffer_a[name] = torch.zeros_like(param)
                    self.buffer_b[name] = torch.zeros_like(param)

            self.current_buffer = 'a'
            self.pending_allreduce = None
            self.comm_stream = torch.cuda.Stream()
            self.step_count = 0

        def get_write_buffer(self):
            """Get buffer for current backward pass."""
            return self.buffer_a if self.current_buffer == 'a' else self.buffer_b

        def get_read_buffer(self):
            """Get buffer being AllReduced."""
            return self.buffer_b if self.current_buffer == 'a' else self.buffer_a

        def swap_buffers(self):
            """Swap buffer roles."""
            self.current_buffer = 'b' if self.current_buffer == 'a' else 'a'

        def backward_step(self, loss):
            """
            Run backward and launch async AllReduce.
            Returns immediately - AllReduce happens in background.
            """
            # Wait for previous AllReduce to complete
            if self.pending_allreduce is not None:
                self.comm_stream.synchronize()
                # Apply averaged gradients from read buffer
                read_buffer = self.get_read_buffer()
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in read_buffer:
                        param.grad = read_buffer[name]

            # Run backward into write buffer
            self.model.zero_grad()
            loss.backward()

            # Copy gradients to write buffer
            write_buffer = self.get_write_buffer()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    write_buffer[name].copy_(param.grad)

            # Launch async AllReduce on write buffer
            self._launch_allreduce(write_buffer)

            # Swap buffers for next iteration
            self.swap_buffers()
            self.step_count += 1

        def _launch_allreduce(self, buffer):
            """Launch AllReduce on communication stream."""
            with torch.cuda.stream(self.comm_stream):
                # Flatten all gradients
                flat_grads = []
                for name in sorted(buffer.keys()):
                    flat_grads.append(buffer[name].view(-1))
                flat = torch.cat(flat_grads)

                # AllReduce
                dist.all_reduce(flat, group=self.pg)
                world_size = dist.get_world_size(self.pg) if self.pg else dist.get_world_size()
                flat.div_(world_size)

                # Unflatten back
                offset = 0
                for name in sorted(buffer.keys()):
                    numel = buffer[name].numel()
                    buffer[name].copy_(flat[offset:offset+numel].view_as(buffer[name]))
                    offset += numel

            self.pending_allreduce = True

        def memory_overhead(self):
            """Calculate memory overhead from double buffering."""
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            bytes_per_param = 4  # FP32 gradients
            buffer_memory = 2 * total_params * bytes_per_param  # Double buffer

            # Normal DDP also stores gradients, so overhead is just the extra buffer
            overhead = total_params * bytes_per_param
            return overhead

    def benchmark_double_buffer():
        """Compare double-buffered vs standard DDP."""
        import time

        dist.init_process_group('nccl')

        model = nn.Sequential(*[nn.Linear(4096, 4096) for _ in range(20)]).cuda()
        ddp_model = DoubleBufferedDDP(model)

        x = torch.randn(32, 4096).cuda()

        # Warmup
        for _ in range(5):
            loss = model(x).sum()
            ddp_model.backward_step(loss)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        num_iters = 100
        for _ in range(num_iters):
            loss = model(x).sum()
            ddp_model.backward_step(loss)
        torch.cuda.synchronize()
        double_buffer_time = (time.time() - start) / num_iters

        # Compare with standard approach
        model.zero_grad()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            loss = model(x).sum()
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad)
            torch.cuda.synchronize()
        standard_time = (time.time() - start) / num_iters

        overhead_gb = ddp_model.memory_overhead() / 1e9

        print(f"Standard DDP: {standard_time*1000:.2f} ms/step")
        print(f"Double-buffered: {double_buffer_time*1000:.2f} ms/step")
        print(f"Speedup: {standard_time/double_buffer_time:.2f}x")
        print(f"Memory overhead: {overhead_gb:.2f} GB")

    # benchmark_double_buffer()
    ```

    **预期结果：**

    | 配置 | 每步时间 | 内存开销 |
    |------|----------|----------|
    | 标准 DDP | ~35 ms | 0 |
    | 双缓冲 | ~22 ms | ~1.3 GB |

    **分析：**

    | 指标 | 标准 | 双缓冲 |
    |------|------|--------|
    | 计算 | 15 ms | 15 ms |
    | 可见通信 | 20 ms | ~7 ms |
    | 总计 | 35 ms | 22 ms |
    | 重叠 | 0% | 65% |

    **内存开销计算：**

    对具有 $\Psi$ 个参数的模型：
    - 标准 DDP：梯度 = $4\Psi$ 字节（FP32）
    - 双缓冲：$2 \times 4\Psi = 8\Psi$ 字节
    - 开销：$4\Psi$ 字节（多一块梯度缓冲区）

    对 350M 参数模型：

    $$\text{Overhead} = 350 \times 10^6 \times 4 = 1.4\text{ GB}$$

    **权衡汇总：**

    | 模型规模 | 内存开销 | 加速比 | 值得吗？ |
    |----------|----------|--------|----------|
    | 350M | 1.4 GB | 1.6× | ✓ 值得 |
    | 7B | 28 GB | 1.6× | 也许 |
    | 70B | 280 GB | 1.6× | ✗ 不值得（内存受限） |

    **注意**：双缓冲引入 1 步的梯度陈旧：
    - 第 $n$ 步应用第 $n-1$ 步的梯度
    - 对大批训练，这通常可接受
    - 可能需要略微调整学习率

    $$\boxed{\text{双缓冲：加速 } \sim1.6\times\text{，代价是额外 } 4\Psi \text{ 字节内存}}$$

## 关键要点

1. **重叠隐藏延迟**：计算期间的通信，其可见成本趋近于零。

2. **分桶使重叠成为可能**：聚合梯度以摊薄延迟并支持流式发送。

3. **CUDA 流是基础**：计算与通信分别使用不同的流，才能并发。

4. **钩子触发通信**：PyTorch 钩子在梯度就绪时启动 AllReduce。

5. **平衡很重要**：重叠受限于每桶计算或通信中较慢的一方。

6. **ZeRO 需要预取**：权重收集必须安排在计算之前。

7. **用剖析验证**：假想的重叠往往与实际情况不同——用 profiler 测量。

8. **内存是代价**：重叠需要缓冲，用内存换延迟。