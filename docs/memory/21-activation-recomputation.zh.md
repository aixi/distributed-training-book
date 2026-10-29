---
title: "激活重算"
subtitle: "通过梯度检查点用计算换内存"
---

<div class="chapter-opener" markdown>
在深度网络中，激活主导内存。一个 7B 模型的参数约需 14 GB，但它的激活可以消耗 100 GB 以上。激活重算（activation recomputation）——也叫梯度检查点（gradient checkpointing）——通过在反向传播期间重算激活而不是存储它们，用计算换内存。
</div>

<div class="investigation-question" markdown>
**问题**：反向传播需要前向传播的激活。把它们全部存下来需要 $O(L)$ 内存（$L$ 层）。我们能把它降到 $O(\sqrt{L})$ 甚至 $O(1)$ 吗？这种权衡的计算成本是多少？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 19 章](19-memory-equation.md)（内存方程分解）

    **核心洞见**：大batch size下激活主导内存。只在层边界做检查点、在反向时重算中间值，用约 33% 的额外计算换取大幅内存节省——从而支持更大的批和更长的序列。

## 激活内存问题

反向传播时，计算梯度需要前向传播的激活：

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial y_l} \cdot \frac{\partial y_l}{\partial W_l} = \delta_l \cdot a_{l-1}^T$$

其中：

- $\delta_l = \partial L / \partial y_l$：误差信号
- $a_{l-1}$：输入激活（来自前向传播）

不存储激活，我们就无法计算梯度。

### 内存增长

对于有 $L$ 层的 transformer，存储的激活包括：

| 组成部分 | 形状 | 每层 |
|-----------|-------|-----------|
| 层输入 | $(B, S, H)$ | $2BSH$ 字节 |
| 注意力 Q、K、V | $(B, S, H) \times 3$ | $6BSH$ 字节 |
| 注意力分数 | $(B, \text{heads}, S, S)$ | $2BnS^2$ 字节 |
| Softmax 输出 | $(B, \text{heads}, S, S)$ | $2BnS^2$ 字节 |
| 注意力输出 | $(B, S, H)$ | $2BSH$ 字节 |
| FFN 中间层 | $(B, S, 4H)$ | $8BSH$ 字节 |
| 各种中间值 | - | 约 $2BSH$ 字节 |

**每层合计**：约 $20BSH + 4BnS^2$ 字节（FP16）。

对于一个 7B 模型（$L=32$、$H=4096$、$n=32$），$B=4$、$S=2048$：

- 每层：$20 \times 4 \times 2048 \times 4096 \times 2 + 4 \times 4 \times 32 \times 2048^2 \times 2$
- 每层：$1.34$ GB + $4.29$ GB = $5.63$ GB
- 合计：$32 \times 5.63$ GB = **180 GB**

这远超 GPU 内存。

## 重算的权衡

**关键洞见**：如果我们可以重算激活，就不必存储它们。

### 基本思路

```
Standard forward/backward:
Forward:  Layer 0 → save a0 → Layer 1 → save a1 → ... → Layer L → Loss
Backward: Load aL-1 → grad L → Load aL-2 → grad L-1 → ... → grad 0

With recomputation:
Forward:  Layer 0 → Layer 1 → ... → Layer L → Loss (save only checkpoints)
Backward: Recompute aL-1 → grad L → Recompute aL-2 → grad L-1 → ...
```

### 根本权衡

设：

- $M$：激活内存
- $C$：前向传播的计算量
- $K$：检查点数量

| 策略 | 内存 | 前向传播次数 |
|----------|--------|----------------|
| 全部存储 | $O(L)$ | $1$ |
| 全部不存 | $O(1)$ | $\approx 2$（多一次前向） |
| 每 $\sqrt{L}$ 层检查点 | $O(\sqrt{L})$ | $\approx 2$ |

最优检查点间隔使总成本最小。

## 检查点策略

### 策略 1：完全重算（不存储）

只存输入，在反向时重算一切。

```python
class FullRecomputeFunction(torch.autograd.Function):
    """Recompute entire forward pass during backward."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, layers: nn.ModuleList):
        ctx.layers = layers

        # Forward through all layers (no intermediate storage)
        output = input
        for layer in layers:
            output = layer(output)

        # Save only input for recomputation
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, = ctx.saved_tensors
        layers = ctx.layers

        # Recompute forward pass
        activations = [input]
        hidden = input
        for layer in layers:
            hidden = layer(hidden)
            activations.append(hidden)

        # Standard backward
        grad = grad_output
        grads = []
        for i in range(len(layers) - 1, -1, -1):
            with torch.enable_grad():
                act = activations[i].detach().requires_grad_(True)
                out = layers[i](act)
                grad = torch.autograd.grad(out, act, grad)[0]

        return grad, None
```

**内存**：$O(1)$——只有输入和当前层。
**计算**：约多一次前向（实践中约 33% 开销）。
**问题**：对大的 $L$ 太昂贵。

### 策略 2：均匀检查点

每 $k$ 层保存一次激活。只在段内做重算。

```python
def uniform_checkpoint_forward(input: torch.Tensor,
                               layers: nn.ModuleList,
                               checkpoint_interval: int) -> torch.Tensor:
    """Forward with uniform checkpointing."""
    checkpoints = [input]
    hidden = input

    for i, layer in enumerate(layers):
        hidden = layer(hidden)
        if (i + 1) % checkpoint_interval == 0:
            checkpoints.append(hidden)

    return hidden, checkpoints

def uniform_checkpoint_backward(grad_output: torch.Tensor,
                                layers: nn.ModuleList,
                                checkpoints: List[torch.Tensor],
                                checkpoint_interval: int):
    """Backward with segment recomputation."""
    num_segments = len(checkpoints)
    grad = grad_output

    for seg_idx in range(num_segments - 1, -1, -1):
        # Determine segment boundaries
        start_layer = seg_idx * checkpoint_interval
        end_layer = min((seg_idx + 1) * checkpoint_interval, len(layers))

        # Recompute segment from checkpoint
        checkpoint = checkpoints[seg_idx]
        activations = [checkpoint]
        hidden = checkpoint
        for i in range(start_layer, end_layer):
            with torch.enable_grad():
                hidden = layers[i](hidden)
                activations.append(hidden)

        # Backward through segment
        for i in range(end_layer - 1, start_layer - 1, -1):
            layer_input = activations[i - start_layer]
            layer_output = activations[i - start_layer + 1]

            # Compute gradients for this layer
            grad = torch.autograd.grad(
                layer_output, layer_input, grad,
                retain_graph=True
            )[0]

    return grad
```

**间隔为 $k$ 时的分析**：

- 检查点数量：$L/k$
- 每个检查点的内存：$M_{\text{layer}}$
- 段内峰值内存：$k \cdot M_{\text{layer}}$

**总内存**：$\frac{L}{k} \cdot M_{\text{layer}} + k \cdot M_{\text{layer}}$

当 $L/k = k$，即 $k = \sqrt{L}$ 时取最小。

**最优检查点间隔**：$k^* = \sqrt{L}$

**最优内存**：$2\sqrt{L} \cdot M_{\text{layer}}$

对于 $L = 32$ 层：$k^* \approx 6$，内存从 $32M$ 降到 $12M$（节省 2.7 倍）。

### 策略 3：选择性检查点

并非所有层消耗相同内存。要有策略地做检查点。

```python
def analyze_layer_memory(layer: nn.Module,
                         input_shape: Tuple[int, ...]) -> int:
    """Estimate activation memory for a layer."""
    # Hook to capture activation sizes
    activation_sizes = []

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())
        elif isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activation_sizes.append(o.numel() * o.element_size())

    handles = []
    for module in layer.modules():
        handles.append(module.register_forward_hook(hook))

    # Dummy forward
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        layer(dummy_input)

    # Clean up hooks
    for h in handles:
        h.remove()

    return sum(activation_sizes)

def select_checkpoints(layers: nn.ModuleList,
                       memory_budget: int,
                       input_shape: Tuple[int, ...]) -> List[int]:
    """Select optimal checkpoint locations given memory budget."""
    layer_memories = [
        analyze_layer_memory(layer, input_shape)
        for layer in layers
    ]

    # Greedy selection: checkpoint before high-memory layers
    total_memory = sum(layer_memories)
    num_checkpoints = total_memory // memory_budget

    # Sort layers by memory, checkpoint before largest
    sorted_indices = sorted(range(len(layers)),
                           key=lambda i: layer_memories[i],
                           reverse=True)

    checkpoint_indices = sorted(sorted_indices[:num_checkpoints])
    return checkpoint_indices
```

## PyTorch 检查点 API

PyTorch 提供内置的检查点支持。

### 基本用法

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class TransformerWithCheckpointing(nn.Module):
    """Transformer with activation checkpointing."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)

        self.use_checkpointing = config.use_checkpointing
        self.checkpoint_ratio = config.checkpoint_ratio  # e.g., 0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)

        num_checkpointed = int(len(self.layers) * self.checkpoint_ratio)

        for i, layer in enumerate(self.layers):
            if self.use_checkpointing and i < num_checkpointed:
                # Checkpoint this layer
                hidden = checkpoint(
                    layer,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
            else:
                # Normal forward
                hidden = layer(hidden)

        return self.head(hidden)
```

### 顺序检查点

对于顺序模型，使用 `checkpoint_sequential`：

```python
def forward_with_sequential_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
    """Forward using checkpoint_sequential for uniform segments."""
    # Divide layers into segments
    num_segments = int(math.sqrt(len(self.layers)))

    # checkpoint_sequential handles segment boundaries automatically
    hidden = checkpoint_sequential(
        self.layers,
        num_segments,
        x,
        use_reentrant=False
    )

    return hidden
```

### 非重入式检查点

PyTorch 提供两种检查点模式：

**重入式**（旧版）：

- 内部使用 `torch.autograd.grad`
- 对某些操作可能有隐蔽 bug
- 正在被弃用

**非重入式**（推荐）：

- 使用已保存张量钩子
- 对复杂计算图更稳健
- 正确保留 RNG 状态

```python
# Always prefer non-reentrant
hidden = checkpoint(
    layer,
    hidden,
    use_reentrant=False,  # Recommended
    preserve_rng_state=True  # Important for dropout
)
```

### 策略 4：注意力专用检查点

注意力有独特的内存模式。$O(S^2)$ 的注意力分数占主导。

```python
class CheckpointedAttention(nn.Module):
    """
    Attention with selective recomputation.

    Stores Q, K, V but recomputes attention scores.
    Saves O(S²) memory per head.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape

        # Compute Q, K, V (stored)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Use checkpointing for attention computation
        attn_output = torch.utils.checkpoint.checkpoint(
            self._attention_forward,
            q, k, v,
            use_reentrant=False
        )

        # Output projection
        output = attn_output.transpose(1, 2).reshape(B, S, H)
        return self.out_proj(output)

    def _attention_forward(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor) -> torch.Tensor:
        """Core attention computation (recomputed in backward)."""
        # Attention scores: O(S²) memory NOT stored
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
```

**内存节省**：

- 不做检查点：分数 + softmax 占 $2 \times B \times n \times S^2$ 字节
- 做检查点：只存 Q、K、V（$6BSH$ 字节）

对于 $S=2048$、$n=32$：每层节省 $4 \times 32 \times 2048^2 \times 2 = 1.07$ GB。

## 计算成本分析

### 单个检查点

对第 $l$ 层做检查点意味着：

- 前向：计算第 $l$ 层一次
- 反向：在计算梯度前重算第 $l$ 层一次

**开销**：每个被检查点的层多一次前向传播。

### 全模型分析

设：

- $F$：一次前向传播的 FLOPs
- $B$：一次反向传播的 FLOPs（通常 $B \approx 2F$）
- $c$：被检查点的层占比

**不做检查点**：

$$C_{\text{total}} = F + B = F + 2F = 3F$$

**做检查点（占比 $c$）**：

$$C_{\text{total}} = F + cF + B = F(1 + c + 2) = F(3 + c)$$

**计算开销**：$c \cdot F$，即相对增加 $\frac{c}{3}$。

| 检查点占比 | 相对开销 |
|--------------------|-------------------|
| 0% | 0% |
| 50% | 16.7% |
| 100% | 33.3% |

**即使完全检查点，最大开销也只有 33%**。

### 内存-计算帕累托前沿

```
Memory
  │
  │●  No checkpointing (3F compute, L memory)
  │
  │    ●  50% checkpointing (3.5F, 0.5L + √L)
  │
  │        ●  √L checkpointing (3.33F, 2√L)
  │
  │            ●  Full checkpointing (4F, O(1))
  │
  └─────────────────────────────────────────→ Compute

Trade-off: 33% more compute for ~√L memory reduction
```

## 高级技术

### 激活压缩

不丢弃激活，而是压缩它们。

```python
class CompressedCheckpoint(torch.autograd.Function):
    """Checkpoint with lossy activation compression."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, layer: nn.Module, compression: str):
        ctx.layer = layer
        ctx.compression = compression

        # Compress input for storage
        if compression == 'fp8':
            compressed = input.to(torch.float8_e4m3fn)
        elif compression == 'quantize':
            # 8-bit quantization
            scale = input.abs().max() / 127
            compressed = (input / scale).round().to(torch.int8)
            ctx.scale = scale
        else:
            compressed = input

        ctx.save_for_backward(compressed)

        # Forward with full precision
        return layer(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        compressed, = ctx.saved_tensors
        layer = ctx.layer

        # Decompress
        if ctx.compression == 'fp8':
            input_approx = compressed.to(torch.float16)
        elif ctx.compression == 'quantize':
            input_approx = compressed.float() * ctx.scale
        else:
            input_approx = compressed

        # Compute gradients with approximate activations
        input_approx.requires_grad_(True)
        with torch.enable_grad():
            output = layer(input_approx)
            grad_input = torch.autograd.grad(output, input_approx, grad_output)[0]

        return grad_input, None, None
```

**权衡**：牺牲一些梯度精度换取显著的内存节省（FP8 为 4 倍，INT8 为 2 倍）。

### 卸载式检查点

将检查点与 CPU 卸载结合。

```python
class OffloadedCheckpoint(torch.autograd.Function):
    """Checkpoint with CPU offloading."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, layer: nn.Module):
        ctx.layer = layer
        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype
        ctx.input_device = input.device

        # Offload input to CPU (async)
        ctx.input_cpu = input.to('cpu', non_blocking=True)

        return layer(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Prefetch input from CPU
        input_gpu = ctx.input_cpu.to(ctx.input_device, non_blocking=True)
        torch.cuda.synchronize()

        input_gpu.requires_grad_(True)
        with torch.enable_grad():
            output = ctx.layer(input_gpu)
            grad_input = torch.autograd.grad(output, input_gpu, grad_output)[0]

        return grad_input, None
```

### 选择性重算

有些操作存储便宜、重算昂贵。要有选择性：

```python
class SelectiveRecompute(nn.Module):
    """Selectively recompute expensive operations."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

        # Operations to always store (cheap to store, expensive to compute)
        self.store_ops = {'embedding', 'layernorm', 'linear'}

        # Operations to recompute (expensive to store, cheap to compute)
        self.recompute_ops = {'attention_scores', 'softmax', 'dropout'}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom forward with selective storage
        return selective_checkpoint(self.layer, x, self.recompute_ops)
```

## 与其他内存优化的相互作用

### 检查点 + ZeRO

这两者是互补的：

- ZeRO 降低模型状态内存（参数、梯度、优化器）
- 检查点降低激活内存

```python
class ZeROWithCheckpointing:
    """Combined ZeRO and checkpointing."""

    def __init__(self, model: nn.Module, zero_stage: int):
        self.model = model
        self.zero_optimizer = ZeROOptimizer(model, stage=zero_stage)

        # Enable checkpointing for all transformer layers
        for layer in model.layers:
            layer.use_checkpoint = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # ZeRO-3: parameters gathered on-demand
        # Checkpointing: activations recomputed on-demand
        return self.model(input)
```

**组合内存**：

- 模型状态：$16\Psi/P$（ZeRO-3）
- 激活：$O(\sqrt{L})$（检查点）

### 检查点 + 张量并行

使用 TP 时要小心检查点，避免冗余通信：

```python
class TPCheckpointedLayer(nn.Module):
    """Tensor-parallel layer with smart checkpointing."""

    def __init__(self, hidden_dim: int, tp_degree: int):
        super().__init__()
        self.tp_degree = tp_degree
        self.local_hidden = hidden_dim // tp_degree

        self.linear = ColumnParallelLinear(hidden_dim, 4 * hidden_dim)
        self.output = RowParallelLinear(4 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Checkpoint the compute-heavy middle section
        # Don't checkpoint the AllReduce (would serialize communication)
        intermediate = checkpoint(
            self.linear,
            x,
            use_reentrant=False
        )

        # AllReduce happens here (in RowParallel backward)
        return self.output(F.gelu(intermediate))
```

### 检查点 + 流水线并行

流水线阶段在阶段边界自然形成检查点：

```python
class PipelineStage(nn.Module):
    """Pipeline stage with internal checkpointing."""

    def __init__(self, layers: nn.ModuleList, checkpoint_internal: bool = True):
        super().__init__()
        self.layers = layers
        self.checkpoint_internal = checkpoint_internal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage input is always stored (for pipeline backward)

        if self.checkpoint_internal and len(self.layers) > 1:
            # Checkpoint within stage
            num_segments = max(1, int(math.sqrt(len(self.layers))))
            x = checkpoint_sequential(
                self.layers,
                num_segments,
                x,
                use_reentrant=False
            )
        else:
            for layer in self.layers:
                x = layer(x)

        return x
```

## 带检查点的内存估计

### 解析模型

```python
def estimate_activation_memory(
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    checkpoint_strategy: str = 'sqrt',
    checkpoint_attention: bool = True,
    dtype_bytes: int = 2
) -> dict:
    """
    Estimate activation memory with various checkpointing strategies.

    Args:
        num_layers: Number of transformer layers
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        batch_size: Batch size
        seq_length: Sequence length
        checkpoint_strategy: 'none', 'full', 'sqrt', or 'selective'
        checkpoint_attention: Whether to checkpoint attention scores (recompute instead of storing)
        dtype_bytes: Bytes per element (2 for FP16)

    Returns:
        Dictionary with memory estimates
    """
    B, S, H, n, L = batch_size, seq_length, hidden_dim, num_heads, num_layers

    # Per-layer activation memory (without attention scores)
    # Input, Q, K, V, attention output, FFN intermediate, misc
    linear_per_layer = 20 * B * S * H * dtype_bytes  # ~20 BSH

    # Attention scores and softmax output
    attention_per_layer = 2 * B * n * S * S * dtype_bytes  # 2 * BnS²

    if checkpoint_strategy == 'none':
        # Store everything (no recomputation)
        stored_per_layer = linear_per_layer + attention_per_layer
        recompute_per_layer = 0
    elif checkpoint_attention:
        # Store Q, K, V; recompute scores
        stored_per_layer = linear_per_layer
        recompute_per_layer = attention_per_layer
    else:
        stored_per_layer = linear_per_layer + attention_per_layer
        recompute_per_layer = 0

    # Apply checkpointing strategy
    if checkpoint_strategy == 'none':
        # Store all layers
        total_stored = L * stored_per_layer
        peak_recompute = 0

    elif checkpoint_strategy == 'full':
        # Store only input, recompute all
        total_stored = B * S * H * dtype_bytes  # Just input
        peak_recompute = stored_per_layer + recompute_per_layer  # One layer at a time

    elif checkpoint_strategy == 'sqrt':
        # Optimal sqrt(L) checkpointing
        num_checkpoints = int(math.ceil(math.sqrt(L)))
        segment_size = (L + num_checkpoints - 1) // num_checkpoints

        # Store checkpoints
        checkpoint_memory = num_checkpoints * B * S * H * dtype_bytes

        # Peak during segment recomputation
        peak_segment = segment_size * stored_per_layer

        total_stored = checkpoint_memory + peak_segment
        peak_recompute = segment_size * recompute_per_layer

    elif checkpoint_strategy == 'selective':
        # Checkpoint every other layer
        num_stored = L // 2
        num_recomputed = L - num_stored

        total_stored = num_stored * stored_per_layer
        peak_recompute = stored_per_layer  # Only 1 layer at a time

    else:
        raise ValueError(f"Unknown strategy: {checkpoint_strategy}")

    return {
        'total_stored_gb': total_stored / (1024**3),
        'peak_recompute_gb': peak_recompute / (1024**3),
        'peak_total_gb': (total_stored + peak_recompute) / (1024**3),
        'strategy': checkpoint_strategy,
        'checkpoint_attention': checkpoint_attention,
        'layers': L,
        'per_layer_mb': stored_per_layer / (1024**2)
    }
```

### 示例计算

```python
# 7B model with batch 4, sequence 2048
config = {
    'num_layers': 32,
    'hidden_dim': 4096,
    'num_heads': 32,
    'batch_size': 4,
    'seq_length': 2048
}

strategies = ['none', 'sqrt', 'selective', 'full']
for strategy in strategies:
    result = estimate_activation_memory(**config, checkpoint_strategy=strategy)
    print(f"{strategy:10}: {result['peak_total_gb']:.1f} GB")
```

输出：
```
none      : 104.0 GB
sqrt      : 20.0 GB  (5× reduction)
selective : 21.3 GB  (~5× reduction)
full      : 3.3 GB   (~31× reduction)
```

## 实践实现指南

### DeepSpeed 激活检查点

DeepSpeed 提供优化过的检查点：

```python
import deepspeed

# Configure in DeepSpeed config
ds_config = {
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,  # Auto-detect
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }
}

# Wrap model
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

### Megatron-LM 检查点

```python
from megatron.core.tensor_parallel import checkpoint

class MegatronTransformerLayer(nn.Module):
    def forward(self, hidden_states, attention_mask):
        if self.checkpoint_activations:
            hidden_states = checkpoint(
                self.attention,
                hidden_states,
                attention_mask
            )
            hidden_states = checkpoint(
                self.mlp,
                hidden_states
            )
        else:
            hidden_states = self.attention(hidden_states, attention_mask)
            hidden_states = self.mlp(hidden_states)

        return hidden_states
```

### HuggingFace 梯度检查点

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Now training uses ~40% less activation memory
```

## 练习题

1. **最优检查点**：对于一个 48 层 transformer，用 $\sqrt{L}$ 规则推导最优检查点间隔。会存多少个检查点？与不做检查点相比内存如何？

??? success "解答"
    **最优检查点间隔：**

    使用 $\sqrt{L}$ 规则：

    $$k^* = \sqrt{L} = \sqrt{48} \approx 6.93$$

    取整到实用值：$k = 7$ 层一个检查点。

    **检查点数量：**
    $$N_{\text{ckpt}} = \left\lceil \frac{L}{k} \right\rceil = \left\lceil \frac{48}{7} \right\rceil = \boxed{7 \text{ checkpoints}}$$

    **内存比较：**

    令 $M_{\text{layer}}$ = 每层激活内存。

    | 策略 | 内存 | 公式 |
    |----------|--------|---------|
    | 不做检查点 | $48 \cdot M_{\text{layer}}$ | $L \cdot M_{\text{layer}}$ |
    | $\sqrt{L}$ 检查点 | $7 \cdot M_{\text{layer}} + 7 \cdot M_{\text{layer}}$ | $2\sqrt{L} \cdot M_{\text{layer}}$ |

    **$\sqrt{L}$ 内存的解释：**
    - 存 $\sqrt{L}$ 个检查点：$\sqrt{L} \cdot M_{\text{layer}}$
    - 反向时最多重算 $\sqrt{L}$ 层：$\sqrt{L} \cdot M_{\text{layer}}$
    - 合计：$2\sqrt{L} \cdot M_{\text{layer}}$

    **内存削减：**
    $$\text{Reduction} = \frac{L}{2\sqrt{L}} = \frac{\sqrt{L}}{2} = \frac{\sqrt{48}}{2} = \boxed{3.46\times}$$

    **数值示例（假设 $M_{\text{layer}} = 1$ GB）：**

    | 策略 | 总内存 | 相对 |
    |----------|--------------|----------|
    | 不做检查点 | 48 GB | 1.00× |
    | $\sqrt{L}$（$k=7$） | 14 GB | 0.29× |
    | 完全检查点（$k=1$） | 2 GB | 0.04× |

    **权衡小结：**

    | 检查点间隔 | 检查点数 | 内存 | 重算开销 |
    |---------------------|-------------|--------|-------------------|
    | $k = 1$（每层） | 48 | $2M$ | 100%（完全重算） |
    | $k = 7$（$\sqrt{L}$） | 7 | $14M$ | 33% |
    | $k = 12$ | 4 | $16M$ | 25% |
    | 不做检查点 | 0 | $48M$ | 0% |

2. **计算开销**：一个训练步不做检查点需要 100ms。完全检查点下预期步时是多少？（假设前向 = 反向计算的 1/3。）

??? success "解答"
    **训练步分解（不做检查点）：**

    总计 = 前向 + 反向 = 100ms

    设 $F$ = 前向时间，$B$ = 反向时间。
    已知：前向 = 反向计算的 1/3。

    反向传播有两个组成部分：
    - 重算前向（若做检查点）：与前向相同
    - 计算梯度：通常是前向的 2 倍

    所以不做检查点时：
    - 前向：$F$
    - 反向（仅梯度）：$2F$
    - 合计：$3F = 100$ ms
    - $F = 33.3$ ms

    **完全检查点时：**

    - 前向：$F = 33.3$ ms（相同，只是不保存激活）
    - 反向：重算 + 梯度 = $F + 2F = 3F = 100$ ms

    **检查点总计：**
    $$T_{\text{ckpt}} = F + 3F = 4F = \frac{4}{3} \times 100 = \boxed{133.3\text{ ms}}$$

    **开销：**
    $$\text{Overhead} = \frac{133.3 - 100}{100} = \boxed{33.3\%}$$

    **用标准公式验证：**

    33% 的开销与众所周知的规律相符：检查点增加约 33% 的计算开销。

    **分解对比：**

    | 阶段 | 不做检查点 | 完全检查点 |
    |-------|------------------|-------------------|
    | 前向 | 33.3 ms | 33.3 ms |
    | 反向（重算） | 0 ms | 33.3 ms |
    | 反向（梯度） | 66.7 ms | 66.7 ms |
    | **合计** | **100 ms** | **133.3 ms** |

3. **混合策略**：设计一种策略：对注意力分数做检查点，但不对 FFN 激活做检查点。计算相对完全检查点的内存节省。

??? success "解答"
    **每层激活内存分解：**

    | 组成部分 | 内存 | 重算成本 |
    |-----------|--------|----------------|
    | 注意力 Q、K、V | $3 \times BSH$ | 低（线性投影） |
    | 注意力分数 | $B \times A \times S^2$ | 高（matmul + softmax） |
    | 注意力输出 | $BSH$ | 中 |
    | FFN 中间层 | $4 \times BSH$ | 低（线性 + 激活） |
    | FFN 输出 | $BSH$ | 低 |

    **对于典型配置（B=4、S=4096、H=4096、A=32）：**

    | 组成部分 | 大小 | 占总计 |
    |-----------|------|------------|
    | 注意力 Q,K,V | 192 MB | 12% |
    | 注意力分数 | 2048 MB | 64% |
    | 注意力输出 | 64 MB | 4% |
    | FFN 中间层 | 256 MB | 16% |
    | FFN 输出 | 64 MB | 4% |
    | **合计** | **2624 MB** | **100%** |

    **混合策略：存 Q/K/V，重算注意力分数 + FFN**

    存储：
    - Q、K、V：192 MB
    - 层输入：64 MB（用于重算 FFN）

    重算：
    - 注意力分数 + softmax：节省 2048 MB
    - FFN 激活：节省 320 MB

    **每层内存：**
    $$M_{\text{mixed}} = 192 + 64 = 256 \text{ MB}$$

    **比较：**

    | 策略 | 内存/层 | 相对 | 重算开销 |
    |----------|--------------|----------|-------------------|
    | 不做检查点 | 2624 MB | 1.00× | 0% |
    | 混合（存 QKV） | 256 MB | 0.10× | 约 20% |
    | 完全检查点 | 64 MB | 0.02× | 33% |

    **何时使用混合策略：**

    ```python
    class MixedCheckpointAttention(nn.Module):
        """Stores Q/K/V, recomputes attention scores and FFN."""

        def forward(self, x):
            q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

            def attn_core(q, k, v):
                scores = torch.einsum('bshd,bthd->bhst', q, k) / math.sqrt(self.head_dim)
                probs = torch.softmax(scores, dim=-1)
                return torch.einsum('bhst,bthd->bshd', probs, v)

            # Recompute attention core in backward (scores not stored)
            attn_out = checkpoint(attn_core, q, k, v, use_reentrant=False)

            # FFN with recomputation (cheap)
            return checkpoint(self.ffn, attn_out, use_reentrant=False)
    ```

    **权衡分析：**

    | 指标 | 混合 | 完全检查点 | 优势方 |
    |--------|-------|-----------|-----------|
    | 内存 | 0.80× | 0.02× | 完全检查点 |
    | 计算 | 1.10× | 1.33× | 混合 |
    | 复杂度 | 中 | 低 | 完全检查点 |

    **建议：** 当内存紧张但 33% 开销不可接受时使用混合策略。20% 的内存节省加仅 10% 开销可能是值得的。

4. **压缩分析**：如果把激活在检查点前压缩到 FP8，相对 FP16 检查点内存削减多少？对梯度精度有何影响？

??? success "解答"
    **内存削减：**

    | 精度 | 字节/元素 | 相对 |
    |-----------|---------------|----------|
    | FP32 | 4 | 2.0× |
    | FP16/BF16 | 2 | 1.0×（基线） |
    | FP8 | 1 | 0.5× |

    **FP8 的内存削减：**
    $$\text{Reduction} = \frac{2}{1} = \boxed{2\times \text{ (50\% savings)}}$$

    **对于 48 层、$\sqrt{L}$ 检查点的模型：**

    | 检查点精度 | 内存 |
    |---------------------|--------|
    | FP16 | $14 \cdot M_{\text{layer}}$ |
    | FP8 | $7 \cdot M_{\text{layer}}$ |

    **梯度精度影响：**

    FP8 的动态范围和精度有限：
    - E4M3：范围 $\pm 448$，3 位尾数
    - E5M2：范围 $\pm 57344$，2 位尾数

    **误差分析：**

    ```python
    import torch
    import numpy as np

    def analyze_fp8_error(tensor_fp16):
        """Analyze quantization error from FP16 → FP8 → FP16."""
        # Simulate FP8 quantization (E4M3)
        scale = tensor_fp16.abs().max() / 448  # FP8 E4M3 max
        quantized = (tensor_fp16 / scale).clamp(-448, 448)
        quantized = (quantized * 8).round() / 8  # 3 mantissa bits
        dequantized = quantized * scale

        # Error metrics
        abs_error = (tensor_fp16 - dequantized).abs()
        rel_error = abs_error / (tensor_fp16.abs() + 1e-8)

        return {
            'max_abs_error': abs_error.max().item(),
            'mean_abs_error': abs_error.mean().item(),
            'max_rel_error': rel_error.max().item(),
            'mean_rel_error': rel_error.mean().item(),
        }

    # Test on typical activation distribution
    activations = torch.randn(1024, 4096) * 0.1  # Typical scale
    errors = analyze_fp8_error(activations)
    # Typical results:
    # mean_rel_error: ~2-5%
    # max_rel_error: ~10-20%
    ```

    **梯度精度影响：**

    | 指标 | FP16 基线 | FP8 检查点 |
    |--------|---------------|-----------------|
    | 梯度相对误差 | 0% | 1–3% |
    | 训练稳定性 | 稳定 | 通常稳定 |
    | 最终损失 | 基线 | +0.1–0.5% |
    | 收敛速度 | 基线 | 大致相同 |

    **缓解策略：**

    1. **随机舍入**：减少量化偏差
    2. **逐张量缩放**：最大化动态范围利用
    3. **混合精度检查点**：大张量用 FP8，小张量用 FP16
    4. **梯度缩放**：补偿降低的精度

    ```python
    class FP8Checkpoint:
        """Checkpoint with FP8 compression."""

        @staticmethod
        def save(tensor):
            scale = tensor.abs().max() / 448
            quantized = (tensor / scale).to(torch.float8_e4m3fn)
            return quantized, scale

        @staticmethod
        def load(quantized, scale):
            return quantized.to(torch.float16) * scale
    ```

    **小结：**

    | 方面 | 影响 |
    |--------|--------|
    | 内存 | 2 倍削减 |
    | 梯度精度 | 1–3% 相对误差 |
    | 训练稳定性 | 总体保持 |
    | 适用于 | 内存受限的训练 |

5. **组合优化**：一个 70B 模型在 64 块 GPU 上用 ZeRO-3 训练，采用 $\sqrt{L}$ 检查点。计算包括模型状态和激活的总每 GPU 内存。

??? success "解答"
    **70B 模型架构（典型）：**

    | 参数 | 值 |
    |-----------|-------|
    | 总参数 | 70B |
    | 层数（$L$） | 80 |
    | 隐藏维度（$H$） | 8192 |
    | 注意力头（$A$） | 64 |
    | FFN 中间层 | 28672（3.5× H） |

    **模型状态内存（ZeRO-3，64 GPU）：**

    $$M_{\text{state}} = \frac{16\Psi}{P} = \frac{16 \times 70 \times 10^9}{64} = \boxed{17.5\text{ GB}}$$

    **带 $\sqrt{L}$ 检查点的激活内存：**

    检查点间隔：$k = \sqrt{80} \approx 9$ 层
    检查点数量：$\lceil 80/9 \rceil = 9$

    假设训练配置：$B = 1$（微批）、$S = 4096$

    **每个检查点的激活大小：**

    层输入：$B \times S \times H \times 2$ 字节 = $1 \times 4096 \times 8192 \times 2 = 67$ MB

    **重算缓冲区（最多 9 层同时活跃）：**

    每层激活（近似）：
    - 注意力：$BSH \times 10 \times 2 = 670$ MB
    - FFN：$BSH \times 5 \times 2 = 335$ MB
    - 每层合计：约 1 GB

    重算缓冲区：$9 \times 1$ GB = 9 GB

    **总激活内存：**
    $$M_{\text{act}} = 9 \times 67\text{ MB} + 9 \times 1\text{ GB} \approx 0.6 + 9 = \boxed{9.6\text{ GB}}$$

    **每 GPU 总内存：**

    | 组成部分 | 内存 |
    |-----------|--------|
    | 模型状态（ZeRO-3） | 17.5 GB |
    | 激活（$\sqrt{L}$ 检查点） | 9.6 GB |
    | 临时缓冲区 | 约 3 GB |
    | CUDA 开销 | 约 2 GB |
    | **合计** | **约 32 GB** |

    **在 80GB H100 上放得下且有余量！**

    **不做优化的比较：**

    | 配置 | 每 GPU 内存 |
    |---------------|------------|
    | 无 ZeRO、无检查点 | $16\Psi + 80 \times 1\text{ GB} = 1200$ GB |
    | 仅 ZeRO-3 | $17.5 + 80 = 97.5$ GB |
    | 仅检查点（8 GPU） | $140 + 9.6 = 150$ GB |
    | **ZeRO-3 + $\sqrt{L}$ 检查点** | **32 GB** |

    **关键洞见：** ZeRO-3 与检查点组合带来乘性收益——两者单独都不足以在 64 块 GPU 上训练 70B。

6. **流水线交互**：在 8 个微批的 4 阶段流水线中，峰值时存了多少份激活拷贝？检查点如何影响这一点？

??? success "解答"
    **流水线配置：**

    - 阶段数（$p$）：4
    - 微批数（$m$）：8
    - 调度：1F1B（一次前向、一次反向）

    **1F1B 调度分析：**

    ```
    Stage 0: F0 F1 F2 F3 F4 F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
    Stage 1:    F0 F1 F2 F3 F4 F5 F6 B0 B1 B2 B3 B4 B5 B6 B7
    Stage 2:       F0 F1 F2 F3 F4 F5 B0 B1 B2 B3 B4 B5 B6 B7
    Stage 3:          F0 F1 F2 F3 F4 B0 B1 B2 B3 B4 B5 B6 B7
                      ↑ Peak activations
    ```

    **峰值激活存储（不做检查点）：**

    在峰值时（第一个反向开始之前）：
    - 阶段 0：持有微批 0-7 的激活 = 8 份拷贝
    - 阶段 1：持有微批 0-6 的激活 = 7 份拷贝
    - 阶段 2：持有微批 0-5 的激活 = 6 份拷贝
    - 阶段 3：持有微批 0-4 的激活 = 5 份拷贝

    等等，让我重新按 1F1B 稳态算……

    **1F1B 稳态：**

    预热之后，每个阶段最多持有 $p$ 个在途微批：
    - 阶段 0：最多 4 份激活集
    - 阶段 1：最多 4 份激活集
    - 等等。

    **每阶段峰值激活：** $\boxed{p = 4 \text{ micro-batches}}$

    **但预热期间（气泡填充）：**

    阶段 0 必须持有激活直到反向到来：
    - 预热微批：$p - 1 = 3$
    - 稳态在途：1

    **总峰值：** 每阶段 $p = 4$（稳态上界）

    **带检查点时：**

    不存完整激活，只存检查点：

    | 策略 | 每微批存储 | 每阶段峰值 |
    |----------|------------------------|----------------|
    | 不做检查点 | 完整激活（$L_{stage} \times M$） | $4 \times L_{stage} \times M$ |
    | 完全检查点 | 仅输入（$M$） | $4 \times M$ |
    | $\sqrt{L}$ 检查点 | $2\sqrt{L_{stage}} \times M$ | $4 \times 2\sqrt{L_{stage}} \times M$ |

    **示例（70B 模型，4 阶段，每阶段 20 层）：**

    $M$ = 层激活约 64 MB（微批大小 1、S=4096、H=8192）

    | 策略 | 每阶段内存 |
    |----------|------------------|
    | 不做检查点 | $4 \times 20 \times 64$ MB = 5.12 GB |
    | 完全检查点 | $4 \times 64$ MB = 256 MB |
    | $\sqrt{L}$（$k=4$） | $4 \times 2 \times 4.5 \times 64$ MB = 2.3 GB |

    **交错流水线（更多在途微批）：**

    采用交错调度（虚拟阶段）时，峰值提高：

    $$\text{Peak} = \frac{p \times v}{v} = p$$

    仍以每个物理阶段的 $p$ 为上界，但所有虚拟阶段合计的总内存更多。

    **内存公式：**

    $$M_{\text{peak}} = p \times \frac{L}{p} \times M_{\text{layer}} \times \text{ckpt\_factor}$$

    其中 ckpt_factor：
    - 不做检查点：1.0
    - $\sqrt{L}$：$\frac{2\sqrt{L/p}}{L/p} = \frac{2}{\sqrt{L/p}}$
    - 完全：$\frac{1}{L/p}$

    **小结：**

    | 指标 | 无检查点 | $\sqrt{L}$ 检查点 | 完全检查点 |
    |--------|---------|-----------------|-----------|
    | 峰值拷贝 | $4 \times 20 = 80$ | $4 \times 9 = 36$ | $4 \times 1 = 4$ |
    | 内存 | 5.12 GB | 2.3 GB | 256 MB |
    | 重算开销 | 0% | 33% | 100% |

## 关键要点

1. **激活主导内存**：在大批/长序列下可超过模型规模 10 倍以上。

2. **检查点用计算换内存**：$\sqrt{L}$ 的内存削减需要 33% 的计算开销。

3. **$\sqrt{L}$ 是最优的**：每 $\sqrt{L}$ 层做检查点，得到最佳的内存-计算权衡。

4. **注意力分数很昂贵**：每层 $O(S^2)$；是重算的首选对象。

5. **可与其他技术组合**：与 ZeRO、TP、PP 配合可带来乘性节省。

6. **使用非重入式检查点**：更稳健，正确处理 RNG 状态。