---
title: "降低精度训练"
subtitle: "计算的比特级经济学"
---

<div class="chapter-opener" markdown>
每一个比特都消耗带宽。每一个尾数位都消耗算力。混合精度训练利用的是硬件高效计算的东西与训练实际需要的东西之间的差距。
</div>

<div class="investigation-question" markdown>
**问题**：FP32 有 23 个尾数位。FP16 有 10 个。BF16 有 7 个。但三者训练出的模型质量相近。这些缺失的比特为什么不重要——又在什么时候重要？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 2 章](../foundations/02-extended-roofline.md)（算术强度与硬件能力）

    **关键洞察**：训练能容忍前向／反向传播中的低精度，因为梯度是在许多样本上平均的。BF16 的 8 位指数保住了动态范围（对大梯度至关重要）；FP32 主权重在许多次微小更新之间保住精度。FP8 借助精细的缩放进一步推进这一前沿。

## 浮点数的解剖

要理解降低精度，先要理解我们到底在降低什么。

### IEEE 754 表示

一个浮点数表示为：

$$x = (-1)^s \times 2^{e-\text{bias}} \times (1 + m)$$

其中：

- $s$：符号位（1 位）
- $e$：指数（决定范围）
- $m$：尾数／有效数（决定精度）

### 常见格式

| 格式 | 符号 | 指数 | 尾数 | 总计 | 范围 | 精度 |
|------|------|------|------|------|------|------|
| FP32 | 1 | 8 | 23 | 32 | $\pm 3.4 \times 10^{38}$ | $\sim 7$ 位十进制 |
| FP16 | 1 | 5 | 10 | 16 | $\pm 6.5 \times 10^4$ | $\sim 3$ 位十进制 |
| BF16 | 1 | 8 | 7 | 16 | $\pm 3.4 \times 10^{38}$ | $\sim 2$ 位十进制 |
| TF32 | 1 | 8 | 10 | 19 | $\pm 3.4 \times 10^{38}$ | $\sim 3$ 位十进制 |
| FP8 (E4M3) | 1 | 4 | 3 | 8 | $\pm 448$ | $\sim 1$ 位十进制 |
| FP8 (E5M2) | 1 | 5 | 2 | 8 | $\pm 5.7 \times 10^4$ | $< 1$ 位十进制 |

### 精度-范围权衡

在固定的比特预算下，我们必须在两者间取舍：

**更多指数位** → 更大的动态范围
**更多尾数位** → 范围内的更高精度

```
FP16:  [S][EEEEE][MMMMMMMMMM]     5-bit exp, 10-bit mantissa
BF16:  [S][EEEEEEEE][MMMMMMM]     8-bit exp, 7-bit mantissa
```

这种权衡正是 BF16 出现的原因：深度学习对范围的需要（梯度横跨多个数量级）超过对精度的需要。

## 为什么精度可以降低

### 统计学论证

神经网络训练本质上是随机的：

$$g_B = \frac{1}{B} \sum_{i=1}^B \nabla L(x_i, \theta) = \nabla \mathbb{E}[L] + \epsilon_B$$

梯度噪声 $\epsilon_B$ 的方差为 $\sigma^2/B$。

**关键洞察**：如果梯度噪声本身就引入了 $\sigma/\sqrt{B}$ 量级的误差，那么只要满足下式，加入量化噪声 $\delta_q$ 就无所谓：

$$\delta_q \ll \frac{\sigma}{\sqrt{B}}$$

对典型训练：

- 梯度噪声：$\sim 10\%$ 相对误差
- FP16 舍入：$\sim 0.1\%$ 相对误差

量化噪声被固有的随机性所主导。

### 损失景观论证

SGD 不需要精确跟随梯度——它只需要跟随一个能降低损失的方向：

$$\nabla L \cdot g_{\text{approx}} > 0$$

降低精度会略微改变梯度方向，但不足以翻转方向导数的符号。

### 正则化论证

量化噪声起到正则化作用，类似于：

- Dropout
- 权重噪声
- 标签平滑

一些研究表明，降低精度反而带来略*更好*的泛化，很可能就是这个隐式正则化的效果。

## 混合精度训练

关键洞察：**并非所有运算都需要全精度**。

### AMP 算法

自动混合精度（Automatic Mixed Precision, AMP），来自 Micikevicius 等人，2018：

```python
# Mixed precision training loop
def train_step_amp(model, optimizer, data, target, scaler):
    optimizer.zero_grad()

    # Forward pass in FP16
    with torch.cuda.amp.autocast():
        output = model(data)  # FP16 compute
        loss = criterion(output, target)  # FP16 compute

    # Backward pass with scaled loss (FP16 gradients)
    scaler.scale(loss).backward()

    # Unscale gradients for clipping (optional)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # Optimizer step with FP32 master weights
    scaler.step(optimizer)
    scaler.update()
```

### 三段式策略

**1. FP32 主权重**

权重以 FP32 存储和更新：

```python
class MixedPrecisionOptimizer:
    def __init__(self, model, base_optimizer, lr: float):
        # Master weights in FP32
        self.master_weights = {
            name: param.data.float().clone()
            for name, param in model.named_parameters()
        }
        self.model = model
        self.base_optimizer = base_optimizer
        self.lr = lr

    def step(self):
        # Gradients accumulated in FP16, now in master copy
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Update master weight (FP32)
                master = self.master_weights[name]
                grad = param.grad.float()
                master.add_(grad, alpha=-self.lr)

                # Copy back to model (FP16)
                param.data.copy_(master.half())
```

**为什么需要 FP32 主权重？**

权重更新可能极小：

$$\Delta w = \eta \cdot g \approx 10^{-4} \times 10^{-3} = 10^{-7}$$

在权重 $\sim 1$ 的 FP16 中：

- 最小可表示差值：$2^{-10} \approx 10^{-3}$
- 更新 $10^{-7}$ 会被舍入为零！

FP32 会累积这些微小更新，直到它们变得显著。

**2. FP16 前向／反向**

计算密集的运算在 FP16 中运行：

| 运算 | 精度 | 原因 |
|------|------|------|
| 矩阵乘 | FP16 | Tensor Core，8× 加速 |
| 卷积 | FP16 | Tensor Core，8× 加速 |
| 激活函数 | FP16 | 简单的逐元素运算 |
| 注意力分数 | FP16 | 由 matmul 主导 |

**3. 敏感运算用 FP32**

有些运算需要更高精度：

| 运算 | 精度 | 原因 |
|------|------|------|
| Softmax | FP32 | 指数会溢出 |
| LayerNorm | FP32 | 方差累积 |
| 损失计算 | FP32 | log-sum-exp 稳定性 |
| 梯度归约 | FP32 | 对多个值累积 |

### 精度选择规则

```python
def select_precision(operation: str, tensor_size: int) -> str:
    """
    Select appropriate precision for an operation.
    """
    # Always FP32: numerically sensitive
    if operation in ['softmax', 'log_softmax', 'layer_norm',
                      'batch_norm', 'loss', 'exp', 'log']:
        return 'fp32'

    # Always FP16: compute-bound, Tensor Core friendly
    if operation in ['matmul', 'conv2d', 'linear']:
        return 'fp16'

    # Depends on size: element-wise operations
    if operation in ['relu', 'gelu', 'add', 'mul']:
        # Small tensors: FP32 has negligible overhead
        # Large tensors: FP16 saves bandwidth
        return 'fp16' if tensor_size > 1024 else 'fp32'

    # Default: FP32 for safety
    return 'fp32'
```

## 损失缩放

### 下溢问题

FP16 的动态范围有限：

- 最小正规格化数：$2^{-14} \approx 6 \times 10^{-5}$
- 最小次正规数：$2^{-24} \approx 6 \times 10^{-8}$

梯度可能更小：

- 典型梯度：$10^{-3}$ 到 $10^{-6}$
- 深层网络梯度：$10^{-7}$ 到 $10^{-10}$

在 FP16 中，许多梯度会下溢为零！

### 缩放方案

在反向传播前把损失乘上一个大的因子：

$$\tilde{L} = s \cdot L$$

梯度按比例缩放：

$$\tilde{g} = s \cdot g$$

反向传播后再把梯度除以 $s$：

$$g = \tilde{g} / s$$

这把梯度值移入 FP16 的可表示范围。

### 动态损失缩放

```python
class DynamicLossScaler:
    """
    Automatically adjust loss scale to avoid overflow/underflow.
    """
    def __init__(self,
                 init_scale: float = 65536.0,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.steps_since_growth = 0
        self.consecutive_good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        return loss * self.scale

    def unscale_gradients(self, optimizer):
        """Unscale gradients after backward pass."""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale

    def check_overflow(self, optimizer) -> bool:
        """Check if any gradient overflowed."""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        return True
        return False

    def update(self, optimizer) -> bool:
        """
        Update loss scale based on gradient health.
        Returns True if step should proceed, False if overflow occurred.
        """
        if self.check_overflow(optimizer):
            # Overflow: reduce scale, skip step
            self.scale *= self.backoff_factor
            self.consecutive_good_steps = 0

            # Zero out bad gradients
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.zero_()

            return False

        # No overflow: maybe increase scale
        self.consecutive_good_steps += 1
        if self.consecutive_good_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self.consecutive_good_steps = 0

        return True
```

### 损失缩放动态

```
Loss Scale over Training:

65536 |        ____
      |       /    \____
32768 |      /          \______
      |_____/                  \____
16384 |                            \_____
      |
      +------------------------------------→ Steps
         ↑       ↑  ↑       ↑
      warmup  spike spike  late
                          training
```

- **预热（warmup）**：缩放通常需要增长
- **尖峰（spike）**：溢出触发缩放降低
- **训练后期**：往往稳定在某个特定缩放值

## BF16：深度学习的格式

### BF16 为何出现

FP16 有限的范围（$\pm 6.5 \times 10^4$）会带来问题：

- 注意力期间的激活尖峰
- 训练早期的大梯度量级
- 深层网络中的梯度爆炸

BF16 用尾数位换取指数位：

- 与 FP32 相同的范围（$\pm 3.4 \times 10^{38}$）
- 更简单的转换：直接截断低 16 位

```python
def fp32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Convert FP32 to BF16 by truncating mantissa."""
    # View as 32-bit unsigned int to avoid sign issues
    x_int = x.view(np.uint32)

    # Round to nearest (add 0x8000 for rounding, not truncation)
    x_int = x_int + 0x8000

    # Zero out lower 16 bits
    x_int = x_int & 0xFFFF0000

    # View back as float
    return x_int.view(np.float32)
```

### BF16 简化训练

使用 BF16 时，损失缩放往往不再必要：

```python
# FP16: needs loss scaling
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)
scaled_loss = scaler.scale(loss)
scaled_loss.backward()
scaler.step(optimizer)
scaler.update()

# BF16: simpler, no scaling needed
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
loss.backward()
optimizer.step()
```

### BF16 与 FP16 对比

| 方面 | FP16 | BF16 |
|------|------|------|
| 范围 | $\pm 6.5 \times 10^4$ | $\pm 3.4 \times 10^{38}$ |
| 精度 | $\sim 0.1\%$ | $\sim 0.8\%$ |
| 损失缩放 | 必需 | 通常不需要 |
| Tensor Core 支持 | A100、H100 | A100、H100 |
| CPU 支持 | 有限 | x86（AVX-512 BF16） |
| 转换为 FP32 | 不平凡 | 平凡（位移） |

## TF32：透明的精度降低

TensorFloat-32（TF32）是 NVIDIA 的折中格式：

```
FP32:  [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]  8-bit exp, 23-bit mantissa
TF32:  [S][EEEEEEEE][MMMMMMMMMM]               8-bit exp, 10-bit mantissa
BF16:  [S][EEEEEEEE][MMMMMMM]                  8-bit exp, 7-bit mantissa
```

### TF32 如何工作

TF32 不是存储格式——它是计算格式：

1. 读取 FP32 输入
2. 把尾数舍入到 10 位
3. 以 TF32 精度计算
4. 把结果存为 FP32

```python
# TF32 is enabled by default on A100+
# Disable for bit-exact reproducibility:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### TF32 性能

| 运算 | FP32 TFLOPs | TF32 TFLOPs | 加速 |
|------|-------------|-------------|------|
| A100 matmul | 19.5 | 156 | 8× |
| A100 conv | 19.5 | 156 | 8× |
| H100 matmul | 67 | 495 | 7.4× |

TF32 无需改动代码即可透明地提供 Tensor Core 加速。

## FP8：下一片前沿

FP8 把精度降低推向更远，瞄准推理，并越来越多地用于训练。

### 两种 FP8 变体

**E4M3**（4 位指数，3 位尾数）：

- 范围：$\pm 448$
- 精度：每个 2 的幂有 8 个值
- 最适合：前向传播（权重、激活）

**E5M2**（5 位指数，2 位尾数）：

- 范围：$\pm 57344$
- 精度：每个 2 的幂有 4 个值
- 最适合：反向传播（梯度需要更大范围）

### FP8 训练配方

```python
class FP8TrainingConfig:
    """Configuration for FP8 training."""

    # Per-tensor scaling for FP8
    def compute_scale(self, tensor: torch.Tensor,
                      fp8_format: str = 'e4m3') -> float:
        """
        Compute scaling factor to maximize FP8 utilization.
        """
        if fp8_format == 'e4m3':
            fp8_max = 448.0
        else:  # e5m2
            fp8_max = 57344.0

        # Scale to use full FP8 range
        tensor_max = tensor.abs().max().item()
        if tensor_max == 0:
            return 1.0

        return fp8_max / tensor_max

    def quantize_to_fp8(self, tensor: torch.Tensor,
                        scale: float,
                        fp8_format: str = 'e4m3') -> torch.Tensor:
        """Quantize tensor to FP8 with given scale."""
        scaled = tensor * scale
        # Clamp to FP8 range
        if fp8_format == 'e4m3':
            clamped = torch.clamp(scaled, -448, 448)
        else:
            clamped = torch.clamp(scaled, -57344, 57344)

        # Round to FP8 representable values
        # (Implementation depends on hardware support)
        return self._round_to_fp8(clamped, fp8_format)
```

### 逐张量 vs 逐通道缩放

FP8 通常使用逐张量或逐通道缩放：

```python
def per_tensor_scale(tensor: torch.Tensor) -> float:
    """Single scale for entire tensor."""
    return 448.0 / tensor.abs().max()

def per_channel_scale(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Different scale per channel."""
    max_vals = tensor.abs().amax(dim=dim, keepdim=True)
    return 448.0 / max_vals
```

逐通道缩放提供更好的精度，但需要跟踪更多缩放因子。

### FP8 的挑战

1. **缩放管理**：需要跟踪并更新逐张量缩放
2. **范围限制**：E4M3 最大值 448，要求谨慎设计激活值
3. **硬件支持**：目前训练需要 H100 及以上
4. **精度敏感**：部分模型／层在 FP8 下会退化

## 硬件加速

### Tensor Core

NVIDIA Tensor Core 加速低精度矩阵运算：

```
Tensor Core Operation:
D = A × B + C

Where:

- A, B: FP16/BF16/FP8 matrices (fragment tiles)
- C, D: FP16/BF16/FP32 matrices
```

各代性能：

| GPU | FP32 | FP16 | BF16 | TF32 | FP8 |
|-----|------|------|------|------|-----|
| V100 | 15.7 | 125 | - | - | - |
| A100 | 19.5 | 312 | 312 | 156 | - |
| H100 | 67 | 990 | 990 | 495 | 1979 |

### 分块尺寸约束

Tensor Core 在 tile 上运算，需要特定维度：

```python
def is_tensor_core_aligned(M: int, N: int, K: int) -> bool:
    """Check if matrix dimensions are Tensor Core friendly."""
    # Requirements vary by GPU generation and precision
    # A100/H100 with FP16/BF16:
    return M % 8 == 0 and N % 8 == 0 and K % 8 == 0

def pad_for_tensor_cores(tensor: torch.Tensor) -> torch.Tensor:
    """Pad tensor dimensions to multiples of 8."""
    *batch_dims, m, k = tensor.shape

    new_m = ((m + 7) // 8) * 8
    new_k = ((k + 7) // 8) * 8

    if new_m == m and new_k == k:
        return tensor

    padded = torch.zeros(*batch_dims, new_m, new_k,
                         dtype=tensor.dtype, device=tensor.device)
    padded[..., :m, :k] = tensor
    return padded
```

### 内存带宽收益

除计算外，降低精度还提升内存效率：

| 格式 | 权重内存 | 激活内存 | 带宽 |
|------|----------|----------|------|
| FP32 | 100% | 100% | 100% |
| FP16/BF16 | 50% | 50% | 50% |
| FP8 | 25% | 25% | 25% |

对内存受限的运算，降低精度提供成正比的加速。

## 数值稳定性分析

### 前向传播中的误差传播

考虑一层：$y = Wx + b$

用 FP16 计算：

$$\hat{y} = \text{fl}(Wx) + b + \epsilon$$

其中 $|\epsilon| \lesssim n \cdot u \cdot |W||x|$，对 FP16，$u = 2^{-11}$。

经过 $L$ 层：

$$|\epsilon_L| \lesssim L \cdot n \cdot u \cdot \prod_{i=1}^L |W_i|$$

**含义**：

- 误差随深度 $L$ 增长
- 误差随层宽 $n$ 增长
- 误差被权重量级放大

### 灾难性抵消

当两个几乎相等的数相减时：

$$\text{fl}(a - b) \approx (a - b)(1 + \delta)$$

若 $a \approx b$，相对误差会爆炸：

$$\frac{|\hat{y} - y|}{|y|} \approx \frac{|a|}{|a - b|} \cdot u$$

这出现在：

- Softmax：当 $x_i \approx x_j$ 时的 $e^{x_i} - e^{x_j}$
- LayerNorm：输入相近时的方差计算
- 残差连接：当 $f(x) \approx 0$ 时的 $x + f(x)$

### Softmax 中的溢出

标准的 softmax 实现会溢出：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

对 FP16 且 $x = 12$：$e^{12} \approx 163000 > 65504$（溢出！）

**解法**：在指数化前减去最大值：

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

```python
def stable_softmax_fp16(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax for FP16."""
    # Compute max in FP32 for safety
    x_max = x.float().max(dim=-1, keepdim=True).values

    # Subtract max in FP16
    x_shifted = x - x_max.half()

    # Exp and normalize
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

## 分层的考量

### 嵌入层

嵌入通常保持在 FP32：

```python
class MixedPrecisionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Store in FP32
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim)
        )

    def forward(self, indices):
        # Cast to FP16 for compute
        return F.embedding(indices, self.weight.half())
```

**原因**：嵌入梯度非常稀疏（只有被访问的索引会得到梯度）。FP32 保证微小更新被正确累积。

### 归一化层

LayerNorm 和 BatchNorm 用 FP32 计算统计量：

```python
def layer_norm_mixed_precision(x: torch.Tensor,
                                weight: torch.Tensor,
                                bias: torch.Tensor,
                                eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm with FP32 statistics."""
    # Compute statistics in FP32
    x_fp32 = x.float()
    mean = x_fp32.mean(dim=-1, keepdim=True)
    var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)

    # Normalize in FP32
    x_norm = (x_fp32 - mean) / torch.sqrt(var + eps)

    # Scale and shift, cast back to input dtype
    return (x_norm * weight.float() + bias.float()).to(x.dtype)
```

**原因**：方差计算对抵消误差敏感。

### 注意力层

注意力分数可以用 FP16 计算，但累积往往需要更高精度：

```python
def attention_with_mixed_precision(Q, K, V, mask=None):
    """Attention with careful precision handling."""
    d_k = K.size(-1)

    # QK^T in FP16 (Tensor Core friendly)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)  # Not -inf for FP16

    # Softmax in FP32 for stability
    attn = F.softmax(scores.float(), dim=-1).to(Q.dtype)

    # Attention * V in FP16
    return torch.matmul(attn, V)
```

### 损失函数

交叉熵损失需要 FP32：

```python
def cross_entropy_mixed_precision(logits: torch.Tensor,
                                   targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with FP32 log-softmax."""
    # Log-softmax in FP32 for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Gather and negate in FP32
    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return loss.mean()
```

## 精度降低何时失败

### 精度问题的迹象

1. **损失尖峰**：损失突然上升
2. **NaN/Inf 梯度**：溢出或非法运算
3. **训练停滞**：更新被舍入为零
4. **准确率退化**：最终质量差于 FP32 基线

### 调试精度问题

```python
class PrecisionDebugger:
    """Tools for debugging mixed-precision issues."""

    @staticmethod
    def check_tensor_stats(tensor: torch.Tensor, name: str):
        """Print tensor statistics for debugging."""
        t = tensor.float()
        stats = {
            'min': t.min().item(),
            'max': t.max().item(),
            'mean': t.mean().item(),
            'std': t.std().item(),
            'num_zeros': (t == 0).sum().item(),
            'num_inf': torch.isinf(t).sum().item(),
            'num_nan': torch.isnan(t).sum().item(),
        }
        print(f"{name}: {stats}")

    @staticmethod
    def check_gradient_magnitudes(model: nn.Module):
        """Check gradient magnitudes across layers."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.float()
                print(f"{name}:")
                print(f"  grad mean: {grad.mean():.2e}")
                print(f"  grad std: {grad.std():.2e}")
                print(f"  grad max: {grad.abs().max():.2e}")
                print(f"  grad min nonzero: {grad[grad != 0].abs().min():.2e}")

    @staticmethod
    def compare_fp16_fp32(model_fp16, model_fp32, input_data):
        """Compare FP16 and FP32 model outputs."""
        with torch.no_grad():
            out_fp16 = model_fp16(input_data).float()
            out_fp32 = model_fp32(input_data.float())

            diff = (out_fp16 - out_fp32).abs()
            rel_diff = diff / (out_fp32.abs() + 1e-8)

            print(f"Absolute diff: mean={diff.mean():.2e}, max={diff.max():.2e}")
            print(f"Relative diff: mean={rel_diff.mean():.2e}, max={rel_diff.max():.2e}")
```

### 补救策略

**1. 增大损失缩放**

```python
# If gradients underflow, use larger initial scale
scaler = GradScaler(init_scale=2**20)  # 1M instead of 65536
```

**2. 将有问题的层保持 FP32**

```python
class HybridPrecisionModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Keep first and last layers in FP32
        self.model.embed.float()
        self.model.output_proj.float()

    def forward(self, x):
        # Embedding in FP32
        x = self.model.embed(x)

        # Main computation in FP16
        with autocast():
            x = self.model.transformer(x.half())

        # Output projection in FP32
        x = self.model.output_proj(x.float())
        return x
```

**3. 用 BF16 代替 FP16**

```python
# Switch from FP16 to BF16
with autocast(dtype=torch.bfloat16):  # Instead of float16
    output = model(data)
```

**4. 在反缩放前做梯度裁剪**

```python
# Clip in scaled space to prevent overflow during unscaling
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

## 通信与精度

### 混合精度下的 AllReduce

梯度归约必须谨慎进行：

```python
def allreduce_gradients_mixed_precision(model: nn.Module,
                                         process_group):
    """AllReduce with precision-aware communication."""
    for param in model.parameters():
        if param.grad is None:
            continue

        # Option 1: Reduce in FP16 (fast, but accumulation error)
        # grad_fp16 = param.grad.half()
        # dist.all_reduce(grad_fp16, group=process_group)
        # param.grad.copy_(grad_fp16)

        # Option 2: Reduce in FP32 (accurate, more bandwidth)
        grad_fp32 = param.grad.float()
        dist.all_reduce(grad_fp32, group=process_group)
        param.grad.copy_(grad_fp32)
```

### 精度-通信权衡

| 策略 | 带宽 | 精度 | 最适合 |
|------|------|------|--------|
| FP32 AllReduce | 高 | 最佳 | 小模型、研究 |
| FP16 AllReduce | 低 | 良好 | 大模型、生产 |
| BF16 AllReduce | 低 | 良好 | 现代硬件（A100+） |
| FP16 + 误差反馈 | 低 | 更好 | 极限扩展 |

## 实用配方

### 推荐的混合精度配置

```python
def create_mixed_precision_trainer(model, optimizer, use_bf16=False):
    """
    Create a mixed-precision training configuration.
    """
    config = {
        'compute_dtype': torch.bfloat16 if use_bf16 else torch.float16,
        'master_weights': True,
        'loss_scaling': not use_bf16,  # BF16 often doesn't need scaling
    }

    # Wrap optimizer for master weights
    if config['master_weights']:
        optimizer = MasterWeightOptimizer(model, optimizer)

    # Create scaler if needed
    scaler = None
    if config['loss_scaling']:
        scaler = torch.cuda.amp.GradScaler()

    return MixedPrecisionTrainer(model, optimizer, scaler, config)

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, scaler, config):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config

    def train_step(self, data, target):
        self.optimizer.zero_grad()

        # Forward pass in reduced precision
        with torch.cuda.amp.autocast(dtype=self.config['compute_dtype']):
            output = self.model(data)
            loss = F.cross_entropy(output, target)

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()
```

### 精度选择流程图

```
                     Start
                       │
                       ▼
              ┌─────────────────┐
              │  H100 or newer? │
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           │ Yes                   │ No
           ▼                       ▼
    ┌─────────────┐        ┌─────────────┐
    │  Use FP8    │        │ A100/A10G?  │
    │  (if model  │        └──────┬──────┘
    │   allows)   │               │
    └─────────────┘       ┌───────┴───────┐
                          │ Yes           │ No
                          ▼               ▼
                   ┌──────────┐    ┌──────────┐
                   │ Use BF16 │    │ V100?    │
                   └──────────┘    └────┬─────┘
                                        │
                                ┌───────┴───────┐
                                │ Yes           │ No
                                ▼               ▼
                         ┌──────────┐    ┌──────────┐
                         │ Use FP16 │    │ Use FP32 │
                         │ + scaling│    │(no accel)│
                         └──────────┘    └──────────┘
```

## 练习题

1. **精度对比**：用 FP32、FP16 和 BF16 实现矩阵乘法。对随机 1000×1000 矩阵，测量逐元素最大差值。误差如何随矩阵规模增长？

??? success "解答"
    **实现：**

    ```python
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    def compare_matmul_precision(n):
        """Compare matrix multiplication across precisions"""
        # Generate random matrices in FP32
        torch.manual_seed(42)
        A_fp32 = torch.randn(n, n, dtype=torch.float32)
        B_fp32 = torch.randn(n, n, dtype=torch.float32)

        # FP32 as ground truth
        C_fp32 = A_fp32 @ B_fp32

        # FP16
        A_fp16 = A_fp32.half()
        B_fp16 = B_fp32.half()
        C_fp16 = (A_fp16 @ B_fp16).float()

        # BF16
        A_bf16 = A_fp32.bfloat16()
        B_bf16 = B_fp32.bfloat16()
        C_bf16 = (A_bf16 @ B_bf16).float()

        # Compute errors
        error_fp16 = (C_fp32 - C_fp16).abs()
        error_bf16 = (C_fp32 - C_bf16).abs()

        return {
            'max_error_fp16': error_fp16.max().item(),
            'max_error_bf16': error_bf16.max().item(),
            'mean_error_fp16': error_fp16.mean().item(),
            'mean_error_bf16': error_bf16.mean().item(),
            'relative_error_fp16': (error_fp16 / C_fp32.abs().clamp(min=1e-8)).mean().item(),
            'relative_error_bf16': (error_bf16 / C_fp32.abs().clamp(min=1e-8)).mean().item(),
        }

    # Test across matrix sizes
    sizes = [100, 500, 1000, 2000, 4000]
    results = []

    for n in sizes:
        metrics = compare_matmul_precision(n)
        metrics['n'] = n
        results.append(metrics)
        print(f"n={n}: FP16 max_err={metrics['max_error_fp16']:.4f}, "
              f"BF16 max_err={metrics['max_error_bf16']:.4f}")
    ```

    **n=1000 的预期结果：**

    | 指标 | FP16 | BF16 |
    |------|------|------|
    | 最大绝对误差 | ~0.05-0.15 | ~0.5-1.5 |
    | 平均绝对误差 | ~0.005-0.01 | ~0.05-0.1 |
    | 相对误差 | ~0.01% | ~0.1% |

    **误差随矩阵规模的增长：**

    | 矩阵规模 (n) | FP16 最大误差 | BF16 最大误差 | 误差增长 |
    |--------------|---------------|---------------|----------|
    | 100 | 0.015 | 0.15 | 基线 |
    | 500 | 0.08 | 0.8 | ~√5 |
    | 1000 | 0.12 | 1.2 | ~√10 |
    | 2000 | 0.18 | 1.8 | ~√20 |
    | 4000 | 0.25 | 2.5 | ~√40 |

    **分析：**

    误差大致按 $O(\sqrt{n})$ 增长，因为：
    - 每个输出元素是 $n$ 个乘积之和
    - 舍入误差像随机游走一样累积：$\sigma_{sum} = \sigma_{single} \cdot \sqrt{n}$

    **关键观察：**

    1. **BF16 的误差约为 FP16 的 10 倍**：因为少了 3 个尾数位（7 vs 10）
    2. **误差按 √n 缩放**：与随机误差累积一致
    3. **相对误差保持很小**：典型矩阵规模下 ~0.01-0.1%

    $$\boxed{\text{Max error } \propto \sqrt{n}, \text{ BF16 error} \approx 10 \times \text{FP16 error}}$$

2. **损失缩放**：人为构造一个在 FP16 下会下溢的梯度（例如 $10^{-6}$）。验证不做损失缩放时该梯度变为零。找出能保住该梯度的最小损失缩放值。

??? success "解答"
    **FP16 下溢阈值：**

    FP16 有：
    - 最小正规格化数：$2^{-14} \approx 6.1 \times 10^{-5}$
    - 最小正次正规数：$2^{-24} \approx 6.0 \times 10^{-8}$

    低于次正规最小值的数会变为零。

    **演示：**

    ```python
    import torch

    def test_underflow():
        # Gradient values that progressively underflow
        test_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

        print("Without loss scaling:")
        for val in test_values:
            grad_fp32 = torch.tensor(val, dtype=torch.float32)
            grad_fp16 = grad_fp32.half()
            recovered = grad_fp16.float()
            print(f"  {val:.0e} -> FP16 -> {recovered.item():.2e} "
                  f"({'ZERO' if recovered == 0 else 'OK'})")

        print("\nWith loss scaling (scale=65536):")
        scale = 65536.0
        for val in test_values:
            grad_fp32 = torch.tensor(val * scale, dtype=torch.float32)
            grad_fp16 = grad_fp32.half()
            recovered = grad_fp16.float() / scale
            print(f"  {val:.0e} -> scaled -> FP16 -> unscaled -> {recovered.item():.2e}")

    test_underflow()
    ```

    **预期输出：**

    ```
    Without loss scaling:
      1e-04 -> FP16 -> 1.00e-04 (OK)
      1e-05 -> FP16 -> 1.00e-05 (OK)
      1e-06 -> FP16 -> 1.00e-06 (OK)
      1e-07 -> FP16 -> 5.96e-08 (OK - subnormal)
      1e-08 -> FP16 -> 0.00e+00 (ZERO)
      1e-09 -> FP16 -> 0.00e+00 (ZERO)

    With loss scaling (scale=65536):
      1e-04 -> scaled -> FP16 -> unscaled -> 1.00e-04
      1e-05 -> scaled -> FP16 -> unscaled -> 1.00e-05
      1e-06 -> scaled -> FP16 -> unscaled -> 1.00e-06
      1e-07 -> scaled -> FP16 -> unscaled -> 1.00e-07
      1e-08 -> scaled -> FP16 -> unscaled -> 1.00e-08
      1e-09 -> scaled -> FP16 -> unscaled -> 1.00e-09
    ```

    **求最小损失缩放：**

    ```python
    def find_minimum_scale(target_gradient):
        """Find minimum scale to preserve gradient in FP16"""
        min_fp16_normal = 2**-14  # ~6.1e-5

        # Need: target * scale >= min_fp16_normal
        min_scale = min_fp16_normal / target_gradient

        # Round up to power of 2 for numerical stability
        import math
        min_scale_pow2 = 2 ** math.ceil(math.log2(min_scale))

        return min_scale_pow2

    # For gradient = 1e-6
    target = 1e-6
    min_scale = find_minimum_scale(target)
    print(f"Minimum scale for {target}: {min_scale}")

    # Verify
    grad = torch.tensor(target * min_scale).half()
    recovered = grad.float() / min_scale
    print(f"Recovered: {recovered.item():.2e}")
    ```

    **最小缩放计算：**

    对梯度 $g = 10^{-6}$：

    $$\text{scale}_{min} = \frac{2^{-14}}{10^{-6}} = \frac{6.1 \times 10^{-5}}{10^{-6}} = 61$$

    向上取整到 2 的幂：$\text{scale}_{min} = 64 = 2^6$

    **汇总表：**

    | 梯度值 | 最小 FP16 规格化数 | 最小缩放 | 取整（2^n） |
    |--------|--------------------|----------|-------------|
    | $10^{-5}$ | $6.1 \times 10^{-5}$ | 6.1 | 8 |
    | $10^{-6}$ | $6.1 \times 10^{-5}$ | 61 | 64 |
    | $10^{-7}$ | $6.1 \times 10^{-5}$ | 610 | 1024 |
    | $10^{-8}$ | $6.1 \times 10^{-5}$ | 6100 | 8192 |

    $$\boxed{\text{For } g = 10^{-6}: \text{minimum scale} = 64}$$

3. **动态范围**：构造一个在 $[10^{-8}, 10^8]$ 上均匀分布的张量。转换为 FP16 时有多少比例的值丢失？转换为 BF16 时呢？

??? success "解答"
    **各格式的动态范围：**

    | 格式 | 指数位 | 最小规格化数 | 最大规格化数 | 总范围 |
    |------|--------|--------------|--------------|--------|
    | FP16 | 5 | $\sim 6 \times 10^{-5}$ | $\sim 6.5 \times 10^4$ | $\sim 10^{10}$ |
    | BF16 | 8 | $\sim 10^{-38}$ | $\sim 10^{38}$ | $\sim 10^{76}$ |
    | FP32 | 8 | $\sim 10^{-38}$ | $\sim 10^{38}$ | $\sim 10^{76}$ |

    **实现：**

    ```python
    import torch
    import numpy as np

    def analyze_dynamic_range_loss():
        # Create log-uniform distribution from 10^-8 to 10^8
        n = 1_000_000
        log_min, log_max = -8, 8
        log_values = np.random.uniform(log_min, log_max, n)
        values_fp32 = torch.tensor(10 ** log_values, dtype=torch.float32)

        # Convert to FP16 and back
        values_fp16 = values_fp32.half().float()

        # Convert to BF16 and back
        values_bf16 = values_fp32.bfloat16().float()

        # Count zeros (underflow) and infs (overflow)
        fp16_underflow = (values_fp16 == 0) & (values_fp32 != 0)
        fp16_overflow = torch.isinf(values_fp16) & ~torch.isinf(values_fp32)
        fp16_lost = fp16_underflow | fp16_overflow

        bf16_underflow = (values_bf16 == 0) & (values_fp32 != 0)
        bf16_overflow = torch.isinf(values_bf16) & ~torch.isinf(values_fp32)
        bf16_lost = bf16_underflow | bf16_overflow

        return {
            'fp16_underflow_frac': fp16_underflow.float().mean().item(),
            'fp16_overflow_frac': fp16_overflow.float().mean().item(),
            'fp16_total_lost': fp16_lost.float().mean().item(),
            'bf16_underflow_frac': bf16_underflow.float().mean().item(),
            'bf16_overflow_frac': bf16_overflow.float().mean().item(),
            'bf16_total_lost': bf16_lost.float().mean().item(),
        }

    results = analyze_dynamic_range_loss()
    print(f"FP16 - Underflow: {results['fp16_underflow_frac']:.2%}, "
          f"Overflow: {results['fp16_overflow_frac']:.2%}")
    print(f"BF16 - Underflow: {results['bf16_underflow_frac']:.2%}, "
          f"Overflow: {results['bf16_overflow_frac']:.2%}")
    ```

    **理论分析：**

    对 $[\log_{10}(10^{-8}), \log_{10}(10^8)] = [-8, 8]$ 上的均匀分布：

    **FP16：**
    - 下溢：低于 $6 \times 10^{-5}$ 的值（log ≈ -4.2）
    - 溢出：高于 $6.5 \times 10^4$ 的值（log ≈ 4.8）
    - 覆盖范围：$[-4.2, 4.8]$，而总范围为 $[-8, 8]$

    $$\text{FP16 lost} = \frac{(-4.2 - (-8)) + (8 - 4.8)}{16} = \frac{3.8 + 3.2}{16} = \frac{7.0}{16} = 43.75\%$$

    **BF16：**
    - 下溢：低于 $\sim 10^{-38}$ → 我们的范围内没有
    - 溢出：高于 $\sim 10^{38}$ → 我们的范围内没有
    - 覆盖范围：整个 $[-8, 8]$

    $$\text{BF16 lost} = 0\%$$

    **预期结果：**

    | 格式 | 下溢 | 溢出 | 总丢失 |
    |------|------|------|--------|
    | FP16 | ~24% | ~20% | **~44%** |
    | BF16 | ~0% | ~0% | **~0%** |

    **可表示范围的可视化：**

    ```
    FP16:        [-----xxxxxxxxx|---------xxxxxxxxx]
                 -8    -4.2     0        4.8      8
                   underflow              overflow

    BF16:        [--------------------------------]
                 -8            0                 8
                        all representable
    ```

    $$\boxed{\text{FP16 loses } \sim44\% \text{ of values; BF16 loses } \sim0\%}$$

4. **Softmax 稳定性**：实现不带减最大值技巧的 softmax。找出在 FP16 中导致溢出的最小输入值。验证稳定版本可用。

??? success "解答"
    **不稳定的 softmax 实现：**

    ```python
    import torch
    import numpy as np

    def softmax_unstable(x):
        """Naive softmax - numerically unstable."""
        exp_x = torch.exp(x)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    def softmax_stable(x):
        """Stable softmax with max subtraction."""
        x_max = x.max(dim=-1, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    # Find overflow threshold in FP16
    def find_overflow_threshold():
        """Binary search for smallest value causing overflow."""
        low, high = 0.0, 100.0

        while high - low > 0.01:
            mid = (low + high) / 2
            x = torch.tensor([mid], dtype=torch.float16)
            exp_x = torch.exp(x)

            if torch.isinf(exp_x).any():
                high = mid
            else:
                low = mid

        return low

    threshold = find_overflow_threshold()
    print(f"FP16 overflow threshold for exp(): {threshold:.2f}")
    # Expected: ~11.09 (since exp(11.09) ≈ 65504 = FP16 max)
    ```

    **理论分析：**

    FP16 最大可表示值：$65504$

    当 $e^x > 65504$ 时溢出：

    $$x > \ln(65504) \approx 11.09$$

    **验证：**

    ```python
    # Test both implementations
    def compare_implementations():
        results = []

        for max_val in [5, 10, 11, 12, 15, 20, 50, 100]:
            # Create logits with specified maximum
            x_fp16 = torch.randn(1000, dtype=torch.float16)
            x_fp16 = x_fp16 * (max_val / x_fp16.abs().max())

            # Compute with both methods
            try:
                unstable = softmax_unstable(x_fp16)
                unstable_nan = torch.isnan(unstable).any().item()
                unstable_inf = torch.isinf(unstable).any().item()
            except:
                unstable_nan, unstable_inf = True, True

            stable = softmax_stable(x_fp16)
            stable_nan = torch.isnan(stable).any().item()
            stable_inf = torch.isinf(stable).any().item()

            # Ground truth in FP32
            x_fp32 = x_fp16.float()
            gt = torch.softmax(x_fp32, dim=-1)

            # Error (if stable succeeded)
            if not (stable_nan or stable_inf):
                error = (stable.float() - gt).abs().max().item()
            else:
                error = float('inf')

            results.append({
                'max_val': max_val,
                'unstable_ok': not (unstable_nan or unstable_inf),
                'stable_ok': not (stable_nan or stable_inf),
                'error': error
            })

        return results

    results = compare_implementations()
    for r in results:
        print(f"max={r['max_val']:3d}: unstable={'OK' if r['unstable_ok'] else 'FAIL':4s}, "
              f"stable={'OK' if r['stable_ok'] else 'FAIL':4s}, error={r['error']:.2e}")
    ```

    **预期输出：**

    ```
    FP16 overflow threshold for exp(): 11.09

    max=  5: unstable=OK  , stable=OK  , error=9.77e-04
    max= 10: unstable=OK  , stable=OK  , error=9.77e-04
    max= 11: unstable=OK  , stable=OK  , error=9.77e-04
    max= 12: unstable=FAIL, stable=OK  , error=9.77e-04
    max= 15: unstable=FAIL, stable=OK  , error=9.77e-04
    max= 20: unstable=FAIL, stable=OK  , error=9.77e-04
    max= 50: unstable=FAIL, stable=OK  , error=9.77e-04
    max=100: unstable=FAIL, stable=OK  , error=9.77e-04
    ```

    **减最大值为何有效：**

    减去最大值后：对所有 $i$，$x_i - x_{max} \leq 0$

    因此：$e^{x_i - x_{max}} \leq 1$——永不溢出！

    数学等价性：

    $$\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - x_{max}}}{\sum_j e^{x_j - x_{max}}}$$

    | 输入范围 | 不稳定版 | 稳定版 |
    |----------|----------|--------|
    | $[-5, 5]$ | ✓ 可用 | ✓ 可用 |
    | $[-10, 10]$ | ✓ 可用 | ✓ 可用 |
    | $[-15, 15]$ | ✗ 溢出/NaN | ✓ 可用 |
    | $[-100, 100]$ | ✗ 溢出/NaN | ✓ 可用 |

    $$\boxed{\text{FP16 overflows at } x \geq 11.09; \text{ max-subtraction fixes it}}$$

5. **混合精度加速**：在你的 GPU 上对 transformer 层做 FP32 与 FP16 的基准测试。加速比是多少？其中多少来自计算、多少来自内存带宽？

??? success "解答"
    **基准测试实现：**

    ```python
    import torch
    import torch.nn as nn
    import time

    class TransformerLayer(nn.Module):
        def __init__(self, hidden_dim, num_heads, dtype=torch.float32):
            super().__init__()
            self.dtype = dtype
            self.hidden_dim = hidden_dim

            # Self-attention
            self.q_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            self.o_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)

            # MLP
            self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim, dtype=dtype)
            self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim, dtype=dtype)

            # Norms
            self.norm1 = nn.LayerNorm(hidden_dim, dtype=dtype)
            self.norm2 = nn.LayerNorm(hidden_dim, dtype=dtype)

            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads

        def forward(self, x):
            # Self-attention
            residual = x
            x = self.norm1(x)

            B, S, H = x.shape
            q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

            out = out.transpose(1, 2).contiguous().view(B, S, H)
            out = self.o_proj(out)
            x = residual + out

            # MLP
            residual = x
            x = self.norm2(x)
            x = self.fc1(x)
            x = torch.gelu(x)
            x = self.fc2(x)
            x = residual + x

            return x

    def benchmark_layer(hidden_dim, num_heads, batch_size, seq_len, dtype, num_iters=100):
        """Benchmark forward + backward pass."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        layer = TransformerLayer(hidden_dim, num_heads, dtype=dtype).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(10):
            out = layer(x)
            loss = out.sum()
            loss.backward()

        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            out = layer(x)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Memory moved per iteration (approximate)
        params = sum(p.numel() for p in layer.parameters())
        bytes_per_elem = 4 if dtype == torch.float32 else 2
        param_bytes = params * bytes_per_elem * 3  # params + grads + opt read

        # Activations (rough estimate)
        act_bytes = batch_size * seq_len * hidden_dim * bytes_per_elem * 10

        total_bytes = (param_bytes + act_bytes) * num_iters

        return {
            'dtype': str(dtype).split('.')[-1],
            'time_per_iter_ms': (elapsed / num_iters) * 1000,
            'total_bytes_gb': total_bytes / 1e9,
            'bandwidth_gbps': total_bytes / elapsed / 1e9
        }

    # Run benchmarks
    configs = [
        {'hidden_dim': 4096, 'num_heads': 32, 'batch_size': 8, 'seq_len': 2048},
    ]

    for cfg in configs:
        print(f"\nConfig: hidden={cfg['hidden_dim']}, batch={cfg['batch_size']}, "
              f"seq={cfg['seq_len']}")

        fp32_result = benchmark_layer(**cfg, dtype=torch.float32)
        fp16_result = benchmark_layer(**cfg, dtype=torch.float16)

        speedup = fp32_result['time_per_iter_ms'] / fp16_result['time_per_iter_ms']

        print(f"  FP32: {fp32_result['time_per_iter_ms']:.2f} ms/iter")
        print(f"  FP16: {fp16_result['time_per_iter_ms']:.2f} ms/iter")
        print(f"  Speedup: {speedup:.2f}x")
    ```

    **H100 上的预期结果：**

    | 指标 | FP32 | FP16 | 比值 |
    |------|------|------|------|
    | 时间（ms） | ~12.5 | ~3.8 | 快 3.3× |
    | 内存带宽使用 | ~1.8 TB/s | ~1.7 TB/s | ~相同 |
    | 计算 TFLOP/s | ~150 | ~500 | 高 3.3× |

    **拆解加速比：**

    ```python
    def decompose_speedup():
        """
        Speedup sources:
        1. Memory bandwidth: 2× (half the bytes)
        2. Tensor Core compute: 4× (FP16 vs FP32)
        3. Cache efficiency: 2× (more fits in cache)
        """

        # Estimate time breakdown for FP32
        # Typical transformer: 60% compute-bound, 40% memory-bound
        compute_frac_fp32 = 0.60
        memory_frac_fp32 = 0.40

        # FP16 speedups
        compute_speedup = 4.0  # Tensor Cores: FP8 1979 vs FP16 495 TFLOP/s
        memory_speedup = 2.0  # Half the bytes

        # Weighted speedup
        # T_fp16 = T_fp32 * (compute_frac / compute_speedup + memory_frac / memory_speedup)
        time_ratio = (compute_frac_fp32 / compute_speedup +
                      memory_frac_fp32 / memory_speedup)
        theoretical_speedup = 1 / time_ratio

        print(f"Compute contribution to speedup: {compute_speedup:.1f}x (weight: {compute_frac_fp32:.0%})")
        print(f"Memory contribution to speedup: {memory_speedup:.1f}x (weight: {memory_frac_fp32:.0%})")
        print(f"Theoretical combined speedup: {theoretical_speedup:.2f}x")

        # Reality check
        print(f"\nTypical observed speedup: 2.5-4.0x")
        print(f"Gap explained by: kernel overhead, non-matmul ops, memory access patterns")

        return theoretical_speedup

    decompose_speedup()
    ```

    **分析：**

    对计算受限的工作负载（大批、长序列）：
    - FP16 Tensor Core 加速：4×
    - 实际观测：3-3.5×

    对内存受限的工作负载（小批）：
    - 内存带宽收益：2×
    - 实际观测：1.8-2.2×

    **加速比拆解：**

    | 组成 | 加速因子 | 贡献 |
    |------|----------|------|
    | Tensor Core（matmul） | 4× | 收益的 ~60% |
    | 内存带宽 | 2× | 收益的 ~30% |
    | 缓存效率 | 1.5× | 收益的 ~10% |

    $$\boxed{\text{Typical speedup: 2.5-4×; } \sim60\% \text{ from compute, } \sim40\% \text{ from memory}}$$

6. **FP8 量化**：实现到 FP8 E4M3 的逐张量量化。对一个预训练模型，比较其相对 FP16/BF16 的精度退化。

??? success "解答"
    **FP8 E4M3 量化实现：**

    ```python
    import torch
    import torch.nn as nn
    import numpy as np

    class FP8Quantizer:
        """
        FP8 E4M3 format:
        - 1 sign bit
        - 4 exponent bits (bias = 7)
        - 3 mantissa bits
        - Range: ±448, precision: ~1/16
        """

        # FP8 E4M3 constants
        E4M3_MAX = 448.0
        E4M3_MIN = 2**-9  # Smallest normal number

        @staticmethod
        def quantize_to_fp8(tensor, scale=None):
            """Quantize FP32/FP16 tensor to FP8 E4M3 (simulated)."""
            if scale is None:
                # Compute per-tensor scale
                amax = tensor.abs().max()
                scale = FP8Quantizer.E4M3_MAX / amax if amax > 0 else 1.0

            # Scale to FP8 range
            scaled = tensor * scale

            # Clamp to FP8 range
            clamped = torch.clamp(scaled, -FP8Quantizer.E4M3_MAX, FP8Quantizer.E4M3_MAX)

            # Simulate FP8 precision (round to 3 mantissa bits)
            # This is an approximation - real FP8 has more complex rounding
            sign = torch.sign(clamped)
            abs_val = torch.abs(clamped)

            # Find exponent and mantissa
            log2_val = torch.log2(abs_val.clamp(min=FP8Quantizer.E4M3_MIN))
            exponent = torch.floor(log2_val)
            mantissa = abs_val / (2.0 ** exponent) - 1.0  # Normalized mantissa [0, 1)

            # Round mantissa to 3 bits (8 levels)
            mantissa_quantized = torch.round(mantissa * 8) / 8

            # Reconstruct
            quantized = sign * (1.0 + mantissa_quantized) * (2.0 ** exponent)

            # Handle zeros and denormals
            quantized = torch.where(abs_val < FP8Quantizer.E4M3_MIN,
                                   torch.zeros_like(quantized), quantized)

            return quantized, scale

        @staticmethod
        def dequantize_from_fp8(quantized, scale):
            """Dequantize from FP8 back to original range."""
            return quantized / scale

    def compare_precision_formats(model, test_loader, device='cuda'):
        """Compare accuracy across precision formats."""
        model = model.to(device)
        model.eval()

        results = {}

        # Test FP32 (baseline)
        correct_fp32 = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_fp32 += predicted.eq(labels).sum().item()
                total += labels.size(0)
        results['fp32'] = 100 * correct_fp32 / total

        # Test FP16
        model_fp16 = model.half()
        correct_fp16 = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.half().to(device), labels.to(device)
                outputs = model_fp16(images)
                _, predicted = outputs.max(1)
                correct_fp16 += predicted.eq(labels).sum().item()
        results['fp16'] = 100 * correct_fp16 / total

        # Test BF16
        model_bf16 = model.to(torch.bfloat16)
        correct_bf16 = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(torch.bfloat16).to(device)
                labels = labels.to(device)
                outputs = model_bf16(images)
                _, predicted = outputs.max(1)
                correct_bf16 += predicted.eq(labels).sum().item()
        results['bf16'] = 100 * correct_bf16 / total

        # Test FP8 (simulated)
        model.float()  # Back to FP32
        correct_fp8 = 0

        # Quantize all weights to FP8
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()
            quantized, scale = FP8Quantizer.quantize_to_fp8(param.data)
            param.data = FP8Quantizer.dequantize_from_fp8(quantized, scale)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_fp8 += predicted.eq(labels).sum().item()
        results['fp8'] = 100 * correct_fp8 / total

        # Restore original weights
        for name, param in model.named_parameters():
            param.data = original_weights[name]

        return results

    # Example usage with a pre-trained model
    def run_comparison():
        import torchvision.models as models
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        from torch.utils.data import DataLoader

        # Load pre-trained ResNet-18
        model = models.resnet18(pretrained=True)

        # Prepare ImageNet validation set (or CIFAR-10 for quick test)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Use CIFAR-10 for demonstration
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        results = compare_precision_formats(model, test_loader)

        print("\nAccuracy comparison:")
        print("-" * 40)
        for fmt, acc in results.items():
            degradation = results['fp32'] - acc
            print(f"{fmt.upper():6s}: {acc:.2f}% (degradation: {degradation:+.2f}%)")

        return results

    results = run_comparison()
    ```

    **预期结果（ImageNet 上的 ResNet-18）：**

    | 格式 | 准确率 | 退化 |
    |------|--------|------|
    | FP32 | 69.76% | 基线 |
    | FP16 | 69.74% | -0.02% |
    | BF16 | 69.71% | -0.05% |
    | FP8 E4M3 | 69.12% | -0.64% |
    | FP8 E5M2 | 68.85% | -0.91% |

    **逐张量 vs 逐通道缩放：**

    ```python
    def compare_scaling_strategies(tensor):
        """Compare per-tensor vs per-channel FP8 quantization."""

        # Per-tensor scaling
        quantized_tensor, scale_tensor = FP8Quantizer.quantize_to_fp8(tensor)
        dequant_tensor = FP8Quantizer.dequantize_from_fp8(quantized_tensor, scale_tensor)
        error_tensor = (tensor - dequant_tensor).abs().mean().item()

        # Per-channel scaling (for conv weights: scale per output channel)
        if tensor.dim() >= 2:
            scales = []
            quantized_channels = []
            for i in range(tensor.shape[0]):
                q, s = FP8Quantizer.quantize_to_fp8(tensor[i])
                scales.append(s)
                quantized_channels.append(q)

            dequant_channel = torch.stack([
                FP8Quantizer.dequantize_from_fp8(q, s)
                for q, s in zip(quantized_channels, scales)
            ])
            error_channel = (tensor - dequant_channel).abs().mean().item()
        else:
            error_channel = error_tensor

        print(f"Per-tensor quantization error: {error_tensor:.6f}")
        print(f"Per-channel quantization error: {error_channel:.6f}")
        print(f"Improvement: {error_tensor / error_channel:.2f}x")

        return error_tensor, error_channel

    # Test with sample weight tensor
    sample_weights = torch.randn(64, 64, 3, 3)  # Conv layer weights
    compare_scaling_strategies(sample_weights)
    ```

    **分析：**

    | 缩放策略 | 量化误差 | 存储开销 |
    |----------|----------|----------|
    | 逐张量 | 较高 | 1 个缩放值 |
    | 逐通道 | 低 2-4× | O(channels) 个缩放 |
    | 逐 token（激活） | 最低 | $O(B \cdot S)$ 个缩放 |

    **关键观察：**

    1. **FP16/BF16**：对推理近乎无损（退化 <0.1%）
    2. **FP8 E4M3**：小幅退化（~0.5-1%），对多数应用可接受
    3. **逐通道缩放**：对 FP8 最小化误差至关重要
    4. **敏感层**：首／末层通常保持更高精度

    $$\boxed{\text{FP8 degrades accuracy by } \sim0.5\text{-}1\%; \text{ per-channel scaling reduces this by } 2\text{-}4\times}$$

## 关键要点

1. **三种格式占主导**：FP32 用于存储／累积，FP16/BF16 用于计算，FP8 正在兴起。

2. **训练时 BF16 > FP16**：与 FP32 相同的范围消除了大部分损失缩放需求。

3. **主权重不可或缺**：微小更新在 FP16 中会消失；要在 FP32 中累积。

4. **损失缩放防止下溢**：动态缩放适应梯度量级。

5. **并非所有运算都平等**：Softmax、LayerNorm 和损失需要 FP32；matmul 可用 FP16。

6. **2× 内存，8× 计算**：降低精度同时帮助带宽和 Tensor Core 吞吐。

7. **FP8 需要逐张量缩放**：有限的范围需要仔细的缩放管理。

8. **系统地调试**：检查梯度量级，与 FP32 基线对比，定位问题层。