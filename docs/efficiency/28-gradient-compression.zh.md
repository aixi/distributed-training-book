---
title: "梯度压缩"
subtitle: "用保真度换取带宽"
---

<div class="chapter-opener" markdown>
通信成本主导大规模训练。梯度压缩提供了一个看似神奇的解法：传输更少的数据，却收敛到同样的结果。它为何有效——以及何时失效——的数学，揭示了优化理论与信息论之间的深层联系。
</div>

<div class="investigation-question" markdown>
**问题**：你把梯度压缩到原始大小的 1%，模型却收敛到同样的质量。这怎么可能？收敛真正需要的信息到底是什么？
</div>

<div class="notation-banner" markdown>
**本章符号：** $\Psi$ = 参数数量，$P$ = GPU 数量，$\rho$ = 压缩比。参见[符号说明](../appendices/notation.md)。
</div>

!!! abstract "承上：第三至第六部分"
    我们在[第三部分](../collectives/11-primitives-properties.md)确定了**AllReduce 成本**随模型规模增长，并在大规模训练中占主导。[第四部分](../parallelism/14-data-parallelism-associativity.md)展示了**数据并行**如何要求每步都做梯度同步。[第六部分](../composition/23-device-mesh.md)讲授了组合。现在我们要问：能否减少通信量本身？本部分探讨把效率推到基本并行之上的技术。

## 通信瓶颈

在数据并行训练中，每一步都需要：

$$\text{AllReduce}(g) = \frac{1}{P} \sum_{i=0}^{P-1} g_i$$

对一个有 $\Psi$ 个 FP32 参数的模型：

- 通信量：每步 $2\Psi \cdot 4$ 字节（环形 AllReduce）
- 对 GPT-3（175B 参数）：每步 1.4 TB

在 400 Gbps 的跨节点带宽下：

$$T_{\text{comm}} = \frac{1.4 \times 10^{12} \times 8}{400 \times 10^9} = 28 \text{ seconds}$$

这是无法接受的。压缩势在必行。

## 压缩框架

一个梯度压缩方案由以下部分组成：

1. **压缩器（compressor）**：$C: \mathbb{R}^d \to \mathcal{M}$（压缩为消息）
2. **解压器（decompressor）**：$D: \mathcal{M} \to \mathbb{R}^d$（重建梯度）
3. **压缩比**：$\rho = \frac{|C(g)|}{d \cdot \text{sizeof(float)}}$

**目标**：在保持收敛的前提下最小化 $\rho$。

### 关键洞察

梯度具有**冗余性**：

1. **时间冗余**：梯度在相邻步之间变化缓慢
2. **空间冗余**：许多梯度元素接近零
3. **幅度冗余**：并不需要全精度

压缩正是利用这些冗余。

## 梯度量化

用低精度表示替代高精度梯度。

### 朴素量化

```python
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class QuantizedTensor:
    """Quantized representation of a tensor."""
    data: np.ndarray        # Quantized values (uint8, int4 packed, etc.)
    scale: float            # Scaling factor
    zero_point: float       # Zero point for asymmetric quantization
    original_shape: Tuple[int, ...]

    def dequantize(self) -> np.ndarray:
        """Reconstruct float tensor."""
        return self.scale * (self.data.astype(np.float32) - self.zero_point)

class NaiveQuantizer:
    """Simple min-max quantization."""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.num_levels = 2 ** bits

    def quantize(self, tensor: np.ndarray) -> QuantizedTensor:
        """Quantize tensor to fixed-point representation."""
        # Compute range
        t_min, t_max = tensor.min(), tensor.max()

        # Handle degenerate case
        if t_max == t_min:
            return QuantizedTensor(
                data=np.zeros(tensor.shape, dtype=np.uint8),
                scale=1.0,
                zero_point=0.0,
                original_shape=tensor.shape
            )

        # Compute scale and zero point
        scale = (t_max - t_min) / (self.num_levels - 1)
        zero_point = t_min / scale

        # Quantize
        quantized = np.round(tensor / scale - zero_point).astype(np.uint8)

        return QuantizedTensor(
            data=quantized,
            scale=scale,
            zero_point=zero_point,
            original_shape=tensor.shape
        )

    def compression_ratio(self) -> float:
        """Bits saved compared to FP32."""
        return self.bits / 32
```

**问题**：取值范围被离群值决定 → 典型值的精度很差。

### QSGD：量化 SGD

Alistarh 等人（2017）提出了随机量化：

$$Q_s(g_i) = ||g||_2 \cdot \text{sign}(g_i) \cdot \xi_i(|g_i|, s)$$

其中 $\xi_i$ 是一个随机取整函数：

$$\xi_i(v, s) = \begin{cases}
\lfloor sv/||g||_2 \rfloor / s & \text{with prob } 1 - (sv/||g||_2 - \lfloor sv/||g||_2 \rfloor) \\
\lceil sv/||g||_2 \rceil / s & \text{otherwise}
\end{cases}$$

**关键性质**：无偏！
$$\mathbb{E}[Q_s(g)] = g$$

```python
class QSGDQuantizer:
    """QSGD: Quantized Stochastic Gradient Descent."""

    def __init__(self, num_levels: int = 256):
        self.num_levels = num_levels
        self.s = num_levels - 1  # Number of quantization intervals

    def quantize(self, gradient: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Stochastic quantization with unbiasedness guarantee.

        Returns:
            signs: Sign of each element
            norm: L2 norm of gradient
            levels: Quantization levels [0, s]
        """
        # Compute norm and signs
        norm = np.linalg.norm(gradient)
        if norm == 0:
            return np.zeros_like(gradient, dtype=np.int8), 0.0, np.zeros(gradient.shape, dtype=np.uint8)

        signs = np.sign(gradient).astype(np.int8)

        # Normalize absolute values
        abs_normalized = np.abs(gradient) / norm

        # Stochastic rounding
        scaled = self.s * abs_normalized
        lower = np.floor(scaled).astype(np.int32)
        prob = scaled - lower  # Probability of rounding up

        # Random rounding
        random_vals = np.random.random(gradient.shape)
        levels = np.where(random_vals < prob, lower + 1, lower).astype(np.uint8)

        return signs, norm, levels

    def dequantize(self, signs: np.ndarray, norm: float,
                   levels: np.ndarray) -> np.ndarray:
        """Reconstruct gradient from quantized representation."""
        return norm * signs * (levels / self.s)

    @property
    def compression_ratio(self) -> float:
        """Communication cost relative to FP32."""
        # 1 bit for sign, log2(s+1) bits for level, plus norm (32 bits amortized)
        bits_per_element = 1 + np.log2(self.num_levels)
        return bits_per_element / 32
```

**方差分析**：

$$\mathbb{E}[||Q_s(g) - g||^2] \leq \min\left(\frac{d}{s^2}, \frac{\sqrt{d}}{s}\right) ||g||^2$$

$s$ 越大 → 层级越多 → 方差越小 → 收敛越好。

### TernGrad：三值梯度

Wen 等人（2017）把量化推到极致：只用三个值 {-1, 0, +1}。

```python
class TernGradQuantizer:
    """TernGrad: Ternary gradient quantization."""

    def quantize(self, gradient: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantize to ternary values {-1, 0, +1}.

        Uses stochastic rounding with threshold based on max absolute value.
        """
        # Scaling factor (per-layer)
        scale = np.max(np.abs(gradient))

        if scale == 0:
            return np.zeros_like(gradient, dtype=np.int8), 0.0

        # Probability of non-zero
        prob = np.abs(gradient) / scale

        # Stochastic ternary rounding
        random_vals = np.random.random(gradient.shape)
        ternary = np.where(
            random_vals < prob,
            np.sign(gradient),
            0
        ).astype(np.int8)

        return ternary, scale

    def dequantize(self, ternary: np.ndarray, scale: float) -> np.ndarray:
        """Reconstruct from ternary values."""
        return scale * ternary.astype(np.float32)

    @property
    def compression_ratio(self) -> float:
        """2 bits per element (can encode 3 values)."""
        return 2 / 32  # 16x compression

class PackedTernaryGradient:
    """Pack ternary gradients efficiently."""

    @staticmethod
    def pack(ternary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack ternary values into bit arrays.

        Each element needs 2 bits: 00=0, 01=+1, 10=-1
        Pack 4 elements per byte.
        """
        # Convert to 0, 1, 2 representation
        encoded = np.where(ternary > 0, 1, np.where(ternary < 0, 2, 0))

        # Pad to multiple of 4
        padded_len = (len(encoded) + 3) // 4 * 4
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(encoded)] = encoded.flatten()

        # Pack 4 values per byte
        packed = (
            padded[0::4] |
            (padded[1::4] << 2) |
            (padded[2::4] << 4) |
            (padded[3::4] << 6)
        )

        return packed, np.array(ternary.shape)

    @staticmethod
    def unpack(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """Unpack to ternary values."""
        # Extract 4 values per byte
        unpacked = np.zeros(len(packed) * 4, dtype=np.int8)
        unpacked[0::4] = packed & 0x03
        unpacked[1::4] = (packed >> 2) & 0x03
        unpacked[2::4] = (packed >> 4) & 0x03
        unpacked[3::4] = (packed >> 6) & 0x03

        # Convert back to -1, 0, +1
        ternary = np.where(unpacked == 1, 1, np.where(unpacked == 2, -1, 0))

        # Reshape
        total_elements = np.prod(shape)
        return ternary[:total_elements].reshape(shape)
```

**权衡**：16 倍压缩，但方差更高。

## 梯度稀疏化

与其量化所有元素，不如只传输最重要的那些。

### Top-K 稀疏化

选出幅值最大的 K 个梯度：

```python
class TopKSparsifier:
    """Top-K gradient sparsification."""

    def __init__(self, k_ratio: float = 0.01):
        """
        Args:
            k_ratio: Fraction of gradients to keep (1% = 0.01)
        """
        self.k_ratio = k_ratio

    def sparsify(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-K elements by magnitude.

        Returns:
            indices: Positions of selected elements
            values: Selected gradient values
        """
        flat = gradient.flatten()
        k = max(1, int(len(flat) * self.k_ratio))

        # Find top-K by magnitude
        abs_flat = np.abs(flat)
        top_k_indices = np.argpartition(abs_flat, -k)[-k:]

        # Sort by index for better compression
        top_k_indices = np.sort(top_k_indices)
        top_k_values = flat[top_k_indices]

        return top_k_indices.astype(np.uint32), top_k_values.astype(np.float32)

    def densify(self, indices: np.ndarray, values: np.ndarray,
                shape: Tuple[int, ...]) -> np.ndarray:
        """Reconstruct dense gradient."""
        flat = np.zeros(np.prod(shape), dtype=np.float32)
        flat[indices] = values
        return flat.reshape(shape)

    @property
    def compression_ratio(self) -> float:
        """Approximate compression ratio."""
        # Index (4 bytes) + value (4 bytes) per selected element
        return self.k_ratio * (32 + 32) / 32

class RandomKSparsifier:
    """Random-K sparsification with unbiased estimation."""

    def __init__(self, k_ratio: float = 0.01):
        self.k_ratio = k_ratio

    def sparsify(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random selection with scaling for unbiasedness.
        """
        flat = gradient.flatten()
        k = max(1, int(len(flat) * self.k_ratio))

        # Random selection
        indices = np.random.choice(len(flat), k, replace=False)
        indices = np.sort(indices)

        # Scale values for unbiased estimation
        scale = len(flat) / k
        values = flat[indices] * scale

        return indices.astype(np.uint32), values.astype(np.float32)
```

**Top-K 的问题**：有偏！小梯度永远不会被传输。

### 误差反馈（记忆压缩）

Seide 等人（2014）与 Stich 等人（2018）的关键洞察：

**累积压缩误差，并把它加到下一个梯度上。**

```python
class ErrorFeedbackCompressor:
    """
    Gradient compression with error feedback.

    Maintains error accumulator to ensure eventual transmission
    of all gradient information.
    """

    def __init__(self, base_compressor, shape: Tuple[int, ...]):
        """
        Args:
            base_compressor: Any compression method (quantizer or sparsifier)
            shape: Shape of gradient tensor
        """
        self.compressor = base_compressor
        self.error_accumulator = np.zeros(shape, dtype=np.float32)

    def compress(self, gradient: np.ndarray) -> Tuple:
        """
        Compress gradient with error feedback.

        Algorithm:
            1. Add accumulated error to gradient
            2. Compress the sum
            3. Store new error (original - transmitted)
        """
        # Add accumulated error
        corrected_gradient = gradient + self.error_accumulator

        # Compress
        compressed = self.compressor.sparsify(corrected_gradient)

        # Compute what was actually transmitted
        transmitted = self.compressor.densify(*compressed, gradient.shape)

        # Update error accumulator
        self.error_accumulator = corrected_gradient - transmitted

        return compressed

    def reset(self):
        """Reset error accumulator (e.g., at optimization reset)."""
        self.error_accumulator.fill(0)
```

**定理（误差反馈收敛性）**：
带误差反馈时，Top-K 稀疏化以与全精度 SGD 相同的速率收敛，只差一个常数因子。

**证明梗概**：误差反馈确保所有梯度信息最终都被传输。设 $e_t$ 为第 $t$ 步的误差，则：

$$\sum_{t=1}^T ||e_t||^2 \leq ||g_1||^2 + ||g_2||^2 + ... + ||g_T||^2$$

累积误差被梯度总量所界定。

### 深度梯度压缩（DGC）

Lin 等人（2018）组合了多种技术：

```python
from collections import deque

class DeepGradientCompressor:
    """
    Deep Gradient Compression: 99.9% sparsity with minimal accuracy loss.

    Combines:
    1. Momentum correction
    2. Local gradient clipping
    3. Momentum factor masking
    4. Warm-up training
    """

    def __init__(self, shape: Tuple[int, ...], sparsity: float = 0.999,
                 momentum: float = 0.9, warmup_epochs: int = 4):
        self.sparsity = sparsity
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Accumulators
        self.velocity = np.zeros(shape, dtype=np.float32)
        self.error = np.zeros(shape, dtype=np.float32)

        # Warmup schedule
        self.warmup_sparsities = self._compute_warmup_schedule()

    def _compute_warmup_schedule(self) -> list:
        """Gradual sparsity increase during warmup."""
        if self.warmup_epochs == 0:
            return []

        # Linear warmup from 75% to target sparsity
        schedule = []
        start = 0.75
        end = self.sparsity
        for i in range(self.warmup_epochs):
            frac = (i + 1) / self.warmup_epochs
            schedule.append(start + (end - start) * frac)
        return schedule

    def compress(self, gradient: np.ndarray, epoch: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress gradient using DGC algorithm.

        Args:
            gradient: Raw gradient
            epoch: Current training epoch

        Returns:
            indices: Selected gradient positions
            values: Compressed gradient values
        """
        # Determine sparsity for this epoch
        if epoch < len(self.warmup_sparsities):
            current_sparsity = self.warmup_sparsities[epoch]
        else:
            current_sparsity = self.sparsity

        # Momentum correction: u_t = m * u_{t-1} + g_t
        self.velocity = self.momentum * self.velocity + gradient

        # Add error feedback
        corrected = self.velocity + self.error

        # Local gradient clipping (per-layer)
        std = np.std(corrected)
        if std > 0:
            corrected = np.clip(corrected, -2.5 * std, 2.5 * std)

        # Top-K selection
        k = max(1, int(corrected.size * (1 - current_sparsity)))
        flat = corrected.flatten()
        abs_flat = np.abs(flat)

        # Find threshold
        if k >= len(flat):
            threshold = 0
        else:
            threshold = np.partition(abs_flat, -k)[-k]

        # Create mask
        mask = abs_flat >= threshold
        indices = np.where(mask)[0].astype(np.uint32)
        values = flat[indices].astype(np.float32)

        # Update error
        transmitted = np.zeros_like(flat)
        transmitted[indices] = values
        self.error = corrected.flatten() - transmitted
        self.error = self.error.reshape(gradient.shape)

        # Momentum factor masking: reset velocity for transmitted gradients
        velocity_flat = self.velocity.flatten()
        velocity_flat[indices] = 0
        self.velocity = velocity_flat.reshape(gradient.shape)

        return indices, values
```

**DGC 实现了 99.9% 稀疏度**（100 倍压缩），在 ImageNet 上精度损失可忽略。

## 误差分析

### 压缩误差界

对压缩器 $C$ 与重建 $D$，定义压缩误差：

$$\epsilon = g - D(C(g))$$

若压缩器满足以下条件，则称其为 $\delta$-收缩的：

$$\mathbb{E}[||\epsilon||^2] \leq \delta^2 ||g||^2$$

**例子**：

- 有 $s$ 层的 QSGD：$\delta^2 = \min(d/s^2, \sqrt{d}/s)$
- Top-K：$\delta^2 = 1 - k/d$（无误差反馈）
- 带误差反馈的 Top-K：随时间 $\delta^2 \to 0$

### 带压缩的收敛

**定理**：对带误差反馈的 $\delta$-收缩压缩，SGD 的收敛速率为：

$$\mathbb{E}[f(w_T) - f(w^*)] = O\left(\frac{1}{\sqrt{T}}\right)$$

与未压缩 SGD 相同！

**证明（梗概）**：
1. 分解更新：$w_{t+1} = w_t - \eta (g_t + \epsilon_t - \epsilon_{t-1})$
2. 误差项随时间望远镜式抵消
3. 有界方差条件保证收敛

```python
class ConvergenceAnalyzer:
    """Analyze convergence with gradient compression."""

    def __init__(self, contraction_delta: float, learning_rate: float):
        self.delta = contraction_delta
        self.lr = learning_rate

    def expected_error_after_T_steps(self, T: int,
                                      gradient_norm: float) -> float:
        """
        Upper bound on expected optimization error.

        With δ-contractive compression and error feedback:
        E[f(w_T) - f*] ≤ O(σ / √T) + O(δ² η G²)

        Where:
            σ: stochastic gradient variance
            G: gradient norm bound
            η: learning rate
        """
        stochastic_term = gradient_norm / np.sqrt(T)
        compression_term = self.delta**2 * self.lr * gradient_norm**2

        return stochastic_term + compression_term

    def optimal_compression_level(self, bandwidth_gbps: float,
                                   compute_time_ms: float,
                                   model_size_gb: float) -> float:
        """
        Find optimal compression ratio.

        Trade-off: Higher compression → faster communication
                   but higher variance → slower convergence
        """
        # Time without compression
        comm_time = (model_size_gb * 8) / bandwidth_gbps * 1000  # ms

        # Find compression ratio where comm_time = compute_time
        # This balances compute and communication
        target_ratio = compute_time_ms / comm_time

        return min(1.0, target_ratio)
```

## 实用系统

### PowerSGD

Vogels 等人（2019）提出了低秩梯度近似：

```python
class PowerSGD:
    """
    PowerSGD: Low-rank gradient compression.

    Approximates gradients as rank-r matrices: G ≈ P @ Q^T
    """

    def __init__(self, shape: Tuple[int, int], rank: int = 4,
                 num_power_iterations: int = 1):
        """
        Args:
            shape: Gradient matrix shape (m, n)
            rank: Approximation rank r
            num_power_iterations: Power iteration steps for better approximation
        """
        self.m, self.n = shape
        self.rank = rank
        self.num_iters = num_power_iterations

        # Initialize Q with orthogonal matrix
        self.Q = np.linalg.qr(np.random.randn(self.n, rank))[0]

        # Error memory
        self.error = np.zeros(shape, dtype=np.float32)

    def compress(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress gradient to low-rank representation.

        Returns:
            P: Left factor (m × r)
            Q: Right factor (n × r)
        """
        # Add error feedback
        G = gradient + self.error

        # Power iteration: P = G @ Q
        P = G @ self.Q

        # Orthogonalize P
        P, _ = np.linalg.qr(P)

        # Compute Q = G^T @ P
        Q = G.T @ P

        # Power iteration refinement
        for _ in range(self.num_iters - 1):
            P = G @ Q
            P, _ = np.linalg.qr(P)
            Q = G.T @ P

        # Reconstruct and compute error
        reconstructed = P @ Q.T
        self.error = G - reconstructed

        # Update Q for next iteration (warm start)
        self.Q, _ = np.linalg.qr(Q)

        return P.astype(np.float32), Q.astype(np.float32)

    def decompress(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Reconstruct gradient from low-rank factors."""
        return P @ Q.T

    @property
    def compression_ratio(self) -> float:
        """Communication cost reduction."""
        original = self.m * self.n
        compressed = self.rank * (self.m + self.n)
        return compressed / original

class PowerSGDOptimizer:
    """Optimizer wrapper with PowerSGD compression."""

    def __init__(self, model_parameters: list, lr: float,
                 rank: int = 4, min_compression_dim: int = 256):
        """
        Args:
            model_parameters: List of parameter tensors
            lr: Learning rate
            rank: PowerSGD rank
            min_compression_dim: Only compress tensors larger than this
        """
        self.lr = lr
        self.rank = rank
        self.min_dim = min_compression_dim

        # Create compressor for each parameter
        self.compressors = {}
        for i, param in enumerate(model_parameters):
            if param.numel() >= min_compression_dim:
                # Reshape to 2D for matrix compression
                shape_2d = self._to_2d_shape(param.shape)
                self.compressors[i] = PowerSGD(shape_2d, rank)

    def _to_2d_shape(self, shape: Tuple) -> Tuple[int, int]:
        """Reshape tensor to 2D for low-rank compression."""
        if len(shape) == 1:
            return (1, shape[0])
        elif len(shape) == 2:
            return shape
        else:
            # Flatten all but last dimension
            return (np.prod(shape[:-1]), shape[-1])

    def step(self, gradients: list) -> list:
        """
        Apply compressed gradients.

        Returns: List of compressed gradients for communication
        """
        compressed = []

        for i, grad in enumerate(gradients):
            if i in self.compressors:
                # Reshape, compress, decompress, reshape back
                grad_2d = grad.reshape(self._to_2d_shape(grad.shape))
                P, Q = self.compressors[i].compress(grad_2d)
                compressed.append((P, Q))
            else:
                # Small tensor: no compression
                compressed.append(grad)

        return compressed
```

**压缩比**：对 $m \times n$ 矩阵、秩 $r$：

$$\rho = \frac{r(m + n)}{mn}$$

对 $m = n = 4096$、$r = 4$：$\rho = 0.002$（500 倍压缩！）。

### 1-Bit Adam

利用 Adam 的方差估计实现 1 比特压缩：

```python
class OneBitAdam:
    """
    1-Bit Adam: Extreme compression using momentum.

    Key insight: Adam's momentum provides the magnitude information,
    so we only need to transmit the sign of gradient corrections.
    """

    def __init__(self, parameters: list, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, warmup_steps: int = 500):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Adam state
        self.m = [np.zeros_like(p) for p in parameters]  # First moment
        self.v = [np.zeros_like(p) for p in parameters]  # Second moment

        # Error compensation
        self.errors = [np.zeros_like(p) for p in parameters]

    def compress_gradient(self, gradient: np.ndarray,
                         idx: int) -> Tuple[np.ndarray, float]:
        """
        Compress gradient to 1 bit per element.

        During warmup: Use full precision
        After warmup: Transmit only signs
        """
        if self.step_count < self.warmup_steps:
            # Warmup: full precision communication
            return gradient, 1.0

        # Add error compensation
        compensated = gradient + self.errors[idx]

        # Extract signs
        signs = np.sign(compensated).astype(np.int8)

        # Scale by local mean absolute value
        scale = np.mean(np.abs(compensated)) + self.eps

        # Compute error
        reconstructed = scale * signs.astype(np.float32)
        self.errors[idx] = compensated - reconstructed

        return signs, scale

    def decompress_gradient(self, signs: np.ndarray,
                            scale: float) -> np.ndarray:
        """Reconstruct gradient from signs and scale."""
        return scale * signs.astype(np.float32)

    def step(self, gradients: list):
        """
        Adam update with 1-bit gradient communication.
        """
        self.step_count += 1

        # Bias correction
        bc1 = 1 - self.beta1 ** self.step_count
        bc2 = 1 - self.beta2 ** self.step_count

        updates = []
        for i, g in enumerate(gradients):
            # Update moments (using decompressed gradients after AllReduce)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Compute update
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            updates.append(update)

        return updates
```

**关键洞察**：预热之后，Adam 的动量已经捕捉了梯度幅度，因此只需传输符号（方向）。

## 带压缩的 AllReduce

压缩后的梯度需要改造过的 AllReduce：

```python
from typing import Callable
import struct

class CompressedAllReduce:
    """
    AllReduce for compressed gradients.

    Standard AllReduce doesn't work with compressed representations.
    Need to either:
    1. AllGather + local aggregation
    2. Custom reduction for compressed format
    """

    def __init__(self, world_size: int, rank: int,
                 compressor: Callable, decompressor: Callable):
        self.world_size = world_size
        self.rank = rank
        self.compressor = compressor
        self.decompressor = decompressor

    def allreduce_sparsified(self, indices: np.ndarray,
                             values: np.ndarray) -> np.ndarray:
        """
        AllReduce for sparse gradients.

        Algorithm:
        1. AllGather indices and values from all ranks
        2. Merge sparse representations
        3. Average overlapping indices
        """
        # Simulate AllGather (in practice, use NCCL/Gloo)
        all_indices = [indices]  # Would gather from all ranks
        all_values = [values]

        # In actual distributed setting:
        # all_indices = allgather(indices)
        # all_values = allgather(values)

        # Merge: accumulate values at each index
        from collections import defaultdict
        accumulated = defaultdict(lambda: (0.0, 0))  # (sum, count)

        for idx_array, val_array in zip(all_indices, all_values):
            for idx, val in zip(idx_array, val_array):
                s, c = accumulated[idx]
                accumulated[idx] = (s + val, c + 1)

        # Average
        merged_indices = np.array(list(accumulated.keys()), dtype=np.uint32)
        merged_values = np.array(
            [s / c for s, c in accumulated.values()],
            dtype=np.float32
        )

        return merged_indices, merged_values

    def allreduce_quantized(self, quantized: QuantizedTensor) -> np.ndarray:
        """
        AllReduce for quantized gradients.

        Cannot directly average quantized values!
        Must dequantize, reduce, then optionally requantize.
        """
        # Dequantize locally
        local_full = quantized.dequantize()

        # Standard AllReduce (simulated)
        # In practice: dist.all_reduce(local_full)
        global_sum = local_full  # Would be sum from all ranks

        # Average
        return global_sum / self.world_size

    def allreduce_low_rank(self, P: np.ndarray,
                          Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        AllReduce for low-rank gradients (PowerSGD style).

        Key insight: AllReduce on P@Q^T factors directly.

        sum_i(P_i @ Q_i^T) = [P_0 | P_1 | ... | P_n] @ [Q_0 | Q_1 | ...]^T
        """
        # AllGather P and Q matrices
        all_P = [P]  # Would gather from all ranks
        all_Q = [Q]

        # Concatenate
        P_concat = np.hstack(all_P)  # m × (r * world_size)
        Q_concat = np.hstack(all_Q)  # n × (r * world_size)

        # Reduce rank back to r (optional, for memory)
        # Could use SVD or just sum and re-orthogonalize

        return P_concat, Q_concat
```

### 高效稀疏 AllReduce

对高度稀疏的梯度，专门算法会有所帮助：

```python
class SparseAllReduce:
    """
    Efficient AllReduce for sparse tensors.

    Uses tree-based aggregation with sparse merge.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def tree_reduce(self, local_sparse: Tuple[np.ndarray, np.ndarray],
                   original_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Binary tree reduction for sparse gradients.

        Complexity: O(k * log(P)) where k = sparsity
        vs O(d * log(P)) for dense reduce
        """
        indices, values = local_sparse

        # Number of tree levels
        levels = int(np.ceil(np.log2(self.world_size)))

        current_indices = indices
        current_values = values

        for level in range(levels):
            step = 2 ** level

            if self.rank % (2 * step) == 0:
                # Receive from partner
                partner = self.rank + step
                if partner < self.world_size:
                    # Simulate receive
                    partner_indices = np.array([], dtype=np.uint32)
                    partner_values = np.array([], dtype=np.float32)

                    # Merge sparse representations
                    current_indices, current_values = self._merge_sparse(
                        (current_indices, current_values),
                        (partner_indices, partner_values)
                    )
            elif self.rank % (2 * step) == step:
                # Send to partner
                partner = self.rank - step
                # Simulate send
                pass

        # Broadcast result
        return current_indices, current_values

    def _merge_sparse(self, sparse1: Tuple[np.ndarray, np.ndarray],
                      sparse2: Tuple[np.ndarray, np.ndarray]
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge two sparse representations, summing overlapping indices."""
        idx1, val1 = sparse1
        idx2, val2 = sparse2

        # Create combined representation
        combined = {}
        for i, v in zip(idx1, val1):
            combined[i] = combined.get(i, 0) + v
        for i, v in zip(idx2, val2):
            combined[i] = combined.get(i, 0) + v

        # Convert back to arrays
        if not combined:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

        merged_indices = np.array(sorted(combined.keys()), dtype=np.uint32)
        merged_values = np.array([combined[i] for i in merged_indices],
                                dtype=np.float32)

        return merged_indices, merged_values
```

## 何时使用压缩

### 决策框架

```python
from dataclasses import dataclass
from enum import Enum

class CompressionMethod(Enum):
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    LOW_RANK = "low_rank"
    HYBRID = "hybrid"

@dataclass
class ClusterSpec:
    """Cluster hardware specification."""
    intra_node_bandwidth_gbps: float  # NVLink/NVSwitch bandwidth
    inter_node_bandwidth_gbps: float  # Network bandwidth
    gpus_per_node: int
    num_nodes: int

class CompressionAdvisor:
    """Recommend compression strategy based on hardware and workload."""

    def __init__(self, cluster: ClusterSpec):
        self.cluster = cluster

    def recommend(self, model_size_bytes: int,
                  compute_time_ms: float,
                  accuracy_tolerance: str = "standard") -> CompressionMethod:
        """
        Recommend compression method.

        Args:
            model_size_bytes: Size of gradients to communicate
            compute_time_ms: Time for forward + backward pass
            accuracy_tolerance: "strict", "standard", or "relaxed"

        Returns:
            Recommended compression method
        """
        # Compute communication time without compression
        # For ring AllReduce: 2 * (P-1)/P * size
        world_size = self.cluster.gpus_per_node * self.cluster.num_nodes

        # Pessimistic estimate using inter-node bandwidth
        bandwidth = self.cluster.inter_node_bandwidth_gbps * 1e9 / 8  # bytes/sec
        comm_time_ms = (2 * model_size_bytes / bandwidth) * 1000

        # Compute-to-communication ratio
        ratio = compute_time_ms / comm_time_ms

        if ratio > 2.0:
            # Compute-bound: no compression needed
            return CompressionMethod.NONE

        if ratio > 0.5:
            # Moderate: light compression
            if accuracy_tolerance == "strict":
                return CompressionMethod.QUANTIZATION  # 8-bit
            else:
                return CompressionMethod.SPARSIFICATION  # Top-K with EF

        if ratio > 0.1:
            # Communication-bound: aggressive compression
            if accuracy_tolerance == "relaxed":
                return CompressionMethod.LOW_RANK
            else:
                return CompressionMethod.SPARSIFICATION

        # Severely communication-bound
        return CompressionMethod.HYBRID

    def estimate_speedup(self, method: CompressionMethod,
                         compression_ratio: float,
                         model_size_bytes: int,
                         compute_time_ms: float) -> float:
        """Estimate training speedup from compression."""
        world_size = self.cluster.gpus_per_node * self.cluster.num_nodes
        bandwidth = self.cluster.inter_node_bandwidth_gbps * 1e9 / 8

        # Original step time
        original_comm = (2 * model_size_bytes / bandwidth) * 1000
        original_step = compute_time_ms + original_comm

        # Compressed step time
        compressed_comm = original_comm * compression_ratio
        # Add overhead for compression/decompression
        overhead = 0.05 * compute_time_ms  # ~5% compute overhead
        compressed_step = compute_time_ms + compressed_comm + overhead

        return original_step / compressed_step
```

### 实践中的压缩

| 场景 | 推荐方法 | 压缩比 |
|------|----------|--------|
| 高带宽（NVLink） | 无 | 1.0 |
| 100 Gbps 以太网 | 8 bit 量化 | 0.25 |
| 25 Gbps 以太网 | TopK + EF | 0.01–0.1 |
| 跨数据中心 | PowerSGD | 0.001–0.01 |
| 联邦学习 | DGC | 0.001 |

## 练习题

1. **QSGD 分析**：证明 QSGD 无偏：$\mathbb{E}[Q_s(g)] = g$。然后计算 $\text{Var}(Q_s(g))$ 关于 $s$ 与 $||g||$ 的函数。

??? success "解答"
    **QSGD 定义：**

    对标量 $g_i$、量化层数 $s$：

    $$Q_s(g_i) = ||g|| \cdot \text{sign}(g_i) \cdot \xi_i$$

    其中 $\xi_i$ 是随机变量，取值：
    - $\frac{\ell + 1}{s}$，概率为 $\frac{|g_i|}{||g||} \cdot s - \ell$
    - $\frac{\ell}{s}$，概率为 $1 - \left(\frac{|g_i|}{||g||} \cdot s - \ell\right)$

    且 $\ell = \lfloor \frac{|g_i|}{||g||} \cdot s \rfloor$。

    **无偏性证明：**

    $$\mathbb{E}[\xi_i] = \frac{\ell + 1}{s} \cdot p + \frac{\ell}{s} \cdot (1 - p)$$

    其中 $p = \frac{|g_i|}{||g||} \cdot s - \ell$

    $$= \frac{\ell + 1}{s} \cdot p + \frac{\ell}{s} - \frac{\ell}{s} \cdot p$$

    $$= \frac{\ell}{s} + \frac{p}{s}$$

    $$= \frac{\ell + p}{s} = \frac{\ell + \frac{|g_i|}{||g||} \cdot s - \ell}{s}$$

    $$= \frac{|g_i|}{||g||}$$

    因此：

    $$\mathbb{E}[Q_s(g_i)] = ||g|| \cdot \text{sign}(g_i) \cdot \frac{|g_i|}{||g||} = g_i$$

    $$\boxed{\mathbb{E}[Q_s(g)] = g}$$

    **方差计算：**

    $$\text{Var}(\xi_i) = \mathbb{E}[\xi_i^2] - \mathbb{E}[\xi_i]^2$$

    $$\mathbb{E}[\xi_i^2] = \left(\frac{\ell + 1}{s}\right)^2 p + \left(\frac{\ell}{s}\right)^2 (1-p)$$

    化简后：

    $$\text{Var}(\xi_i) = \frac{p(1-p)}{s^2}$$

    由于 $p \leq 1$、$p(1-p) \leq 1/4$：

    $$\text{Var}(Q_s(g_i)) = ||g||^2 \cdot \text{Var}(\xi_i) \leq \frac{||g||^2}{4s^2}$$

    对完整向量（各分量独立）：

    $$\boxed{\text{Var}(Q_s(g)) \leq \frac{d \cdot ||g||^2}{4s^2}}$$

    当 $s \to \infty$，方差 $\to 0$（全精度）。

2. **误差反馈**：为 Top-K 稀疏化实现误差反馈。验证累积误差在 1000 个训练步内保持有界。

??? success "解答"
    **误差反馈实现：**

    ```python
    import torch

    class TopKWithErrorFeedback:
        def __init__(self, k_ratio=0.01):
            self.k_ratio = k_ratio
            self.error = None

        def compress(self, gradient):
            # Add accumulated error to current gradient
            if self.error is None:
                self.error = torch.zeros_like(gradient)

            g_accumulated = gradient + self.error

            # Top-K selection
            k = max(1, int(gradient.numel() * self.k_ratio))
            values, indices = torch.topk(
                g_accumulated.abs().flatten(), k
            )
            threshold = values[-1]

            # Create sparse gradient
            mask = g_accumulated.abs() >= threshold
            sparse_g = g_accumulated * mask

            # Update error: what we couldn't send this round
            self.error = g_accumulated - sparse_g

            return sparse_g, mask

        def get_error_norm(self):
            return self.error.norm().item() if self.error is not None else 0

    # Verification experiment
    def verify_bounded_errors():
        compressor = TopKWithErrorFeedback(k_ratio=0.01)
        error_norms = []

        for step in range(1000):
            # Simulate gradient with some structure
            gradient = torch.randn(10000) * (1.0 / (step + 1) ** 0.5)

            _, _ = compressor.compress(gradient)
            error_norms.append(compressor.get_error_norm())

        # Check boundedness
        max_error = max(error_norms)
        avg_error = sum(error_norms) / len(error_norms)

        return max_error, avg_error
    ```

    **预期结果：**

    | 指标 | 数值 |
    |------|------|
    | 最大误差范数 | ~0.5–2.0 |
    | 平均误差范数 | ~0.2–0.5 |
    | 趋势 | 稳定，不增长 |

    **误差为何保持有界：**

    误差反馈保证：

    $$e_{t+1} = e_t + g_t - \text{TopK}(e_t + g_t)$$

    关键洞察：TopK 总是移除最大的值，因此：

    $$||e_{t+1}||_\infty \leq ||e_t + g_t||_\infty \cdot (1 - k/d)$$

    加上训练中梯度通常衰减，误差保持有界。

    **验证图的观察：**

    $$\boxed{\text{Error norms oscillate but do not grow over 1000 steps}}$$

3. **压缩方案选择**：你有 64 个节点，由 100 Gbps 以太网连接。每个节点有 8 块 GPU，由 NVLink 连接。模型大小为 1B 参数。前向＋反向耗时 100ms。跨节点数据并行应使用什么压缩？

??? success "解答"
    **通信分析：**

    | 参数 | 数值 |
    |------|------|
    | 节点数 | 64 |
    | 每节点 GPU 数 | 8 |
    | GPU 总数 | 512 |
    | 跨节点带宽 | 100 Gbps = 12.5 GB/s |
    | 节点内带宽 | NVLink ~600 GB/s |
    | 模型参数 | 1B |
    | 梯度大小（FP16） | 2 GB |
    | 前向＋反向时间 | 100 ms |

    **不压缩时的 AllReduce 时间：**

    对分层 AllReduce（节点内归约、跨节点 AllReduce、节点内广播）：

    跨节点 AllReduce 占主导（64 个节点，环形算法）：

    $$T_{inter} = \frac{2 \times 2 \text{ GB}}{12.5 \text{ GB/s}} \times \frac{63}{64} \approx 315 \text{ ms}$$

    **通信与计算之比：**
    $$\frac{T_{comm}}{T_{comp}} = \frac{315}{100} = 3.15$$

    通信耗时是计算的 3 倍——无法接受！

    **所需压缩比：**

    要让通信 ≤ 计算：

    $$\text{Compression ratio} \geq \frac{315}{100} \approx 3.2×$$

    要与计算完全重叠：

    $$\text{Compression ratio} \geq \frac{315}{100} \approx 3.2×$$

    为留出安全余量，目标定为 **10 倍压缩**。

    **压缩方案选择：**

    | 方法 | 压缩比 | 开销 | 是否合适？ |
    |------|--------|------|------------|
    | FP16→FP8 | 2× | 极低 | 不足 |
    | 4 bit 量化 | 4× | 低 | 勉强 |
    | 1-bit SGD | 16× | 低 | ✓ 好 |
    | Top-10% 稀疏化 | 10× | 中 | ✓ 好 |
    | PowerSGD rank-16 | ~50× | 中 | ✓ 极佳 |

    **推荐：**

    鉴于通信造成的 3 倍减速：

    1. **首选：PowerSGD，秩 16–32**
       - 压缩比：32–64 倍
       - 误差反馈处理偏差
       - 对 1B 模型精度影响极小

    2. **备选：带误差反馈的 1 比特量化**
       - 16 倍压缩
       - 计算开销极低
       - 延迟可预测

    **在 10 倍压缩下：**
    $$T_{comm}^{compressed} = \frac{315}{10} = 31.5 \text{ ms}$$

    **最终效率：**
    $$\text{Throughput improvement} = \frac{100 + 315}{100 + 31.5} = \frac{415}{131.5} = \boxed{3.16× \text{ speedup}}$$

4. **PowerSGD 秩选择**：对一个 4096 × 4096 的权重矩阵，多少秩能带来 100 倍压缩？对典型梯度矩阵，近似误差（Frobenius 范数）是多少？

??? success "解答"
    **PowerSGD 压缩比：**

    对矩阵 $M \in \mathbb{R}^{m \times n}$，PowerSGD 近似为：

    $$M \approx P Q^T$$
    其中 $P \in \mathbb{R}^{m \times r}$、$Q \in \mathbb{R}^{n \times r}$。

    **存储对比：**

    | 表示 | 大小 |
    |------|------|
    | 原矩阵 | $m \times n$ |
    | 低秩因子 | $r(m + n)$ |
    | 压缩比 | $\frac{mn}{r(m+n)}$ |

    **对 4096 × 4096 矩阵：**

    $$\text{Compression ratio} = \frac{4096 \times 4096}{r(4096 + 4096)} = \frac{16,777,216}{8192r}$$

    要 100 倍压缩：

    $$100 = \frac{16,777,216}{8192r}$$

    $$r = \frac{16,777,216}{8192 \times 100} = \frac{16,777,216}{819,200} \approx 20.48$$

    $$\boxed{r = 20 \text{ (or } r = 21 \text{ for slightly less compression)}}$$

    **验证：**
    - $r = 20$：压缩比 = $\frac{16,777,216}{8192 \times 20} = 102.4×$
    - $r = 21$：压缩比 = $\frac{16,777,216}{8192 \times 21} = 97.5×$

    **近似误差分析：**

    对奇异值 $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_n$ 的矩阵 $G$：

    最佳秩 $r$ 近似误差（Eckart-Young 定理）：

    $$||G - G_r||_F = \sqrt{\sum_{i=r+1}^{n} \sigma_i^2}$$

    **典型梯度矩阵的谱：**

    经验上，梯度矩阵的奇异值衰减很快：

    $$\sigma_i \approx \sigma_1 \cdot i^{-\alpha}$$

    对典型训练梯度，$\alpha \approx 0.5$ 到 $1.0$。

    **数值例子**（假设幂律衰减，$\alpha = 0.7$）：

    ```python
    import numpy as np

    # Simulate typical gradient singular value distribution
    n = 4096
    r = 20
    sigma_1 = 1.0
    alpha = 0.7

    singular_values = sigma_1 * np.arange(1, n+1) ** (-alpha)

    # Total Frobenius norm
    total_norm = np.sqrt(np.sum(singular_values ** 2))

    # Error (tail beyond rank r)
    error_norm = np.sqrt(np.sum(singular_values[r:] ** 2))

    # Relative error
    relative_error = error_norm / total_norm
    ```

    **预期结果：**

    | α（衰减速率） | 相对误差 | 质量 |
    |---------------|----------|------|
    | 0.5（慢衰减） | ~15–20% | 中等 |
    | 0.7（典型） | ~8–12% | 好 |
    | 1.0（快衰减） | ~3–5% | 极佳 |

    **对典型梯度（α ≈ 0.7，r = 20）：**
    $$\boxed{\text{Relative error} \approx 10\%}$$

    **注**：带误差反馈的 PowerSGD 会随时间修正这一近似误差，因此即便每步 10–15% 的误差，训练也能收敛。

5. **混合压缩**：设计一个压缩方案：小梯度（< 1MB）用量化，大梯度用稀疏化。实现并测量开销。

??? success "解答"
    **设计理由：**

    | 梯度大小 | 最佳方法 | 原因 |
    |----------|----------|------|
    | < 1 MB | 量化 | 开销低、压缩比固定 |
    | ≥ 1 MB | 稀疏化 | 压缩比更高、扩展性好 |

    **对小梯度用量化：**
    - 固定 4 bit 量化 → 4 倍压缩
    - 开销：量化/反量化 O(n)
    - 在 n 小时最优（开销占主导）

    **对大梯度用稀疏化：**
    - Top-1% 加误差反馈 → 100 倍压缩
    - 开销：top-k 选择 O(n log k)
    - 在大张量上摊销良好

    **实现：**

    ```python
    import torch
    import time

    class HybridCompressor:
        def __init__(self, size_threshold=1_000_000, k_ratio=0.01, bits=4):
            self.size_threshold = size_threshold  # 1MB = 250K float32
            self.k_ratio = k_ratio
            self.bits = bits
            self.error_buffers = {}  # Error feedback for sparsification

        def quantize(self, tensor):
            """4-bit quantization with scaling"""
            min_val = tensor.min()
            max_val = tensor.max()

            # Scale to [0, 2^bits - 1]
            scale = (max_val - min_val) / (2**self.bits - 1)
            if scale == 0:
                return torch.zeros_like(tensor, dtype=torch.uint8), min_val, scale

            quantized = ((tensor - min_val) / scale).round().to(torch.uint8)
            return quantized, min_val, scale

        def dequantize(self, quantized, min_val, scale):
            """Reconstruct from quantized values"""
            return quantized.float() * scale + min_val

        def sparsify(self, tensor, name):
            """Top-K sparsification with error feedback"""
            # Add error feedback
            if name not in self.error_buffers:
                self.error_buffers[name] = torch.zeros_like(tensor)

            accumulated = tensor + self.error_buffers[name]

            # Select top-K
            k = max(1, int(tensor.numel() * self.k_ratio))
            values, indices = torch.topk(accumulated.abs().flatten(), k)
            threshold = values[-1]

            # Create sparse representation
            mask = accumulated.abs() >= threshold
            sparse_values = accumulated[mask]

            # Update error buffer
            self.error_buffers[name] = accumulated.clone()
            self.error_buffers[name][mask] = 0

            return sparse_values, mask

        def desparsify(self, sparse_values, mask, shape):
            """Reconstruct from sparse representation"""
            result = torch.zeros(shape).flatten()
            result[mask.flatten()] = sparse_values
            return result.reshape(shape)

        def compress(self, gradient, name="default"):
            """Hybrid compression based on size"""
            numel = gradient.numel()

            if numel < self.size_threshold:
                # Use quantization for small tensors
                quantized, min_val, scale = self.quantize(gradient)
                return ('quant', quantized, min_val, scale, gradient.shape)
            else:
                # Use sparsification for large tensors
                sparse_values, mask = self.sparsify(gradient, name)
                return ('sparse', sparse_values, mask, gradient.shape)

        def decompress(self, compressed):
            """Decompress based on method used"""
            if compressed[0] == 'quant':
                _, quantized, min_val, scale, shape = compressed
                return self.dequantize(quantized, min_val, scale).reshape(shape)
            else:
                _, sparse_values, mask, shape = compressed
                return self.desparsify(sparse_values, mask, shape)

    def benchmark_hybrid():
        compressor = HybridCompressor()
        results = []

        test_sizes = [
            (100_000, "Small (100K)"),      # < 1MB
            (500_000, "Medium (500K)"),     # < 1MB
            (1_000_000, "Threshold (1M)"),  # = 1MB
            (10_000_000, "Large (10M)"),    # > 1MB
            (50_000_000, "XLarge (50M)"),   # >> 1MB
        ]

        for size, label in test_sizes:
            gradient = torch.randn(size)

            # Warmup
            for _ in range(3):
                compressed = compressor.compress(gradient, f"layer_{size}")
                _ = compressor.decompress(compressed)

            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            for _ in range(10):
                compressed = compressor.compress(gradient, f"layer_{size}")
                reconstructed = compressor.decompress(compressed)

            elapsed = (time.perf_counter() - start) / 10

            # Calculate compression ratio
            original_bytes = gradient.numel() * 4
            if compressed[0] == 'quant':
                compressed_bytes = compressed[1].numel() + 8  # quantized + metadata
            else:
                compressed_bytes = compressed[1].numel() * 4 + compressed[2].numel() // 8

            compression_ratio = original_bytes / compressed_bytes

            results.append({
                'size': label,
                'method': compressed[0],
                'time_ms': elapsed * 1000,
                'compression': compression_ratio,
                'overhead_per_elem_ns': elapsed * 1e9 / size
            })

        return results
    ```

    **预期开销测量：**

    | 梯度大小 | 方法 | 压缩时间 | 压缩比 | 每元素开销 |
    |----------|------|----------|--------|------------|
    | 100K | 量化 | 0.2 ms | 4× | 2 ns |
    | 500K | 量化 | 0.8 ms | 4× | 1.6 ns |
    | 1M | 稀疏 | 5 ms | 100× | 5 ns |
    | 10M | 稀疏 | 40 ms | 100× | 4 ns |
    | 50M | 稀疏 | 180 ms | 100× | 3.6 ns |

    **设计权衡：**

    | 考量 | 量化 | 稀疏化 |
    |------|------|--------|
    | 压缩比 | 固定（4×） | 高（100×） |
    | 开销扩展 | O(n) | O(n log k) |
    | 误差所需内存 | 无 | 每张量 O(n) |
    | 精度影响 | 有界误差 | 需要误差反馈 |

    **小结：**

    $$\boxed{\text{Hybrid: Quantize if } n < 1\text{M, else TopK-1\% with error feedback}}$$

    1M 元素处的转折权衡了：
    - 量化对小张量的低开销
    - 稀疏化对大张量的高压缩比

6. **收敛实验**：用 ResNet-18 在 CIFAR-10 上训练：(a) 不压缩，(b) 8 bit 量化，(c) Top-1% 稀疏化，(d) PowerSGD rank-4。比较收敛曲线与最终精度。

??? success "解答"
    **实验设置：**

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Compression methods
    class NoCompression:
        def compress(self, grad):
            return grad

    class Quantize8Bit:
        def compress(self, grad):
            min_val, max_val = grad.min(), grad.max()
            scale = (max_val - min_val) / 255
            quantized = ((grad - min_val) / scale).round()
            return quantized * scale + min_val

    class TopKSparsify:
        def __init__(self, k_ratio=0.01):
            self.k_ratio = k_ratio
            self.error = {}

        def compress(self, grad, name="default"):
            if name not in self.error:
                self.error[name] = torch.zeros_like(grad)

            accumulated = grad + self.error[name]
            k = max(1, int(grad.numel() * self.k_ratio))
            vals, idx = torch.topk(accumulated.abs().flatten(), k)
            mask = accumulated.abs() >= vals[-1]

            sparse = accumulated * mask
            self.error[name] = accumulated - sparse
            return sparse

    class PowerSGDRank4:
        def __init__(self, rank=4):
            self.rank = rank
            self.P = {}
            self.Q = {}

        def compress(self, grad, name="default"):
            if grad.dim() < 2:
                return grad  # Skip 1D tensors

            m, n = grad.shape[0], grad.numel() // grad.shape[0]
            g = grad.reshape(m, n)

            if name not in self.Q:
                self.Q[name] = torch.randn(n, self.rank, device=grad.device)
                self.Q[name], _ = torch.linalg.qr(self.Q[name])

            # Power iteration
            P = g @ self.Q[name]
            P, _ = torch.linalg.qr(P)
            Q = g.T @ P
            Q, _ = torch.linalg.qr(Q)

            self.P[name] = P
            self.Q[name] = Q

            # Reconstruct
            return (P @ (P.T @ g @ Q) @ Q.T).reshape(grad.shape)

    def train_with_compression(compressor, compressor_name, epochs=100):
        # Data
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=transform)
        testloader = DataLoader(testset, batch_size=100, shuffle=False)

        # Model
        model = torchvision.models.resnet18(num_classes=10)
        model = model.cuda() if torch.cuda.is_available() else model

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'train_loss': [], 'test_acc': []}

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for inputs, targets in trainloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Apply compression to gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if hasattr(compressor, 'error'):
                            param.grad.data = compressor.compress(
                                param.grad.data, name)
                        else:
                            param.grad.data = compressor.compress(param.grad.data)

                optimizer.step()
                total_loss += loss.item()

            # Evaluate
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            history['train_loss'].append(total_loss / len(trainloader))
            history['test_acc'].append(acc)

            scheduler.step()

        return history
    ```

    **预期结果：**

    | 方法 | 最终精度 | 精度差距 | 收敛 |
    |------|----------|----------|------|
    | 不压缩 | 94.5–95.5% | 基线 | 正常 |
    | 8 bit 量化 | 94.0–95.0% | -0.3 至 -0.5% | 基本相同 |
    | Top-1% 稀疏化 | 93.5–94.5% | -0.5 至 -1.0% | 稍慢 |
    | PowerSGD rank-4 | 93.0–94.0% | -1.0 至 -1.5% | 慢 5–10% |

    **收敛曲线观察：**

    | 轮次范围 | 表现 |
    |----------|------|
    | 1–20 | 所有方法紧跟 |
    | 20–50 | 稀疏化略微落后 |
    | 50–80 | PowerSGD 出现差距 |
    | 80–100 | 所有方法趋于平台 |

    **关键发现：**

    1. **8 bit 量化**：对 ResNet-18 几乎无损
       - 4 倍压缩，精度损失 <0.5%
       - 收敛速度与基线一致

    2. **Top-1% 稀疏化**：激进但可行
       - 100 倍压缩，精度损失约 1%
       - 误差反馈对收敛至关重要
       - 早期轮次方差更大

    3. **PowerSGD rank-4**：近似误差最高
       - 对大权重矩阵约 100 倍压缩
       - 对小层（1D 偏置）吃力
       - 受益于预热期

    **精度与压缩的权衡：**

    | 方法 | 压缩比 | 精度损失 | 效率分数 |
    |------|--------|----------|----------|
    | 8 bit | 4× | 0.3% | 13.3 |
    | Top-1% | 100× | 0.8% | 125 |
    | PowerSGD r4 | ~80× | 1.2% | 67 |

    效率分数 = 压缩比 / 精度损失

    **建议：**

    $$\boxed{\text{For CIFAR-10/ResNet-18: 8-bit quantization offers best accuracy; Top-1\% best efficiency}}$$

    对更大的模型（数十亿参数），PowerSGD 会更有吸引力，因为：
    - 大权重矩阵最受益于低秩
    - 通信节省压过计算开销
    - 误差反馈会累积改进

## 关键要点

1. **梯度冗余使压缩成为可能**：时间、空间和幅度冗余支持 100–1000 倍压缩。

2. **误差反馈至关重要**：没有它，有偏压缩会损害收敛。

3. **无偏压缩保持收敛速率**：QSGD、Random-K 等无偏方法维持 O(1/√T) 的收敛。

4. **不同区制用不同方法**：中等压缩用量化，高压缩用稀疏化，极端情形用低秩。

5. **压缩开销不可忽视**：压缩的计算成本不能超过通信上的节省。

6. **预热有帮助**：从较低压缩开始、训练中逐步提高，能改善稳定性。

7. **大矩阵用 PowerSGD**：低秩近似对大模型的权重梯度矩阵最理想。