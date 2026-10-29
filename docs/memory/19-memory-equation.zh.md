---
title: "内存方程"
subtitle: "训练中 GPU 内存的解剖"
---

<div class="chapter-opener" markdown>
训练大模型本质上是一个内存问题。在解决它之前，我们必须精确地理解它：什么在消耗内存、消耗多少、以及何时消耗？内存方程给了我们这种理解。
</div>

<div class="investigation-question" markdown>
**问题**：一个 7B 参数模型在 fp16 下是 14GB。但训练它却需要 120GB 以上的内存。另外 106GB 去哪了？
</div>

<div class="notation-banner" markdown>
**本章符号：** $\Psi$ = 参数量，$B$ = 批 token 数，$S$ = 序列长度，$H$ = 隐藏维度，$L$ = 层数。参见[符号表](../appendices/notation.md)。
</div>

!!! abstract "承上启下：第一部分与第四部分"
    我们在[第 1 章](../foundations/01-scale-imperative.md)简要介绍过 **16Ψ 内存法则**。现在我们来完整剖析它。本部分还建立在**数据并行**（[第 14 章](../parallelism/14-data-parallelism-associativity.md)）之上——理解梯度如何分布，对下一章 ZeRO 的内存优化至关重要。第三部分的集合通信（[第 11 章](../collectives/11-primitives-properties.md)，AllGather、ReduceScatter）将再次作为内存高效分片的机制出现。

## 内存预算

训练期间 GPU 内存承载四类数据：

```
┌─────────────────────────────────────────────────────┐
│                  GPU Memory                          │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐    │
│  │           Model State Memory                │    │
│  │  ┌─────────────┬──────────┬──────────────┐  │    │
│  │  │ Parameters  │Gradients │  Optimizer   │  │    │
│  │  │     P       │    P     │    States    │  │    │
│  │  └─────────────┴──────────┴──────────────┘  │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Activation Memory                  │    │
│  │  ┌─────────────────────────────────────┐    │    │
│  │  │ Intermediate values saved for       │    │    │
│  │  │ backward pass                       │    │    │
│  │  └─────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Temporary Memory                   │    │
│  │  ┌─────────────────────────────────────┐    │    │
│  │  │ Workspace for operations (matmul,   │    │    │
│  │  │ flash attention, communication)     │    │    │
│  │  └─────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Fragmentation                      │    │
│  │  ┌─────────────────────────────────────┐    │    │
│  │  │ Unusable holes in allocated memory  │    │    │
│  │  └─────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## 模型状态内存

模型状态内存在整个训练过程中常驻。它保存了表示和更新模型所需的一切。

### 参数

模型权重本身：

$$M_{\text{params}} = \Psi \times b_p$$

其中：

- $\Psi$ = 参数量
- $b_p$ = 每参数字节数（fp16/bf16 为 2，fp32 为 4）

**示例**：7B 参数、fp16 = $7 \times 10^9 \times 2$ = 14 GB

### 梯度

反向传播期间，梯度被计算并累积：

$$M_{\text{grads}} = \Psi \times b_g$$

梯度通常与参数保持相同精度。

**示例**：7B 参数 = 梯度占 14 GB

### 优化器状态

优化器为每个参数维护额外状态。

**带动量的 SGD**：

- 动量缓冲区：$\Psi \times b_m$ 字节

$$M_{\text{opt}}^{\text{SGD}} = \Psi \times b_m$$

**Adam/AdamW**：

- 一阶矩（$m$）：$\Psi \times b_m$ 字节
- 二阶矩（$v$）：$\Psi \times b_v$ 字节

两个矩通常都用 fp32 以保证数值稳定：

$$M_{\text{opt}}^{\text{Adam}} = 2 \times \Psi \times 4 = 8\Psi \text{ bytes}$$

### 模型状态方程

对于使用 AdamW 的混合精度（mixed precision）训练：

| 组成部分 | 精度 | 每参数字节数 |
|-----------|-----------|---------------------|
| 参数 | fp16/bf16 | 2 |
| 梯度 | fp16/bf16 | 2 |
| 主权重 | fp32 | 4 |
| 一阶矩 | fp32 | 4 |
| 二阶矩 | fp32 | 4 |
| **合计** | | **16** |

$$M_{\text{model}} = 16\Psi \text{ bytes}$$

!!! note "实践"
    大规模训练开始前，先算出 $16\Psi$ 并与 GPU HBM 比较。如果 $16\Psi$ > HBM，就必须先做分片或卸载（offloading），再调批大小。

对于 7B 参数：$16 \times 7 \times 10^9 = 112$ GB

**这就是为什么训练一个 7B 模型需要约 8 倍于仅存放权重所需的内存。**

### 为什么需要主权重？

混合精度训练保留 fp32 的"主"权重副本：

```python
# Simplified mixed-precision training step
def training_step(model_fp16, optimizer, loss_fn, data):
    # Forward pass in fp16
    with autocast(dtype=torch.float16):
        output = model_fp16(data)
        loss = loss_fn(output)

    # Backward pass: gradients in fp16
    scaler.scale(loss).backward()

    # Unscale gradients
    scaler.unscale_(optimizer)

    # Optimizer step: updates fp32 master weights
    optimizer.step()  # optimizer holds fp32 weights

    # Copy back to fp16 model
    copy_fp32_to_fp16(optimizer.param_groups, model_fp16)
```

fp32 主权重是必要的，因为：
1. 微小的梯度更新可能在 fp16 下下溢
2. fp16 只有 3–4 位十进制精度
3. 累积的舍入误差会导致训练不稳定

## 激活内存

激活是前向传播期间计算并保存下来供反向计算使用的中间值。

### 为什么要保存激活？

反向传播用链式法则计算梯度：

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$$

其中 $y = f(W, x)$。

计算 $\frac{\partial y}{\partial W}$ 通常需要 $x$ 和/或 $y$：

- 线性层：$y = Wx$，$\frac{\partial y}{\partial W} = x^T$，需要 $x$
- ReLU：$y = \max(0, x)$，$\frac{\partial y}{\partial x} = \mathbf{1}[x > 0]$，需要掩码
- Softmax：需要 $y$ 来算雅可比
- LayerNorm：需要 $\mu, \sigma, \hat{x}$

### Transformer 中的激活内存

对于一个具有以下参数的 transformer：

- $L$ = 层数
- $B$ = 批大小
- $S$ = 序列长度
- $H$ = 隐藏维度
- $A$ = 注意力头数

每个 transformer 层存储：

| 激活 | 形状 | 大小 |
|------------|-------|------|
| LayerNorm 1 的输入 | $[B, S, H]$ | $BSH$ |
| 注意力 QKV | $[3, B, A, S, H/A]$ | $3BSH$ |
| 注意力分数 | $[B, A, S, S]$ | $BAS^2$ |
| 注意力输出 | $[B, S, H]$ | $BSH$ |
| LayerNorm 2 的输入 | $[B, S, H]$ | $BSH$ |
| FFN 中间层 | $[B, S, 4H]$ | $4BSH$ |
| FFN 输出 | $[B, S, H]$ | $BSH$ |

每层合计（线性项求和：1+3+1+1+4+1 = 11）：

$$M_{\text{act/layer}} \approx 11BSH + BAS^2$$

对整个模型：

$$M_{\text{activations}} = L \cdot (11BSH + BAS^2) \cdot b_a$$

其中 $b_a$ 是每个激活的字节数（fp16/bf16 通常为 2）。

### 序列长度问题

注意力分数随序列长度二次增长：

$$M_{\text{attention}} = L \cdot B \cdot A \cdot S^2 \cdot b_a$$

**示例**：L=32、B=1、A=32、S=8192、bf16：

$$32 \times 1 \times 32 \times 8192^2 \times 2 = 137.4 \text{ GB}$$

仅仅注意力分数！这就是长上下文训练需要 Flash Attention 和序列并行（Sequence Parallelism）的原因。

### 激活内存公式

transformer 的完整激活内存：

$$M_{\text{act}} = 2 \cdot L \cdot B \cdot S \cdot H \cdot \left(11 + \frac{A \cdot S}{H}\right)$$

对于典型 transformer 架构，每个注意力头的维度是 $d_h = H/A \approx 128$（例如 GPT-3 有 $H=12288$、$A=96$，所以 $d_h=128$）。这意味着 $H = 128A$，化简为：

$$M_{\text{act}} \approx 2 \cdot L \cdot B \cdot S \cdot H \cdot \left(11 + \frac{S}{128}\right)$$

当 $S < 1408$：线性项占主导（11 > S/128）
当 $S > 1408$：二次项占主导

## 临时内存

操作需要仅在计算期间存在的 workspace 内存。

### 矩阵乘法 workspace

cuBLAS 和 cuDNN 使用 workspace 来：

- 对中间结果做分块
- 高效执行转置
- 存放算法特定的数据

workspace 大小取决于算法选择：

| 算法 | Workspace | 速度 |
|-----------|-----------|-------|
| GEMM（默认） | 极小 | 基线 |
| GEMM + Tensor Core | 0–100 MB | 更快 |
| Strassen 类 | 大 | 渐近更快 |

### Flash Attention workspace

Flash Attention 使用不同的内存画像：

**标准注意力**：
```
Q, K, V → Scores (S²) → Softmax (S²) → Output
Memory: O(S²) for attention matrix
```

**Flash Attention**：
```
Q, K, V → Tiled computation → Output
Memory: O(S) - only current tile in SRAM
```

Flash Attention 避免将完整的 $S \times S$ 注意力矩阵落地，但需要 SRAM workspace 来存放：

- 当前的 query/key/value 分块
- 运行中的 softmax 统计量（最大值、和）
- 部分输出累积

### 通信缓冲区

分布式训练需要缓冲区用于：

| 操作 | 缓冲区大小 |
|-----------|-------------|
| AllReduce（环形） | 2 × 分块大小 |
| AllGather | (P-1) × 分块大小 |
| ReduceScatter | (P-1) × 分块大小 |
| AlltoAll | 依路由而变 |

这些缓冲区通常从一个专用池中分配。

## 完整内存方程

GPU 总内存需求：

$$M_{\text{total}} = M_{\text{model}} + M_{\text{act}} + M_{\text{temp}} + M_{\text{frag}}$$

展开每一项：

$$M_{\text{total}} = \underbrace{16\Psi}_{\text{model state}} + \underbrace{2LBSH(11 + S/128)}_{\text{activations}} + \underbrace{M_{\text{workspace}}}_{\text{temporary}} + \underbrace{f \cdot M_{\text{peak}}}_{\text{fragmentation}}$$

其中 $f$ 是碎片（fragmentation）因子（通常为 5–20%）。

### 完整示例：LLaMA-7B

模型规格：

- 参数：$\Psi = 7 \times 10^9$
- 层数：$L = 32$
- 隐藏：$H = 4096$
- 头数：$A = 32$
- 头维度：128

训练配置：

- 批大小：$B = 1$
- 序列长度：$S = 2048$

**模型状态内存**：

$$M_{\text{model}} = 16 \times 7 \times 10^9 = 112 \text{ GB}$$

**激活内存**：

$$M_{\text{act}} = 2 \times 32 \times 1 \times 2048 \times 4096 \times (11 + 2048/128)$$

$$= 2 \times 32 \times 2048 \times 4096 \times 27$$

$$= 2 \times 7.25 \times 10^9 \text{ bytes} = 14.5 \text{ GB}$$

**临时 + 碎片**：约 10–15 GB

**总计**：约 137–142 GB

这超过了 A100 的 80GB！因此需要内存优化技术。

## 内存缩放分析

### 随模型规模缩放

模型状态随参数线性增长：

$$M_{\text{model}} \propto \Psi$$

激活内存也随 $\Psi$ 增长（因为通常 $H \propto \sqrt{\Psi}$、$L \propto \sqrt{\Psi}$）：

$$M_{\text{act}} \propto L \cdot H \propto \Psi$$

**总内存随模型规模线性增长**（在批和序列固定时）。

### 随批大小缩放

模型状态与批大小无关。

激活内存随批线性增长：

$$M_{\text{act}} \propto B$$

**批大小翻倍大致使激活内存翻倍。**

### 随序列长度缩放

模型状态与序列长度无关。

激活内存有一个二次分量：

$$M_{\text{act}} \propto S + S^2$$

**长序列很费内存。** 在 S=8192 时，二次项占主导。

### 内存墙

存在一种根本性的张力：

```
Want: Larger batch → Better GPU utilization
Want: Longer sequence → Better context understanding
Want: Larger model → Better capability

Have: Fixed GPU memory

Reality: Can't have all three
```

这推动了所有内存优化技术：

- **ZeRO**：跨 GPU 分片模型状态
- **激活检查点（checkpoint）**：用计算换激活内存
- **卸载**：把 CPU/NVMe 当作扩展内存
- **序列并行**：分片激活内存
- **Flash Attention**：减少注意力内存

## 内存剖析

### 理解峰值内存

峰值内存往往出现在特定时刻：

```
Memory
  │    ╭──╮
  │   ╱    ╲     Peak during
  │  ╱      ╲    backward pass
  │ ╱        ╲
  ├╱──────────╲────────
  │            ╲
  └───────────────────→ Time
    Forward  Backward  Optimizer
```

峰值通常出现在反向传播期间，因为：
1. 所有激活都已被保存
2. 梯度正在被计算
3. 某些操作需要 workspace

### PyTorch 内存跟踪

```python
import torch

def memory_stats():
    """Get current GPU memory statistics."""
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
    }

def profile_training_step(model, batch, optimizer, loss_fn):
    """Profile memory during a training step."""
    torch.cuda.reset_peak_memory_stats()

    # Before forward
    print(f"Before forward: {memory_stats()}")

    # Forward
    output = model(batch)
    print(f"After forward: {memory_stats()}")

    loss = loss_fn(output)

    # Backward
    loss.backward()
    print(f"After backward: {memory_stats()}")

    # Optimizer
    optimizer.step()
    optimizer.zero_grad()
    print(f"After optimizer: {memory_stats()}")

    return memory_stats()['max_allocated']
```

### 内存时间线可视化

```python
def create_memory_timeline(model, batch, optimizer, loss_fn):
    """Create a detailed memory timeline."""
    import torch.cuda.memory as mem

    # Start recording
    mem.reset_peak_memory_stats()

    snapshots = []

    def capture(label):
        snapshots.append({
            'label': label,
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
        })

    capture('start')

    # Hook into forward
    hooks = []
    for name, module in model.named_modules():
        def forward_hook(m, inp, out, name=name):
            capture(f'after_{name}')
        hooks.append(module.register_forward_hook(forward_hook))

    output = model(batch)
    capture('forward_complete')

    # Remove hooks
    for h in hooks:
        h.remove()

    loss = loss_fn(output)
    loss.backward()
    capture('backward_complete')

    optimizer.step()
    optimizer.zero_grad()
    capture('optimizer_complete')

    return snapshots
```

## 各组成部分的内存分解

### 经验分析

对于一个 LLaMA-7B 风格的模型，典型分解：

```
Memory Breakdown (% of total):
├── Model State: 75-80%
│   ├── Parameters: 10%
│   ├── Gradients: 10%
│   └── Optimizer States: 55-60%
├── Activations: 10-15%
├── Temporary: 5-10%
└── Fragmentation: 5-10%
```

**关键洞见**：优化器状态（optimizer states）占主导。这就是 ZeRO 优先分片优化器状态的原因。

### 逐层分析

```python
def analyze_layer_memory(model, sample_input):
    """Analyze memory usage per layer."""
    layer_memory = {}

    def make_hook(name):
        def hook(module, input, output):
            # Compute output size
            if isinstance(output, torch.Tensor):
                size = output.numel() * output.element_size()
            elif isinstance(output, tuple):
                size = sum(o.numel() * o.element_size()
                          for o in output if isinstance(o, torch.Tensor))
            else:
                size = 0
            layer_memory[name] = size
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()

    # Sort by memory usage
    sorted_layers = sorted(layer_memory.items(),
                          key=lambda x: x[1], reverse=True)

    return sorted_layers
```

## 训练前的内存估计

### 理论估计

```python
def estimate_memory_requirements(
    num_params: int,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    precision: str = 'bf16',
    optimizer: str = 'adamw'
) -> dict:
    """
    Estimate GPU memory requirements for training.

    Returns:
        Dictionary with memory breakdown in bytes
    """
    # Bytes per element
    param_bytes = 2 if precision in ('fp16', 'bf16') else 4

    # Model state memory
    params_mem = num_params * param_bytes
    grads_mem = num_params * param_bytes

    if optimizer == 'adamw':
        # fp32 master weights + first moment + second moment
        optimizer_mem = num_params * (4 + 4 + 4)
    elif optimizer == 'sgd_momentum':
        optimizer_mem = num_params * 4  # momentum buffer
    else:
        optimizer_mem = 0

    model_state_mem = params_mem + grads_mem + optimizer_mem

    # Activation memory per layer
    head_dim = hidden_dim // num_heads

    # Linear activations: 11 * BSH (approximate)
    linear_act = 11 * batch_size * seq_length * hidden_dim * param_bytes

    # Attention scores: B * A * S * S
    attention_scores = (batch_size * num_heads * seq_length * seq_length
                       * param_bytes)

    activation_mem = num_layers * (linear_act + attention_scores)

    # Temporary memory (rough estimate: 10% of activations)
    temp_mem = int(activation_mem * 0.1)

    # Fragmentation (rough estimate: 10% of total)
    subtotal = model_state_mem + activation_mem + temp_mem
    fragmentation = int(subtotal * 0.1)

    total = subtotal + fragmentation

    return {
        'parameters': params_mem,
        'gradients': grads_mem,
        'optimizer_states': optimizer_mem,
        'model_state_total': model_state_mem,
        'activations': activation_mem,
        'temporary': temp_mem,
        'fragmentation': fragmentation,
        'total': total,
        'total_gb': total / (1024**3)
    }

# Example usage
memory = estimate_memory_requirements(
    num_params=7_000_000_000,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    batch_size=1,
    seq_length=2048
)

print(f"Estimated memory: {memory['total_gb']:.1f} GB")
print(f"  Model state: {memory['model_state_total']/1e9:.1f} GB")
print(f"  Activations: {memory['activations']/1e9:.1f} GB")
```

### 与实际用量对照验证

始终用实际剖析来验证估计：

```python
def validate_memory_estimate(model, batch, estimate):
    """Compare estimated vs actual memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure actual
    optimizer = torch.optim.AdamW(model.parameters())

    output = model(batch)
    loss = output.mean()
    loss.backward()
    optimizer.step()

    actual = torch.cuda.max_memory_allocated()
    estimated = estimate['total']

    error = abs(actual - estimated) / actual * 100

    print(f"Estimated: {estimated/1e9:.2f} GB")
    print(f"Actual: {actual/1e9:.2f} GB")
    print(f"Error: {error:.1f}%")

    return actual, estimated, error
```

## 内存效率比

把内存效率定义为：

$$\eta_{\text{mem}} = \frac{M_{\text{params}}}{M_{\text{total}}} = \frac{\Psi \cdot b_p}{M_{\text{total}}}$$

对于使用 AdamW 的标准训练：

$$\eta_{\text{mem}} = \frac{2\Psi}{16\Psi + M_{\text{act}}} \approx \frac{2}{16} = 12.5\%$$

只有 12.5% 的内存存放真正的模型参数！

**内存优化的目标**：通过减小分母来提高 $\eta_{\text{mem}}$。

| 技术 | $\eta_{\text{mem}}$ |
|-----------|---------------------|
| 标准 AdamW | 12.5% |
| ZeRO-1 | ~15% |
| ZeRO-2 | ~20% |
| ZeRO-3 | ~40% |
| ZeRO-3 + CPU 卸载 | ~60% |

## 练习题

1. **内存计算**：一个 13B 参数模型使用 bf16 训练和 AdamW。计算模型状态内存。如果 GPU 有 80GB，还剩多少给激活？

??? success "解答"
    **使用 AdamW 的模型状态内存（bf16 训练）：**

    | 组成部分 | 精度 | 字节/参数 | 合计 |
    |-----------|-----------|-------------|-------|
    | 参数 | bf16 | 2 | $2 \times 13\text{B} = 26\text{ GB}$ |
    | 梯度 | bf16 | 2 | $2 \times 13\text{B} = 26\text{ GB}$ |
    | 主权重 | fp32 | 4 | $4 \times 13\text{B} = 52\text{ GB}$ |
    | 动量 | fp32 | 4 | $4 \times 13\text{B} = 52\text{ GB}$ |
    | 方差 | fp32 | 4 | $4 \times 13\text{B} = 52\text{ GB}$ |

    **模型状态合计：**
    $$M_{\text{state}} = (2 + 2 + 4 + 4 + 4) \times 13 \times 10^9 = 16 \times 13\text{B} = \boxed{208\text{ GB}}$$

    **留给激活的内存：**
    $$M_{\text{act}} = 80\text{ GB} - 208\text{ GB} = \boxed{-128\text{ GB}}$$

    **分析：** 仅模型状态就已超出 GPU 内存！这个 13B 模型不借助内存优化技术，无法在单块 80GB GPU 上训练：

    | 技术 | 内存削减 | 优化后 |
    |-----------|------------------|--------------------|
    | ZeRO-1（8 GPU） | 优化器：156 GB → 19.5 GB/GPU | 71.5 GB/GPU |
    | ZeRO-2（8 GPU） | + 梯度：26 GB → 3.25 GB/GPU | 48.25 GB/GPU |
    | ZeRO-3（8 GPU） | + 参数：26 GB → 3.25 GB/GPU | 26 GB/GPU |

    在 8 块 GPU 上使用 ZeRO-3：模型状态 26 GB，留出 **54 GB 给激活**。

2. **激活占主导**：对于一个 H=4096、A=32 的模型，在什么序列长度下注意力激活会超过线性激活？

??? success "解答"
    **每层线性（非注意力）激活：**

    由正文推导，线性项约为：

    $$M_{\text{linear}} \approx 11BSH \text{ elements} \;\; \Rightarrow \;\; 22BSH \text{ bytes (bf16)}$$

    **每层注意力激活：**

    - QKV 投影：$3 \times BSH \times 2 = 6BSH$
    - 注意力分数：$BAS^2 \times 2 = 2BAS^2$（每头存储 $S \times S$）
    - 注意力输出：$BSH \times 2 = 2BSH$

    $$M_{\text{attn}} = 8BSH + 2BAS^2 \text{ bytes}$$

    **交叉条件：**

    当以下条件成立时注意力超过线性：

    $$2BAS^2 > 22BSH$$

    $$AS^2 > 11SH$$

    $$S > \frac{11H}{A}$$

    代入 $H = 4096$、$A = 32$：

    $$S > \frac{11 \times 4096}{32} = \frac{45056}{32} = \boxed{1408}$$

    **解释：**

    | 序列长度 | 主导分量 | 比值（注意力/线性） |
    |-----------------|-------------------|---------------------|
    | 512 | 线性 | 0.36× |
    | 1024 | 线性 | 0.73× |
    | 2048 | 注意力 | 1.45× |
    | 4096 | 注意力 | 2.91× |
    | 8192 | 注意力 | 5.82× |

    对于 $S = 8192+$ 的现代长上下文模型，注意力激活严重占主导。

3. **批大小上限**：给定 40GB GPU、7B 参数模型、2048 序列长度，最大批大小是多少？

??? success "解答"
    **模型状态内存（7B，bf16 + AdamW）：**
    $$M_{\text{state}} = 16 \times 7 \times 10^9 = 112\text{ GB}$$

    **问题：** 模型状态超出 GPU 内存！我们需要优化。

    **假设使用 8 块 GPU 的 ZeRO-3：**
    $$M_{\text{state/GPU}} = \frac{112}{8} = 14\text{ GB}$$

    **可用于激活：**
    $$M_{\text{avail}} = 40 - 14 - 2 \text{ (overhead)} = 24\text{ GB}$$

    **每层激活内存（bf16）：**

    使用正文估计：$M_{\text{act}}^{\text{layer}} \approx (11BSH + BAS^2) \times 2$ 字节。

    对 7B：$L \approx 32$、$H \approx 4096$、$A \approx 32$、$S=2048$：

    $$M_{\text{act}}^{\text{layer}} \approx B \times 0.45 \text{ GB}$$

    每 4 层做一次检查点，近似峰值：

    $$M_{\text{act}} \approx \left(\frac{L}{4} + 4\right) \times 0.45B \approx 5.4B \text{ GB}$$

    **求解批大小：**

    $$5.4B \leq 24 \quad \Rightarrow \quad B \leq 4.4$$

    **最大批大小：** $\boxed{B = 4}$（精细调优可到 5）

    **实践验证：**
    - 每步 token 数：$4 \times 2048 = 8{,}192$ 个 token
    - 这对 40GB GPU 上的 7B 模型是典型配置

4. **内存剖析**：编写代码找出 transformer 中哪一层消耗的激活内存最多。

??? success "解答"
    ```python
    import torch
    import torch.nn as nn
    from contextlib import contextmanager
    from typing import Dict, List, Tuple

    class MemoryProfiler:
        """Profiles activation memory per layer in a transformer."""

        def __init__(self):
            self.layer_memory: Dict[str, int] = {}
            self.hooks = []

        def _get_tensor_memory(self, tensor: torch.Tensor) -> int:
            """Get memory in bytes for a tensor."""
            return tensor.numel() * tensor.element_size()

        def _forward_hook(self, name: str):
            def hook(module, input, output):
                mem = 0
                if isinstance(output, torch.Tensor):
                    mem = self._get_tensor_memory(output)
                elif isinstance(output, tuple):
                    for o in output:
                        if isinstance(o, torch.Tensor):
                            mem += self._get_tensor_memory(o)
                self.layer_memory[name] = mem
            return hook

        def register_hooks(self, model: nn.Module, prefix: str = ""):
            """Register hooks on all layers."""
            for name, module in model.named_modules():
                full_name = f"{prefix}.{name}" if prefix else name
                if len(list(module.children())) == 0:  # Leaf module
                    hook = module.register_forward_hook(
                        self._forward_hook(full_name)
                    )
                    self.hooks.append(hook)

        def remove_hooks(self):
            """Remove all registered hooks."""
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

        def profile(self, model: nn.Module, input_tensor: torch.Tensor
                   ) -> List[Tuple[str, int]]:
            """Run forward pass and return sorted memory usage."""
            self.layer_memory.clear()
            self.register_hooks(model)

            with torch.no_grad():
                model(input_tensor)

            self.remove_hooks()

            # Sort by memory usage (descending)
            sorted_layers = sorted(
                self.layer_memory.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_layers

        def report(self, top_k: int = 10) -> str:
            """Generate a formatted report."""
            sorted_layers = sorted(
                self.layer_memory.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            total = sum(self.layer_memory.values())
            report = "Top Activation Memory Consumers:\n"
            report += "-" * 60 + "\n"
            report += f"{'Layer':<40} {'Memory':>10} {'%':>6}\n"
            report += "-" * 60 + "\n"

            for name, mem in sorted_layers:
                pct = 100 * mem / total if total > 0 else 0
                report += f"{name:<40} {mem/1e6:>8.2f}MB {pct:>5.1f}%\n"

            report += "-" * 60 + "\n"
            report += f"{'Total':<40} {total/1e6:>8.2f}MB\n"
            return report

    # Usage example
    def profile_transformer():
        from transformers import AutoModel

        model = AutoModel.from_pretrained("gpt2")
        model.eval()

        # Create dummy input
        batch_size, seq_len = 4, 1024
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))

        profiler = MemoryProfiler()
        results = profiler.profile(model, input_ids)

        print(profiler.report(top_k=15))

        # Identify layer type with highest memory
        attention_mem = sum(m for n, m in results if 'attn' in n.lower())
        ffn_mem = sum(m for n, m in results if 'mlp' in n.lower() or 'fc' in n.lower())

        print(f"\nAttention total: {attention_mem/1e6:.2f} MB")
        print(f"FFN total: {ffn_mem/1e6:.2f} MB")
    ```

    **预期输出结构：**

    | 层类型 | 典型内存占比 | 原因 |
    |------------|------------------|-----|
    | 注意力 QKV 投影 | 15–20% | $3 \times BSH$ 张量 |
    | 注意力分数 | 25–40% | $BAS^2$（二次！） |
    | FFN 中间层 | 20–30% | $4 \times$ 扩展 |
    | LayerNorm | 5–10% | 每个 norm $BSH$ |

    **关键洞见：** 对于长序列，注意力分数矩阵（query-key 乘积）占主导。

5. **缩放分析**：对 S ∈ [512, 16384] 画出预期内存使用随序列长度的变化。找出二次项占主导的交叉点。

??? success "解答"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def activation_memory(S, B=4, H=4096, A=32, L=32):
        """
        Calculate activation memory in GB.

        Components:
        - Linear: O(BSH) - embeddings, FFN, LayerNorm
        - Quadratic: O(BAS²) - attention scores
        """
        # Linear components (bytes)
        linear = L * B * S * H * 34 * 2  # bf16

        # Quadratic component (attention scores)
        quadratic = L * B * A * S * S * 2  # bf16

        return linear / 1e9, quadratic / 1e9

    # Sequence lengths to analyze
    seq_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384])

    linear_mem = []
    quad_mem = []

    for S in seq_lengths:
        lin, quad = activation_memory(S)
        linear_mem.append(lin)
        quad_mem.append(quad)

    linear_mem = np.array(linear_mem)
    quad_mem = np.array(quad_mem)
    total_mem = linear_mem + quad_mem

    # Find crossover point (where quadratic = linear)
    # BAS² = 34BSH → AS = 34H → S = 34H/A
    H, A = 4096, 32
    crossover = 34 * H / A
    print(f"Theoretical crossover: S = {crossover:.0f}")

    # Create results table
    print("\nMemory Scaling Analysis (B=4, H=4096, A=32, L=32):")
    print("-" * 70)
    print(f"{'Seq Len':>8} {'Linear (GB)':>12} {'Quadratic (GB)':>14} "
          f"{'Total (GB)':>12} {'Quad %':>8}")
    print("-" * 70)
    for i, S in enumerate(seq_lengths):
        pct = 100 * quad_mem[i] / total_mem[i]
        print(f"{S:>8} {linear_mem[i]:>12.2f} {quad_mem[i]:>14.2f} "
              f"{total_mem[i]:>12.2f} {pct:>7.1f}%")
    print("-" * 70)
    ```

    **结果：**

    | 序列长度 | 线性（GB） | 二次（GB） | 合计（GB） | 二次占比 |
    |---------|-------------|----------------|------------|--------|
    | 512 | 4.56 | 0.54 | 5.10 | 10.5% |
    | 1024 | 9.13 | 2.15 | 11.28 | 19.0% |
    | 2048 | 18.25 | 8.59 | 26.84 | 32.0% |
    | **4096** | **36.51** | **34.36** | **70.87** | **48.5%** |
    | 8192 | 73.01 | 137.44 | 210.45 | 65.3% |
    | 16384 | 146.03 | 549.76 | 695.79 | 79.0% |

    **交叉点：** $S = \frac{34H}{A} = \frac{34 \times 4096}{32} = \boxed{4352}$

    在 $S \approx 4096$ 处，二次项与线性项大致相等。超过这一点后，注意力内存呈指数级占主导。

6. **效率计算**：一次训练运行对 7B 模型使用 120GB。计算 $\eta_{\text{mem}}$。

??? success "解答"
    **内存效率定义：**
    $$\eta_{\text{mem}} = \frac{M_{\text{params}}}{M_{\text{total}}}$$

    **参数内存（7B，bf16）：**
    $$M_{\text{params}} = 2 \times 7 \times 10^9 = 14\text{ GB}$$

    **使用的总内存：**
    $$M_{\text{total}} = 120\text{ GB}$$

    **内存效率：**
    $$\eta_{\text{mem}} = \frac{14}{120} = \boxed{11.7\%}$$

    **内存分解（典型）：**

    | 组成部分 | 估计大小 | 占总计 |
    |-----------|----------------|------------|
    | 参数（bf16） | 14 GB | 11.7% |
    | 梯度（bf16） | 14 GB | 11.7% |
    | 主权重（fp32） | 28 GB | 23.3% |
    | 优化器状态（fp32） | 56 GB | 46.7% |
    | **模型状态小计** | **112 GB** | **93.3%** |
    | 激活 | ~6 GB | 5.0% |
    | 开销/碎片 | ~2 GB | 1.7% |
    | **合计** | **120 GB** | **100%** |

    **分析：** 只有 11.7% 的内存存放真正的模型权重。这种低效率对未做内存优化的 Adam 训练是典型的。

    **提高 $\eta_{\text{mem}}$ 的方法：**

    | 技术 | 新效率 |
    |-----------|----------------|
    | ZeRO-3（8 GPU） | ~14/(120/8) = 93% |
    | SGD（无动量） | 14/(14+14+14) = 33% |
    | 激活检查点 | 减少激活内存 |

7. **优化器对比**：比较 70B 模型分别用 Adam、带动量的 SGD、不带动量的 SGD 训练时的内存需求。

??? success "解答"
    **每种优化器的内存公式：**

    | 组成部分 | Adam | SGD+动量 | SGD |
    |-----------|------|--------------|-----|
    | 参数（bf16） | $2\Psi$ | $2\Psi$ | $2\Psi$ |
    | 梯度（bf16） | $2\Psi$ | $2\Psi$ | $2\Psi$ |
    | 主权重（fp32） | $4\Psi$ | $4\Psi$ | $4\Psi$ |
    | 动量（fp32） | $4\Psi$ | $4\Psi$ | 0 |
    | 方差（fp32） | $4\Psi$ | 0 | 0 |
    | **每参数字节数** | **16** | **12** | **8** |

    **对 70B 模型（$\Psi = 70 \times 10^9$）：**

    | 优化器 | 字节/参数 | 总内存 |
    |-----------|-------------|--------------|
    | **Adam** | 16 | $16 \times 70\text{B} = \boxed{1120\text{ GB}}$ |
    | **SGD+动量** | 12 | $12 \times 70\text{B} = \boxed{840\text{ GB}}$ |
    | **SGD** | 8 | $8 \times 70\text{B} = \boxed{560\text{ GB}}$ |

    **最少所需 GPU 数（80GB H100）：**

    | 优化器 | 最少 GPU（无 ZeRO） | 有 ZeRO-3 |
    |-----------|-------------------|-------------|
    | Adam | $\lceil 1120/80 \rceil = 14$ | 2 GPU |
    | SGD+动量 | $\lceil 840/80 \rceil = 11$ | 2 GPU |
    | SGD | $\lceil 560/80 \rceil = 7$ | 1 GPU（紧张） |

    **权衡：**

    | 优化器 | 内存 | 收敛性 | 使用场景 |
    |-----------|--------|-------------|----------|
    | Adam | 最高 | 最好 | 标准 LLM 训练 |
    | SGD+动量 | 中等 | 良好 | 微调、内存受限 |
    | SGD | 最低 | 慢/不稳定 | 很少用于 LLM |

    **注：** 8-bit Adam 或 CAME 等较新的优化器通过量化优化器状态来降低内存。

8. **碎片分析**：内存报告显示分配了 100GB，但实际只用了 85GB。碎片率是多少？可能由什么造成？

??? success "解答"
    **碎片率：**
    $$\text{Fragmentation} = \frac{M_{\text{allocated}} - M_{\text{used}}}{M_{\text{allocated}}}$$

    $$= \frac{100 - 85}{100} = \boxed{15\%}$$

    **替代度量（开销比）：**
    $$\text{Overhead} = \frac{M_{\text{allocated}}}{M_{\text{used}}} - 1 = \frac{100}{85} - 1 = 17.6\%$$

    **GPU 内存碎片的常见成因：**

    | 成因 | 描述 | 解决方案 |
    |-------|-------------|----------|
    | **分配模式** | 变长分配造成空洞 | 使用内存池 |
    | **缓存分配器** | PyTorch 缓存已释放内存 | `torch.cuda.empty_cache()` |
    | **峰值内存** | 临时张量留下的高水位 | 梯度检查点 |
    | **张量形状未对齐** | 非 2 的幂形状浪费内存 | 填充到对齐尺寸 |
    | **动态形状** | 变化的批/序列造成碎片 | 固定形状 |
    | **锁页内存开销** | 异步传输缓冲区 | 减少并发 |

    **调试碎片：**

    ```python
    import torch

    def diagnose_fragmentation():
        # Current allocation state
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Max alloc: {max_allocated:.2f} GB")
        print(f"Internal frag: {(reserved - allocated)/reserved*100:.1f}%")
        print(f"Peak overhead: {(max_allocated - allocated)/allocated*100:.1f}%")

        # Memory snapshot for detailed analysis
        if hasattr(torch.cuda, 'memory_snapshot'):
            snapshot = torch.cuda.memory_snapshot()
            # Analyze blocks, gaps, etc.

    # Mitigation strategies
    def reduce_fragmentation():
        # 1. Pre-allocate with consistent shapes
        torch.cuda.set_per_process_memory_fraction(0.95)

        # 2. Use memory-efficient attention
        # (Flash Attention allocates less temporary memory)

        # 3. Clear cache periodically
        torch.cuda.empty_cache()

        # 4. Use memory pools
        # CUDA memory pools reduce fragmentation
        torch.cuda.memory.set_allocator_settings("expandable_segments:True")
    ```

    **碎片严重程度指南：**

    | 碎片率 | 严重程度 | 行动 |
    |---------------|----------|--------|
    | < 5% | 正常 | 无需处理 |
    | 5–15% | 中等 | 监控，考虑 empty_cache |
    | 15–25% | 高 | 优化分配模式 |
    | > 25% | 严重 | 需要大规模重构 |

    观察到的 15% 落在"高"区间——值得调查分配模式。

## 关键要点

1. **16× 法则**：使用 AdamW 训练大约需要每参数 16 字节的模型状态。

2. **优化器状态占主导**：模型状态内存的 75% 以上是优化器状态，而非参数。

3. **激活随批和序列缩放**：$M_{\text{act}} \propto B \cdot S \cdot (1 + S/128)$。

4. **序列长度是二次的**：由于注意力，长序列很费内存。

5. **峰值出现在反向**：最大内存用量通常出现在反向传播期间。

6. **内存效率很低**：标准训练中只有约 12% 的内存存放真正的模型权重。

7. **要剖析，不要猜**：理论估计有用，但始终要用实际剖析验证。

8. **内存优化是必需的**：没有优化技术，任何单块 GPU 都无法训练大模型。