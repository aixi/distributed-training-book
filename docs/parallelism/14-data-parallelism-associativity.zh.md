---
title: "从结合律推导数据并行"
subtitle: "为什么梯度累积使分布式训练成为可能"
---

<div class="chapter-opener" markdown>
数据并行（Data Parallelism）之所以成立，是因为梯度累积（Gradient Accumulation）满足结合律。这不是实现上的细节——它是让整套方法得以成立的数学基础。理解这一基础，既能揭示数据并行的威力，也能揭示它的局限。
</div>

<div class="investigation-question" markdown>
**问题**：我们在不同的 GPU 上用不同的批次计算梯度，然后把它们加起来。为什么这样得到的结果与在完整批次上计算相同？它什么时候会失效？
</div>

<div class="notation-banner" markdown>
**本章符号**：$\Psi$ = 参数量，$B$ = 全局批大小，$b$ = 每 GPU 批大小，$P$ = GPU 数量。见[符号表](../appendices/notation.md)。
</div>

!!! abstract "承上启下：第三部分 集合通信"
    本部分假定你已掌握**集合通信（collective communication）操作**——尤其是 [AllReduce（第 11 章）](../collectives/11-primitives-properties.md)、[环形与树形算法（第 12 章）](../collectives/12-ring-tree-algorithms.md)以及[成本建模（第 13 章）](../collectives/13-collective-cost-model.md)。我们在此推导的每种并行策略都依赖特定的集合通信操作。数据并行——我们的第一种策略——使用 AllReduce 来同步梯度。

## 数学基础

### 损失函数的结构

考虑数据集 $\mathcal{D}$ 上的损失函数：

$$L(\theta) = \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \ell(x, \theta)$$

其中 $\ell(x, \theta)$ 是单样本损失。

在小批量 SGD 中，我们用批次 $B$ 来近似它：

$$L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \ell(x, \theta)$$

其梯度为：

$$\nabla_\theta L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \nabla_\theta \ell(x, \theta)$$

### 划分定理

!!! question "直觉检查"
    我们在不同 GPU 上对不同的数据子集计算梯度，然后取平均。这会得到与在完整批次上计算完全相同的梯度吗？是什么数学性质使之成立？

**定理**：若批次 $B$ 被划分为 $P$ 个互不相交的子集 $B = B_1 \cup B_2 \cup \cdots \cup B_P$，且对所有 $i$ 有 $|B_i| = |B|/P$：

$$\nabla_\theta L_B(\theta) = \frac{1}{P} \sum_{i=1}^{P} \nabla_\theta L_{B_i}(\theta)$$

**证明**：

从定义出发：

$$\nabla_\theta L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \nabla_\theta \ell(x, \theta)$$

因为 $B = \bigcup_{i=1}^{P} B_i$ 且各 $B_i$ 互不相交：

$$= \frac{1}{|B|} \sum_{i=1}^{P} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

又因 $|B_i| = |B|/P$：

$$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \frac{1}{|B_i|} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

$$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \nabla_\theta L_{B_i}(\theta)$$

$$= \frac{1}{|B|} \sum_{i=1}^{P} \frac{|B|}{P} \cdot \nabla_\theta L_{B_i}(\theta)$$

$$= \frac{1}{P} \sum_{i=1}^{P} \nabla_\theta L_{B_i}(\theta) \quad \square$$

这一划分之所以成立，是因为：

1. **加法满足结合律**：$(a + b) + c = a + (b + c)$
2. **加法满足交换律**：$a + b = b + a$
3. **梯度是向量空间中的元素**：它们从实数继承了这些性质

### 关键洞见

和的梯度等于梯度的和：

$$\nabla_\theta \left( \sum_i f_i(\theta) \right) = \sum_i \nabla_\theta f_i(\theta)$$

**梯度算子的线性性**与**加法的结合律**相结合，正是数据并行在数学上成立的原因。

## 同步数据并行算法

### 基本算法

```
Input: Model θ, Dataset D, P GPUs, learning rate η
Output: Trained model θ

1. Broadcast θ to all P GPUs
2. For each training step:
   a. Sample batch B of size |B| = P × b
   b. Partition: Bi = samples [(i-1)b : ib] for GPU i
   c. On each GPU i in parallel:

      - Forward: yi = f(Bi; θ)
      - Loss: Li = L(yi, targets)
      - Backward: gi = ∇θLi
   d. AllReduce: g = (1/P) Σi gi
   e. Update: θ ← θ - η·g
   f. (Implicit) All GPUs now have identical θ
3. Return θ
```

### 正确性不变式

**不变式**：在每一步开始时，所有 GPU 持有相同的模型参数。

**归纳法证明**：

*基础情形*：第 0 步广播 θ，因此所有 GPU 起始状态相同。

*归纳步骤*：假设在第 $t$ 步所有 GPU 的 θ 相同。经 AllReduce 后，所有 GPU 得到相同的梯度 $g$。施加相同学习率的相同更新规则：

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot g$$

会在所有 GPU 上产生相同的 $\theta^{(t+1)}$。$\square$

正是这一不变式让我们无需每一步都广播参数——相同的更新自动维持了同步。

### 伪代码实现

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_data_parallel(model, dataloader, optimizer, epochs):
    """Synchronous data parallel training."""

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Wrap model in DDP (handles gradient synchronization)
    model = DDP(model, device_ids=[rank])

    for epoch in range(epochs):
        # Distributed sampler ensures non-overlapping batches
        dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            # Forward pass (identical on each GPU with different data)
            loss = model(batch)

            # Backward pass (computes local gradients)
            loss.backward()

            # DDP automatically performs AllReduce here
            # Gradients are averaged across all ranks

            # Update (identical update on all GPUs)
            optimizer.step()
            optimizer.zero_grad()
```

## 通信分析

### 单步通信量

设 $\Psi$ 为模型参数数量（按元素计，而非字节）。

**AllReduce 通信量**：每个参数的梯度都必须同步。

使用环形 AllReduce：

$$V = 2 \cdot \frac{P-1}{P} \cdot \Psi \cdot \text{sizeof}(\text{dtype})$$

对 32 位浮点数：

$$V = 8 \cdot \frac{P-1}{P} \cdot \Psi \text{ bytes}$$

当 $P \to \infty$：

$$V \to 8\Psi \text{ bytes}$$

### 通信时间

使用 α-β 模型和环形 AllReduce：

$$T_{\text{comm}} = 2(P-1) \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{\Psi \cdot \text{sizeof}(\text{dtype})}{\beta}$$

对大 $P$、大 $\Psi$（受带宽支配时）：

$$T_{\text{comm}} \approx \frac{2\Psi \cdot \text{sizeof}(\text{dtype})}{\beta}$$

### 计算时间

在每块 GPU 上，本地批大小为 $b$：

$$T_{\text{compute}} = \frac{6\Psi \cdot b}{F}$$

其中 $F$ 是该 GPU 的 FLOP/s。

### 计算-通信比

关键的比值：

$$R = \frac{T_{\text{compute}}}{T_{\text{comm}}} = \frac{6\Psi \cdot b / F}{2\Psi \cdot \text{sizeof}(\text{dtype}) / \beta}$$

化简得：

$$R = \frac{3b \cdot \beta}{F \cdot \text{sizeof}(\text{dtype})}$$

对 FP16 训练：

$$R = \frac{3b \cdot \beta}{2F}$$

!!! note "实践"
    如果扩展停滞，就用你实际的 $b$、$F$ 和网络 $\beta$ 算出 $R$。当 $R < 1$ 时，你受限于通信——增大本地批、累积梯度，或改用层次化 AllReduce。

**示例**：H100 的 $F = 989 \times 10^{12}$ FLOP/s（密集 BF16，见[硬件假设](../appendices/hardware-assumptions.md)），NVLink $\beta = 900$ GB/s：

$$R = \frac{3b \cdot 900 \times 10^9}{2 \times 989 \times 10^{12}} = \frac{2700b}{1.978 \times 10^6} \approx \frac{1365b}{10^6}$$

要达到 $R > 1$（受计算支配），需要：

$$b > \frac{10^6}{1365} \approx 733 \text{ samples}$$

### 扩展效率

定义扩展效率（scaling efficiency）：

$$E(P) = \frac{T_1}{P \cdot T_P}$$

其中 $T_1$ 是同样的*全局*批次在单 GPU 上的时间，$T_P$ 是 $P$ 块 GPU 每步的时间（强扩展）。

完全重叠时：

$$T_P = \max\left(\frac{T_{\text{compute}}}{P},\; T_{\text{comm}}\right)$$

**受计算支配区间**（$T_{\text{compute}} > T_{\text{comm}}$）：

$$E(P) = \frac{T_{\text{compute}}}{P \cdot (T_{\text{compute}}/P)} = 1$$

完美扩展！但这假设了：
1. 通信与计算完全重叠
2. 无额外开销

**受通信支配区间**（$T_{\text{comm}} > T_{\text{compute}}$）：

$$E(P) = \frac{T_{\text{compute}}}{P \cdot T_{\text{comm}}} = \frac{R}{P}$$

效率随 $P$ 增大而下降。

## 重叠通信与计算

### 重叠的机会

反向传播从输出到输入逐层计算梯度。一旦某层的梯度算完，就可以在计算更早层梯度的同时把它发出去。

```
Time →
Layer 4: [backward][  AllReduce  ]
Layer 3:          [backward][  AllReduce  ]
Layer 2:                   [backward][  AllReduce  ]
Layer 1:                            [backward][  AllReduce  ]
                                             ↑
                                    Overlap region
```

### 基于桶的重叠

PyTorch DDP 把梯度归入多个桶（bucket）：

```python
# DDP bucket configuration
model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=25,  # Bucket size in MB
)
```

**分桶算法**：
1. 按反向计算顺序注册梯度
2. 当桶达到容量时，触发 AllReduce
3. AllReduce 进行时继续计算

**最优桶大小**：

- 太小：大量小的 AllReduce（延迟开销）
- 太大：重叠机会减少

经验上：10-50 MB 的桶效果良好。

### 梯度就绪顺序

DDP 用钩子（hook）检测梯度何时就绪：

```python
class DDPGradientHook:
    def __init__(self, bucket_manager):
        self.bucket_manager = bucket_manager

    def __call__(self, grad):
        # Called when gradient is computed
        self.bucket_manager.add_gradient(grad)
        if self.bucket_manager.bucket_ready():
            # Launch async AllReduce
            self.bucket_manager.flush_bucket()
        return grad
```

第 $L$ 层的通信与第 $1, ..., L-1$ 层的反向计算重叠。

### 理论重叠效率

设 $f_i$ 为第 $i$ 层反向传播计算时间所占的比例。

可达到的重叠：

$$\text{Overlap} = \sum_{i=1}^{L-1} f_i \cdot \min\left(1, \frac{T_{\text{comm},i}}{T_{\text{compute},<i}}\right)$$

其中 $T_{\text{compute},<i}$ 是 $i$ 之前各层的计算时间。

实践中可达到 70%~90% 的重叠。

## 结合律何时失效

### 浮点数的非结合性

IEEE 754 浮点加法**不满足结合律**：

$$(a + b) + c \neq a + (b + c)$$

**示例**：
```python
import numpy as np

a = np.float32(1e20)
b = np.float32(-1e20)
c = np.float32(1.0)

print((a + b) + c)  # = 1.0
print(a + (b + c))  # = 0.0  (c absorbed into b)
```

### 归约顺序会影响结果

不同的 AllReduce 实现使用不同的归约顺序：

**环形 AllReduce**：

$$g = (\cdots((g_0 + g_1) + g_2) + \cdots + g_{P-1})$$

**树形 AllReduce**（二叉树）：

$$g = ((g_0 + g_1) + (g_2 + g_3)) + ((g_4 + g_5) + (g_6 + g_7))$$

对相同输入，两者会给出略有不同的结果！

### 非确定性的来源

| 来源 | 原因 | 缓解措施 |
|--------|-------|------------|
| AllReduce 顺序 | 环形方向可变 | 固定环形顺序 |
| 树形归约 | 分组方式不同 | 使用一致的树 |
| 异步完成 | 竞态条件 | 同步归约 |
| 融合算子 | 算法不同 | 禁用融合 |
| 硬件差异 | GPU 不同 | 同构集群 |

### 实现确定性

```python
import torch

def enable_deterministic_training():
    """Enable deterministic training for reproducibility."""

    # Set seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Disable auto-tuning (chooses different algorithms)
    torch.backends.cudnn.benchmark = False

    # Set deterministic NCCL reduction order (still not a strict bitwise guarantee)
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
```

即便有这些设置，NCCL 仍可能改变通道选择或拓扑；严格的逐位确定性通常需要固定拓扑并限制通道数。

**确定性的代价**：由于禁用优化，通常慢 10%~20%。

### 更高精度的累积

对梯度累积，使用更高精度：

```python
class HighPrecisionAccumulator:
    """Accumulate gradients in FP32 even if model is FP16."""

    def __init__(self, model):
        self.fp32_grads = {
            name: torch.zeros_like(p, dtype=torch.float32)
            for name, p in model.named_parameters()
        }

    def accumulate(self, model):
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.fp32_grads[name] += p.grad.float()

    def get_and_reset(self, model):
        for name, p in model.named_parameters():
            p.grad = self.fp32_grads[name].to(p.dtype)
            self.fp32_grads[name].zero_()
```

## 梯度压缩（预告）

梯度维度很高，但往往可以压缩。我们可以用计算换通信带宽。主要方法包括：

- **稀疏化**（Top-K、随机）：只发送最大的梯度元素
- **量化**（1-bit SGD、QSGD）：降低所传输梯度的精度
- **低秩**（PowerSGD）：用低秩因子近似梯度矩阵

这些技术可以把通信压缩 10-100 倍，在配合误差反馈（error feedback）时仍能保持收敛。

!!! info "详见第七部分"
    [第 28 章（梯度压缩）](../efficiency/28-gradient-compression.md)将深入讲解这些方法，包括收敛性证明、无偏性分析、误差反馈理论和实践实现指导。

## 梯度累积

### 当内存限制了批大小

如果本地批 $b$ 放不进内存，就在多个微批次上累积：

```python
def train_with_accumulation(model, batch, accumulation_steps, optimizer):
    """Gradient accumulation for effective large batches."""

    optimizer.zero_grad()

    # Split batch into micro-batches
    micro_batches = batch.chunk(accumulation_steps)

    for micro_batch in micro_batches:
        # Forward and backward (gradients accumulate)
        loss = model(micro_batch) / accumulation_steps
        loss.backward()

    # Now perform AllReduce and update
    # (DDP would AllReduce here automatically)
    optimizer.step()
```

### 数学等价性

对 $A$ 个大小 $b/A$ 的微批次做累积：

$$g_{\text{accum}} = \frac{1}{A} \sum_{j=1}^{A} \nabla L_{B_j}$$

由划分定理，它等于 $\nabla L_B$。

### 通信模式

无累积：每个微批次都做 AllReduce
有累积：每 $A$ 个微批次做一次 AllReduce

**通信减少带来的加速**：

$$\frac{T_{\text{no-accum}}}{T_{\text{accum}}} = \frac{A \cdot T_{\text{compute/A}} + A \cdot T_{\text{comm}}}{A \cdot T_{\text{compute/A}} + T_{\text{comm}}} = \frac{A \cdot T_c + A \cdot T_r}{A \cdot T_c + T_r}$$

其中 $T_c$ 是每个微批次的计算时间，$T_r$ 是一次梯度同步的 AllReduce 时间。

当 $T_r = T_c$ 时：加速比 = $(2A)/(A+1)$ → 当 $A \to \infty$ 时趋近 2 倍。

## 扩展行为

### 弱扩展

保持本地批大小 $b$ 不变，增加 GPU：

- 总批大小：$B = P \cdot b$
- 每 GPU 计算量：不变
- 通信：随 $P$ 增加（延迟项）

**效率**：

$$E_{\text{weak}}(P) = \frac{T_1}{T_P} = \frac{T_c}{T_c + T_r(P)}$$

其中 $T_r(P) = 2(P-1)\alpha + 2\frac{P-1}{P} \cdot \Psi/\beta$。

### 强扩展

保持总批大小 $B$ 不变，增加 GPU：

- 本地批大小：$b = B/P$
- 每 GPU 计算量：按 $1/P$ 下降
- 通信：大致不变（受带宽项支配）

**效率**：

$$E_{\text{strong}}(P) = \frac{T_1}{P \cdot T_P} = \frac{T_c(1)}{P \cdot (T_c(1)/P + T_r)}$$

随着 $P$ 增大，$T_c/P$ 相对 $T_r$ 变得很小，效率随之下降。

### 实践中的极限

这些极限只是数量级上的估计，强烈依赖于硬件、模型架构和批大小。

| 模型规模 | 典型 P 上限（90% 效率） |
|------------|----------------------------------|
| 1B 参数 | ~256 块 GPU |
| 10B 参数 | ~128 块 GPU |
| 100B 参数 | ~64 块 GPU |
| 1T 参数 | ~16 块 GPU |

超过这些极限后，单靠数据并行就会受限于通信。

## 异步数据并行

### 同步瓶颈

同步数据并行要等待最慢的 worker（“拖后腿问题”）：

$$T_{\text{step}} = \max_i T_i + T_{\text{AllReduce}}$$

$T_i$ 的方差不一降低了效率。

### 异步 SGD

各 worker 互不等待：

```
Worker 1: [compute][push grad][pull params][compute]...
Worker 2: [compute][push][pull][compute][push]...
Worker 3: [compute][push][pull][compute]...
          ← Staleness →
```

**陈旧度**（staleness）：worker 把更新应用到参数 $\theta^{(t-\tau)}$ 上，其中 $\tau$ 是延迟。

### 陈旧度修正的更新

为了补偿陈旧度：

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot f(\tau) \cdot g^{(t-\tau)}$$

其中 $f(\tau)$ 是陈旧度惩罚：

- $f(\tau) = 1$：忽略陈旧度（常常发散）
- $f(\tau) = 1/\tau$：按倒数缩放
- $f(\tau) = e^{-\lambda\tau}$：指数衰减

### 权衡

| 方面 | 同步 | 异步 |
|--------|-------------|--------------|
| 正确性 | 精确 | 近似 |
| 陈旧度 | 0 | 可变 |
| 拖后腿处理 | 差 | 好 |
| 收敛 | 每步更快 | 需要更多步 |
| 实现 | 简单 | 复杂 |
| 调试 | 容易 | 困难 |

**趋势**：由于收敛保证更简单，同步方式在现代大规模训练中占主导。

## 实现：PyTorch DDP

### 架构

```
DDP Module
    ├── Forward Hook: Sync forward (no-op for most cases)
    ├── Backward Hooks: Register gradient ready callbacks
    ├── Bucket Manager: Group gradients for AllReduce
    ├── Reducer: Execute AllReduce operations
    └── Comm Hook: Customizable communication (compression, etc.)
```

### 自定义通信钩子

```python
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default

def compression_hook(state, bucket):
    """Custom hook for gradient compression."""
    tensor = bucket.buffer()

    # Compress
    compressed, ctx = my_compressor.compress(tensor)

    # AllReduce compressed gradient
    fut = dist.all_reduce(compressed, async_op=True).get_future()

    def decompress(fut):
        synced = fut.value()[0]
        return my_compressor.decompress(synced, ctx)

    return fut.then(decompress)

# Register hook
model = DDP(model)
model.register_comm_hook(state=None, hook=compression_hook)
```

### 梯度同步点

DDP 在以下时机同步：
1. **首次前向**（若 `find_unused_parameters=True`）
2. **反向之后**（主要的梯度同步）

```python
# For models with unused parameters
model = DDP(
    model,
    device_ids=[rank],
    find_unused_parameters=True,  # Extra overhead
)
```

## 练习题

1. **划分证明**：把划分定理推广到大小不均的划分。若 $|B_i| = w_i \cdot |B|$ 且 $\sum_i w_i = 1$，正确的平均公式是什么？

??? success "解答"
    **大小不均的推广划分定理**

    给定批次 $B$ 划分为 $P$ 个互不相交的子集 $B_1, B_2, \ldots, B_P$，其中 $|B_i| = w_i \cdot |B|$ 且 $\sum_{i=1}^P w_i = 1$。

    **命题：**
    $$\nabla_\theta L_B(\theta) = \sum_{i=1}^{P} w_i \cdot \nabla_\theta L_{B_i}(\theta)$$

    **证明：**

    从定义出发：

    $$\nabla_\theta L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \nabla_\theta \ell(x, \theta)$$

    因为 $B = \bigcup_{i=1}^{P} B_i$ 且各 $B_i$ 互不相交：

    $$= \frac{1}{|B|} \sum_{i=1}^{P} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

    乘以再除以 $|B_i|$：

    $$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \frac{1}{|B_i|} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

    $$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \nabla_\theta L_{B_i}(\theta)$$

    代入 $|B_i| = w_i \cdot |B|$：

    $$= \frac{1}{|B|} \sum_{i=1}^{P} w_i \cdot |B| \cdot \nabla_\theta L_{B_i}(\theta)$$

    $$= \boxed{\sum_{i=1}^{P} w_i \cdot \nabla_\theta L_{B_i}(\theta)} \quad \square$$

    **实践含义：**

    当各 GPU 的批大小不等时（例如由于负载均衡），AllReduce 应计算**加权平均**，而非简单平均：

    ```python
    # Weighted gradient averaging
    local_batch_size = len(local_batch)
    total_batch_size = dist.all_reduce(
        torch.tensor(local_batch_size), op=dist.ReduceOp.SUM
    )
    weight = local_batch_size / total_batch_size

    for param in model.parameters():
        param.grad *= weight
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    ```

    **特例：** 当所有 $w_i = 1/P$ 时，我们得到原定理的简单平均形式。

2. **计算-通信平衡**：一个 7B 参数模型运行在 8 块 H100 上，使用 NVLink（900 GB/s），要令 $R = 2$（计算是通信的 2 倍），本地批大小应是多少？

??? success "解答"
    **已知：**

    - $\Psi = 7 \times 10^9$ 参数
    - $P = 8$ 块 GPU
    - $\beta = 900$ GB/s（NVLink 带宽）
    - $F = 989 \times 10^{12}$ FLOP/s（H100 密集 FP16/BF16 峰值）
    - FP16 训练：sizeof(dtype) = 2 字节
    - 目标：$R = 2$

    **计算-通信比公式（来自本章）：**

    $$R = \frac{3b \cdot \beta}{F \cdot \text{sizeof}(\text{dtype})}$$

    **求解批大小 $b$：**

    $$b = \frac{R \cdot F \cdot \text{sizeof}(\text{dtype})}{3 \cdot \beta}$$

    $$b = \frac{2 \times 989 \times 10^{12} \times 2}{3 \times 900 \times 10^9}$$

    $$b = \frac{3.956 \times 10^{15}}{2.7 \times 10^{12}}$$

    $$b = \boxed{1,465 \text{ samples}}$$

    **验证：**

    *通信时间：*
    $$T_{\text{comm}} = 2 \times \frac{P-1}{P} \times \frac{\Psi \times 2}{\beta}$$

    $$= 2 \times \frac{7}{8} \times \frac{7 \times 10^9 \times 2}{900 \times 10^9}$$

    $$= 1.75 \times 15.56 \text{ ms} = 27.2 \text{ ms}$$

    *计算时间：*
    $$T_{\text{compute}} = \frac{6\Psi \times b}{F}$$

    $$= \frac{6 \times 7 \times 10^9 \times 1465}{989 \times 10^{12}}$$

    $$= \frac{6.16 \times 10^{13}}{9.89 \times 10^{14}} = 62.2 \text{ ms}$$

    *比值核对：*
    $$R = \frac{62.2}{27.2} = 2.29 \approx 2 \checkmark$$

    **实践考量：**

    | 序列长度 | 样本数（b=1465） | 每 GPU token 数 |
    |-----------------|------------------|----------------|
    | 512 | 1,465 | 750K |
    | 2048 | 1,465 | 3.0M |
    | 4096 | 1,465 | 6.0M |

    序列长度为 4096 时，这需要每步每 GPU 约 1200 万 token——很可能超出内存。此时就需要梯度累积。

3. **重叠分析**：一个模型有 100 个规模相同的层。如果全部梯度的 AllReduce 耗时 100 毫秒，全部层的反向耗时 80 毫秒，那么通信可重叠的比例是多少？有效步时间是多少？

??? success "解答"
    **已知：**

    - $L = 100$ 个规模相同的层
    - $T_{\text{AR}}^{\text{total}} = 100$ 毫秒（AllReduce 总时间）
    - $T_{\text{bwd}}^{\text{total}} = 80$ 毫秒（反向总时间）
    - 所有层相同 → 每层时间均匀

    **每层时间：**

    $$T_{\text{AR}}^{\text{layer}} = \frac{100}{100} = 1 \text{ ms}$$

    $$T_{\text{bwd}}^{\text{layer}} = \frac{80}{100} = 0.8 \text{ ms}$$

    **重叠分析：**

    基于桶的重叠，第 $i$ 层的 AllReduce 可以与第 $1, \ldots, i-1$ 层的反向计算重叠。

    ```
    Layer 100: [bwd 0.8ms][  AR 1ms  ]
    Layer 99:            [bwd 0.8ms][  AR 1ms  ]
    Layer 98:                      [bwd 0.8ms][  AR 1ms  ]
    ...
    ```

    **对除第一层外的每一层：**
    - AllReduce（1 毫秒）有可能与更早层的反向重叠

    由于反向快于 AllReduce（0.8 毫秒 < 1 毫秒），AllReduce 才是瓶颈。

    **关键路径分析：**

    - 第 100 层：t=0 开始，反向于 t=0.8ms 结束，AR 于 t=1.8ms 结束
    - 第 99 层：反向可在 t=0.8ms 开始（第 100 层反向结束后），于 t=1.6ms 结束
      - 但第 100 层的 AR 仍在运行，直到 t=1.8ms
      - 第 99 层 AR 于 t=1.8ms 开始，t=2.8ms 结束
    - 如此继续，AR 一直是瓶颈

    **实际上，重叠受限于相对速度。**

    更仔细地计算：

    *无重叠时的反向总时间：* 80 毫秒

    *无重叠时的 AR 总时间：* 100 毫秒

    *完全流水线化时：*
    - 第一层反向：0.8 毫秒
    - 之后 99 个 AllReduce 依次进行：99 × 1 毫秒 = 99 毫秒（部分重叠）
    - 最后一个 AllReduce 在最后一个反向后完成

    可重叠的部分是能与 AllReduce 并发运行的各层反向时间：

    $$\text{Overlap time} = T_{\text{bwd}}^{\text{total}} - T_{\text{bwd}}^{\text{first layer}} = 80 - 0.8 = 79.2 \text{ ms}$$

    由于 AR 总时间为 100 毫秒、第一层之后的反向为 79.2 毫秒：

    $$\text{Overlapped communication} = \min(79.2, 100) = 79.2 \text{ ms}$$

    **重叠比例：**
    $$\text{Overlap fraction} = \frac{79.2}{100} = \boxed{79.2\%}$$

    **有效步时间：**

    $$T_{\text{step}} = T_{\text{bwd}}^{\text{first layer}} + \max(T_{\text{bwd}}^{\text{remaining}}, T_{\text{AR}}^{\text{total}})$$

    由于 AR（100 毫秒）> 剩余反向（79.2 毫秒）：

    $$T_{\text{step}} = 0.8 + 100 = \boxed{100.8 \text{ ms}}$$

    **对比：**

    | 场景 | 步时间 |
    |----------|-----------|
    | 无重叠 | 80 + 100 = 180 ms |
    | 有重叠 | 100.8 ms |
    | 加速比 | 1.79× |

    **关键洞见：** 当 AR 慢于反向时，步时间趋近于 AR 时间。系统受限于通信。

4. **压缩分析**：Top-1% 稀疏化可以把梯度压缩 100 倍。如果原来的 AllReduce 耗时 50 毫秒，新时间是多少？计入压缩/解压计算（各假设 5 毫秒）。

??? success "解答"
    **已知：**

    - 原 AllReduce 时间：$T_{\text{AR}}^{\text{orig}} = 50$ 毫秒
    - 压缩比：100×（Top-1% 稀疏化）
    - 压缩计算：$T_{\text{compress}} = 5$ 毫秒
    - 解压计算：$T_{\text{decompress}} = 5$ 毫秒

    **分析：**

    *原来的通信受带宽支配（梯度很大）：*
    $$T_{\text{AR}}^{\text{orig}} \approx \frac{2(P-1)}{P} \cdot \frac{n}{\beta} \approx \frac{2n}{\beta} = 50 \text{ ms}$$

    *压缩后的通信：*
    $$T_{\text{AR}}^{\text{compressed}} \approx \frac{2n/100}{\beta} = \frac{50}{100} = 0.5 \text{ ms}$$

    **然而，稀疏 AllReduce 更复杂：**

    稀疏张量需要对索引 + 值做 AllGather，而不是标准的 AllReduce：

    - 索引：$0.01n$ 个元素 × 4 字节（int32）= $0.04n$ 字节
    - 值：$0.01n$ 个元素 × 2 字节（FP16）= $0.02n$ 字节
    - 总计：$0.06n$ 字节（对比原来的 $2n$ 字节）

    有效压缩：$2n / 0.06n = 33\times$（不是 100 倍！）

    *修正后的压缩通信：*
    $$T_{\text{AR}}^{\text{compressed}} = \frac{50}{33} \approx 1.5 \text{ ms}$$

    **带压缩的总时间：**

    $$T_{\text{total}} = T_{\text{compress}} + T_{\text{AR}}^{\text{compressed}} + T_{\text{decompress}}$$

    $$= 5 + 1.5 + 5 = \boxed{11.5 \text{ ms}}$$

    **加速比：**
    $$\text{Speedup} = \frac{T_{\text{orig}}}{T_{\text{total}}} = \frac{50}{11.5} = \boxed{4.3\times}$$

    **汇总：**

    | 组成部分 | 时间 |
    |-----------|------|
    | 压缩 | 5 ms |
    | 稀疏 AllReduce | 1.5 ms |
    | 解压 | 5 ms |
    | **总计** | **11.5 ms** |

    **关键洞见：**

    1. **压缩开销不可忽视**：10 毫秒的计算开销相当显著
    2. **索引开销降低了有效压缩比**：100 倍稀疏 ≠ 100 倍带宽下降
    3. **仍然值得**：对受限于通信的训练，4.3 倍加速很可观
    4. **盈亏平衡分析**：压缩只有在 $T_{\text{compress}} + T_{\text{decompress}} < T_{\text{AR}}^{\text{orig}} - T_{\text{AR}}^{\text{compressed}}$ 时才有收益

    $$10 < 50 - 1.5 = 48.5 \text{ ms} \checkmark$$

5. **累积的权衡**：以本地批 32 训练，每个微批次计算耗时 100 毫秒、AllReduce 耗时 40 毫秒。如果累积 4 个微批次，每个有效步的时间是多少？如果内存允许直接用批 128（单次前反向）呢？

??? success "解答"
    **已知：**

    - 微批次大小：32
    - $T_{\text{compute}}^{\text{micro}} = 100$ 毫秒每微批次
    - $T_{\text{AR}} = 40$ 毫秒每次 AllReduce
    - 累积步数：$A = 4$
    - 有效批大小：$32 \times 4 = 128$

    **情形 1：梯度累积（4 个微批次）**

    累积时，我们计算 4 个微批次，但只做一次 AllReduce：

    $$T_{\text{accum}} = A \times T_{\text{compute}}^{\text{micro}} + T_{\text{AR}}$$

    $$= 4 \times 100 + 40 = \boxed{440 \text{ ms}}$$

    **情形 2：直接批 128（无累积）**

    计算时间随批大小线性增长：

    $$T_{\text{compute}}^{\text{direct}} = \frac{128}{32} \times 100 = 400 \text{ ms}$$

    总计：

    $$T_{\text{direct}} = T_{\text{compute}}^{\text{direct}} + T_{\text{AR}} = 400 + 40 = \boxed{440 \text{ ms}}$$

    **对比：**

    | 方法 | 计算 | AllReduce | 总计 |
    |--------|---------|-----------|-------|
    | 累积（4×32） | 4 × 100 = 400 ms | 40 ms | 440 ms |
    | 直接（128） | 400 ms | 40 ms | 440 ms |

    $$\boxed{\text{Both methods take the same time: 440 ms}}$$

    **为什么相等？**

    梯度累积不增加开销——它只是把计算拆成多次。两种情形下 AllReduce 都只发生一次。

    **直接大批何时胜出？**

    如果计算与通信之间有重叠：

    *有重叠（直接）：*
    $$T_{\text{direct}}^{\text{overlap}} = \max(T_{\text{compute}}, T_{\text{AR}}) = \max(400, 40) = 400 \text{ ms}$$

    *有重叠（累积）：*
    - 微批次 1-3 的 AR 无法重叠（我们不做 AR）
    - 最后的 AR 只能与……无物重叠（计算已结束）
    $$T_{\text{accum}}^{\text{overlap}} = 4 \times 100 + 40 = 440 \text{ ms}$$

    **直接大批配合重叠的加速：**
    $$\frac{440}{400} = 1.1\times$$

    **关键洞见：** 直接的大批相比梯度累积能实现更好的计算-通信重叠。

6. **陈旧度上界**：在 $P = 16$ 个 worker 的异步 SGD 中，使用指数陈旧度惩罚 $f(\tau) = e^{-0.1\tau}$，若最大陈旧度为 $\tau = 15$，有效学习率缩放是多少？

??? success "解答"
    **已知：**

    - $P = 16$ 个 worker
    - 陈旧度惩罚：$f(\tau) = e^{-0.1\tau}$
    - 最大陈旧度：$\tau_{\max} = 15$

    **陈旧度修正的更新规则：**

    $$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot f(\tau) \cdot g^{(t-\tau)}$$

    有效学习率为：$\eta_{\text{eff}} = \eta \cdot f(\tau)$

    **陈旧度分布分析：**

    在 $P$ 个 worker 的异步 SGD 中，最坏情况下陈旧度 $\tau$ 的范围是 0 到 $P-1$（某个 worker 始终领先）。

    当 $\tau_{\max} = 15$（接近 $P - 1 = 15$）时，我们处于最大异步程度。

    **不同陈旧度下的有效学习率：**

    | 陈旧度 $\tau$ | $f(\tau) = e^{-0.1\tau}$ | $\eta_{\text{eff}}/\eta$ |
    |------------------|--------------------------|--------------------------|
    | 0 | $e^0 = 1.000$ | 100% |
    | 5 | $e^{-0.5} = 0.607$ | 60.7% |
    | 10 | $e^{-1.0} = 0.368$ | 36.8% |
    | 15 | $e^{-1.5} = 0.223$ | 22.3% |

    **最大陈旧度下（$\tau = 15$）：**

    $$\eta_{\text{eff}} = \eta \cdot e^{-0.1 \times 15} = \eta \cdot e^{-1.5} = \boxed{0.223 \cdot \eta}$$

    学习率被缩小到**基准的 22.3%**。

    **平均有效学习率：**

    假设陈旧度在 0 到 $\tau_{\max}$ 之间均匀分布：

    $$\mathbb{E}[f(\tau)] = \frac{1}{\tau_{\max} + 1} \sum_{\tau=0}^{\tau_{\max}} e^{-0.1\tau}$$

    这是一个等比级数：

    $$= \frac{1}{16} \cdot \frac{1 - e^{-0.1 \times 16}}{1 - e^{-0.1}} = \frac{1}{16} \cdot \frac{1 - e^{-1.6}}{1 - e^{-0.1}}$$

    $$= \frac{1}{16} \cdot \frac{1 - 0.202}{1 - 0.905} = \frac{1}{16} \cdot \frac{0.798}{0.095} = \frac{8.4}{16} = 0.525$$

    **平均有效学习率：** 基准的 $\boxed{52.5\%}$。

    **实践含义：**

    | 方面 | 数值 |
    |--------|-------|
    | 最小缩放（新鲜梯度） | 100% |
    | 最大缩放（最陈旧梯度） | 22.3% |
    | 平均缩放 | 52.5% |
    | 有效吞吐损失 | ~47.5% |

    要与同步 SGD 的收敛速度持平，你需要把基础学习率提高约 $1/0.525 \approx 1.9$ 倍，但这可能使新鲜梯度导致不稳定。

7. **非确定性的量化**：两种归约顺序 $((g_0 + g_1) + g_2)$ 与 $(g_0 + (g_1 + g_2))$ 至多相差 $\epsilon_{\text{machine}} \cdot |g_0 + g_1 + g_2|$。对一个 1B 参数、FP16 梯度（$\epsilon \approx 10^{-3}$）的模型，估算最大梯度差异。

??? success "解答"
    **已知：**

    - 模型参数：$\Psi = 1 \times 10^9$
    - 数据类型：FP16
    - 机器精度：$\epsilon \approx 10^{-3}$（实际 $\epsilon_{\text{FP16}} = 2^{-10} \approx 9.77 \times 10^{-4}$）
    - 每次归约的误差界：$\epsilon \cdot |g_0 + g_1 + g_2|$

    **单次归约误差：**

    对两个数 $a$、$b$ 的浮点加法：

    $$\text{fl}(a + b) = (a + b)(1 + \delta), \quad |\delta| \leq \epsilon$$

    绝对误差为：

    $$|error| \leq \epsilon \cdot |a + b|$$

    **AllReduce 中的误差累积：**

    对 $P$ 块 GPU 的环形 AllReduce，每个参数要做 $P-1$ 次加法：

    $$g = ((((g_0 + g_1) + g_2) + g_3) + \cdots + g_{P-1})$$

    每一步都累积误差。根据标准误差分析：

    $$|\text{accumulated error}| \leq (P-1) \cdot \epsilon \cdot |g_{\text{sum}}|$$

    **每个参数的误差估计：**

    假设梯度幅值典型为 $|g_i| \approx 10^{-4}$（归一化训练中常见）：

    - $P = 8$ 个梯度之和：$|g_{\text{sum}}| \approx 8 \times 10^{-4}$（若同向）
    - 每参数误差：$\leq 7 \times 10^{-3} \times 8 \times 10^{-4} = 5.6 \times 10^{-6}$

    **不同归约顺序：**

    两种顺序至多相差：

    $$|\text{difference}| \leq 2 \cdot (P-1) \cdot \epsilon \cdot |g_{\text{sum}}|$$

    （系数 2 是因为每种顺序可能朝相反方向出错）

    **全模型误差：**

    对 1B 参数，梯度总差异：

    *差异的 L2 范数：*

    若误差在参数之间独立、方差为 $\sigma^2 = \epsilon^2 \cdot |g|^2$：

    $$\|g_{\text{order1}} - g_{\text{order2}}\|_2 \approx \sqrt{\Psi} \cdot \sigma$$

    $$= \sqrt{10^9} \cdot \epsilon \cdot |g_{\text{avg}}| \cdot (P-1)$$

    取 $|g_{\text{avg}}| \approx 10^{-4}$、$P = 8$、$\epsilon = 10^{-3}$：

    $$= 3.16 \times 10^4 \cdot 10^{-3} \cdot 10^{-4} \cdot 7 = \boxed{2.2 \times 10^{-2}}$$

    **每参数最大差异：**

    $$|\Delta g_i|_{\max} \approx 2 \cdot (P-1) \cdot \epsilon \cdot |g_i| \approx 14 \cdot 10^{-3} \cdot 10^{-4} = \boxed{1.4 \times 10^{-6}}$$

    **相对误差：**

    $$\frac{|\Delta g|}{|g|} \approx 2(P-1) \cdot \epsilon = 14 \times 10^{-3} = \boxed{1.4\%}$$

    **汇总：**

    | 指标 | 数值 |
    |--------|-------|
    | 每参数最大差异 | $1.4 \times 10^{-6}$ |
    | 每参数相对误差 | ~1.4% |
    | 差异的 L2 范数（1B 参数） | ~0.022 |

    **实践含义：**

    1. **训练稳定性**：每步 1.4% 的方差会在数百万步中累积
    2. **可复现性**：相同代码、不同 GPU 数量 → 不同结果
    3. **调试**：跨运行做逐位比较是不可能的

    **缓解策略：**

    | 策略 | 开销 | 确定性 |
    |----------|----------|-------------|
    | 固定归约顺序 | ~5% | 每种配置下逐位一致 |
    | FP32 累积 | 2 倍内存 | 误差降低约 1000 倍 |
    | Kahan 求和 | 2 倍计算 | 接近 FP64 精度 |
    | 树形归约（平衡） | 相同 | 比环形更稳定 |

## 旋钮与权衡

| 旋钮 | 主要效果 | 代价 |
|---|---|---|
| 数据并行度（P） | 更高的总算力 | 更多梯度通信、对延迟更敏感 |
| 本地批大小（b） | 更高的通信强度 | 内存压力与优化器稳定性 |
| 梯度累积 | 更大的有效批 | 步延迟更高、响应性更差 |
| 分桶大小 | 更好的重叠 | 峰值内存更高、更新被延迟 |

## 关键要点

1. **结合律使数据并行成立**：划分定理是数学基础——和的梯度等于梯度的和。

2. **AllReduce 是通信原语**：每步 $2(P-1)/P \cdot \Psi$ 字节。

3. **计算-通信比决定效率**：本地批必须足够大，才能摊薄通信。

4. **重叠至关重要**：反向传播期间基于桶的 AllReduce 可实现 70%~90% 的重叠。

5. **浮点数破坏了精确结合律**：要确定性，需固定归约顺序并禁用优化。

6. **梯度压缩（Gradient Compression）以计算换带宽**：Top-K、量化和低秩方法可实现 10-100 倍压缩。

7. **扩展有极限**：对大模型，超过约 100 块 GPU 后，单靠数据并行就会受限于通信。

8. **同步方式占主导**：更简单的正确性与收敛性使同步方式更受青睐，尽管它对拖后腿敏感。