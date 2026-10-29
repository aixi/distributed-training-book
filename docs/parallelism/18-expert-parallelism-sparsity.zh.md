---
title: "从稀疏性推导专家并行"
subtitle: "把 token 路由到分布式专家"
---

<div class="chapter-opener" markdown>
混合专家（Mixture of Experts, MoE）模型对每个 token 只激活一部分参数。这种稀疏性使得模型可以在不按比例增加计算成本的前提下拥有海量参数——但需要精心设计的路由，才能让负载在分布式专家之间保持均衡。
</div>

<div class="investigation-question" markdown>
**问题**：一个模型有 8 个专家，分布在 8 块 GPU 上。某个 token 被路由到 GPU 3 和 GPU 7 上的专家。这个 token 如何到达那里，又如何返回？如果所有 token 都想去找同一个专家，会发生什么？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 11 章](../collectives/11-primitives-properties.md)（AlltoAll 原语）、[第 14–15 章](14-data-parallelism-associativity.md)（数据并行与张量并行的背景；另见[第 15 章](15-tensor-parallelism-linearity.md)）

    **核心洞见**：MoE 通过为每个 token 只激活少数专家，实现了"参数海量、计算恒定"。通信模式是 AlltoAll（token 去专家、结果回来）。负载均衡至关重要——辅助损失和容量因子可以防止专家塌缩。

## 什么是混合专家？

标准 transformer 让每个 token 都经过同一个前馈网络（FFN）——每个参数都参与每一次计算。**混合专家（Mixture of Experts, MoE）**把这一单一 FFN 替换为多个并行的 FFN，称为*专家*（expert），外加一个轻量的*路由器*（router，或称*门控* gate），由它决定每个 token 交给哪个（哪些）专家处理。

核心洞见是：不同的 token 可能受益于不同的计算。MoE 不强迫所有 token 经过完全相同的权重，而是让模型学会特化：

- **路由器**检查每个 token，为每个专家产出一个分数
- 只有得分最高的专家被激活——通常是从众多专家（8、64 甚至 128 个）中选 1 或 2 个
- 被选中的专家处理该 token 并按路由分数加权返回输出

这造成了一种强大的不对称性：模型可以拥有多得多的参数（更多专家 = 更大容量），而计算成本不必按比例增长（每个 token 只跑少数几个专家）。一个有 64 个专家、top-2 路由的模型，**FFN 参数多 64 倍**（每个专家是一个完整 FFN），而 FFN 计算量只有稠密模型的 **2 倍**（因为每个 token 只激活 2 个专家）。

挑战在于分布式：当专家分布在不同 GPU 上时，token 必须被送到正确的专家那里再送回来。本章探讨这种通信如何工作，以及如何让负载在专家间保持均衡。

## 稀疏性性质

稠密神经网络有一个基本性质：每个参数都参与每一次计算。对于一个稠密前馈层：

$$y = W_2(\sigma(W_1 x))$$

$W_1$ 和 $W_2$ 的每一个元素都对每个输出有贡献。

**定义（条件计算）**：如果对于每个输入只有一部分参数被激活，则该计算是*稀疏*的：

$$y = \sum_{i \in S(x)} w_i \cdot f_i(x)$$

其中 $S(x) \subseteq \{1, \ldots, E\}$ 是一个依赖于输入的*选择函数*，且 $|S(x)| \ll E$。

这带来一个关键性质。

**定理（稀疏性缩放）**：一个包含 $E$ 个专家（每个大小为 $d_{\text{model}} \times d_{\text{ff}}$）的混合专家层具有：

- **参数**：$E \cdot d_{\text{model}} \cdot d_{\text{ff}}$（随 $E$ 增长）
- **每 token 的 FLOPs**：$k \cdot d_{\text{model}} \cdot d_{\text{ff}}$（与 $E$ 无关）

其中 $k$ 是每个 token 激活的专家数。

*证明*：每个 token 只流经 $k$ 个被选中的专家，与专家总数无关。参数数量随 $E$ 线性增长，但每 token 的计算量保持在 $O(k \cdot d_{\text{model}} \cdot d_{\text{ff}})$ 不变。□

这种不对称性非常强大：我们可以扩展参数（容量）而不扩展计算（成本）。

## 混合专家架构

### MoE 层

混合专家层替换 transformer 块中的标准前馈网络（FFN）：

```
Standard Transformer:      MoE Transformer:
┌─────────────────┐        ┌─────────────────┐
│   Attention     │        │   Attention     │
├─────────────────┤        ├─────────────────┤
│   Add & Norm    │        │   Add & Norm    │
├─────────────────┤        ├─────────────────┤
│      FFN        │   →    │   Router (Gate) │
│                 │        │    ↙  ↓  ↘      │
│                 │        │   E₀ E₁ ... E_n │
├─────────────────┤        ├─────────────────┤
│   Add & Norm    │        │   Add & Norm    │
└─────────────────┘        └─────────────────┘
```

每个专家 $E_i$ 本身就是一个完整的 FFN：

$$E_i(x) = W_2^{(i)} \cdot \text{GeLU}(W_1^{(i)} x)$$

路由器（门控网络）决定哪些专家处理每个 token。

### 数学表述

对于输入 token $x \in \mathbb{R}^{d}$：

**第 1 步：计算路由 logits**
$$h = W_g x$$

其中 $W_g \in \mathbb{R}^{E \times d}$ 是门控权重矩阵。

**第 2 步：应用门控函数**
$$g = \text{Softmax}(h)$$

**第 3 步：选择 top-k 专家**
$$S = \text{TopK}(g, k)$$

**第 4 步：计算加权输出**
$$y = \sum_{i \in S} \frac{g_i}{\sum_{j \in S} g_j} \cdot E_i(x)$$

归一化确保权重在被选专家上之和为 1。

### 稀疏性为何有效

**直觉**：并非所有参数都需要对所有输入激活。不同专家可以特化：

- 专家 1：处理语法相关的计算
- 专家 2：处理事实性知识
- 专家 3：处理推理链
- ……

路由器学会把 token 导向合适的专家。

**经验观察**：在容量充足时，专家确实会发展出不同的特化，尽管这些特化往往难以解释。

## 门控机制

路由器至关重要：它决定了选择函数 $S(x)$。

### Softmax 门控（最初的 MoE）

最简单的方法：

$$g_i = \frac{e^{h_i}}{\sum_{j=1}^E e^{h_j}}, \quad h = W_g x$$

**问题**：Softmax 是"软"的——所有专家都得到一点权重。要实现真正的稀疏，我们需要硬选择。

### Top-k 门控

选择路由分数最高的 $k$ 个专家：

$$S = \{i : g_i \text{ is among top-}k\text{ values}\}$$

重新归一化权重：

$$\tilde{g}_i = \frac{g_i}{\sum_{j \in S} g_j} \text{ for } i \in S$$

**典型取值**：$k = 1$（Switch Transformer）或 $k = 2$（GShard、最初的 MoE）。

### 带噪 Top-k 门控

训练时加入噪声以鼓励探索：

$$h_i = (W_g x)_i + \epsilon_i \cdot \text{Softplus}((W_{\text{noise}} x)_i)$$

其中 $\epsilon_i \sim \mathcal{N}(0, 1)$。

可学习的噪声让模型能够：
1. 探索不同的专家分配
2. 逃离路由上的糟糕局部最优
3. 形成更均衡的负载分布

### 专家选择路由

与其让 token 选专家，不如让专家选 token：

**标准方式（token 选择）**：

- 每个 token 挑选自己的 top-k 专家
- 导致负载不均衡

**专家选择**（Zhou et al., 2022）：

- 每个专家挑选自己的 top-k token
- 保证完美的负载均衡
- 每个专家恰好处理 $\text{capacity} = k \cdot T / E$ 个 token

```python
def expert_choice_routing(tokens, router_logits, capacity):
    """
    Expert choice: experts select their top tokens.

    Args:
        tokens: [batch, seq, dim] - input tokens
        router_logits: [batch, seq, num_experts] - routing scores
        capacity: int - tokens per expert
    """
    batch, seq, dim = tokens.shape
    num_tokens = batch * seq
    num_experts = router_logits.shape[-1]

    # Reshape for routing
    tokens_flat = tokens.view(num_tokens, dim)
    logits_flat = router_logits.view(num_tokens, -1)

    # Transpose: [num_experts, num_tokens]
    expert_scores = logits_flat.t()

    # Each expert selects top-capacity tokens
    _, indices = expert_scores.topk(capacity, dim=1)

    # Gather selected tokens for each expert
    expert_inputs = []
    for e in range(num_experts):
        selected = tokens_flat[indices[e]]  # [capacity, dim]
        expert_inputs.append(selected)

    return expert_inputs, indices
```

## AlltoAll 通信模式

专家并行需要一种独特的通信模式：AlltoAll。

### 为什么是 AlltoAll？

考虑 4 块 GPU、每块 1 个专家、处理一批 token：

```
Before routing:
GPU 0: [t0, t1, t2, t3] - tokens 0-3 need various experts
GPU 1: [t4, t5, t6, t7] - tokens 4-7 need various experts
GPU 2: [t8, t9, t10, t11]
GPU 3: [t12, t13, t14, t15]

After routing decision:
t0 → E2    t4 → E0    t8 → E1     t12 → E3
t1 → E0    t5 → E3    t9 → E0     t13 → E1
t2 → E1    t6 → E2    t10 → E2    t14 → E0
t3 → E3    t7 → E1    t11 → E3    t15 → E2
```

每块 GPU 需要：
1. 把自己的 token 送到正确专家的 GPU
2. 接收发往本地专家的 token

这正好就是 AlltoAll 模式：

```
AlltoAll:
GPU 0 sends:  [to_E0, to_E1, to_E2, to_E3]
GPU 1 sends:  [to_E0, to_E1, to_E2, to_E3]
GPU 2 sends:  [to_E0, to_E1, to_E2, to_E3]
GPU 3 sends:  [to_E0, to_E1, to_E2, to_E3]

After AlltoAll:
GPU 0 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E0)
GPU 1 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E1)
GPU 2 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E2)
GPU 3 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E3)
```

### 通信量

对于有 $P$ 个参与者、每人共持有 $n$ 字节的 AlltoAll：

$$\text{Volume per GPU} = \frac{(P-1) \cdot n}{P}$$

每块 GPU 把自己数据的 $(P-1)/P$ 发送给其他 GPU。

**总量**（所有 GPU 求和）：

$$\text{Total} = P \cdot \frac{(P-1) \cdot n}{P} = (P-1) \cdot n$$

### AlltoAll-AlltoAll 模式

一个 MoE 层需要两次 AlltoAll 操作：

```
┌────────────────┐
│ Token inputs   │  Local tokens on each GPU
├────────────────┤
│ Routing (Gate) │  Compute expert assignments
├────────────────┤
│ AlltoAll       │  ← Dispatch: tokens → expert GPUs
├────────────────┤
│ Expert FFN     │  Each GPU runs its local expert(s)
├────────────────┤
│ AlltoAll       │  ← Combine: outputs → original GPUs
├────────────────┤
│ Weighted sum   │  Combine expert outputs
└────────────────┘
```

**实现**：

```python
class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        expert_group: dist.ProcessGroup
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_group = expert_group
        self.world_size = dist.get_world_size(expert_group)
        self.rank = dist.get_rank(expert_group)

        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Local experts (this GPU's experts)
        experts_per_rank = num_experts // self.world_size
        self.local_experts = nn.ModuleList([
            FFN(hidden_dim, ffn_dim)
            for _ in range(experts_per_rank)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, hidden] input tokens
        Returns:
            [batch, seq, hidden] output tokens
        """
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)  # [batch*seq, hidden]
        num_tokens = x_flat.shape[0]

        # Step 1: Compute routing
        router_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        routing_weights, selected_experts = self._top_k_gating(router_logits)

        # Step 2: Prepare for AlltoAll dispatch
        # Group tokens by destination expert
        dispatch_data, dispatch_indices = self._prepare_dispatch(
            x_flat, selected_experts, routing_weights
        )

        # Step 3: AlltoAll dispatch
        # Each GPU sends tokens to expert-owning GPUs
        received_data = self._all_to_all(dispatch_data)

        # Step 4: Process through local experts
        expert_outputs = self._run_local_experts(received_data)

        # Step 5: AlltoAll combine
        # Send outputs back to original GPUs
        combined_outputs = self._all_to_all(expert_outputs)

        # Step 6: Weighted combination
        output = self._combine_outputs(
            combined_outputs, dispatch_indices, routing_weights, num_tokens, hidden
        )

        return output.view(batch, seq, hidden)

    def _top_k_gating(self, logits: torch.Tensor):
        """Select top-k experts per token."""
        weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = weights.topk(
            self.num_experts_per_tok, dim=-1
        )
        # Renormalize
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return top_weights, top_indices

    def _all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        """Perform AlltoAll communication."""
        return all_to_all_single(
            x,
            output_split_sizes=None,
            input_split_sizes=None,
            group=self.expert_group
        )
```

## 负载均衡

关键的挑战在于：如果所有 token 都想去找同一个专家，会怎样？

### 问题

如果不做均衡，专家负载可能高度倾斜：

```
Unbalanced:
Expert 0: ████████████████████ (80% of tokens)
Expert 1: ██ (5%)
Expert 2: ██ (5%)
Expert 3: ███ (10%)

→ Expert 0's GPU is the bottleneck
→ Other GPUs mostly idle
→ Training throughput collapses
```

### 方案 1：辅助负载均衡损失

加入一个惩罚不均衡路由的损失项：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

其中：

- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[\text{token } t \text{ routes to expert } i]$（路由到专家 $i$ 的 token 占比）
- $p_i = \frac{1}{T} \sum_{t=1}^{T} g_i^{(t)}$（专家 $i$ 的平均路由概率）
- $\alpha$ 是均衡系数（通常为 0.01–0.1）

**为什么有效**：乘积 $f_i \cdot p_i$ 在负载均匀时取最小值：

- 如果 $f_i$ 较高（token 很多），则 $p_i$ 必须较低才能使损失最小
- 梯度会把路由器推向均衡分配

**推导**：对于均匀路由，所有 $i$ 有 $f_i = p_i = 1/E$：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} \frac{1}{E} \cdot \frac{1}{E} = \alpha \cdot E \cdot E \cdot \frac{1}{E^2} = \alpha$$

任何偏离均匀分布的情况都会使 $\mathcal{L}_{\text{aux}}$ 高于 $\alpha$。

### 方案 2：容量因子

限制每个专家能处理的 token 数：

$$C = \left\lceil \text{capacity\_factor} \times \frac{T \times k}{E} \right\rceil$$

其中：

- $T$ = token 总数
- $k$ = 每个 token 的专家数
- $E$ = 专家总数
- $\text{capacity\_factor} \geq 1.0$（通常为 1.25–2.0）

超出容量的 token 会被：

- **丢弃**（dropped）：输出置零或跳过（Switch Transformer）
- **溢出**（overflowed）：路由到次优专家（GShard）

```python
def apply_capacity_limit(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    capacity_factor: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply capacity limit to expert assignments.

    Returns:
        tokens, indices, weights with capacity limits applied
    """
    num_tokens, k = expert_indices.shape
    num_experts = expert_indices.max().item() + 1

    # Compute capacity per expert
    capacity = int(math.ceil(capacity_factor * num_tokens * k / num_experts))

    # Count assignments per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=tokens.device)

    # Mask for valid assignments
    valid_mask = torch.zeros_like(expert_indices, dtype=torch.bool)

    for tok_idx in range(num_tokens):
        for k_idx in range(k):
            expert_id = expert_indices[tok_idx, k_idx].item()
            if expert_counts[expert_id] < capacity:
                valid_mask[tok_idx, k_idx] = True
                expert_counts[expert_id] += 1

    # Zero out weights for dropped assignments
    expert_weights = expert_weights * valid_mask.float()

    # Renormalize weights
    weight_sum = expert_weights.sum(dim=-1, keepdim=True)
    weight_sum = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
    expert_weights = expert_weights / weight_sum

    return tokens, expert_indices, expert_weights
```

### 方案 3：Z-Loss

正则化路由 logits，避免路由过于自信：

$$\mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \log^2 \left( \sum_{i=1}^{E} e^{h_i^{(t)}} \right)$$

它惩罚过大的 logits，促使路由分布更平滑。

### 均衡策略比较

| 策略 | 优点 | 缺点 |
|------|------|------|
| 辅助损失 | 可微、端到端 | 可能干扰主损失 |
| 容量因子 | 对均衡有硬保证 | 会丢 token、不可微 |
| 专家选择 | 完美均衡 | 需要改动架构 |
| Z-loss | 稳定训练 | 不直接均衡 |

**最佳实践**：辅助损失 + 容量因子 + z-loss 组合使用。

## Token 丢弃分析

token 被丢弃时会发生什么？

### 丢弃 token 的影响

设 $d$ 为丢弃率。有效批大小变为：

$$B_{\text{eff}} = B \times (1 - d)$$

**梯度估计**：被丢弃的 token 不产生梯度：

$$\nabla_{\text{observed}} = \frac{1}{B_{\text{eff}}} \sum_{i \in \text{not dropped}} \nabla L_i$$

如果丢弃是随机的，这是梯度的无偏估计；但基于路由的丢弃并不是随机的。

### 偏差问题

被路由到热门专家的 token 更容易被丢弃。这些 token 往往是：

- 更常见的模式
- 对泛化很重要
- 模型"希望"以相似方式处理的 token

系统性地丢弃这些 token 会损害模型质量。

### 缓解措施：No-Token-Left-Behind

替代策略：
1. **溢出路由**：被丢弃的 token 送往次优专家
2. **辅助缓冲区**：暂存被丢弃的 token，在下一批中处理
3. **增大容量**：令 capacity_factor = 2.0 或更高

## 专家并行的实现

### 在 GPU 上放置专家

有 $E$ 个专家和 $P$ 块 GPU：

**情形 1：$E = P$（每 GPU 一个专家）**
```
GPU 0: Expert 0
GPU 1: Expert 1
...
GPU P-1: Expert P-1
```

这是最简单的情形：每个 AlltoAll 参与者恰好是一个专家。

**情形 2：$E > P$（每 GPU 多个专家）**
```
GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
...
GPU P-1: Expert 2(P-1), Expert 2P-1
```

本地路由无需通信；只有跨 GPU 路由才使用 AlltoAll。

**情形 3：$E < P$（专家被分片到多块 GPU）**

每个专家内部使用张量并行：
```
Expert 0: GPU 0, GPU 1 (TP=2)
Expert 1: GPU 2, GPU 3 (TP=2)
...
```

### 完整实现

```python
class DistributedMoE(nn.Module):
    """
    Distributed Mixture of Experts layer with AlltoAll communication.
    """
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        expert_group: Optional[dist.ProcessGroup] = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef

        # Expert parallel group
        if expert_group is None:
            expert_group = dist.distributed_c10d._get_default_group()
        self.expert_group = expert_group
        self.world_size = dist.get_world_size(expert_group)
        self.rank = dist.get_rank(expert_group)

        assert num_experts % self.world_size == 0
        self.experts_per_rank = num_experts // self.world_size

        # Router (on all ranks)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Local experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim, bias=False),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim, bias=False)
            )
            for _ in range(self.experts_per_rank)
        ])

        # Auxiliary loss storage
        self.aux_loss = 0.0
        self.z_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with AlltoAll communication.

        Args:
            x: [batch, seq, hidden]
        Returns:
            [batch, seq, hidden]
        """
        batch, seq, hidden = x.shape
        num_tokens = batch * seq
        x_flat = x.view(num_tokens, hidden)

        # ===== ROUTING =====
        router_logits = self.gate(x_flat)  # [num_tokens, num_experts]

        # Z-loss for stability
        self.z_loss = self.z_loss_coef * torch.logsumexp(
            router_logits, dim=-1
        ).square().mean()

        # Softmax routing
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_weights, top_indices = router_probs.topk(self.top_k, dim=-1)

        # Renormalize
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Auxiliary load balancing loss
        self.aux_loss = self._compute_aux_loss(router_probs, top_indices)

        # ===== CAPACITY LIMITING =====
        capacity = int(math.ceil(
            self.capacity_factor * num_tokens * self.top_k / self.num_experts
        ))

        # Build dispatch mask and positions
        dispatch_mask, combine_weights, tokens_per_expert = self._build_dispatch(
            x_flat, top_indices, top_weights, capacity
        )

        # ===== ALLTOALL DISPATCH =====
        # Prepare data: group tokens by target expert
        dispatch_tokens = self._gather_for_dispatch(x_flat, dispatch_mask, capacity)

        # AlltoAll: send tokens to expert-owning ranks
        # Shape: [experts_per_rank * capacity, hidden] per rank
        recv_tokens = self._dispatch_all_to_all(dispatch_tokens, tokens_per_expert)

        # ===== EXPERT COMPUTATION =====
        expert_outputs = self._run_experts(recv_tokens, capacity)

        # ===== ALLTOALL COMBINE =====
        # AlltoAll: send outputs back
        combined = self._combine_all_to_all(expert_outputs, tokens_per_expert)

        # ===== WEIGHTED SUM =====
        output = self._weighted_combine(combined, combine_weights, dispatch_mask)

        return output.view(batch, seq, hidden)

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        num_tokens = router_probs.shape[0]

        # f_i: fraction of tokens routed to expert i
        # Use one-hot encoding of selections
        one_hot = F.one_hot(
            selected_experts.view(-1),
            num_classes=self.num_experts
        ).float()
        tokens_per_expert = one_hot.sum(dim=0)
        f = tokens_per_expert / (num_tokens * self.top_k)

        # p_i: average routing probability for expert i
        p = router_probs.mean(dim=0)

        # Aux loss: encourages uniform f and p
        aux_loss = self.aux_loss_coef * self.num_experts * (f * p).sum()

        return aux_loss

    def _dispatch_all_to_all(
        self,
        tokens: torch.Tensor,
        tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        """
        AlltoAll to dispatch tokens to expert-owning ranks.
        """
        # Compute send/recv sizes
        send_sizes = tokens_per_expert.tolist()
        recv_sizes = [0] * self.world_size

        # Exchange sizes
        dist.all_to_all_single(
            torch.tensor(recv_sizes, device=tokens.device),
            torch.tensor(send_sizes, device=tokens.device),
            group=self.expert_group
        )

        # Allocate receive buffer
        total_recv = sum(recv_sizes)
        recv_buffer = torch.empty(
            total_recv, self.hidden_dim,
            dtype=tokens.dtype, device=tokens.device
        )

        # AlltoAll data
        dist.all_to_all_single(
            recv_buffer,
            tokens,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
            group=self.expert_group
        )

        return recv_buffer

    def _run_experts(
        self,
        tokens: torch.Tensor,
        capacity: int
    ) -> torch.Tensor:
        """Run tokens through local experts."""
        outputs = []

        # Tokens are grouped by expert
        offset = 0
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens for this expert
            expert_tokens = tokens[offset:offset + capacity]

            # Forward through expert
            expert_out = expert(expert_tokens)
            outputs.append(expert_out)

            offset += capacity

        return torch.cat(outputs, dim=0)

    def get_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss (add to main loss)."""
        return self.aux_loss + self.z_loss
```

## 稀疏路由中的梯度流

梯度如何穿过离散的专家选择？

### 可微性问题

Top-k 选择不可微：

$$\frac{\partial \text{TopK}(g)}{\partial g} = \text{undefined}$$

我们无法穿过选择操作反向传播。

### 直通估计

使用被选中的权重——它们本身是可微的：

**前向**：

$$y = \sum_{i \in S} \tilde{g}_i \cdot E_i(x)$$

**反向**：

$$\frac{\partial L}{\partial g_i} = \tilde{g}_i \cdot \frac{\partial L}{\partial E_i(x)} \cdot E_i(x)$$

梯度通过路由权重 $\tilde{g}_i$ 流动，而不是穿过选择操作。

### 路由器梯度

路由器从以下来源接收梯度：
1. **被选专家的输出**：被选专家表现如何？
2. **辅助损失**：推向负载均衡

这意味着路由器学会选择这样的专家：

- 对 token 产生好的输出
- 不使任何单个专家过载

## 与其他并行维度的组合

专家并行能自然地与其他形式结合。

### EP + DP（专家并行 + 数据并行）

最常见的组合：

```
                    Data Parallel Group (replicas)
                    ┌───────────────────────────┐
                    │                           │
     ┌──────────────┼─────────────┬─────────────┼──────────────┐
     │              │             │             │              │
     ▼              ▼             ▼             ▼              │
 ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐          │
 │Replica│     │Replica│     │Replica│     │Replica│          │
 │   0   │     │   1   │     │   2   │     │   3   │          │
 └───┬───┘     └───┬───┘     └───┬───┘     └───┬───┘          │
     │             │             │             │              │
     └─────────────┴──────┬──────┴─────────────┘              │
                          │                                    │
                 Expert Parallel Group                         │
                 ┌────────┴────────┐                          │
                 │                 │                          │
            ┌────┼────┐       ┌────┼────┐                     │
            ▼    ▼    ▼       ▼    ▼    ▼                     │
           E0   E1   E2      E3   E4   E5                     │
           │    │    │       │    │    │                      │
           GPU0 GPU1 GPU2   GPU3 GPU4 GPU5                    │
```

**通信模式**：

- EP 组内 AlltoAll（专家路由）
- DP 副本间 AllReduce（梯度同步）

### EP + TP（专家并行 + 张量并行）

对于非常大的专家，对每个专家做分片：

```
Expert 0:        Expert 1:
┌────┬────┐      ┌────┬────┐
│TP0 │TP1 │      │TP0 │TP1 │
│GPU0│GPU1│      │GPU2│GPU3│
└────┴────┘      └────┴────┘
     │                │
     └────────────────┘
          AlltoAll
```

每个专家内部使用张量并行。

### 完整三维：EP + DP + TP

```python
def create_3d_moe_groups(
    world_size: int,
    tp_size: int,
    ep_size: int
) -> Tuple[ProcessGroup, ProcessGroup, ProcessGroup]:
    """
    Create process groups for 3D parallelism with MoE.

    world_size = tp_size * ep_size * dp_size
    """
    dp_size = world_size // (tp_size * ep_size)

    rank = dist.get_rank()

    # Tensor parallel: consecutive ranks
    tp_group_id = rank // tp_size
    tp_ranks = list(range(tp_group_id * tp_size, (tp_group_id + 1) * tp_size))
    tp_group = dist.new_group(tp_ranks)

    # Expert parallel: strided by TP size
    ep_group_id = (rank % (tp_size * ep_size)) // tp_size
    ep_ranks = [
        (rank // (tp_size * ep_size)) * (tp_size * ep_size) +
        i * tp_size + rank % tp_size
        for i in range(ep_size)
    ]
    ep_group = dist.new_group(ep_ranks)

    # Data parallel: strided by TP * EP size
    dp_ranks = [
        rank % (tp_size * ep_size) + i * (tp_size * ep_size)
        for i in range(dp_size)
    ]
    dp_group = dist.new_group(dp_ranks)

    return tp_group, ep_group, dp_group
```

### 通信成本

对于一个具有以下参数的模型：

- $H$ = 隐藏维度
- $E$ = 专家数
- $T$ = 每批 token 数
- $k$ = 每 token 的专家数

| 并行方式 | 操作 | 通信量（每 GPU） |
|-------------|-----------|------------------|
| 仅 EP | 2 × AlltoAll | $2 \times \frac{(P-1)}{P} \times T \times H$ |
| EP + DP | AlltoAll + AllReduce | AlltoAll + $2 \times \frac{(P_{DP}-1)}{P_{DP}} \times \text{params}$ |
| EP + TP | AlltoAll + AllReduce | AlltoAll + $2 \times \frac{(P_{TP}-1)}{P_{TP}} \times T \times H$ |

## 实践考量

### 何时使用 MoE

**适合 MoE**：

- 稠密模型成本高到难以承受的超大模型
- 高吞吐需求（更多参数、相同计算）
- 能从特化中受益的任务

**不适合 MoE**：

- 小模型（路由开销占主导）
- 低延迟推理（路由会增加延迟）
- 互连有限的 GPU 集群（AlltoAll 是带宽密集型操作）

### 超参数指南

| 参数 | 典型范围 | 说明 |
|-----------|---------------|-------|
| num_experts | 8–128 | 取 2 的幂便于分片 |
| top_k | 1–2 | k=2 更稳定，k=1 更高效 |
| capacity_factor | 1.25–2.0 | 越高丢弃越少、内存越多 |
| aux_loss_coef | 0.01–0.1 | 过大会损害主任务 |
| z_loss_coef | 0.001–0.01 | 稳定训练 |

### 常见陷阱

**陷阱 1：路由塌缩**

所有 token 都路由到同一个专家。迹象：

- 一个专家处理 90% 以上的 token
- 辅助损失居高不下
- 模型质量下降

修复：增大 aux_loss_coef、加入抖动噪声、检查初始化。

**陷阱 2：容量溢出**

丢弃的 token 太多。迹象：

- 丢弃率高（>10%）
- 训练损失不稳定
- 梯度方差增大

修复：增大 capacity_factor、减小批大小、增加专家数。

**陷阱 3：AlltoAll 瓶颈**

通信主导计算。迹象：

- GPU 利用率低
- 训练比预期慢得多
- AlltoAll 占步时 >50%

修复：增大每批 token 数、减小 EP 规模、改善网络。

**陷阱 4：专家利用不足**

部分专家很少被使用。迹象：

- 负载不均衡指标显示倾斜
- 部分专家的参数几乎不更新
- 模型容量被浪费

修复：专家选择路由、增大 aux_loss_coef、随机路由正则化。

## 完整的 MoE Transformer 块

```python
class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts FFN.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        expert_group: Optional[ProcessGroup] = None,
        use_moe: bool = True  # Some layers can be dense
    ):
        super().__init__()

        # Attention
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # FFN (MoE or dense)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        if use_moe:
            self.ffn = DistributedMoE(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                expert_group=expert_group
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim)
            )

        self.use_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            output, aux_loss
        """
        # Attention block
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = residual + x

        # FFN block
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        # Get auxiliary loss if MoE
        aux_loss = self.ffn.get_aux_loss() if self.use_moe else 0.0

        return x, aux_loss

class MoETransformer(nn.Module):
    """
    Full MoE Transformer with alternating dense and MoE layers.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        moe_frequency: int = 2,  # Every Nth layer is MoE
        expert_group: Optional[ProcessGroup] = None,
        **moe_kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MoETransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                expert_group=expert_group,
                use_moe=(i % moe_frequency == moe_frequency - 1),
                **moe_kwargs
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning output and total auxiliary loss.
        """
        total_aux_loss = 0.0

        for layer in self.layers:
            x, aux_loss = layer(x, attention_mask)
            total_aux_loss = total_aux_loss + aux_loss

        return x, total_aux_loss
```

## 练习题

1. **AlltoAll 通信量分析**：8 个专家分布在 8 块 GPU 上，每块 GPU 1024 个 token，隐藏维度 4096，计算 AlltoAll 分发通信量 (a) 假设均匀分布，(b) 如果 50% 的 token 都去专家 0。

??? success "解答"
    **已知：**

    - 专家：$E = 8$，分布在 8 块 GPU 上
    - 每 GPU token 数：$T = 1024$
    - 隐藏维度：$H = 4096$
    - 假设使用 bf16（每元素 2 字节）

    **MoE 中的 AlltoAll：**

    每块 GPU 把自己的 token 送给相应的专家 GPU，并接收来自所有其他 GPU、发往本地专家的 token。

    **(a) 均匀分布：**

    均匀路由下，每块 GPU 向每个专家发送 $T/E = 1024/8 = 128$ 个 token。

    **每 GPU 发送量：**

    每块 GPU 向所有 8 块 GPU（包括自己）发送 token：
    - 发往其他 GPU 的 token：$\frac{E-1}{E} \times T = \frac{7}{8} \times 1024 = 896$ 个
    - 通信量：$896 \times H \times 2 = 896 \times 4096 \times 2 = 7.34$ MB

    **AlltoAll 总通信量（仅分发）：**

    $$V_{\text{dispatch}} = 896 \times 4096 \times 2 = \boxed{7.34 \text{ MB per GPU}}$$

    对于完整前向（分发 + 合并）：

    $$V_{\text{total}} = 2 \times V_{\text{dispatch}} = \boxed{14.68 \text{ MB per GPU}}$$

    **(b) 50% 的 token 去专家 0：**

    现在分布是倾斜的：专家 0 拿到 50%，其余专家分剩下的 50%。

    每块 GPU：
    - 发往专家 0：$0.5 \times 1024 = 512$ 个 token
    - 发往其他每个专家：各 $\frac{0.5 \times 1024}{7} = 73$ 个 token

    **GPU 0（托管专家 0）的发送量：**

    GPU 0 发给自己 512 个 token（不经过网络），发给其他 GPU $7 \times 73 = 511$ 个：

    $$V_{\text{GPU0 send}} = 511 \times 4096 \times 2 = 4.19 \text{ MB}$$

    **GPU $i \neq 0$ 的发送量：**

    发给 GPU 0 512 个 token，发给其余 6 块 GPU（不含自己）各 73 个：

    $$V_{\text{GPUi send}} = (512 + 6 \times 73) \times 4096 \times 2 = 950 \times 4096 \times 2 = 7.78 \text{ MB}$$

    **GPU 0 的接收量：**

    接收来自其余 7 块 GPU 各 512 个 token：

    $$V_{\text{GPU0 recv}} = 7 \times 512 \times 4096 \times 2 = 29.4 \text{ MB}$$

    **对比：**

    | 分布 | 发送量（每 GPU 平均） | 接收量（GPU 0） |
    |--------------|--------------------------|------------------------|
    | 均匀 | 7.34 MB | 7.34 MB |
    | 50% 去专家 0 | 7.34 MB | 29.4 MB |

    **关键洞见：** 倾斜路由会造成热点——专家 0 收到的数据多 4 倍，成为瓶颈。

    $$\boxed{\text{Uniform: 14.68 MB total, Skewed: GPU 0 receives 4× more}}$$

2. **容量因子选择**：你有 64 个 token、8 个专家、top_k=2，观察到 capacity_factor=1.25 时丢弃率为 10%。capacity_factor 取多少才能把丢弃降到 <1%？

??? success "解答"
    **已知：**

    - token 总数：$T = 64$
    - 专家数：$E = 8$
    - Top-k：$k = 2$
    - 当前容量因子：$c = 1.25$
    - 当前丢弃率：10%

    **容量计算：**

    专家容量 = 每个专家能处理的 token 数：

    $$C = c \times \frac{T \times k}{E} = 1.25 \times \frac{64 \times 2}{8} = 1.25 \times 16 = 20 \text{ tokens}$$

    **理解丢弃：**

    token-专家分配总数：$T \times k = 64 \times 2 = 128$

    若均匀分布，每个专家得到 $128/8 = 16$ 个分配——不会丢弃。

    丢弃率 10% 即丢弃 12.8 个分配，意味着有些专家收到的超过容量 20。

    **对分布建模：**

    当专家负载超过容量时就会发生丢弃。对于容量因子 $c$：

    $$\text{Max load per expert} = c \times \frac{Tk}{E}$$

    丢弃率取决于负载分布的方差。假设近似二项分布：

    - 每专家期望负载：$\mu = Tk/E = 16$
    - 标准差：$\sigma \approx \sqrt{Tk/E} = 4$（粗略近似）

    对于 $c = 1.25$（容量 = 20）时的 10% 丢弃：

    超过 20 的尾部包含约 10% 的概率质量。

    **目标：丢弃率 <1%**

    我们需要容量满足 P(负载 > 容量) < 1%。

    使用正态近似：
    - 对 10% 尾部：$z_{0.10} \approx 1.28$，因此 $20 \approx 16 + 1.28 \times \sigma$
    - 得到 $\sigma \approx 3.1$

    对 1% 尾部：$z_{0.01} \approx 2.33$
    - 所需容量：$16 + 2.33 \times 3.1 \approx 23.2$ 个 token

    **所需容量因子：**

    $$c_{\text{new}} = \frac{C_{\text{new}}}{Tk/E} = \frac{23.2}{16} = \boxed{1.45}$$

    **实践建议：**

    为稳妥起见，使用 $c = 1.5$ 或略高。

    | 容量因子 | 每专家容量 | 预期丢弃率 |
    |-----------------|--------------------|--------------------|
    | 1.25 | 20 | ~10% |
    | 1.45 | 23.2 | ~1% |
    | 1.50 | 24 | <1% |
    | 2.00 | 32 | ~0% |

    **权衡：** 更高的容量因子意味着每个专家缓冲区占用更多内存：

    $$\text{Memory increase} = \frac{1.50}{1.25} = 1.2\times \text{ (20% more memory)}$$

    **替代方案：专家选择路由**

    使用专家选择时，每个专家恰好选择 $C$ 个 token，从设计上保证 0% 丢弃。这通常比增大容量因子更可取。

3. **负载均衡损失**：推导辅助负载均衡损失对路由器 logits 的梯度。说明梯度会推向均匀的专家选择。

??? success "解答"
    **设定：**

    设 $g_i = \text{softmax}(z)_i$ 是专家 $i$ 的路由概率，其中 $z$ 是路由器 logits。

    **辅助损失定义（Switch Transformer 风格）：**

    $$L_{\text{aux}} = E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

    其中：
    - $f_i$：路由到专家 $i$ 的 token 占比（离散量）
    - $p_i$：专家 $i$ 的平均路由概率
    - $E$：专家数

    在梯度计算中，我们用 $p_i$ 作为 $f_i$ 的可微代理：

    $$L_{\text{aux}} \approx E \cdot \sum_{i=1}^{E} p_i^2$$

    其中 $p_i = \frac{1}{T} \sum_{t=1}^{T} g_{t,i}$ 是所有 token 上的平均概率。

    **梯度推导：**

    对单个 token（logits $z$、概率 $g = \text{softmax}(z)$）：

    $$\frac{\partial L_{\text{aux}}}{\partial z_j} = \frac{\partial L_{\text{aux}}}{\partial g} \cdot \frac{\partial g}{\partial z_j}$$

    **第 1 步：对路由概率的梯度**

    $$\frac{\partial L_{\text{aux}}}{\partial g_i} = \frac{\partial}{\partial g_i} \left( E \cdot \sum_k p_k^2 \right) = 2E \cdot p_i \cdot \frac{\partial p_i}{\partial g_i} = \frac{2E \cdot p_i}{T}$$

    **第 2 步：Softmax 雅可比**

    $$\frac{\partial g_i}{\partial z_j} = g_i(\delta_{ij} - g_j)$$

    **第 3 步：链式法则**

    $$\frac{\partial L_{\text{aux}}}{\partial z_j} = \sum_i \frac{2E \cdot p_i}{T} \cdot g_i(\delta_{ij} - g_j)$$

    $$= \frac{2E}{T} \left[ p_j \cdot g_j - g_j \sum_i p_i \cdot g_i \right]$$

    $$= \frac{2E \cdot g_j}{T} \left[ p_j - \sum_i p_i \cdot g_i \right]$$

    **解释：**

    专家 $j$ 的梯度正比于：

    $$\nabla_{z_j} L_{\text{aux}} \propto g_j \cdot (p_j - \bar{p})$$

    其中 $\bar{p} = \sum_i p_i \cdot g_i$ 是加权平均负载。

    **为什么这会推向均匀：**

    | 条件 | 梯度符号 | 效果 |
    |-----------|---------------|--------|
    | $p_j > \bar{p}$（过载） | 正 | 减小 $z_j$ → 降低 $g_j$ |
    | $p_j < \bar{p}$（欠载） | 负 | 增大 $z_j$ → 提高 $g_j$ |
    | $p_j = \bar{p}$（均衡） | 零 | 不变 |

    **平衡分析：**

    在平衡点，所有专家的 $\nabla L_{\text{aux}} = 0$：

    $$p_j = \bar{p} \quad \forall j$$

    这意味着所有专家负载相等：所有 $i$ 有 $p_i = 1/E$。

    **直观图示：**

    ```
    Gradient pushes probability mass from overloaded to underloaded experts:

    Before:     After gradient step:
    p₀ ████████     p₀ ██████
    p₁ ██           p₁ ████
    p₂ ██           p₂ ████
    p₃ ████████     p₃ ██████
    ```

    $$\boxed{\nabla_{z_j} L_{\text{aux}} \propto g_j(p_j - \bar{p}) \text{ pushes toward } p_i = 1/E}$$

4. **专家选择的实现**：实现专家选择路由，让每个专家选择自己的 top-C token。证明这保证了完美的负载均衡。

??? success "解答"
    **专家选择路由：**

    不是 token 选择专家（top-k），而是专家选择 token（top-C）。

    **实现：**

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ExpertChoiceRouter(nn.Module):
        """
        Expert Choice routing: each expert selects its top-C tokens.
        Guarantees perfect load balance by construction.
        """
        def __init__(
            self,
            hidden_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0
        ):
            super().__init__()
            self.num_experts = num_experts
            self.capacity_factor = capacity_factor

            # Router: projects tokens to expert scores
            self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        def forward(
            self,
            tokens: torch.Tensor  # [batch, seq, hidden]
        ) -> tuple:
            """
            Returns:
                dispatch_mask: [num_experts, capacity, batch*seq]
                combine_weights: [num_experts, capacity, batch*seq]
                expert_indices: [num_experts, capacity] - which tokens selected
            """
            batch, seq, hidden = tokens.shape
            num_tokens = batch * seq

            # Compute expert capacity
            capacity = int(self.capacity_factor * num_tokens / self.num_experts)

            # Flatten tokens
            flat_tokens = tokens.view(num_tokens, hidden)  # [T, H]

            # Compute routing scores: which tokens does each expert want?
            scores = self.router(flat_tokens)  # [T, E]

            # Transpose: [E, T] - each row is an expert's preference over tokens
            expert_scores = scores.T  # [E, T]

            # Apply softmax over tokens for each expert
            expert_probs = F.softmax(expert_scores, dim=-1)  # [E, T]

            # Each expert selects top-C tokens
            top_values, top_indices = torch.topk(
                expert_probs, k=capacity, dim=-1
            )  # [E, C], [E, C]

            # Create dispatch mask: one-hot encoding of selections
            dispatch_mask = torch.zeros(
                self.num_experts, capacity, num_tokens,
                device=tokens.device
            )
            for e in range(self.num_experts):
                dispatch_mask[e, torch.arange(capacity), top_indices[e]] = 1.0

            # Combine weights: normalized probabilities for selected tokens
            combine_weights = top_values  # [E, C]

            # Renormalize weights per token (a token may be selected by multiple experts)
            token_expert_weights = torch.zeros(num_tokens, device=tokens.device)
            for e in range(self.num_experts):
                token_expert_weights.scatter_add_(
                    0, top_indices[e], top_values[e]
                )

            # Normalize combine weights
            for e in range(self.num_experts):
                normalizer = token_expert_weights[top_indices[e]]
                combine_weights[e] = top_values[e] / (normalizer + 1e-9)

            return dispatch_mask, combine_weights, top_indices

    class ExpertChoiceMoE(nn.Module):
        """MoE layer using Expert Choice routing."""
        def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0
        ):
            super().__init__()
            self.router = ExpertChoiceRouter(
                hidden_dim, num_experts, capacity_factor
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, hidden_dim)
                )
                for _ in range(num_experts)
            ])
            self.num_experts = num_experts

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq, hidden = x.shape
            num_tokens = batch * seq
            flat_x = x.view(num_tokens, hidden)

            # Get routing
            dispatch_mask, combine_weights, indices = self.router(x)

            # Initialize output
            output = torch.zeros_like(flat_x)

            # Process each expert
            for e, expert in enumerate(self.experts):
                # Gather tokens for this expert
                expert_tokens = flat_x[indices[e]]  # [C, H]

                # Process through expert
                expert_output = expert(expert_tokens)  # [C, H]

                # Weighted scatter back
                weights = combine_weights[e].unsqueeze(-1)  # [C, 1]
                output.scatter_add_(
                    0,
                    indices[e].unsqueeze(-1).expand(-1, hidden),
                    expert_output * weights
                )

            return output.view(batch, seq, hidden)
    ```

    **完美负载均衡的证明：**

    **定理：** 专家选择路由保证每个专家恰好处理 $C$ 个 token。

    **证明：**

    1. **由构造**：每个专家用 top-k 恰好选择 $C = \lceil \frac{T \cdot c}{E} \rceil$ 个 token。

    2. **容量分配**：共有 $T$ 个 token、$E$ 个专家：

       $$\text{Tokens per expert} = C = \frac{T \cdot c}{E}$$

    3. **总容量**：共有 $E \times C = T \cdot c$ 个槽位。

    4. **无溢出**：与 token 选择中热门专家会溢出不同，每个专家的选择由构造本身限定。

    5. **负载方差**：$\text{Var}(\text{load}) = 0$，因为每个专家恰好得到 $C$ 个 token。

    $$\boxed{\text{Load}_i = C \quad \forall i \in \{1, \ldots, E\}}$$

    **对比：**

    | 方面 | Token 选择（top-k） | 专家选择（top-C） |
    |--------|---------------------|----------------------|
    | 负载均衡 | 需要辅助损失 | 设计上完美均衡 |
    | token 丢弃 | 溢出时可能发生 | 从不发生 |
    | token 覆盖 | 所有 token 都被处理 | 部分 token 可能被跳过 |
    | 梯度流 | 经由路由权重 | 经由选择权重 |

    **关键权衡：** 专家选择可能完全跳过某些 token（如果没有专家选中它们）。可通过对策缓解：
    1. 令 $c > 1$，使总槽位 $> T$
    2. 与一个共享稠密层组合
    3. 用辅助损失鼓励覆盖

5. **通信重叠**：设计一种方案，把 AlltoAll 分发与上一层的注意力计算重叠。约束是什么？

??? success "解答"
    **目标：** 通过重叠把 AlltoAll 的延迟隐藏起来。

    **MoE 层结构：**

    ```
    Layer N:   Attention → MoE (dispatch → experts → combine)
    Layer N+1: Attention → MoE (dispatch → experts → combine)
    ```

    **重叠策略：**

    把第 N+1 层 MoE 的 AlltoAll 分发与第 N+1 层的注意力计算重叠：

    ```
    Timeline:
    ────────────────────────────────────────────────────────────────

    Layer N MoE:    [dispatch]──[experts]──[combine]
                                                    │
                                                    ▼
    Layer N+1:                              [Attention Compute]
                                            ────────────────────
                                                    ▲
    Overlap:                                [AlltoAll dispatch]
                                            (for layer N+1 MoE)
    ```

    **实现：**

    ```python
    import torch
    import torch.distributed as dist
    from typing import Optional

    class OverlappedMoEBlock(nn.Module):
        """
        Transformer block with overlapped AlltoAll and attention.
        """
        def __init__(self, hidden_dim, num_heads, ffn_dim, num_experts):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
            self.moe = DistributedMoE(hidden_dim, ffn_dim, num_experts)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

            # Buffers for async communication
            self.dispatch_buffer = None
            self.dispatch_handle = None

        def start_dispatch(self, x: torch.Tensor):
            """
            Start async AlltoAll dispatch.
            Called before attention to overlap.
            """
            # Compute routing
            routing_weights, expert_indices = self.moe.router(x)

            # Prepare tokens for dispatch
            tokens_to_send = self.moe.prepare_dispatch(x, expert_indices)

            # Allocate receive buffer
            self.dispatch_buffer = torch.empty_like(tokens_to_send)

            # Start async AlltoAll
            self.dispatch_handle = dist.all_to_all_single(
                self.dispatch_buffer,
                tokens_to_send,
                async_op=True  # Non-blocking!
            )

            return routing_weights, expert_indices

        def finish_dispatch_and_compute(
            self,
            routing_weights,
            expert_indices
        ):
            """
            Wait for dispatch and run experts.
            Called after attention completes.
            """
            # Wait for AlltoAll to complete
            self.dispatch_handle.wait()

            # Run tokens through local experts
            expert_outputs = self.moe.run_experts(self.dispatch_buffer)

            # AlltoAll combine (could also be overlapped with next layer)
            combined = self.moe.combine(expert_outputs)

            return combined

        def forward(
            self,
            x: torch.Tensor,
            prefetched_routing: Optional[tuple] = None
        ):
            """
            Forward with optional overlapped dispatch from previous call.
            """
            # Attention block
            residual = x
            x = self.norm1(x)

            # Start next dispatch while computing attention
            next_routing = self.start_dispatch(x)

            # Compute attention (overlapped with dispatch)
            x, _ = self.attention(x, x, x)
            x = residual + x

            # Now finish the MoE from the *current* routing
            if prefetched_routing is not None:
                moe_out = self.finish_dispatch_and_compute(*prefetched_routing)
                residual = x
                x = self.norm2(x)
                x = residual + moe_out

            return x, next_routing

    class OverlappedMoEModel(nn.Module):
        """Model with pipelined AlltoAll overlap."""
        def __init__(self, num_layers, hidden_dim, num_heads, ffn_dim, num_experts):
            super().__init__()
            self.layers = nn.ModuleList([
                OverlappedMoEBlock(hidden_dim, num_heads, ffn_dim, num_experts)
                for _ in range(num_layers)
            ])

        def forward(self, x):
            routing = None

            for layer in self.layers:
                x, routing = layer(x, prefetched_routing=routing)

            # Handle final MoE
            if routing is not None:
                x = self.layers[-1].finish_dispatch_and_compute(*routing)

            return x
    ```

    **约束：**

    | 约束 | 原因 | 缓解 |
    |------------|--------|------------|
    | **注意力时间 > AlltoAll 时间** | 否则分发无法被完全隐藏 | 增大批大小或减小 EP |
    | **缓冲区内存** | 需要同时持有发送 + 接收缓冲区 | 预留额外内存 |
    | **路由需提前计算** | 必须在注意力前知道目的地 | 增加依赖关系的复杂性 |
    | **反向传播复杂性** | 梯度要穿过异步操作 | 谨慎使用梯度检查点 |
    | **无数据依赖** | 分发的输入不能依赖注意力输出 | 层架构的约束 |

    **重叠何时有效：**

    $$T_{\text{attention}} \geq T_{\text{AlltoAll}}$$

    对于典型配置：
    - 注意力：$O(S^2 \cdot H)$ 计算
    - AlltoAll：$O(T \cdot H / \beta)$ 通信

    **可达的重叠比例：**

    $$\text{Overlap fraction} = \min\left(1, \frac{T_{\text{attention}}}{T_{\text{AlltoAll}}}\right)$$

    如果注意力耗时 10ms、AlltoAll 耗时 5ms：

    $$\text{Overlap} = 100\% \text{ (fully hidden)}$$

    如果注意力耗时 5ms、AlltoAll 耗时 10ms：

    $$\text{Overlap} = 50\% \text{ (5ms exposed)}$$

    $$\boxed{\text{Overlap requires: } T_{\text{compute}} > T_{\text{comm}} \text{ and no data dependencies}}$$

6. **三维并行组**：world_size=64，TP=4，EP=8，DP=2，枚举 rank 13 的所有进程组。

??? success "解答"
    **已知：**

    - World size：64 块 GPU
    - 张量并行（TP）：4
    - 专家并行（EP）：8
    - 数据并行（DP）：2

    **验证：** $TP \times EP \times DP = 4 \times 8 \times 2 = 64$ ✓

    **组布局：**

    标准布局是：变化最快 → 变化最慢 = TP → EP → DP

    ```
    Rank = dp_id * (TP * EP) + ep_id * TP + tp_id
    ```

    对于 rank 13：
    ```
    rank = 13
    TP * EP = 4 * 8 = 32

    dp_id = 13 // 32 = 0
    remainder = 13 % 32 = 13
    ep_id = 13 // 4 = 3
    tp_id = 13 % 4 = 1
    ```

    **Rank 13 的坐标：** $(tp=1, ep=3, dp=0)$

    **张量并行组：**

    所有具有相同 $(ep, dp)$、变化 $tp$ 的 rank：

    $$\text{TP group for rank 13} = \{dp=0, ep=3, tp=0,1,2,3\}$$

    Rank：$0 \times 32 + 3 \times 4 + \{0,1,2,3\} = \{12, 13, 14, 15\}$

    $$\boxed{\text{TP group: } \{12, 13, 14, 15\}}$$

    **专家并行组：**

    所有具有相同 $(tp, dp)$、变化 $ep$ 的 rank：

    $$\text{EP group for rank 13} = \{dp=0, tp=1, ep=0,1,2,...,7\}$$

    Rank：$0 \times 32 + \{0,1,...,7\} \times 4 + 1 = \{1, 5, 9, 13, 17, 21, 25, 29\}$

    $$\boxed{\text{EP group: } \{1, 5, 9, 13, 17, 21, 25, 29\}}$$

    **数据并行组：**

    所有具有相同 $(tp, ep)$、变化 $dp$ 的 rank：

    $$\text{DP group for rank 13} = \{tp=1, ep=3, dp=0,1\}$$

    Rank：$\{0,1\} \times 32 + 3 \times 4 + 1 = \{13, 45\}$

    $$\boxed{\text{DP group: } \{13, 45\}}$$

    **Rank 13 汇总：**

    | 组 | 大小 | 成员 |
    |-------|------|---------|
    | 张量并行 | 4 | {12, 13, 14, 15} |
    | 专家并行 | 8 | {1, 5, 9, 13, 17, 21, 25, 29} |
    | 数据并行 | 2 | {13, 45} |

    **验证代码：**

    ```python
    def get_groups_for_rank(rank, tp_size, ep_size, dp_size):
        """Compute all process groups for a given rank."""
        tp_ep = tp_size * ep_size

        # Decompose rank into coordinates
        dp_id = rank // tp_ep
        remainder = rank % tp_ep
        ep_id = remainder // tp_size
        tp_id = remainder % tp_size

        print(f"Rank {rank}: tp_id={tp_id}, ep_id={ep_id}, dp_id={dp_id}")

        # TP group: same (ep_id, dp_id), vary tp_id
        tp_group = [dp_id * tp_ep + ep_id * tp_size + t for t in range(tp_size)]

        # EP group: same (tp_id, dp_id), vary ep_id
        ep_group = [dp_id * tp_ep + e * tp_size + tp_id for e in range(ep_size)]

        # DP group: same (tp_id, ep_id), vary dp_id
        dp_group = [d * tp_ep + ep_id * tp_size + tp_id for d in range(dp_size)]

        return tp_group, ep_group, dp_group

    tp, ep, dp = get_groups_for_rank(13, tp_size=4, ep_size=8, dp_size=2)
    # Output:
    # Rank 13: tp_id=1, ep_id=3, dp_id=0
    # TP group: [12, 13, 14, 15]
    # EP group: [1, 5, 9, 13, 17, 21, 25, 29]
    # DP group: [13, 45]
    ```

    **可视化：**

    ```
    DP=0 (ranks 0-31):                    DP=1 (ranks 32-63):
    ┌────┬────┬────┬────┬────┬────┬────┬────┐  ┌────┬────┬────┬────┬────┬────┬────┬────┐
    │EP0 │EP1 │EP2 │EP3 │EP4 │EP5 │EP6 │EP7 │  │EP0 │EP1 │EP2 │EP3 │EP4 │EP5 │EP6 │EP7 │
    ├────┼────┼────┼────┼────┼────┼────┼────┤  ├────┼────┼────┼────┼────┼────┼────┼────┤
    │0-3 │4-7 │8-11│12- │16- │20- │24- │28- │  │32- │36- │40- │44- │48- │52- │56- │60- │
    │    │    │    │15  │19  │23  │27  │31  │  │35  │39  │43  │47  │51  │55  │59  │63  │
    └────┴────┴────┴────┴────┴────┴────┴────┘  └────┴────┴────┴────┴────┴────┴────┴────┘

    Rank 13 is in EP3 block (ranks 12-15) at DP=0
    - TP group: ranks 12, 13, 14, 15 (same EP block)
    - EP group: rank 1,5,9,13,17,21,25,29 (tp_id=1 across all EP blocks)
    - DP group: ranks 13, 45 (same position in each DP replica)
    ```

7. **路由塌缩检测**：设计一个监控系统，在训练早期检测路由塌缩。你会跟踪哪些指标，什么阈值会触发告警？

??? success "解答"
    **路由塌缩**指路由器学会把大多数 token 送到一小部分专家，浪费模型容量。

    **需要跟踪的关键指标：**

    | 指标 | 公式 | 健康范围 |
    |--------|---------|---------------|
    | 专家利用率熵 | $H = -\sum_i p_i \log p_i$ | $> 0.9 \times \log(E)$ |
    | 最大专家负载 | $\max_i(\text{tokens}_i) / \text{avg}$ | $< 2.0$ |
    | 最小专家负载 | $\min_i(\text{tokens}_i) / \text{avg}$ | $> 0.3$ |
    | 负载基尼系数 | $G = \frac{\sum_{i,j}|l_i - l_j|}{2n\sum_i l_i}$ | $< 0.3$ |
    | 丢弃 token 率 | $\text{dropped} / \text{total}$ | $< 5\%$ |
    | 路由权重熵 | 每 token 的 softmax 熵 | $> 1.0$ |

    **监控系统设计：**

    ```python
    import numpy as np
    from collections import deque
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class CollapseAlert:
        severity: str  # "warning", "critical"
        metric: str
        value: float
        threshold: float
        step: int

    class RoutingCollapseMonitor:
        def __init__(self, num_experts: int, window_size: int = 100):
            self.num_experts = num_experts
            self.max_entropy = np.log(num_experts)
            self.window_size = window_size

            # Rolling windows for trend detection
            self.entropy_history = deque(maxlen=window_size)
            self.gini_history = deque(maxlen=window_size)
            self.drop_history = deque(maxlen=window_size)

            # Thresholds
            self.thresholds = {
                "entropy_warning": 0.85 * self.max_entropy,
                "entropy_critical": 0.70 * self.max_entropy,
                "gini_warning": 0.35,
                "gini_critical": 0.50,
                "max_load_warning": 2.5,
                "max_load_critical": 4.0,
                "drop_rate_warning": 0.05,
                "drop_rate_critical": 0.15,
                "trend_threshold": -0.01,  # Entropy decreasing
            }

        def compute_metrics(self, expert_counts: np.ndarray,
                          dropped: int, total: int) -> dict:
            """Compute all routing health metrics."""
            # Normalize to probabilities
            probs = expert_counts / expert_counts.sum()
            probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)

            # Entropy (higher = more balanced)
            entropy = -np.sum(probs * np.log(probs))

            # Gini coefficient (lower = more balanced)
            sorted_loads = np.sort(expert_counts)
            n = len(sorted_loads)
            cumsum = np.cumsum(sorted_loads)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_loads)) /
                   (n * cumsum[-1])) - (n + 1) / n

            # Load imbalance
            avg_load = expert_counts.mean()
            max_load_ratio = expert_counts.max() / avg_load
            min_load_ratio = expert_counts.min() / avg_load

            # Drop rate
            drop_rate = dropped / total if total > 0 else 0

            return {
                "entropy": entropy,
                "normalized_entropy": entropy / self.max_entropy,
                "gini": gini,
                "max_load_ratio": max_load_ratio,
                "min_load_ratio": min_load_ratio,
                "drop_rate": drop_rate,
                "expert_counts": expert_counts,
            }

        def check_alerts(self, metrics: dict, step: int) -> list[CollapseAlert]:
            """Check metrics against thresholds."""
            alerts = []
            t = self.thresholds

            # Entropy checks
            if metrics["entropy"] < t["entropy_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "entropy", metrics["entropy"],
                    t["entropy_critical"], step
                ))
            elif metrics["entropy"] < t["entropy_warning"]:
                alerts.append(CollapseAlert(
                    "warning", "entropy", metrics["entropy"],
                    t["entropy_warning"], step
                ))

            # Gini checks
            if metrics["gini"] > t["gini_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "gini", metrics["gini"],
                    t["gini_critical"], step
                ))
            elif metrics["gini"] > t["gini_warning"]:
                alerts.append(CollapseAlert(
                    "warning", "gini", metrics["gini"],
                    t["gini_warning"], step
                ))

            # Max load checks
            if metrics["max_load_ratio"] > t["max_load_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "max_load_ratio",
                    metrics["max_load_ratio"],
                    t["max_load_critical"], step
                ))

            # Drop rate checks
            if metrics["drop_rate"] > t["drop_rate_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "drop_rate", metrics["drop_rate"],
                    t["drop_rate_critical"], step
                ))

            # Trend detection (entropy declining over time)
            if len(self.entropy_history) >= 50:
                recent = list(self.entropy_history)[-50:]
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                if slope < t["trend_threshold"]:
                    alerts.append(CollapseAlert(
                        "warning", "entropy_trend", slope,
                        t["trend_threshold"], step
                    ))

            return alerts

        def step(self, expert_counts: np.ndarray,
                dropped: int, total: int, step: int):
            """Process one step and return any alerts."""
            metrics = self.compute_metrics(expert_counts, dropped, total)

            # Update history
            self.entropy_history.append(metrics["entropy"])
            self.gini_history.append(metrics["gini"])
            self.drop_history.append(metrics["drop_rate"])

            return self.check_alerts(metrics, step), metrics
    ```

    **告警阈值汇总：**

    | 指标 | 警告 | 严重 | 行动 |
    |--------|---------|----------|--------|
    | 归一化熵 | $< 0.85$ | $< 0.70$ | 提高辅助损失权重 |
    | 基尼系数 | $> 0.35$ | $> 0.50$ | 加入抖动，检查路由器初始化 |
    | 最大负载比 | $> 2.5$ | $> 4.0$ | 降低容量因子 |
    | 丢弃率 | $> 5\%$ | $> 15\%$ | 提高容量因子 |
    | 熵趋势 | 下降 | - | 及早干预 |

    **早期预警信号：**

    1. **前 1000 步熵下降**：通常表明路由器初始化有问题
    2. **早期一个专家独大**：用更大方差重置路由器权重
    3. **负载震荡**：辅助损失权重太高，导致过度校正
    4. **稳定期后逐渐塌缩**：路由器学习率过高

    **建议的响应动作：**

    - **警告**：记录到仪表盘，继续监控
    - **严重（熵）**：把 `aux_loss_weight` 提高 2×，给路由器加噪声
    - **严重（丢弃）**：把容量因子提高 0.2
    - **严重（不均衡）**：考虑切换到专家选择路由

8. **梯度分析**：在一个 top_k=2 的模型中，专家 0 处理 60% 的 token，专家 1 处理 40%。分析辅助损失梯度如何影响这些比例。

??? success "解答"
    **设定：**

    - top_k = 2（每个 token 路由到 2 个专家）
    - 专家 0：接收 60% 的 token → $f_0 = 0.60$
    - 专家 1：接收 40% 的 token → $f_1 = 0.40$
    - 为简化假设只有 2 个专家（可推广到更多）

    **辅助损失表述：**

    负载均衡辅助损失为：

    $$\mathcal{L}_{aux} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

    其中：

    - $E$ = 专家数（此处为 2）
    - $f_i$ = 路由到专家 $i$ 的 token 占比
    - $p_i$ = 专家 $i$ 的平均路由概率（softmax 输出的均值）
    - $\alpha$ = 辅助损失系数

    **梯度分析：**

    对路由器 logits $z_{t,i}$（token $t$、专家 $i$）的梯度：

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_{t,i}} = \alpha \cdot E \cdot \sum_j f_j \cdot \frac{\partial p_j}{\partial z_{t,i}}$$

    对 softmax：$p_i = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

    导数：

    $$\frac{\partial p_j}{\partial z_{t,i}} = p_i(\delta_{ij} - p_j) / T$$

    其中 $T$ = token 数，$\delta_{ij}$ 是 Kronecker delta。

    **简化梯度（每 token）：**

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_{t,i}} = \alpha \cdot E \cdot p_i \left(f_i - \sum_j f_j p_j\right)$$

    令 $\bar{f} = \sum_j f_j p_j$（加权平均负载）：

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_{t,i}} = \alpha \cdot E \cdot p_i (f_i - \bar{f})$$

    **代入本例：**

    已知 $f_0 = 0.60$，$f_1 = 0.40$：

    假设路由概率近似等于负载：$p_0 \approx 0.60$，$p_1 \approx 0.40$

    $$\bar{f} = f_0 \cdot p_0 + f_1 \cdot p_1 = 0.60 \times 0.60 + 0.40 \times 0.40 = 0.52$$

    **梯度：**

    对专家 0（过载）：

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_0} \propto p_0 (f_0 - \bar{f}) = 0.60 \times (0.60 - 0.52) = 0.60 \times 0.08 = \boxed{+0.048}$$

    对专家 1（欠载）：

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_1} \propto p_1 (f_1 - \bar{f}) = 0.40 \times (0.40 - 0.52) = 0.40 \times (-0.12) = \boxed{-0.048}$$

    **解释：**

    | 专家 | 负载 $f_i$ | 梯度符号 | 效果 |
    |--------|------------|---------------|--------|
    | 专家 0 | 0.60（高） | 正 | 减小 logit → 降低概率 |
    | 专家 1 | 0.40（低） | 负 | 增大 logit → 提高概率 |

    辅助损失把系统推向**均衡**：

    - 过载专家得到正梯度 → 其路由概率下降
    - 欠载专家得到负梯度 → 其路由概率上升

    **平衡分析：**

    在平衡点，$f_0 = f_1 = 0.50$（2 个专家时）：

    $$\bar{f} = 0.50 \times 0.50 + 0.50 \times 0.50 = 0.50$$

    梯度变为：

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_i} \propto 0.50 \times (0.50 - 0.50) = 0$$

    **完美均衡处无梯度**——该平衡点是稳定的。

    **带训练动态的数值示例：**

    ```python
    import numpy as np

    def simulate_load_balancing(f0_init=0.60, alpha=0.01, steps=100, lr=0.1):
        """Simulate how aux loss corrects imbalance."""
        f0, f1 = f0_init, 1 - f0_init

        trajectory = [(f0, f1)]

        for _ in range(steps):
            # Assume p ≈ f (router matches current loads)
            p0, p1 = f0, f1
            f_bar = f0 * p0 + f1 * p1

            # Gradients
            grad_0 = alpha * 2 * p0 * (f0 - f_bar)
            grad_1 = alpha * 2 * p1 * (f1 - f_bar)

            # Update (gradient descent on logits affects f)
            # Simplified: treat as direct adjustment
            f0_new = f0 - lr * grad_0
            f1_new = f1 - lr * grad_1

            # Normalize to valid distribution
            total = f0_new + f1_new
            f0, f1 = f0_new / total, f1_new / total

            trajectory.append((f0, f1))

        return trajectory

    # Run simulation
    traj = simulate_load_balancing(f0_init=0.60)
    print(f"Start: f0={traj[0][0]:.3f}, f1={traj[0][1]:.3f}")
    print(f"End:   f0={traj[-1][0]:.3f}, f1={traj[-1][1]:.3f}")
    # Output:
    # Start: f0=0.600, f1=0.400
    # End:   f0=0.500, f1=0.500  (converges to balance)
    ```

    **小结：**

    专家 $i$ 的辅助损失梯度正比于：

    $$\nabla_{z_i} \mathcal{L}_{aux} \propto p_i \cdot (f_i - \bar{f})$$

    - **过载专家**（$f_i > \bar{f}$）：正梯度 → 概率下降
    - **欠载专家**（$f_i < \bar{f}$）：负梯度 → 概率上升
    - **均衡时**：零梯度 → 稳定平衡

    60/40 的不均衡会制造一个纠正力，在训练步中把分布推向 50/50。

## 旋钮与权衡

| 旋钮 | 主要作用 | 代价 |
|---|---|---|
| 每层专家数（E） | 更大容量 | 路由复杂度和负载不均衡 |
| Top-k 路由 | 每 token 计算量固定 | AlltoAll 通信量和分发开销 |
| 容量因子 | 更少丢弃 token | 更高的专家内存和填充开销 |
| 辅助损失权重 | 更好的均衡 | 轻微的优化偏差 |

## 关键要点

1. **稀疏性成就规模**：MoE 模型可以拥有多 10–100 倍的参数，而每 token 计算量保持不变。

2. **AlltoAll 是通信模式**：token 去专家、输出回来——每个 MoE 层需要两次 AlltoAll 操作。

3. **负载均衡至关重要**：没有均衡，路由塌缩会让稀疏性变得毫无用处。使用辅助损失 + 容量限制。

4. **专家选择改善均衡**：让专家选择 token 可以减少塌缩，但仍需容量限制和辅助损失来避免过载。

5. **容量因子在丢弃与内存之间权衡**：更高的容量 = 更少的丢弃 token，但每个专家的内存更多。

6. **组合是自然的**：EP 通过正交的进程组与 DP 和 TP 干净地组合。

7. **梯度经由权重流动**：top-k 选择不可微，但路由权重为路由器提供梯度信号。

8. **路由开销不可忽视**：对于小模型或低延迟需求，路由计算和 AlltoAll 通信可能占主导。