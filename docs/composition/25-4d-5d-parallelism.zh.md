---
title: "四维与五维并行"
subtitle: "上下文并行与专家并行的混入"
---

<div class="chapter-opener" markdown>
当上下文窗口增长到数百万 token、稀疏混合专家模型达到数万亿参数时，三维并行会撞上新的墙。第四维——上下文并行（Context Parallelism）——处理长序列。第五维——专家并行（Expert Parallelism）——处理稀疏容量。两者合在一起，才让现存的那些最大模型成为可能。
</div>

<div class="investigation-question" markdown>
**问题**：你正在 16,384 块 GPU 上训练一个 1T 参数的 MoE 模型，上下文长度为 128K。三维并行既应付不了序列内存，也应付不了专家路由。你还需要哪些额外的并行维度，它们又如何与已有的三个维度组合？
</div>

## 超越三维：新的约束

### 上下文长度问题

在三维并行（DP × TP × PP）下，激活内存按如下方式增长：

$$M_{\text{act}} = \frac{B \times S \times H}{T} \times L_{\text{stage}}$$

其中 $S$ 是序列长度。当 $S \to 128K$ 乃至更大：

| 序列长度 | 激活内存（每层，TP=8） |
|-----------------|-------------------------------------|
| 2K | 0.4 GB |
| 8K | 1.6 GB |
| 32K | 6.4 GB |
| 128K | 25.6 GB |
| 1M | 200 GB |

即便用 TP=8 和 PP=16，128K 的序列仅激活值就会耗尽 GPU 内存。

### 专家扩展问题

混合专家模型的特性不同：

**稠密模型**：所有参数对所有 token 都激活。

**MoE 模型**：每个 token 只激活 $E$ 个专家中的 $k$ 个，但所有专家都必须存储。

$$M_{\text{experts}} = \frac{E \times \text{ExpertSize}}{T \times P}$$

对于一个有 128 个专家的 1T MoE：

$$M_{\text{experts}} = \frac{128 \times 8\text{B} \times 2}{8 \times 16} = 16\text{ GB just for expert parameters}$$

而且路由会造成动态的负载不均衡。

## 第四维：上下文并行

上下文并行（context parallelism, CP）把序列维度切分到各设备上。

### CP 的工作方式

```
Without CP:
GPU 0: [Token 0, Token 1, ..., Token 128K]  ← OOM

With CP=4:
GPU 0: [Token 0, ..., Token 32K]
GPU 1: [Token 32K, ..., Token 64K]
GPU 2: [Token 64K, ..., Token 96K]
GPU 3: [Token 96K, ..., Token 128K]
```

每块 GPU 处理序列的 1/CP。

### CP 的通信模式

注意力要求所有 token 彼此相看：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

在序列切分的情况下，这需要对 K 和 V 做 AllGather：

```
Forward pass:
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ Q0  │  │ Q1  │  │ Q2  │  │ Q3  │  Local queries
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │
   └────────┴────────┴────────┘
          AllGather K,V
   ┌────────┬────────┬────────┐
   │        │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│K0-3 │  │K0-3 │  │K0-3 │  │K0-3 │  Full K,V on each
│V0-3 │  │V0-3 │  │V0-3 │  │V0-3 │
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│Attn │  │Attn │  │Attn │  │Attn │  Local attention
│ O0  │  │ O1  │  │ O2  │  │ O3  │
└─────┘  └─────┘  └─────┘  └─────┘
```

### 环形注意力优化

不用 AllGather（内存开销大），而用环形注意力（ring attention）：

```python
from typing import Optional
import torch
import torch.distributed as dist

class RingAttention:
    """Memory-efficient ring attention for context parallelism."""

    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        cp_size: int,
        causal: bool = True
    ):
        self.cp_group = cp_group
        self.cp_size = cp_size
        self.causal = causal
        self.cp_rank = dist.get_rank(cp_group)

    def forward(
        self,
        q: torch.Tensor,  # [batch, seq_local, heads, dim]
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Ring attention forward pass."""
        batch, seq_local, heads, dim = q.shape

        # Initialize output accumulator
        output = torch.zeros_like(q)
        normalizer = torch.zeros(batch, seq_local, heads, 1, device=q.device)
        running_max = torch.full_like(normalizer, float('-inf'))

        # Ring buffers
        k_recv = torch.empty_like(k)
        v_recv = torch.empty_like(v)

        k_send = k.contiguous()
        v_send = v.contiguous()

        for step in range(self.cp_size):
            # Compute which KV chunk we have
            kv_rank = (self.cp_rank - step) % self.cp_size

            # Compute attention scores for this KV chunk
            scores = torch.einsum('bshd,bShd->bshS', q, k_send)
            scores = scores / (dim ** 0.5)

            # Apply causal mask if needed
            if self.causal:
                scores = self._apply_causal_mask(
                    scores, step, kv_rank
                )

            # Online softmax update
            max_scores = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - max_scores)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)

            # Update output with this chunk's contribution
            chunk_output = torch.einsum('bshS,bShd->bshd', exp_scores, v_send)

            # Numerically stable accumulation
            output, normalizer, running_max = self._online_softmax_update(
                output, normalizer, running_max, chunk_output, sum_exp, max_scores
            )

            # Ring communication (except last step)
            if step < self.cp_size - 1:
                # Send to next, receive from prev
                next_rank = (self.cp_rank + 1) % self.cp_size
                prev_rank = (self.cp_rank - 1) % self.cp_size

                send_ops = [
                    dist.P2POp(dist.isend, k_send, next_rank, self.cp_group),
                    dist.P2POp(dist.isend, v_send, next_rank, self.cp_group),
                ]
                recv_ops = [
                    dist.P2POp(dist.irecv, k_recv, prev_rank, self.cp_group),
                    dist.P2POp(dist.irecv, v_recv, prev_rank, self.cp_group),
                ]

                reqs = dist.batch_isend_irecv(send_ops + recv_ops)
                for req in reqs:
                    req.wait()

                # Swap buffers
                k_send, k_recv = k_recv, k_send
                v_send, v_recv = v_recv, v_send

        # Final normalization
        output = output / normalizer

        return output

    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        step: int,
        kv_rank: int
    ) -> torch.Tensor:
        """Apply causal masking for ring attention."""
        batch, seq_q, heads, seq_kv = scores.shape

        # Query positions: [cp_rank * seq_local, (cp_rank+1) * seq_local)
        # KV positions: [kv_rank * seq_local, (kv_rank+1) * seq_local)

        if kv_rank > self.cp_rank:
            # All KV positions are in the future - mask all
            return torch.full_like(scores, float('-inf'))
        elif kv_rank < self.cp_rank:
            # All KV positions are in the past - no masking
            return scores
        else:
            # Same chunk - standard causal mask
            mask = torch.triu(
                torch.ones(seq_q, seq_kv, device=scores.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            return scores

    def _online_softmax_update(
        self,
        output: torch.Tensor,
        normalizer: torch.Tensor,
        running_max: torch.Tensor,
        chunk_output: torch.Tensor,
        chunk_sum: torch.Tensor,
        chunk_max: torch.Tensor
    ) -> tuple:
        """Online softmax accumulation with running max for stability."""
        # First chunk
        if torch.isinf(running_max).all():
            return chunk_output, chunk_sum, chunk_max

        # Update running max
        new_max = torch.maximum(running_max, chunk_max)

        # Rescale old and new contributions
        scale_old = torch.exp(running_max - new_max)
        scale_new = torch.exp(chunk_max - new_max)

        new_output = output * scale_old + chunk_output * scale_new
        new_normalizer = normalizer * scale_old + chunk_sum * scale_new

        return new_output, new_normalizer, new_max
```

### CP 的内存节省

上下文并行带来的内存下降：

| 组件 | 不使用 CP | 使用 CP=C |
|-----------|-----------|-----------|
| 激活值 | $B \times S \times H$ | $B \times S/C \times H$ |
| KV 缓存 | $2 \times L \times B \times S \times H$ | $2 \times L \times B \times \frac{S}{C} \times H$（+ 环形缓冲） |
| 峰值内存 | 注意力需 $O(S^2)$ | 环形下为 $O(S \times S/C)$ |

**关键洞见**：环形注意力避免了对注意力矩阵的 $O(S^2)$ 内存需求。

## 四维并行：DP × TP × PP × CP

### 网格配置

```
4D Mesh: shape = (D, P, T, C)

Example: 1024 GPUs for 100B model with 128K context
- DP = 4   (data parallel replicas)
- PP = 8   (pipeline stages)
- TP = 8   (tensor parallel)
- CP = 4   (context parallel)

Total: 4 × 8 × 8 × 4 = 1024
```

### 进程组结构

```python
from dataclasses import dataclass
from typing import Tuple, Optional
import torch.distributed as dist
import numpy as np

@dataclass
class FourDConfig:
    """Configuration for 4D parallelism."""
    dp_size: int
    pp_size: int
    tp_size: int
    cp_size: int

    @property
    def world_size(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size * self.cp_size

class FourDMesh:
    """Device mesh for 4D parallelism: DP × PP × TP × CP."""

    def __init__(self, config: FourDConfig):
        self.config = config

        world_size = config.world_size
        devices = np.arange(world_size).reshape(
            config.dp_size,
            config.pp_size,
            config.tp_size,
            config.cp_size
        )

        self.mesh = devices

        # Get my coordinates
        rank = dist.get_rank()
        coords = np.argwhere(devices == rank)[0]
        self.dp_rank = coords[0]
        self.pp_rank = coords[1]
        self.tp_rank = coords[2]
        self.cp_rank = coords[3]

        # Create process groups
        self._create_groups()

    def _create_groups(self):
        """Create all necessary process groups."""
        # DP group: vary DP, fix others
        dp_ranks = self.mesh[:, self.pp_rank, self.tp_rank, self.cp_rank].tolist()
        self.dp_group = dist.new_group(dp_ranks)

        # PP group: fix DP, vary PP, fix TP, fix CP
        pp_ranks = self.mesh[self.dp_rank, :, self.tp_rank, self.cp_rank].tolist()
        self.pp_group = dist.new_group(pp_ranks)

        # TP group: fix DP, fix PP, vary TP, fix CP
        tp_ranks = self.mesh[self.dp_rank, self.pp_rank, :, self.cp_rank].tolist()
        self.tp_group = dist.new_group(tp_ranks)

        # CP group: fix DP, fix PP, fix TP, vary CP
        cp_ranks = self.mesh[self.dp_rank, self.pp_rank, self.tp_rank, :].tolist()
        self.cp_group = dist.new_group(cp_ranks)

        # TP-CP group (for some fused operations): vary both TP and CP
        tp_cp_ranks = self.mesh[self.dp_rank, self.pp_rank, :, :].flatten().tolist()
        self.tp_cp_group = dist.new_group(tp_cp_ranks)

    def get_ranks(self) -> Tuple[int, int, int, int]:
        """Return (dp_rank, pp_rank, tp_rank, cp_rank)."""
        return (self.dp_rank, self.pp_rank, self.tp_rank, self.cp_rank)
```

### 四维通信分析

四维并行中的通信模式：

| 维度 | 操作 | 频率 | 通信量 |
|-----------|-----------|-----------|--------|
| TP | AllReduce | 每层 | $2 \times \frac{T-1}{T} \times H \times B \times S/C$ |
| CP | 环形 P2P | 每次注意力 | $2 \times B \times S/C \times H$（K+V） |
| PP | P2P Send/Recv | 每个微批 | $B \times S/C \times H$ |
| DP | AllReduce | 每步 | $2 \times \frac{D-1}{D} \times \text{Params}/(T \times P)$ |

### 面向硬件的维度排序

把维度映射到硬件拓扑：

```
Optimal for 8-GPU DGX nodes:
┌─────────────────────────────────┐
│         Within Node             │
│  TP (NVLink) + CP (NVLink)      │
│         8 GPUs                  │
├─────────────────────────────────┤
│         Across Nodes            │
│  PP (IB) + DP (IB)              │
│        Many nodes               │
└─────────────────────────────────┘

Recommended: TP × CP ≤ 8 (within NVLink domain)
```

## 第五维：专家并行

专家并行（expert parallelism, EP）把 MoE 专家分布到各设备上。

### MoE 内存分解

对于混合专家层：

```
Standard MoE Layer:

- Router: H → E weights (small)
- Experts: E × (H → 4H → H) (large)
- Typically E = 64-256 experts
```

每个专家的内存：

$$M_{\text{expert}} = 2 \times H \times 4H \times 2 \text{ bytes (FP16)} = 16H^2$$

对于 H=12288、E=128：

$$M_{\text{all\_experts}} = 128 \times 16 \times 12288^2 \approx 310\text{ GB}$$

### EP 的工作方式

```
Without EP (all experts on each GPU):
GPU 0: [Expert 0-127]  ← 310 GB just for experts

With EP=8:
GPU 0: [Expert 0-15]    ← 39 GB
GPU 1: [Expert 16-31]   ← 39 GB
...
GPU 7: [Expert 112-127] ← 39 GB
```

### EP 的通信：AlltoAll

MoE 路由需要 AlltoAll 通信：

```
Token Routing with EP=4:
┌──────────────────────────────────────────┐
│ GPU 0: Tokens [T0, T1, T2, T3]           │
│ Router decides:                          │
│   T0 → Expert 2 (GPU 0)                  │
│   T1 → Expert 5 (GPU 1)                  │
│   T2 → Expert 3 (GPU 0)                  │
│   T3 → Expert 9 (GPU 2)                  │
└──────────────────────────────────────────┘
              │
              ▼ AlltoAll
┌──────────────────────────────────────────┐
│ GPU 0: [T0, T2] → Experts 0-3            │
│ GPU 1: [T1]     → Experts 4-7            │
│ GPU 2: [T3]     → Experts 8-11           │
│ GPU 3: []       → Experts 12-15          │
└──────────────────────────────────────────┘
              │
              ▼ Expert Computation
              │
              ▼ AlltoAll (reverse)
┌──────────────────────────────────────────┐
│ GPU 0: [T0, T1, T2, T3] (results)        │
└──────────────────────────────────────────┘
```

### EP 的实现

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

class ExpertParallelMoE(nn.Module):
    """MoE layer with expert parallelism."""

    def __init__(
        self,
        hidden_dim: int,
        expert_dim: int,
        num_experts: int,
        top_k: int,
        ep_group: dist.ProcessGroup,
        ep_size: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.ep_rank = dist.get_rank(ep_group)

        # Number of local experts
        assert num_experts % ep_size == 0
        self.num_local_experts = num_experts // ep_size

        # Router
        self.router = nn.Linear(hidden_dim, num_experts)

        # Local experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, hidden_dim)
            )
            for _ in range(self.num_local_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert parallelism.

        Args:
            x: [batch * seq, hidden_dim]

        Returns:
            [batch * seq, hidden_dim]
        """
        batch_seq = x.shape[0]

        # Route tokens
        router_logits = self.router(x)  # [batch*seq, num_experts]
        routing_weights, selected_experts = self._route(router_logits)

        # Prepare for AlltoAll
        # Group tokens by destination EP rank
        tokens_per_ep, send_counts = self._prepare_alltoall(
            x, selected_experts
        )

        # AlltoAll: send tokens to their expert's EP rank
        received_tokens, recv_counts = self._alltoall(
            tokens_per_ep, send_counts
        )

        # Process through local experts
        expert_outputs = self._process_local_experts(
            received_tokens, recv_counts
        )

        # AlltoAll: return results
        final_outputs, _ = self._alltoall_reverse(
            expert_outputs, recv_counts, send_counts
        )

        # Combine with routing weights
        output = self._combine_outputs(
            final_outputs, routing_weights, selected_experts
        )

        return output

    def _route(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k routing."""
        routing_weights = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices

    def _prepare_alltoall(
        self,
        tokens: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """Prepare tokens for AlltoAll dispatch."""
        batch_seq = tokens.shape[0]

        # Count tokens going to each EP rank
        send_counts = [0] * self.ep_size

        # Flatten expert selection
        flat_experts = selected_experts.flatten()

        for expert_id in flat_experts:
            ep_rank = expert_id.item() // self.num_local_experts
            send_counts[ep_rank] += 1

        # Sort tokens by destination EP rank
        # (In practice, use more efficient GPU-based sorting)
        sorted_indices = torch.argsort(
            flat_experts // self.num_local_experts
        )

        # Expand tokens for top-k
        expanded_tokens = tokens.unsqueeze(1).expand(-1, self.top_k, -1)
        expanded_tokens = expanded_tokens.reshape(-1, self.hidden_dim)

        sorted_tokens = expanded_tokens[sorted_indices]

        return sorted_tokens, send_counts

    def _alltoall(
        self,
        send_data: torch.Tensor,
        send_counts: list
    ) -> Tuple[torch.Tensor, list]:
        """AlltoAll communication."""
        # Exchange counts
        recv_counts = [0] * self.ep_size
        dist.all_to_all_single(
            torch.tensor(recv_counts, device=send_data.device),
            torch.tensor(send_counts, device=send_data.device),
            group=self.ep_group
        )
        recv_counts = recv_counts

        # Exchange data
        total_recv = sum(recv_counts)
        recv_data = torch.empty(
            total_recv, self.hidden_dim,
            device=send_data.device, dtype=send_data.dtype
        )

        dist.all_to_all_single(
            recv_data, send_data,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group
        )

        return recv_data, recv_counts

    def _process_local_experts(
        self,
        tokens: torch.Tensor,
        counts_per_source: list
    ) -> torch.Tensor:
        """Process tokens through local experts."""
        # For each local expert, process its assigned tokens
        outputs = []

        # In practice, need to track which tokens go to which expert
        # This is simplified - real implementation tracks expert assignments
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens for this expert
            expert_tokens = self._get_tokens_for_expert(
                tokens, expert_idx, counts_per_source
            )

            if expert_tokens.shape[0] > 0:
                expert_output = expert(expert_tokens)
                outputs.append(expert_output)

        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, self.hidden_dim)

    def _get_tokens_for_expert(
        self,
        tokens: torch.Tensor,
        expert_idx: int,
        counts: list
    ) -> torch.Tensor:
        """Get tokens assigned to a specific local expert."""
        # Simplified - in practice, track indices during routing
        return tokens  # Placeholder
```

## 五维并行：DP × TP × PP × CP × EP

### 完整图景

```
5D Mesh: shape = (D, P, T, C, E)

Example: 16,384 GPUs for 1T MoE with 128K context
- DP = 8    (data parallel replicas)
- PP = 16   (pipeline stages)
- TP = 8    (tensor parallel)
- CP = 4    (context parallel)
- EP = 4    (expert parallel)

Total: 8 × 16 × 8 × 4 × 4 = 16,384
```

### 五维网格的实现

```python
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import torch.distributed as dist
import numpy as np

@dataclass
class FiveDConfig:
    """Configuration for 5D parallelism."""
    dp_size: int
    pp_size: int
    tp_size: int
    cp_size: int
    ep_size: int

    @property
    def world_size(self) -> int:
        return (self.dp_size * self.pp_size * self.tp_size *
                self.cp_size * self.ep_size)

    def validate(self) -> None:
        """Validate configuration constraints."""
        # TP × CP should fit within node for NVLink
        if self.tp_size * self.cp_size > 8:
            print("Warning: TP × CP > 8 may cross NVLink boundary")

        # EP should not be too large (load imbalance)
        if self.ep_size > 32:
            print("Warning: Large EP may cause load imbalance")

class FiveDMesh:
    """Device mesh for 5D parallelism."""

    def __init__(self, config: FiveDConfig):
        self.config = config
        config.validate()

        world_size = config.world_size
        devices = np.arange(world_size).reshape(
            config.dp_size,
            config.pp_size,
            config.tp_size,
            config.cp_size,
            config.ep_size
        )

        self.mesh = devices

        # Get my coordinates
        rank = dist.get_rank()
        coords = np.argwhere(devices == rank)[0]
        self.dp_rank = coords[0]
        self.pp_rank = coords[1]
        self.tp_rank = coords[2]
        self.cp_rank = coords[3]
        self.ep_rank = coords[4]

        # Create all process groups
        self._create_groups()

    def _create_groups(self):
        """Create all necessary process groups."""
        # Single-dimension groups
        self.dp_group = self._create_group_along(0)
        self.pp_group = self._create_group_along(1)
        self.tp_group = self._create_group_along(2)
        self.cp_group = self._create_group_along(3)
        self.ep_group = self._create_group_along(4)

        # Composite groups for fused operations
        self.tp_cp_group = self._create_group_along([2, 3])
        self.tp_ep_group = self._create_group_along([2, 4])

        # DP group that spans EP (for gradient sync in MoE)
        self.dp_ep_group = self._create_group_along([0, 4])

    def _create_group_along(self, dims) -> dist.ProcessGroup:
        """Create process group varying along specified dimension(s)."""
        if isinstance(dims, int):
            dims = [dims]

        # Build slice that fixes all dims except those specified
        slices = []
        for i, size in enumerate([
            self.config.dp_size,
            self.config.pp_size,
            self.config.tp_size,
            self.config.cp_size,
            self.config.ep_size
        ]):
            if i in dims:
                slices.append(slice(None))
            else:
                my_coords = [self.dp_rank, self.pp_rank, self.tp_rank,
                            self.cp_rank, self.ep_rank]
                slices.append(my_coords[i])

        ranks = self.mesh[tuple(slices)].flatten().tolist()
        return dist.new_group(ranks)

    def get_all_ranks(self) -> Dict[str, int]:
        """Return all ranks in a dictionary."""
        return {
            'dp': self.dp_rank,
            'pp': self.pp_rank,
            'tp': self.tp_rank,
            'cp': self.cp_rank,
            'ep': self.ep_rank
        }
```

### 五维通信矩阵

| 从 \ 到 | TP | CP | PP | DP | EP |
|-----------|----|----|----|----|-----|
| TP | AllReduce | - | - | - | - |
| CP | - | 环形 P2P | - | - | - |
| PP | - | - | P2P | - | - |
| DP | - | - | - | AllReduce | AllReduce（MoE 梯度） |
| EP | - | - | - | - | AlltoAll |

### 五维的内存分析

五维并行下每块 GPU 的内存：

**参数**：

$$M_{\text{params}} = \frac{N_{\text{dense}}}{T \times P} + \frac{N_{\text{expert}}}{T \times P \times E}$$

**优化器状态（optimizer states）**：

$$M_{\text{optimizer}} = 4 \times M_{\text{params}}$$

**激活值**：

$$M_{\text{activations}} = \frac{B \times S \times H}{T \times C} \times L_{\text{stage}} \times k_{\text{buffer}}$$

**示例**：1T MoE（200B 稠密 + 800B 专家）、128K 上下文、16K GPU、微批（micro-batch） $B_\mu=16$、$k_{\text{buffer}}=4$

配置：DP=8、PP=16、TP=8、CP=4、EP=4（全局批 $B = B_\mu \times \text{DP} = 128$）

- 稠密参数：$\frac{200\text{B} \times 2}{8 \times 16} = 3.1\text{ GB}$
- 专家参数：$\frac{800\text{B} \times 2}{8 \times 16 \times 4} = 3.1\text{ GB}$
- 优化器：$4 \times 6.2 = 24.8\text{ GB}$
- 激活值：$\frac{16 \times 128\text{K} \times 8192 \times 2}{8 \times 4} \times 4 \approx 4.3\text{ GB}$
- **总计**：约 35 GB（能放进 80GB A100）

## 实践考量

### EP 中的负载均衡

专家并行会引入负载不均衡：

```python
class LoadBalancer:
    """Auxiliary loss for expert load balancing."""

    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.25,
        balance_loss_weight: float = 0.01
    ):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.balance_loss_weight = balance_loss_weight

    def compute_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.

        Args:
            router_logits: [batch * seq, num_experts]
            expert_mask: [batch * seq, num_experts] binary

        Returns:
            Scalar balance loss
        """
        # Fraction of tokens routed to each expert
        tokens_per_expert = expert_mask.float().mean(dim=0)

        # Average routing probability to each expert
        router_prob = torch.softmax(router_logits, dim=-1)
        router_prob_per_expert = router_prob.mean(dim=0)

        # Balance loss: minimize product
        # (encourages uniform distribution)
        balance_loss = (
            self.num_experts *
            (tokens_per_expert * router_prob_per_expert).sum()
        )

        return self.balance_loss_weight * balance_loss

    def get_capacity(self, num_tokens: int) -> int:
        """Get capacity per expert (max tokens it can handle)."""
        avg_tokens_per_expert = num_tokens / self.num_experts
        return int(avg_tokens_per_expert * self.capacity_factor)
```

### 通信重叠

五维并行中的重叠机会：

```
Timeline for 5D training step:

│ GPU │ Op Type           │ Overlap With │
├─────┼───────────────────┼──────────────┤
│ All │ Forward compute   │              │
│ CP  │ Ring attention    │ FFN compute  │
│ EP  │ AlltoAll dispatch │ Self-attn    │
│ TP  │ AllReduce         │ Next layer   │
│ EP  │ AlltoAll combine  │ Next layer   │
│ All │ Backward compute  │              │
│ PP  │ Send activation   │ Next micro   │
│ DP  │ AllReduce grads   │ Next step    │
```

### 选择维度

五维配置的算法：

```python
def choose_5d_config(
    total_gpus: int,
    model_params_dense: int,
    model_params_expert: int,
    num_experts: int,
    sequence_length: int,
    gpu_memory_gb: float = 80,
    gpus_per_node: int = 8
) -> FiveDConfig:
    """
    Choose optimal 5D parallelism configuration.

    Priority:
    1. TP × CP ≤ gpus_per_node (NVLink)
    2. EP divides num_experts evenly
    3. PP minimizes bubble
    4. DP maximizes throughput
    """

    best_config = None
    best_efficiency = 0

    # TP options: powers of 2 up to gpus_per_node
    for tp in [1, 2, 4, 8]:
        if tp > gpus_per_node:
            continue

        # CP options: fill remaining NVLink bandwidth
        for cp in [1, 2, 4, 8]:
            if tp * cp > gpus_per_node:
                continue

            # EP options: must divide num_experts
            for ep in [1, 2, 4, 8, 16, 32]:
                if num_experts % ep != 0:
                    continue

                # PP options
                for pp in [1, 2, 4, 8, 16, 32]:
                    # Check if DP is valid
                    dp = total_gpus // (tp * cp * ep * pp)
                    if dp < 1:
                        continue
                    if dp * tp * cp * ep * pp != total_gpus:
                        continue

                    # Check memory
                    config = FiveDConfig(dp, pp, tp, cp, ep)
                    mem = estimate_5d_memory(
                        config, model_params_dense, model_params_expert,
                        num_experts, sequence_length
                    )

                    if mem > gpu_memory_gb * 0.9:
                        continue

                    # Estimate efficiency
                    eff = estimate_5d_efficiency(
                        config, model_params_dense, model_params_expert,
                        num_experts, sequence_length
                    )

                    if eff > best_efficiency:
                        best_efficiency = eff
                        best_config = config

    if best_config is None:
        raise ValueError("No valid configuration found")

    return best_config
```

## 案例研究

### LLaMA 3 405B（假想的四维配置）

训练一个 405B 的稠密模型、上下文 128K：

| 维度 | 值 | 理由 |
|-----------|-------|-----------|
| DP | 32 | 最大化吞吐 |
| PP | 16 | 96 层 / 每阶段 6 层 |
| TP | 8 | 在 NVLink 内 |
| CP | 4 | 处理 128K 上下文 |

**总计**：32 × 16 × 8 × 4 = 16,384 块 GPU

### Mixtral 8x22B（四维 + EP）

训练有 8 个专家的 Mixtral：

| 维度 | 值 | 理由 |
|-----------|-------|-----------|
| DP | 64 | 高吞吐 |
| PP | 4 | 较浅的 MoE |
| TP | 4 | 隐藏维度适中 |
| EP | 8 | 每个 EP rank 一个专家 |

**总计**：64 × 4 × 4 × 8 = 8,192 块 GPU

### GPT-4 规模的 MoE（假想的五维配置）

1T+ 参数、128 个专家的 MoE：

| 维度 | 值 | 理由 |
|-----------|-------|-----------|
| DP | 16 | 数据并行（Data Parallelism） |
| PP | 32 | 深层模型 |
| TP | 8 | 节点内 |
| CP | 2 | 128K 上下文 |
| EP | 16 | 128/16 = 每个 rank 8 个专家 |

**总计**：16 × 32 × 8 × 2 × 16 = 131,072 块 GPU

## 练习题

1. **四维设计**：你有 4,096 块 GPU，需要训练一个 70B 参数、256K 上下文的模型。设计一个四维配置。每块 GPU 的内存是多少？

??? success "解答"
    **需求：**

    - 模型：70B 参数
    - 上下文：256K token
    - GPU：4,096

    **四维配置设计：**

    | 维度 | 值 | 理由 |
    |-----------|-------|-----------|
    | TP | 8 | 节点内（NVLink） |
    | CP | 8 | 256K / 8 = 每个 CP rank 32K |
    | PP | 8 | 每阶段约 8.75B 参数 |
    | DP | 8 | 4096 / (8×8×8) = 8 个副本 |

    **验证：** $8 \times 8 \times 8 \times 8 = 4096$ ✓

    **内存分析：**

    **每块 GPU 的静态内存：**

    $$\Psi_{GPU} = \frac{70B}{PP \times TP} = \frac{70B}{64} = 1.09B \text{ params}$$

    | 组件 | 内存 |
    |-----------|--------|
    | 参数（fp16） | 2.18 GB |
    | 梯度（fp16） | 2.18 GB |
    | 优化器（fp32） | 13.1 GB |
    | **静态总计** | **17.5 GB** |

    在 DP=8 上使用 ZeRO-1：

    $$M_{opt}^{ZeRO1} = 13.1 / 8 = 1.64 \text{ GB}$$

    **使用 ZeRO-1 的静态总计：** 5.96 GB

    **激活内存：**

    每块 GPU 的序列：$256K / CP = 32K$ 个 token

    每层（TP=8）：

    $$M_{act}^{layer} = B \times \frac{S}{CP} \times H \times 34 / TP$$

    假设 B=2、H=8192、80 层、每阶段 10 层：

    $$M_{act} = 10 \times 2 \times 32768 \times 8192 \times 34 / 8 = 18.2 \text{ GB (no ckpt)}$$

    使用激活检查点（√10 ≈ 3 个区间）：

    $$M_{act}^{ckpt} \approx 6 \text{ GB}$$

    **环形注意力的 KV 缓存：**

    $$M_{KV} = 2 \times 2 \times B \times \frac{S}{CP} \times H / TP = 2 \times 2 \times 2 \times 32768 \times 8192 / 8$$

    $$M_{KV} = 268 \text{ MB per peer}$$

    双缓冲下：536 MB

    **每块 GPU 总计：**

    | 组件 | 内存 |
    |-----------|--------|
    | 静态（ZeRO-1） | 5.96 GB |
    | 激活值（检查点） | 约 6 GB |
    | KV 环形缓冲 | 约 0.5 GB |
    | 工作内存 | 约 2 GB |
    | **总计** | $\boxed{\sim 15 \text{ GB}}$ |

    轻松放进 H100 80GB。

2. **EP 通信**：对于一个有 64 个专家、EP=8、batch×seq=16384 个 token、top-2 路由的 MoE：

   - 每个 EP rank 在 AlltoAll 中发送多少 token？
   - AlltoAll 的通信量是多少？

??? success "解答"
    **设定：**

    - 64 个专家分布到 EP=8 个 rank → 每个 rank 8 个专家
    - token 总数：16,384
    - Top-2 路由：每个 token 被送到 2 个专家

    **token 分发的计算：**

    在均匀路由下，每个专家收到：

    $$\text{tokens per expert} = \frac{\text{total tokens} \times k}{\text{num experts}} = \frac{16384 \times 2}{64} = 512$$

    **每个 EP rank 发送的 token：**

    每个 EP rank 初始有 $16384 / 8 = 2048$ 个 token。

    在 top-2 路由下，每个 token 生成 2 份 → 每个 rank 有 4096 个已路由的 token。

    在 8 个 EP rank 上均匀分布：
    - 本地保留：$4096 / 8 = 512$ 个 token（送往本地专家）
    - 送往每个远端 rank：$512$ 个 token

    **AlltoAll 中每个 rank 发送的 token 总数：**
    $$\boxed{4096 - 512 = 3584 \text{ tokens}}$$

    或者说，向 7 个远端 rank 各发送 512 个 token。

    **AlltoAll 通信量：**

    假设隐藏维度 H=4096、fp16：

    $$\text{bytes per token} = H \times 2 = 8192 \text{ bytes}$$

    **每个 rank 的发送量：**
    $$V_{send} = 3584 \times 8192 = 29.4 \text{ MB}$$

    **AlltoAll 总通信量（所有 rank 都发送）：**
    $$V_{total} = 8 \times 29.4 = \boxed{235 \text{ MB}}$$

    **计入接收（双向）：**

    AlltoAll 是对称的，因此每个 rank 也接收约 29.4 MB。

    | 指标 | 值 |
    |--------|-------|
    | 每个 rank 的 token（输入） | 2,048 |
    | 每个 rank 路由的 token（top-2） | 4,096 |
    | 发送的 token（非本地） | 3,584 |
    | 每个 token 字节数 | 8 KB |
    | 每个 rank 的 AlltoAll 通信量 | 29.4 MB |
    | AlltoAll 总通信量 | 235 MB |

3. **环形注意力分析**：对于 CP=8、序列长度 128K：

   - 需要多少环形步？
   - K、V 缓冲的内存是多少？
   - 与 AllGather 的内存需求相比如何？

??? success "解答"
    **配置：**

    - CP = 8（上下文并行度）
    - S = 128K token 总数
    - 每个 CP rank 的序列：$S_{local} = 128K / 8 = 16K$ 个 token

    **所需环形步数：**

    在环形注意力中，每个 rank 计算其本地 Q 与来自所有 rank（包括自己）的 K、V 之间的注意力。这要求 K、V 沿环传递。

    $$\text{Ring steps} = CP - 1 = \boxed{7 \text{ steps}}$$

    （加上 1 次本地步 = 共 8 次注意力计算）

    **K、V 缓冲的内存：**

    假设 H=8192 隐藏维度、80 层、fp16：

    每层中 K 和 V 各自的形状：[batch, seq_local, hidden]

    $$M_{KV}^{layer} = 2 \times B \times S_{local} \times H \times 2 \text{ bytes}$$

    当 B=2：

    $$M_{KV}^{layer} = 2 \times 2 \times 16384 \times 8192 \times 2 = 1.07 \text{ GB}$$

    **为重叠而做的双缓冲：**

    环形注意力把通信与计算重叠，需要 2 个缓冲区：

    $$M_{KV}^{ring} = 2 \times 1.07 = \boxed{2.14 \text{ GB per layer}}$$

    **与 AllGather 内存的对比：**

    AllGather 会实体化完整序列的 K、V：

    $$M_{AllGather}^{layer} = 2 \times B \times S_{full} \times H \times 2$$

    $$M_{AllGather}^{layer} = 2 \times 2 \times 131072 \times 8192 \times 2 = 8.59 \text{ GB per layer}$$

    **内存比较：**

    | 方案 | 每层内存 | 80 层合计 | 伸缩性 |
    |----------|------------------|---------------|---------|
    | 环形注意力 | 2.14 GB | 171 GB | O(S/CP) |
    | AllGather | 8.59 GB | 687 GB | O(S) |
    | **节省** | **4×** | **4×** | **CP×** |

    **小结：**

    CP=8 的环形注意力比 AllGather 少用 $\boxed{4\times}$ 内存：

    - 环形：同一时刻只持有 2 块（当前 + 下一块）
    - AllGather：同时实体化全部 8 块

    代价是延迟：环形注意力有 $CP-1$ 个串行通信步，但每一步都与计算重叠。

    **内存伸缩性：**

    $$M_{ring} = O\left(\frac{S}{CP}\right) \quad \text{vs} \quad M_{AllGather} = O(S)$$

    对于极长的上下文，环形注意力是必需的——AllGather 会 OOM。

4. **负载不均衡**：一个有 8 个专家的 MoE，路由分布为 [30%, 5%, 25%, 10%, 5%, 10%, 10%, 5%]。当 capacity_factor=1.25 时，哪些专家会丢弃 token？

??? success "解答"
    **给定的路由分布：**

    | 专家 | 路由 % | token 数（若共 1000 个） |
    |--------|-----------|------------------------|
    | 0 | 30% | 300 |
    | 1 | 5% | 50 |
    | 2 | 25% | 250 |
    | 3 | 10% | 100 |
    | 4 | 5% | 50 |
    | 5 | 10% | 100 |
    | 6 | 10% | 100 |
    | 7 | 5% | 50 |

    **容量计算：**

    8 个专家、均匀路由时，每个专家将收到：

    $$\text{expected tokens} = \frac{\text{total tokens}}{8} = 12.5\%$$

    capacity_factor=1.25 时的专家容量：

    $$\text{capacity} = \text{capacity\_factor} \times \frac{\text{total}}{8} = 1.25 \times 12.5\% = 15.625\%$$

    对于 1000 个 token：每个专家容量 = 156 个 token

    **识别被丢弃的 token：**

    | 专家 | token | 容量 | 丢弃 | 丢弃率 |
    |--------|--------|----------|---------|-----------|
    | 0 | 300 | 156 | 144 | 48% |
    | 1 | 50 | 156 | 0 | 0% |
    | 2 | 250 | 156 | 94 | 38% |
    | 3 | 100 | 156 | 0 | 0% |
    | 4 | 50 | 156 | 0 | 0% |
    | 5 | 100 | 156 | 0 | 0% |
    | 6 | 100 | 156 | 0 | 0% |
    | 7 | 50 | 156 | 0 | 0% |

    **答案：**

    $$\boxed{\text{Experts 0 and 2 will drop tokens}}$$

    - 专家 0：丢弃 144 个 token（其路由 token 的 48%）
    - 专家 2：丢弃 94 个 token（其路由 token 的 38%）
    - 总丢弃：238 个 token（占全部 token 的 23.8%！）

    **影响分析：**

    这是严重的不均衡：

    - 2 个专家承担了 55% 的流量，却只能处理 31.25%
    - 23.8% 的 token 丢弃率会显著影响模型质量
    - 专家 1、4、7 利用不足（只用了 32% 的容量）

    **缓解策略：**

    1. **提高 capacity_factor**：2.0 能让专家 0 处理 250 个 token
    2. **负载均衡损失**：加入辅助损失以鼓励均匀路由
    3. **专家选择路由**：让专家选择 token，而不是 token 选择专家
    4. **Top-k > 2**：每个 token 用更多专家来分散负载

    **避免丢弃所需的 capacity_factor：**

    $$\text{capacity\_factor}_{min} = \frac{\max(\text{routing})}{1/E} = \frac{30\%}{12.5\%} = \boxed{2.4}$$

5. **五维扩展**：一个五维配置在 16K GPU 上达到 40% MFU。扩展到 64K GPU（4× DP）时，预测新的 MFU 并指出瓶颈。

??? success "解答"
    **基线配置（16K GPU，40% MFU）：**

    假设五维配置：(DP=16, PP=8, TP=8, CP=4, EP=4)
    - 总计：16 × 8 × 8 × 4 × 4 = 16,384 块 GPU ✓

    **扩展后的配置（64K GPU，4× DP）：**

    新的：(DP=64, PP=8, TP=8, CP=4, EP=4)
    - 总计：64 × 8 × 8 × 4 × 4 = 65,536 块 GPU ✓

    **影响分析：**

    增加 DP 会影响：

    1. **AllReduce 通信代价随 DP 扩展：**

    环形 AllReduce 时间：$T_{AR} = \frac{2M}{\beta} \cdot \frac{DP-1}{DP}$

    对于较大的 DP，这趋近于 $\frac{2M}{\beta}$（常数），但延迟会增加。

    | DP | $\frac{DP-1}{DP}$ | 相对 AR 时间 |
    |----|-------------------|------------------|
    | 16 | 0.9375 | 1.0× |
    | 64 | 0.9844 | 1.05× |

    带宽部分：增加约 5%（可忽略）

    2. **AllReduce 延迟随环步数扩展：**

    基于树：$O(\log DP)$ → $\log_2(64)/\log_2(16) = 6/4 = 1.5\times$ 更多延迟
    基于环：$2(DP-1)$ 步 → $126/30 = 4.2\times$ 更多延迟

    3. **ZeRO 分片的内存改善：**

    在 DP=64 时，优化器状态被多分片 4 倍 → 内存压力缓解

    **MFU 预测：**

    原始分解（16K GPU 上 40% MFU）：

    | 组件 | 时间占比 | 效率 |
    |-----------|------------------|------------|
    | 计算 | 60% | 67% |
    | DP AllReduce | 15% | - |
    | PP 通信 | 10% | - |
    | 其他开销 | 15% | - |

    DP 扩展 4 倍之后：

    | 组件 | 旧 | 新 | 变化 |
    |-----------|-----|-----|--------|
    | 计算 | 60% | 60% | 不变 |
    | DP AllReduce（延迟） | 15% | ~22% | +50%（树）到 +4×（环） |
    | PP 通信 | 10% | 10% | 不变 |
    | 其他开销 | 15% | 15% | 不变 |

    **保守估计（基于树的 AllReduce）：**

    新的总时间：60% + 22% + 10% + 15% = 原始的 107%

    $$MFU_{new} = \frac{40\%}{1.07} \approx 37\%$$

    **悲观估计（大规模下基于环）：**

    如果在 64K 规模下延迟占主导：

    $$MFU_{new} \approx 30-35\%$$

    **预测 MFU：**

    $$\boxed{MFU \approx 35-38\%}$$

    **已识别的瓶颈：**

    1. **DP AllReduce 延迟**：主要瓶颈
       - 归约组内 GPU 更多
       - 若跨数据中心，链路更长
       - 解决方案：层次化 AllReduce、梯度压缩

    2. **网络拥塞**：主干网上 4 倍流量
       - 解决方案：更好的网络拓扑、流量整形

    3. **掉队者**：GPU 越多 → 完成时间的方差越大
       - 解决方案：备份 worker、异步 SGD

    4. **梯度陈旧**：若用异步/本地 SGD 来缓解
       - 解决方案：学习率调优、动量修正

    **64K 扩展的建议：**

    | 策略 | 影响 |
    |----------|--------|
    | 层次化 AllReduce | 降低延迟 2–3× |
    | 梯度压缩（Gradient Compression） | 降低带宽 4–10× |
    | 本地 SGD（K 步） | 降低 AllReduce 频率 |
    | 改为增加 PP 或 CP | 比 DP 更好的扩展性 |

6. **维度排序**：提出一种替代 (DP, PP, TP, CP, EP) 的排序。论证它在什么情况下更好。

??? success "解答"
    **标准排序：(DP, PP, TP, CP, EP)**

    这意味着：
    - 最内层（连续 rank）：EP 组
    - 然后：CP 组
    - 然后：TP 组
    - 然后：PP 组
    - 最外层：DP 组

    **提出的替代方案：(DP, EP, PP, CP, TP)**

    **关键变化：**

    | 维度 | 标准 | 替代 | 变化 |
    |-----------|----------|-------------|--------|
    | TP | 中间 | 最内层 | ↑ 优先级 |
    | EP | 最内层 | DP 之后 | ↓ 优先级 |
    | CP | TP 之后 | TP 之前 | 小幅移动 |

    **论证：替代排序更优的情形：**

    **1. 当 TP 通信占主导时：**

    如果注意力层是瓶颈（隐藏维度大、头数多）：

    - TP：每层 4 次 AllReduce × 每次 $O(BSH)$
    - EP：只在 MoE 层有 2 次 AlltoAll

    让 TP 位于最内层可确保 TP 的 rank 在同一节点（NVLink）。

    **标准排序下的 8 GPU 节点：**

    ```
    Node 0: EP=0-7 (expert parallel)
    Node 1: EP=0-7 for next CP rank
    ```

    TP 通信会跨节点 → 慢！

    **替代排序：**

    ```
    Node 0: TP=0-7 (tensor parallel)
    Node 1: TP=0-7 for next CP rank
    ```

    TP 通信保持在节点内 → 快！

    **2. 当使用层数很少的稀疏 MoE 时：**

    如果每 4 层才有一个 MoE：
    - EP AlltoAll：25% 的层
    - TP AllReduce：100% 的层

    EP 可以容忍更慢的节点间链路。

    **3. 当 EP 可以使用层次化 AlltoAll 时：**

    在 (DP, EP, ...) 排序下：
    - EP 有意跨越节点
    - 可实现两级 AlltoAll：节点内 + 节点间
    - 大规模下带宽利用更好

    **通信放置分析：**

    | 排序 | TP 位置 | EP 位置 | 最适合 |
    |----------|--------------|--------------|-----------|
    | (DP,PP,TP,CP,EP) | 节点中部 | 节点内 | 稠密 MoE、隐藏维度小 |
    | (DP,EP,PP,CP,TP) | 节点内 | 跨节点 | 稀疏 MoE、隐藏维度大 |
    | (DP,PP,CP,TP,EP) | 靠近内层 | 最内层 | 两者的平衡 |

    **定量示例：**

    配置：512 块 GPU、每节点 8 块、H=12288、16 个专家、每 4 层一个 MoE

    | 排序 | TP 带宽 | EP 带宽 | 步时间 |
    |----------|-------|-------|-----------|
    | 标准（EP 在内） | 50 GB/s（IB） | 900 GB/s（NVLink） | 1.2s |
    | 替代（TP 在内） | 900 GB/s（NVLink） | 50 GB/s（IB） | 0.9s |

    **加速比：** $\boxed{1.33\times}$（替代排序）

    **一般原则：**

    $$\text{Innermost dimension} = \arg\max(\text{comm volume} \times \text{frequency})$$

    把带宽需求最高的通信放在最快的互联上。

    **其他替代排序：**

    | 排序 | 用例 |
    |----------|----------|
    | (PP, DP, TP, CP, EP) | 流水线优先，以改善内存局部性 |
    | (DP, CP, PP, TP, EP) | 长上下文且 PP 便宜 |
    | (EP, DP, PP, TP, CP) | 专家专用节点 |

## 关键要点

1. **第四维加入上下文并行**：为长上下文切分序列维度。

2. **第五维加入专家并行**：把 MoE 专家分布到各设备。

3. **环形注意力避免 O(S²) 内存**：通过环流式传递 KV。

4. **EP 需要 AlltoAll**：token 路由是一个置换，而不是归约。

5. **TP × CP ≤ 节点规模**：把最高带宽的通信留在 NVLink 上。

6. **负载均衡对 EP 至关重要**：辅助损失可防止专家坍塌。

7. **复合进程组使融合成为可能**：TP-CP 组、DP-EP 组用于组合操作。

8. **配置是多目标优化**：在内存、计算、通信与负载均衡之间取得平衡。