---
title: "集群的内存层次结构"
subtitle: "从寄存器到数据中心"
---

<div class="chapter-opener" markdown>
单台机器有内存层次：寄存器、L1、L2、L3、DRAM。集群把这一层次结构延伸到多台机器之间：HBM、NVLink、InfiniBand、以太网。理解每一层的带宽与延迟，决定了我们能把数据放在哪里。
</div>

<div class="investigation-question" markdown>
**问题**：为什么张量并行在节点内行得通，跨节点就吃力？为什么跨节点通信更偏好流水线并行？答案藏在 NVLink 与 InfiniBand 之间 20 倍的带宽差异里。
</div>

## 扩展后的层次结构

| 层级 | 技术 | 带宽 | 延迟 | 典型用途 |
|------|------|------|------|----------|
| 片上 | SRAM（L1/L2） | ~80 TB/s | ~1-10 ns | kernel 中间结果 |
| 设备内存 | HBM3 | ~3.35 TB/s | ~200-500 ns | 模型权重、激活值 |
| 节点内 | NVLink 4.0 | ~900 GB/s | ~1 μs | 张量并行 |
| 跨节点（快） | InfiniBand NDR | ~50 GB/s | ~1-2 μs | 流水线并行、数据并行 |
| 跨节点（慢） | 以太网 100G | ~12 GB/s | ~10-50 μs | 兜底、部分云环境 |

从 NVLink 到 InfiniBand，带宽下降约 **20 倍**。这一点，加上延迟和拓扑，解释了大多数放置决策。

## 节点内拓扑

### DGX H100（8 块 GPU）

```
    GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3
      ↕        ↕        ↕        ↕
    GPU4 ←→ GPU5 ←→ GPU6 ←→ GPU7

    All connected via NVSwitch
    Full bisection bandwidth: 900 GB/s per GPU
```

NVSwitch 提供**非阻塞的全对分带宽（full bisection bandwidth）**：任意两块 GPU 都能同时以全速通信。这就是节点内 8 路张量并行能良好工作的原因。

### 仅 PCIe 的系统

没有 NVLink 时，GPU 通过 PCIe（Gen5 x16：每方向约 32 GB/s，双向约 64 GB/s）通信，可能还要经过 CPU 内存。这会增加延迟、降低带宽，使跨 GPU 的张量并行吸引力下降。

## 跨节点拓扑

### 胖树（Fat-Tree）

```
        Core Switch
       /    |    \
      /     |     \
   Agg    Agg    Agg  (Aggregation switches)
   /|\    /|\    /|\
  N N N  N N N  N N N  (Nodes)
```

- 在每一层都提供全对分带宽
- 昂贵（交换机数量多）
- 高性能集群的标准选择

### 轨道优化（Dragonfly+）

Meta 的 Grand Teton 等现代集群采用轨道优化（rail-optimized）拓扑：跨节点处于相同位置的 GPU 共享一条网络轨道：

```
Node 0:  GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
           |     |     |     |     |     |     |     |
Node 1:  GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
           |     |     |     |     |     |     |     |
         Rail0 Rail1 Rail2 Rail3 Rail4 Rail5 Rail6 Rail7
```

这优化了最常见的情形：同序号 GPU 之间通信（数据并行）。

## 带宽与延迟的权衡

### α-β 模型（预览）

通信时间满足：

$$T(n) = \alpha + \frac{n}{\beta}$$

其中：

- $\alpha$：延迟（固定的启动开销）
- $n$：消息大小（字节）
- $\beta$：带宽（字节/秒）

小消息时延迟占主导，大消息时带宽占主导。

**交叉点（crossover point）**：$n^* = \alpha \cdot \beta$

| 链路类型 | α（延迟） | β（带宽） | 交叉点 |
|----------|-----------|-----------|--------|
| NVLink | ~1 μs | 900 GB/s | ~1 MB |
| InfiniBand | ~2 μs | 50 GB/s | ~100 KB |

小于交叉点的消息就是延迟受限的。

## 对并行策略的启示

### 张量并行

需要**每一层**都对激活张量做集合通信（常见 TP 方案中的 AllGather/ReduceScatter）。通信量与批 × 序列 × 隐层维成正比。

- 节点内（NVLink）：约 900 GB/s → 可行
- 跨节点（InfiniBand）：约 50 GB/s → 往往是瓶颈

**准则**：把 TP 度数限制在节点内的 GPU 数（通常为 8）。

### 流水线并行

只在阶段边界通信激活值。是点对点而非集合通信，能容忍更高的延迟。

- 激活大小：批 × 序列 × 隐层维（通常几十 MB）
- 对 InfiniBand 而言已接近交叉点：延迟影响较小
- 总体带宽需求更低

**准则**：跨节点使用 PP。

### 数据并行

每步通信一次梯度。通信量是 O(参数)，而非 O(激活 × 层数)。

- 可用梯度累积提高通信强度
- 压缩技术能降低带宽需求
- AllReduce 可以与反向传播重叠

**准则**：把 DP 作为最外层的并行维度。

## 三维并行的放置

```
Within DGX Node (8 GPUs):
├── Tensor Parallelism (TP=8)
│   └── Uses NVLink at 900 GB/s

Across Nodes:
├── Pipeline Parallelism (PP=k)
│   └── Uses InfiniBand for P2P
│   └── Point-to-point, latency-tolerant
│
└── Data Parallelism (DP=N/(8*k))
    └── Uses InfiniBand for AllReduce
    └── Can overlap with compute
```

这一层次结构直接源自集群的带宽层次。

## 练习题

1. 你有一台 256 GPU 的集群，每节点 8 块 GPU（NVLink），节点间为 400 Gbps InfiniBand。计算以下情形的最大集合通信带宽：(a) 节点内 8 路 AllReduce，(b) 跨 4 个节点的 32 路 AllReduce。

??? success "解答"
    **环形 AllReduce 的有效带宽：**

    对环形 AllReduce，有效带宽 ≈ $\beta \times \frac{P}{2(P-1)}$，其中 $\beta$ 为链路带宽。

    **(a) 节点内 8 路 AllReduce（NVLink）：**

    - NVLink 带宽：$\beta = 900$ GB/s
    - $P = 8$ 块 GPU

    $$BW_{eff} = 900 \times \frac{8}{2 \times 7} = 900 \times \frac{8}{14} \approx 514 \text{ GB/s}$$

    实践中，NCCL 在大消息下可达约 500–600 GB/s，具体取决于拓扑和测量口径。

    **(b) 跨 4 个节点的 32 路 AllReduce：**

    使用分层 AllReduce：

    1. 每个节点内先归约（NVLink，快）
    2. 4 个节点代表之间归约（InfiniBand，慢）
    3. 各节点内广播（NVLink，快）

    跨节点瓶颈：

    - InfiniBand：400 Gbps = 每方向 50 GB/s
    - 跨节点 4 路 AllReduce

    $$BW_{eff}^{inter} = 50 \times \frac{4}{2 \times 3} \approx 33 \text{ GB/s}$$

    跨节点通信占主导，把整个 32 路集合通信的有效带宽限制在**约 30–40 GB/s**。

2. 一个 transformer 层为张量并行做两次 AllReduce。隐层维为 8192，批大小为 32，序列长度为 2048。计算每层传输的总字节数，以及 (a) NVLink、(b) InfiniBand 上所需的时间。

??? success "解答"
    **激活张量大小（BF16）：**

    $$\text{Size} = B \times S \times H \times 2 = 32 \times 2048 \times 8192 \times 2 = 1.07 \text{ GB}$$

    **单次 AllReduce 通信量（环形，大 $P$）：**

    $$V_{AR} \approx 2 \times \text{Size} = 2.14 \text{ GB}$$

    **两次 AllReduce 合计：**

    $$V_{total} = 2 \times 2.14 = 4.28 \text{ GB per layer}$$

    **时间计算：**

    (a) **NVLink**（900 GB/s）：

    $$T_{NVLink} = \frac{4.28 \text{ GB}}{900 \text{ GB/s}} = 4.8 \text{ ms}$$

    (b) **InfiniBand**（50 GB/s）：

    $$T_{IB} = \frac{4.28 \text{ GB}}{50 \text{ GB/s}} = 86 \text{ ms}$$

    **结论：** 走 InfiniBand 的张量并行比 NVLink 慢 18 倍。这就是 TP 只留在 NVLink 域内的原因。

3. 为什么流水线并行使用点对点通信而不是集合通信？如果每个阶段都必须与其他所有阶段做 AllReduce 会怎样？

??? success "解答"
    **为什么用点对点：**

    流水线并行是线性的数据流：Stage 0 → Stage 1 → Stage 2 → ... → Stage N-1。

    每个阶段只需要：

    - 从上一阶段**接收**激活值
    - 向下一阶段**发送**激活值

    这天然就是**点对点**（Send/Recv），不是集合通信。

    **如果必须用 AllReduce：**

    | 问题 | 影响 |
    |------|------|
    | 全局同步 | 所有阶段等待最慢者 |
    | 延迟随 P 增长 | $O(P)$ 而非 $O(1)$ |
    | 破坏流水 | 无法重叠各阶段 |
    | 不必要的数据移动 | 各阶段并不需要彼此的数据 |

    **定量例子（8 个阶段）：**

    - 点对点：1 跳延迟（每次激活传输约 1–5 μs）
    - AllReduce：至少 7 跳，所有阶段同步

    对典型的 $M$，流水线气泡会从 $\frac{P-1}{M+P-1}$ 增至大得多的比例，可能吞没有效工作。

    **关键洞察：** PP 利用了神经网络层的*顺序*特性。集合通信适用于对同一数据的*并行*运算。

## 关键要点

1. **集群是分层的**：NVLink、PCIe、网卡和交换机定义了不同的带宽层级。
2. **并行放置必须尊重带宽落差**：TP 在节点内、DP 跨节点、PP 跨阶段。
3. **拓扑感知能避免隐性减速**：策略与链路速度不匹配时，可能带来 10–20 倍开销。