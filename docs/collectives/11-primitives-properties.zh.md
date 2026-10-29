---
title: "集合通信原语与性质"
subtitle: "分布式通信的构建块"
---

<div class="chapter-opener" markdown>
每一个分布式训练算法都由一小组集合通信（collective communication）操作构建而成。理解它们的代数性质，能揭示为什么某些组合可行，而另一些会失败。
</div>

<div class="investigation-question" markdown>
**问题**：为什么 AllReduce = AllGather ∘ ReduceScatter？这种分解成立需要满足哪些性质？当这些性质被违背时会发生什么？
</div>

<div class="notation-banner" markdown>
**本章符号：** $P$ = 进程/GPU 数量，$n$ = 消息大小（字节），$\alpha$ = 延迟，$\beta$ = 带宽。参见[符号说明](../appendices/notation.md)。

**关于符号复用：** 从本章开始，$\alpha$ 表示网络延迟，$\beta$ 表示网络带宽。在第二部分（扩展定律）中，这两个符号用于缩放指数。含义应从上下文中清晰可辨——第三至第八部分只讨论通信与并行。
</div>

!!! abstract "承上启下：第一至第二部分"
    本部分建立在[扩展屋顶线（第 2 章）](../foundations/02-extended-roofline.md)的**网络上限**概念、**α-β 成本模型**（[第 4 章](../foundations/04-alpha-beta-model.md)）以及集群的**内存层次结构**（[第 3 章](../foundations/03-cluster-memory-hierarchy.md)）之上。从第二部分我们知道，大规模训练需要海量数据移动。现在我们来形式化使这种移动成为可能的通信原语。

## 七种原语

集合通信操作是分布式系统的词汇表。所有分布式训练通信都可以用这些原语表达。

### 1. Broadcast（广播）

一个进程向所有其他进程发送数据。

```
之前:                之后:
P0: [A B C D]        P0: [A B C D]
P1: [. . . .]   →    P1: [A B C D]
P2: [. . . .]        P2: [A B C D]
P3: [. . . .]        P3: [A B C D]
```

**使用场景**：初始化时分发模型权重。

**通信量**：根进程注入 $n$ 字节；总网络流量取决于算法（例如树形为 $n \log P$，朴素为 $n(P-1)$）。

### 2. Reduce（归约）

所有进程贡献数据；一个进程接收归约结果。

```
之前:                之后（求和）:
P0: [1 2 3 4]        P0: [10 20 30 40]
P1: [2 4 6 8]   →    P1: [. . . .]
P2: [3 6 9 12]       P2: [. . . .]
P3: [4 8 12 16]      P3: [. . . .]
```

**使用场景**：把梯度聚合到参数服务器。

**通信量**：每个非根进程发送 $n$ 字节。

### 3. AllReduce

所有进程贡献数据；所有进程都接收归约结果。

```
之前:                之后（求和）:
P0: [1 2 3 4]        P0: [10 20 30 40]
P1: [2 4 6 8]   →    P1: [10 20 30 40]
P2: [3 6 9 12]       P2: [10 20 30 40]
P3: [4 8 12 16]      P3: [10 20 30 40]
```

**使用场景**：数据并行（Data Parallelism）中的梯度同步。

**通信量**：每个进程 $2(P-1)/P \cdot n$ 字节（环形算法）。

### 4. Scatter（分散）

一个进程把不同的数据块分发给各个进程。

```
之前:                之后:
P0: [A B C D]        P0: [A]
P1: [. . . .]   →    P1: [B]
P2: [. . . .]        P2: [C]
P3: [. . . .]        P3: [D]
```

**使用场景**：分发不同的数据批次。

**通信量**：根进程总共发送 $n$ 字节。

### 5. Gather（收集）

每个进程贡献一个数据块；一个进程接收拼接后的结果。

```
之前:                之后:
P0: [A]              P0: [A B C D]
P1: [B]         →    P1: [.]
P2: [C]              P2: [.]
P3: [D]              P3: [.]
```

**使用场景**：收集各 worker 的输出。

**通信量**：根进程总共接收 $n$ 字节。

### 6. AllGather

每个进程贡献一个数据块；所有进程都接收拼接后的结果。

```
之前:                之后:
P0: [A]              P0: [A B C D]
P1: [B]         →    P1: [A B C D]
P2: [C]              P2: [A B C D]
P3: [D]              P3: [A B C D]
```

**使用场景**：在 ZeRO-3 中重建完整张量。

!!! info "什么是 ZeRO？"
    **ZeRO（Zero Redundancy Optimizer，零冗余优化器）** 是第 20 章介绍的一种内存优化技术。它在数据并行 GPU 之间分片优化器状态（optimizer states）、梯度和/或参数，而不是复制它们。ZeRO-3 对三者都做分片，需要 AllGather 在前向/反向传播时重建参数，并用 ReduceScatter 分发归约后的梯度。

**通信量**：每个进程接收 $(P-1) \cdot n/P$ 字节。

### 7. ReduceScatter

所有进程贡献完整数据；每个进程接收归约结果的一个分片。

```
之前:                之后（求和）:
P0: [1 2 3 4]        P0: [10]
P1: [2 4 6 8]   →    P1: [20]
P2: [3 6 9 12]       P2: [30]
P3: [4 8 12 16]      P3: [40]
```

**使用场景**：ZeRO 中的梯度归约 + 分片。

**通信量**：每个进程发送 $(P-1)/P \cdot n$ 字节。

### 附加：All-to-All

每个进程向其他每个进程发送不同的数据。

```
之前:                之后:
P0: [A0 A1 A2 A3]    P0: [A0 B0 C0 D0]
P1: [B0 B1 B2 B3] →  P1: [A1 B1 C1 D1]
P2: [C0 C1 C2 C3]    P2: [A2 B2 C2 D2]
P3: [D0 D1 D2 D3]    P3: [A3 B3 C3 D3]
```

**使用场景**：混合专家中的专家路由。

**通信量**：每个进程发送并接收 $(P-1)/P \cdot n$ 字节。

## 汇总表

下表使用 α-β 成本模型（见第 4 章），其中 $P$ = 进程数，$n$ = 消息大小（字节），$\alpha$ = 每条消息的延迟，$\beta$ = 带宽（字节/秒）。

| 原语 | 输入/输出 | 数据移动 | 最优复杂度 |
|------|-----------|----------|------------|
| Broadcast | 1 → N 份 | root → all | $\log_2 P \cdot \left(\alpha + \frac{n}{\beta}\right)$ |
| Reduce | N → 1 求和 | all → root | $\log_2 P \cdot \left(\alpha + \frac{n}{\beta}\right)$ |
| AllReduce | N → N 求和 | all ↔ all | $2\alpha(P-1) + 2\frac{P-1}{P}\frac{n}{\beta}$ |
| Scatter | 1 → N 分片 | root → all | $\alpha \log P + \frac{P-1}{P}\frac{n}{\beta}$ |
| Gather | N → 1 拼接 | all → root | $\alpha \log P + \frac{P-1}{P}\frac{n}{\beta}$ |
| AllGather | N → N 拼接 | all ↔ all | $\alpha(P-1) + \frac{P-1}{P}\frac{n}{\beta}$ |
| ReduceScatter | N → N 分片求和 | all ↔ all | $\alpha(P-1) + \frac{P-1}{P}\frac{n}{\beta}$ |
| All-to-All | N → N 转置 | all ↔ all | $\alpha(P-1) + \frac{P-1}{P}\frac{n}{\beta}$ |

## 代数性质

集合通信操作构成一种代数。理解这种代数，能揭示哪些变换是合法的。

### 结合律

二元运算 $\oplus$ 满足**结合律（associativity）**，如果：

$$(a \oplus b) \oplus c = a \oplus (b \oplus c)$$

**为什么重要**：归约可以按任意分组顺序进行。

```
树形归约（满足结合律的运算）：
      sum
     /   \
   sum   sum
   / \   / \
  P0 P1 P2 P3

合法，如果 (g0 ⊕ g1) ⊕ (g2 ⊕ g3) = ((g0 ⊕ g1) ⊕ g2) ⊕ g3
```

**浮点注意事项**：加法并不完美满足结合律！

$$\text{float}((a + b) + c) \neq \text{float}(a + (b + c))$$

例如：$a = 10^{20}$，$b = -10^{20}$，$c = 1$
- $(a + b) + c = 0 + 1 = 1$
- $a + (b + c) = 10^{20} + (-10^{20} + 1) = 10^{20} - 10^{20} = 0$

这会影响分布式训练的可复现性。

### 交换律

二元运算 $\oplus$ 满足**交换律（commutativity）**，如果：

$$a \oplus b = b \oplus a$$

**为什么重要**：各进程贡献的顺序不影响结果。

**例子**：

- 求和：满足交换律 ✓
- Max：满足交换律 ✓
- 拼接：不满足交换律 ✗

### 单位元

**单位元（identity element）** $e$ 满足：

$$a \oplus e = e \oplus a = a$$

**例子**：

- 求和：$e = 0$
- 乘积：$e = 1$
- Max：$e = -\infty$

单位元用于初始化归约操作中的缓冲区。

### 复合封闭性

如果我们复合两个集合通信操作，会得到另一个合法操作吗？

$$\text{AllReduce} = \text{AllGather} \circ \text{ReduceScatter}$$

这种分解是理解集合通信实现的基础。

## 分解定理

**定理**：对满足结合律和交换律的归约运算 $\oplus$：

$$\text{AllReduce}(x_0, x_1, ..., x_{P-1}) = \text{AllGather}(\text{ReduceScatter}(x_0, x_1, ..., x_{P-1}))$$

**证明**：

设每个进程 $i$ 拥有大小为 $n$ 的向量 $x_i$。

**第 1 步**：ReduceScatter 把每个向量分成 $P$ 个数据块，并把块 $j$ 归约到进程 $j$：

$$y_j = \bigoplus_{i=0}^{P-1} x_i^{(j)}$$

其中 $x_i^{(j)}$ 是进程 $i$ 数据的第 $j$ 块。

ReduceScatter 之后，进程 $j$ 持有大小为 $n/P$ 的 $y_j$。

**第 2 步**：AllGather 把所有 $y_j$ 收集到所有进程：

每个进程收到 $[y_0, y_1, ..., y_{P-1}]$。

**结果**：

$$[y_0, y_1, ..., y_{P-1}] = \left[\bigoplus_{i} x_i^{(0)}, \bigoplus_{i} x_i^{(1)}, ..., \bigoplus_{i} x_i^{(P-1)}\right]$$

由交换律和结合律：

$$= \bigoplus_{i} x_i$$

这等于 AllReduce 的结果。$\square$

### 为什么这很重要

1. **实现灵活性**：可以把 AllReduce 实现为 RS + AG
2. **带宽最优性**：RS 和 AG 都达到带宽下界
3. **融合机会**：可以在 RS 和 AG 之间融合计算
4. **理解 ZeRO**：ZeRO 直接利用了这种分解

## 对偶操作

一些集合通信操作有自然的对偶（并非代数意义上的严格逆运算）：

| 操作 | 对偶 |
|------|------|
| Broadcast | Reduce（仅对单位元归约才是逆运算） |
| Scatter | Gather |
| AllGather | ReduceScatter（带 reshape） |

**Gather ∘ Scatter = Identity**（在根进程上）：
```
Scatter: P0:[ABCD] → P0:[A], P1:[B], P2:[C], P3:[D]
Gather:  P0:[A], P1:[B], P2:[C], P3:[D] → P0:[ABCD]
```

## 集合通信的层次结构

集合通信可以用更简单的原语实现：

```
                AllReduce
                /       \
        ReduceScatter   AllGather
           /    \        /    \
       Reduce  Scatter  Gather  Broadcast
          \      |        |      /
           \     |        |     /
            Send/Recv (点对点)
```

每一层都增加抽象。较低层为较高层提供构建块。

## 实现：NCCL

NVIDIA Collective Communications Library（NCCL）提供优化实现：

```cpp
// AllReduce
ncclAllReduce(sendbuff, recvbuff, count, datatype,
              ncclSum, comm, stream);

// ReduceScatter
ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype,
                   ncclSum, comm, stream);

// AllGather
ncclAllGather(sendbuff, recvbuff, sendcount, datatype,
              comm, stream);
```

**NCCL 会根据以下因素自动选择算法**：

- 消息大小（环形 vs 树形 vs 直接）
- 拓扑（NVLink vs PCIe vs 网络）
- GPU 数量

### 算法选择

| 大小 | 算法 | 原因 |
|------|------|------|
| < 256 KB | 树形 | 延迟受限，最小化跳数 |
| 256 KB - 4 MB | 混合 | 平衡延迟与带宽 |
| > 4 MB | 环形 | 带宽受限，最大化吞吐 |

## 一致性模型

### 整体同步并行（BSP）

所有进程完成集合通信后，任何进程才继续。

```python
# 集合通信中隐含屏障
gradients = allreduce(local_gradients)
# 此处所有进程拥有相同梯度
weights -= lr * gradients
```

**优点**：确定性，易于推理
**缺点**：最慢的进程决定速度（掉队者）

### 异步

进程不等待，使用陈旧的值。

```python
# 非阻塞
handle = allreduce_async(local_gradients)
# ... 做其他工作 ...
gradients = handle.wait()  # 最终完成
```

**优点**：没有掉队者问题
**缺点**：收敛可能受陈旧性影响

## 练习题

1. **分解验证**：用 ReduceScatter 后接 AllGather 实现 AllReduce，验证两者产生相同结果。

??? success "解答"
    **用 ReduceScatter + AllGather 实现 AllReduce：**

    ```python
    import numpy as np

    def reduce_scatter(data, P, op=np.sum):
        """跨进程归约，并分散结果分片。"""
        # 堆叠所有数据：形状 (P, n)
        stacked = np.stack(data)
        # 沿进程维度归约
        reduced = op(stacked, axis=0)
        # 分散：每个进程得到结果的 1/P
        chunk_size = len(reduced) // P
        return [reduced[i*chunk_size:(i+1)*chunk_size] for i in range(P)]

    def all_gather(shards, P):
        """把所有分片收集到所有进程。"""
        gathered = np.concatenate(shards)
        return [gathered.copy() for _ in range(P)]

    def allreduce_via_decomposition(data, P):
        """AllReduce = AllGather ∘ ReduceScatter"""
        shards = reduce_scatter(data, P)
        return all_gather(shards, P)

    def allreduce_direct(data, P, op=np.sum):
        """用于对比的直接 AllReduce。"""
        stacked = np.stack(data)
        reduced = op(stacked, axis=0)
        return [reduced.copy() for _ in range(P)]

    # 测试
    P = 4
    data = [np.array([1, 2, 3, 4]),
            np.array([2, 4, 6, 8]),
            np.array([3, 6, 9, 12]),
            np.array([4, 8, 12, 16])]

    decomposed = allreduce_via_decomposition(data, P)
    direct = allreduce_direct(data, P)

    # 验证：两者都应在所有进程上给出 [10, 20, 30, 40]
    assert all(np.array_equal(d, [10, 20, 30, 40]) for d in decomposed)
    assert all(np.array_equal(d, [10, 20, 30, 40]) for d in direct)
    print("✓ 分解验证通过！")
    ```

    **结果：** 两者在全部 4 个进程上都产生 `[10, 20, 30, 40]`。

2. **结合律测试**：编写代码演示浮点不满足结合律。找出树形归约与顺序归约结果不同的输入。

??? success "解答"
    **演示浮点不满足结合律：**

    ```python
    import numpy as np

    def sequential_sum(values):
        """从左到右的顺序归约。"""
        result = values[0]
        for v in values[1:]:
            result = result + v
        return result

    def tree_sum(values):
        """成对树形归约。"""
        if len(values) == 1:
            return values[0]
        pairs = []
        for i in range(0, len(values), 2):
            if i + 1 < len(values):
                pairs.append(values[i] + values[i+1])
            else:
                pairs.append(values[i])
        return tree_sum(pairs)

    # 例子 1：灾难性抵消
    a = np.float32(1e20)
    b = np.float32(-1e20)
    c = np.float32(1.0)

    seq_result = (a + b) + c  # = 0 + 1 = 1
    alt_result = a + (b + c)  # = 1e20 + (-1e20) = 0（c 被吸收）

    print(f"顺序: (a + b) + c = {seq_result}")
    print(f"另一种: a + (b + c) = {alt_result}")
    print(f"差异: {abs(seq_result - alt_result)}")

    # 例子 2：许多小值
    np.random.seed(42)
    values = [np.float32(1e-8) for _ in range(1000000)]
    values.append(np.float32(1.0))

    seq = sequential_sum(values)
    tree = tree_sum(values)
    exact = 1.0 + 1000000 * 1e-8  # = 1.01

    print(f"\n顺序求和: {seq}")
    print(f"树形求和: {tree}")
    print(f"精确值: {exact}")
    ```

    **输出：**
    ```
    Sequential: (a + b) + c = 1.0
    Alternate:  a + (b + c) = 0.0
    Difference: 1.0

    Sequential sum: 1.0100002
    Tree sum: 1.0099999
    Exact value: 1.01
    ```

    **关键洞察：** 由于运算分组不同，树形归约可能与顺序归约给出不同结果。两者都不等于精确的数学结果。

3. **通信量**：对 $P$ 个进程、$n$ 字节张量的 AllReduce：

   - 计算朴素 reduce + broadcast 的总发送字节数
   - 计算环形 AllReduce 的总发送字节数
   - 改善因子是多少？

??? success "解答"
    **通信量对比：**

    **朴素 Reduce + Broadcast：**

    - Reduce：$P-1$ 个进程各向根发送 $n$ 字节
      - 总发送：$(P-1) \times n$
    - Broadcast：根向 $P-1$ 个进程各发送 $n$ 字节
      - 总发送：$(P-1) \times n$（树形为 $n \log P$）

    **朴素总计：** $2(P-1) \times n$ 字节（树形广播为 $(P-1 + \log P) \times n$）

    **环形 AllReduce：**

    - 每个进程发送：$2 \times \frac{P-1}{P} \times n$ 字节
    - 所有进程合计：$2 \times \frac{P-1}{P} \times n \times P = 2(P-1)n$

    **等等——总量相同？** 是的，但差异在于**每进程**和**每链路**负载：

    | 指标 | 朴素 | 环形 |
    |------|------|------|
    | 根发送 | $(P-1)n$ | $\frac{2(P-1)}{P}n$ |
    | 根接收 | $(P-1)n$ | $\frac{2(P-1)}{P}n$ |
    | 最大链路负载 | $(P-1)n$ | 每步 $\frac{n}{P}$ |

    **根瓶颈的改善因子：**
    $$\frac{(P-1)n}{\frac{2(P-1)}{P}n} = \frac{P}{2}$$

    对 $P = 64$：环形把根负载降低 **32×**。

    **关键洞察：** 总字节数相近，但环形把负载均匀分布到各进程和时间上，消除了根瓶颈。

4. **Gather/Scatter 恒等**：证明对任意向量 $x$：$\text{Gather}(\text{Scatter}(x)) = x$（在根进程上）。

??? success "解答"
    **证明 Gather(Scatter(x)) = x（在根进程上）：**

    **Scatter 操作：** 根把块 $i$ 发送给进程 $i$：

    $$\text{Scatter}([x_0, x_1, ..., x_{P-1}]) \to (P_0: x_0, P_1: x_1, ..., P_{P-1}: x_{P-1})$$

    **Gather 操作：** 根从每个进程接收块并拼接：

    $$\text{Gather}(P_0: y_0, P_1: y_1, ..., P_{P-1}: y_{P-1}) \to [y_0, y_1, ..., y_{P-1}]$$

    **复合：**
    $$\text{Gather}(\text{Scatter}([x_0, x_1, ..., x_{P-1}]))$$

    $$= \text{Gather}(P_0: x_0, P_1: x_1, ..., P_{P-1}: x_{P-1})$$

    $$= [x_0, x_1, ..., x_{P-1}]$$

    $$= x$$

    **因此：** $\text{Gather} \circ \text{Scatter} = \text{Identity}$（在根进程上）$\square$

    **注意：** 这只在根进程上成立。Gather 之后其他进程没有完整结果。

5. **AlltoAll 分析**：推导 AlltoAll 的通信量。语义不同，为什么通信量与 AllGather 相同？

??? success "解答"
    **AlltoAll 通信量：**

    **设定：** 每个进程 $i$ 有数据 $[D_{i,0}, D_{i,1}, ..., D_{i,P-1}]$，其中 $D_{i,j}$ 发往进程 $j$。

    **数据移动：** 进程 $i$ 对每个 $j \neq i$ 把 $D_{i,j}$ 发送给进程 $j$。

    **每进程：**
    - 发送：$(P-1)$ 个大小为 $n/P$ 的块 = $\frac{P-1}{P} \times n$ 字节
    - 接收：$(P-1)$ 个大小为 $n/P$ 的块 = $\frac{P-1}{P} \times n$ 字节

    **每进程总计：** 发送 $\frac{P-1}{P} \times n$ + 接收 $\frac{P-1}{P} \times n$

    **AllGather 对比：**

    - 每个进程贡献 $n/P$，总共接收 $n$
    - 发送：$\frac{P-1}{P} \times n$ 字节（把自己的块发给另外 $P-1$ 个进程）
    - 接收：$\frac{P-1}{P} \times n$ 字节（从另外 $P-1$ 个进程接收）

    **通信量相同！** 两者每进程都移动 $\frac{P-1}{P} \times n$ 字节。

    **为什么语义不同但量相同？**

    | 操作 | 发送 | 接收 | 模式 |
    |------|------|------|------|
    | AllGather | 相同数据给所有进程 | 从每个进程接收不同数据 | 一对多 |
    | AlltoAll | 不同数据给每个进程 | 从每个进程接收不同数据 | 多对多 |

    总数据移动量完全相同；只是**模式**不同。AlltoAll 是一种“个性化”的 AllGather：每个目的地收到唯一数据。

6. **算法选择**：给定 α = 1μs，β = 100 GB/s，对 P = 64，消息多大时环形 AllReduce 开始快于树形 AllReduce？

??? success "解答"
    **P = 64 时环形与树形的交叉点：**

    **已知：** $\alpha = 1\mu s = 10^{-6}$ s，$\beta = 100$ GB/s $= 10^{11}$ bytes/s，$P = 64$

    **环形 AllReduce 时间：**
    $$T_{\text{ring}} = 2(P-1)\alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

    $$= 2(63)(10^{-6}) + 2 \cdot \frac{63}{64} \cdot \frac{n}{10^{11}}$$

    $$= 126 \mu s + 1.97 \cdot 10^{-11} \cdot n$$

    **树形 AllReduce 时间：**
    $$T_{\text{tree}} = 2\log_2 P \cdot \alpha + 2\log_2 P \cdot \frac{n}{\beta}$$

    $$= 2(6)(10^{-6}) + 2(6) \cdot \frac{n}{10^{11}}$$

    $$= 12 \mu s + 1.2 \cdot 10^{-10} \cdot n$$

    **令两者相等求交叉点：**
    $$126 \mu s + 1.97 \times 10^{-11} n = 12 \mu s + 1.2 \times 10^{-10} n$$

    $$114 \mu s = (1.2 \times 10^{-10} - 1.97 \times 10^{-11}) n$$

    $$114 \times 10^{-6} = 1.003 \times 10^{-10} \cdot n$$

    $$n = \frac{114 \times 10^{-6}}{1.003 \times 10^{-10}} = \boxed{1.14 \text{ MB}}$$

    **结论：**
    - 消息 < 1.14 MB：使用**树形**（更低延迟）
    - 消息 > 1.14 MB：使用**环形**（更好带宽）

    | 消息大小 | 环形时间 | 树形时间 | 胜者 |
    |----------|----------|----------|------|
    | 100 KB | 128 μs | 24 μs | 树形 |
    | 1 MB | 146 μs | 132 μs | 树形 |
    | 10 MB | 324 μs | 1.2 ms | 环形 |
    | 100 MB | 2.1 ms | 12 ms | 环形 |

## 关键要点

1. **七种原语就够了**：所有分布式训练通信都使用这些构建块。

2. **AllReduce = ReduceScatter + AllGather**：这种分解是基础性的。

3. **代数性质很重要**：结合律支持树形归约；交换律支持顺序无关。

4. **浮点破坏结合律**：这会影响可复现性。

5. **算法选择取决于大小**：小消息 → 树形；大消息 → 环形。

6. **NCCL 处理复杂性**：根据拓扑和大小自动选择算法。