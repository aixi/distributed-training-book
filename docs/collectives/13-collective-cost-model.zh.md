---
title: "集合通信成本模型"
subtitle: "预测任意操作的通信时间"
---

<div class="chapter-opener" markdown>
借助 α-β 分析和算法知识，我们可以在运行之前就预测任意集合通信（collective communication）操作的成本。这种预测能力对容量规划（capacity planning）至关重要。
</div>

<div class="investigation-question" markdown>
**问题**：你的一次 AllReduce 耗时 500 毫秒。这个成绩是好是坏？你怎么知道自己是接近了最优性能？理论值与实测值之间的差距有多大，这个差距又是从哪里来的？
</div>

!!! abstract "本章导览"
    **前置知识**：[第 4 章](../foundations/04-alpha-beta-model.md)（α-β 模型基础）、[第 12 章](12-ring-tree-algorithms.md)（环形与树形算法）

    **核心洞见**：每一种集合通信操作都有一个可预测的成本公式。将实测时间与这些公式对比，就能找出低效之处——差距揭示了你是受限于延迟、带宽，还是实现开销。

## 完整的成本公式

利用 α-β 模型，我们可以预测任意集合通信操作的通信时间。

!!! note "理论"
    以下公式是在理想假设下（完全重叠、无竞争）得到的算法下界。

!!! note "实践"
    真实系统通常能达到这些下界的约 70%~90%，具体取决于拓扑、内核效率以及重叠程度。

### 点对点操作

**Send/Recv**：

$$T = \alpha + \frac{n}{\beta}$$

这是基础。所有集合通信的成本都来自点对点操作的组合。

### Broadcast 与 Reduce

**Broadcast**（树形算法）：

$$T_{\text{broadcast}} = \log_2 P \cdot \alpha + \log_2 P \cdot \frac{n}{\beta}$$

**Reduce**（树形算法）：

$$T_{\text{reduce}} = \log_2 P \cdot \alpha + \log_2 P \cdot \frac{n}{\beta}$$

**注意**：两者都使用树形算法，因为它们只有一个源或一个目的地。

### Scatter 与 Gather

**Scatter**（二项树）：

$$T_{\text{scatter}} = \log_2 P \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

带宽项之所以更优，是因为每下降一层，数据量就减半。

**Gather**（二项树）：

$$T_{\text{gather}} = \log_2 P \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

### AllReduce

**环形算法**（大消息下最优）：

$$T_{\text{AllReduce}}^{\text{ring}} = 2(P-1) \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

**树形算法**（小消息下最优）：

$$T_{\text{AllReduce}}^{\text{tree}} = 2\log_2 P \cdot \alpha + 2\log_2 P \cdot \frac{n}{\beta}$$

**递归减半-倍增**（RHD，当 P 为 2 的幂时兼具两者之长）：

$$T_{\text{AllReduce}}^{\text{RHD}} = 2\log_2 P \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

!!! note "实践"
    将实测的 NCCL 时间与上述公式对比。如果低于模型值的 70%，那你很可能要么受限于延迟（小桶太多），要么受限于带宽（竞争）。对于延迟，用更大的桶；对于带宽，减少重叠带来的竞争。

### AllGather

**环形算法**：

$$T_{\text{AllGather}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

其中 $n$ 是输出总大小（每个进程贡献 $n/P$）。

### ReduceScatter

**环形算法**：

$$T_{\text{ReduceScatter}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

其中 $n$ 是输入总大小（每个进程输出 $n/P$）。

### All-to-All

**两两交换**：

$$T_{\text{All-to-All}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

其中 $n$ 是每个进程的总数据量（向其他每个进程各发送 $n/P$）。

### 汇总表

| 集合通信 | 延迟项 | 带宽项 | 算法 |
|------------|--------------|----------------|-----------|
| Broadcast | $\log_2 P \cdot \alpha$ | $\log_2 P \cdot \frac{n}{\beta}$ | 树形 |
| Reduce | $\log_2 P \cdot \alpha$ | $\log_2 P \cdot \frac{n}{\beta}$ | 树形 |
| Scatter | $\log_2 P \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | 二项树 |
| Gather | $\log_2 P \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | 二项树 |
| AllReduce | $2(P-1) \cdot \alpha$ | $2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$ | 环形 |
| AllGather | $(P-1) \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | 环形 |
| ReduceScatter | $(P-1) \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | 环形 |
| All-to-All | $(P-1) \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | 两两交换 |

## 算法带宽

**算法带宽**（algorithmic bandwidth，algbw）衡量相对于朴素基准的数据搬移速率。

### 定义

$$\text{algbw} = \frac{n}{T_{\text{collective}}}$$

这是"有效"带宽：从应用视角看，数据"表面上"移动得有多快。

### 总线带宽

对 NCCL 而言，**总线带宽**（bus bandwidth，busbw）考虑了一个事实：数据必须流经多条链路：

$$\text{busbw} = \text{algbw} \times \text{correction factor}$$

校正因子取决于具体的集合通信操作：

| 集合通信 | 校正因子 |
|------------|-------------------|
| Broadcast | 1 |
| Reduce | 1 |
| AllReduce | $\frac{2(P-1)}{P}$ |
| AllGather | $\frac{P-1}{P}$ |
| ReduceScatter | $\frac{P-1}{P}$ |
| All-to-All（AlltoAll） | $\frac{P-1}{P}$ |

### 示例计算

在 8 块 GPU 上对 1 GB 做 AllReduce，耗时 50 毫秒。

**算法带宽**：

$$\text{algbw} = \frac{10^9 \text{ bytes}}{0.05 \text{ s}} = 20 \text{ GB/s}$$

**总线带宽**：

$$\text{busbw} = 20 \times \frac{2(8-1)}{8} = 20 \times \frac{14}{8} = 35 \text{ GB/s}$$

如果你的网卡是 400 Gbps = 50 GB/s，那么你达到了峰值的 70%——性能相当不错！

## 层次化成本分析

真实集群存在网络层次结构。一台 DGX 集群可能有：

- **节点内（Intra-node）**：8 块 GPU 通过 NVLink 互联（每块 GPU 900 GB/s）
- **节点间（Inter-node）**：8 块 400 Gbps 网卡（每节点 400 GB/s）

### 两层次 AllReduce 模型

设每节点 $G$ 块 GPU、共 $N$ 个节点（总数 $P = GN$）：

**阶段 1：节点内 ReduceScatter**
$$T_1 = (G-1) \cdot \alpha_{\text{NV}} + \frac{G-1}{G} \cdot \frac{n}{\beta_{\text{NV}}}$$

现在每块 GPU 持有 $n/G$ 的部分结果。

**阶段 2：节点间 AllReduce**
$$T_2 = 2(N-1) \cdot \alpha_{\text{net}} + 2 \cdot \frac{N-1}{N} \cdot \frac{n/G}{\beta_{\text{net}}}$$

每块 GPU 只有 $n/G$ 字节跨网络传输！

**阶段 3：节点内 AllGather**
$$T_3 = (G-1) \cdot \alpha_{\text{NV}} + \frac{G-1}{G} \cdot \frac{n}{\beta_{\text{NV}}}$$

**总计**：

$$T_{\text{hier}} = T_1 + T_2 + T_3$$

### 数值示例

**场景**：8 节点 × 8 GPU = 64 块 GPU，对 2 GB 做 AllReduce

**参数**：

- $\alpha_{\text{NV}} = 1 \mu s$，$\beta_{\text{NV}} = 300$ GB/s（NVSwitch）
- $\alpha_{\text{net}} = 5 \mu s$，$\beta_{\text{net}} = 50$ GB/s（400 GbE）

**阶段 1（节点内 RS）**：

$$T_1 = 7 \times 10^{-6} + \frac{7}{8} \times \frac{2 \times 10^9}{3 \times 10^{11}} = 7\mu s + 5.8\text{ms} = 5.81\text{ms}$$

**阶段 2（节点间 AR，每块 GPU 256 MB）**：

$$T_2 = 14 \times 5 \times 10^{-6} + \frac{14}{8} \times \frac{2.5 \times 10^8}{5 \times 10^{10}} = 70\mu s + 8.75\text{ms} = 8.82\text{ms}$$

**阶段 3（节点内 AG）**：

$$T_3 = 7 \times 10^{-6} + \frac{7}{8} \times \frac{2 \times 10^9}{3 \times 10^{11}} = 5.81\text{ms}$$

**层次化总计**：$T_{\text{hier}} = 5.81 + 8.82 + 5.81 = 20.44$ 毫秒

**与扁平环形对比**（64 块 GPU，瓶颈在网络上）：

$$T_{\text{flat}} = 126 \times 5\mu s + \frac{126}{64} \times \frac{2 \times 10^9}{5 \times 10^{10}} = 0.63\text{ms} + 78.75\text{ms} = 79.38\text{ms}$$

**层次化快了 3.9 倍**，因为只有 1/8 的数据跨过慢速网络！

## 效率差距

在实践中，实测性能总是低于理论值。差距来自以下几个方面：

### 1. 软件开销

NCCL、驱动和内核启动都会在 $\alpha$ 之外引入额外延迟：

$$T_{\text{actual}} = T_{\text{theory}} + T_{\text{sw}}$$

典型的软件开销为每次操作 10-50 微秒。

### 2. 协议开销

包头、校验和、重传都会带来带宽开销：

$$\beta_{\text{effective}} = \beta_{\text{raw}} \times (1 - \text{overhead})$$

典型开销为原始带宽的 5%~15%。

### 3. 竞争

多个集合通信操作，或来自其他作业的流量：

$$\beta_{\text{contention}} = \frac{\beta_{\text{raw}}}{\text{contention factor}}$$

共享集群上常常出现 30%~50% 的带宽下降。

### 4. 内存拷贝

从 GPU 显存到网卡暂存缓冲区（staging buffer）会带来额外时间：

$$T_{\text{copy}} = \frac{n}{\beta_{\text{PCIe}}}$$

对 1 GB 数据走 PCIe Gen5：$\frac{10^9}{64 \times 10^9} \approx 16$ 毫秒

### 5. 同步

BSP（整体同步并行）模型要求所有进程等待最慢的那一个：

$$T_{\text{sync}} = T_{\text{max}} - T_{\text{median}}$$

拖后腿（straggler）效应通常带来 5%~20% 的额外开销。

## 测量参数

### 延迟（$\alpha$）

用极小的消息测量往返时间：

```python
import torch.distributed as dist
import time

def measure_alpha(iterations=1000):
    tensor = torch.zeros(1, device='cuda')

    # Warmup
    for _ in range(100):
        dist.all_reduce(tensor)

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # For ring AllReduce: T ≈ 2(P-1)α for tiny messages
    P = dist.get_world_size()
    alpha = elapsed / (iterations * 2 * (P - 1))

    return alpha  # seconds
```

### 带宽（$\beta$）

用大消息测量吞吐量：

```python
def measure_beta(size_gb=1.0, iterations=10):
    size = int(size_gb * 1e9 / 4)  # float32 elements
    tensor = torch.zeros(size, device='cuda')

    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor)

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # For ring AllReduce: T ≈ 2(P-1)/P * n/β for large messages
    P = dist.get_world_size()
    n = size_gb * 1e9  # bytes
    factor = 2 * (P - 1) / P

    beta = factor * n * iterations / elapsed

    return beta  # bytes/second
```

### NCCL-tests

集合通信基准测试的标准工具：

```bash
# Build
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make

# Run AllReduce benchmark
./build/all_reduce_perf -b 1M -e 1G -f 2 -g 8

# Output interpretation:
#   busbw: bus bandwidth (accounting for algorithm)
#   algbw: algorithmic bandwidth (raw n/T)
```

## 预测训练的通信时间

对一次训练迭代（step），总通信时间包括：

### 数据并行 AllReduce

$$T_{\text{DP}} = 2(P-1)\alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n_{\text{grad}}}{\beta}$$

其中 $n_{\text{grad}}$ = 梯度总大小（约等于模型参数量 × 4 字节，FP32 情况）。

### ZeRO-3 AllGather + ReduceScatter

每一层两次（前向 + 反向）：

$$T_{\text{ZeRO}} = 2 \times \left[ (P-1)\alpha + \frac{P-1}{P} \cdot \frac{n_{\text{layer}}}{\beta} \right]$$

### 张量并行 AllReduce

每个 Transformer 层：

$$T_{\text{TP}} = 4 \times T_{\text{AllReduce}}(n_{\text{activation}})$$

（每层前向 2 次 AllReduce、反向 2 次）

### 流水线并行（Pipeline Parallelism） Send/Recv

每个微批次：

$$T_{\text{PP}} = 2 \times \left( \alpha + \frac{n_{\text{activation}}}{\beta} \right)$$

### 完整示例：70B 模型

**场景**：70B 参数，FP16，512 块 GPU，8 路 TP、8 路 DP、8 路 PP

**参数**：

- 梯度大小：70B × 2 字节 = 140 GB（但按 8 路分片，因此每个 DP 组 17.5 GB）
- TP 激活：8192 × 4096 × 2 = 每层 64 MB
- PP 激活：8192 × 4096 × 2 = 每个微批次 64 MB
- 80 层，8 个微批次

**TP 通信**（8 块 GPU，NVLink β = 300 GB/s）：

$$T_{\text{TP}} = 80 \times 4 \times \left( 14 \times 1\mu s + \frac{14}{8} \times \frac{64 \times 10^6}{3 \times 10^{11}} \right)$$

$$= 320 \times (14\mu s + 0.37\text{ms}) = 123\text{ms}$$

**DP 通信**（8 块 GPU，网络 β = 50 GB/s）：

$$T_{\text{DP}} = 14 \times 5\mu s + \frac{14}{8} \times \frac{17.5 \times 10^9}{5 \times 10^{10}}$$

$$= 70\mu s + 612.5\text{ms} = 612.6\text{ms}$$

**PP 通信**（8 个阶段，8 个微批次）：

$$T_{\text{PP}} = 8 \times 2 \times \left( 5\mu s + \frac{64 \times 10^6}{5 \times 10^{10}} \right) = 16 \times 1.28\text{ms} = 20.5\text{ms}$$

**总通信估算**：123 + 613 + 21 = **757 毫秒**

如果计算时间是 1500 毫秒，那么通信开销占 **50%**——必须优化！

## 优化策略

有了成本模型，我们就可以推理各种优化手段：

### 重叠通信与计算

把 $T_{\text{comm}}$ 隐藏在 $T_{\text{compute}}$ 之下：

$$T_{\text{total}} = \max(T_{\text{compute}}, T_{\text{comm}})$$

而不是：

$$T_{\text{total}} = T_{\text{compute}} + T_{\text{comm}}$$

### 缩小消息大小

- **梯度压缩（Gradient Compression）**：将 $n$ 缩小 10-100 倍
- **混合精度（mixed precision）**：FP16 相比 FP32 把 $n$ 减半
- **梯度累积（Gradient Accumulation）**：减少 AllReduce 调用次数

### 提升有效带宽

- **更好的算法**：环形与树形的选择
- **层次化集合通信**：利用拓扑结构
- **桶融合**（bucket fusion）：摊薄延迟

### 降低并行度

如果 $P$ 很大而消息很小：

$$T \approx P \cdot \alpha \gg \frac{n}{\beta}$$

考虑：更小的 DP 组，更多 TP/PP。

## 验证：预测与实测

务必用实测数据验证你的模型。

### 方法论

1. 在你自己的具体集群上**测量 α 和 β**
2. 用公式**预测时间**
3. 用性能分析器（profiler）**测量实际时间**
4. **计算误差**：$\frac{|T_{\text{pred}} - T_{\text{actual}}|}{T_{\text{actual}}}$

### 可接受的误差

- < 10%：模型拟合极佳
- 10%~30%：可用于规划
- > 30%：模型的假设被违反了

### 误差过大的常见原因

1. **竞争**：共享网络上有其他作业
2. **拓扑不符**：α、β 是在不同路径上测得的
3. **协议效应**：小消息的协议开销
4. **显存压力**：频繁换页影响拷贝带宽

!!! tip "下一章"
    本章假设网络是均匀的。下一章[拓扑感知的集合通信](13a-topology-aware-collectives.md)将把模型扩展到层次化拓扑以及感知竞争（contention-aware）的调度。

## 练习题

1. **公式验证**：对 P=16、n=100 MB、α=10μs、β=100 GB/s，计算：

   - AllReduce 时间（环形算法）
   - AllGather 时间
   - 层次化（4 节点 × 4 GPU）相对扁平的加速比

??? success "解答"
    **已知参数：**

    - P = 16 块 GPU
    - n = 100 MB = 10⁸ 字节
    - α = 10 μs = 10⁻⁵ 秒
    - β = 100 GB/s = 10¹¹ 字节/秒

    **第 1 部分：AllReduce 时间（环形算法）**

    使用环形 AllReduce 公式：

    $$T_{\text{AllReduce}} = 2(P-1) \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

    延迟项：

    $$T_\alpha = 2(16-1) \times 10^{-5} = 30 \times 10^{-5} = 0.3 \text{ ms}$$

    带宽项：

    $$T_\beta = 2 \times \frac{15}{16} \times \frac{10^8}{10^{11}} = 2 \times 0.9375 \times 10^{-3} = 1.875 \text{ ms}$$

    $$\boxed{T_{\text{AllReduce}} = 0.3 + 1.875 = 2.175 \text{ ms}}$$

    **第 2 部分：AllGather 时间**

    使用环形 AllGather 公式：

    $$T_{\text{AllGather}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

    $$T_{\text{AllGather}} = 15 \times 10^{-5} + \frac{15}{16} \times \frac{10^8}{10^{11}}$$

    $$= 0.15 \text{ ms} + 0.9375 \text{ ms} = \boxed{1.09 \text{ ms}}$$

    **第 3 部分：层次化（4×4）相对扁平的加速比**

    **扁平环形（P=16）：** 已算出：T_flat = 2.175 ms

    **层次化（4 节点 × 每节点 4 GPU）：**

    为简单起见假设 α 和 β 相同（实际中节点内 NVLink 会更快）：

    *阶段 1：节点内 ReduceScatter（G=4）：*
    $$T_1 = (4-1) \times 10^{-5} + \frac{3}{4} \times \frac{10^8}{10^{11}} = 0.03 + 0.75 = 0.78 \text{ ms}$$

    *阶段 2：节点间 AllReduce（N=4，数据量 = n/G = 25 MB）：*
    $$T_2 = 2(4-1) \times 10^{-5} + 2 \times \frac{3}{4} \times \frac{2.5 \times 10^7}{10^{11}}$$

    $$= 0.06 + 0.375 = 0.435 \text{ ms}$$

    *阶段 3：节点内 AllGather（G=4）：*
    $$T_3 = (4-1) \times 10^{-5} + \frac{3}{4} \times \frac{10^8}{10^{11}} = 0.78 \text{ ms}$$

    $$T_{\text{hier}} = 0.78 + 0.435 + 0.78 = 1.995 \text{ ms}$$

    **加速比：**
    $$\text{Speedup} = \frac{T_{\text{flat}}}{T_{\text{hier}}} = \frac{2.175}{1.995} = \boxed{1.09\times}$$

    **注意：** 在 α、β 均匀的情况下，层次化的加速比很有限。真正的收益出现在网络 β 远慢于 NVLink β 时（例如 50 GB/s 对 300 GB/s），此时可达 3-4 倍加速。

2. **效率分析**：你在 8 块 GPU 上对 1 GB 做 AllReduce 耗时 80 毫秒。网络为 400 Gbps。计算：

   - 算法带宽
   - 总线带宽
   - 相对理论峰值的效率

??? success "解答"
    **已知：**

    - P = 8 块 GPU
    - n = 1 GB = 10⁹ 字节
    - T = 80 ms = 0.08 秒
    - 网络 = 400 Gbps = 50 GB/s

    **第 1 部分：算法带宽**

    $$\text{algbw} = \frac{n}{T} = \frac{10^9}{0.08} = 12.5 \text{ GB/s}$$

    $$\boxed{\text{algbw} = 12.5 \text{ GB/s}}$$

    **第 2 部分：总线带宽**

    对 AllReduce，校正因子为 $\frac{2(P-1)}{P}$：

    $$\text{busbw} = \text{algbw} \times \frac{2(P-1)}{P} = 12.5 \times \frac{2 \times 7}{8}$$

    $$= 12.5 \times 1.75 = \boxed{21.875 \text{ GB/s}}$$

    **第 3 部分：相对理论峰值的效率**

    理论峰值即网络带宽：

    $$\text{Efficiency} = \frac{\text{busbw}}{\beta_{\text{peak}}} = \frac{21.875}{50} = \boxed{43.75\%}$$

    **分析：**

    | 指标 | 数值 |
    |--------|-------|
    | 算法带宽 | 12.5 GB/s |
    | 总线带宽 | 21.875 GB/s |
    | 峰值带宽 | 50 GB/s |
    | 效率 | 43.75% |

    **这个效率令人担忧。** 可能的原因：

    1. **软件开销**：NCCL 内核启动、同步
    2. **协议开销**：包头、校验和降低了有效带宽
    3. **竞争**：共享网络上的其他流量
    4. **内存拷贝**：GPU 与网卡之间的 PCIe 瓶颈

    **良好优化系统的预期效率**：70%~85%

    **要排查**，可运行不同消息大小的 NCCL-tests，判断问题出在受限于延迟（小消息）还是受限于带宽（大消息）。

3. **训练时间预测**：一个 13B 模型满足：

   - 注意力部分有 4B 参数（TP）
   - FFN 部分有 9B 参数（TP）
   - 序列长度 4096，隐藏维度 5120，批 2M token
   - 64 块 GPU：8 TP × 8 DP

   估算每次迭代的通信时间。哪种并行方式占主导？

??? success "解答"
    **场景：**

    - 总计 13B 参数（4B 注意力 + 9B FFN）
    - TP = 8，DP = 8（共 64 块 GPU）
    - 序列 S = 4096，隐藏维度 H = 5120
    - 批 2M token → 每个 DP 副本的微批次 = 2M/8 = 250K token
    - 每 GPU 批大小：B = 250K / 4096 ≈ 61 条序列

    **假设：**

    - NVLink β = 300 GB/s（节点内，用于 TP）
    - 网络 β = 50 GB/s（节点间，用于 DP）
    - α_NV = 1 μs，α_net = 5 μs

    **第 1 部分：张量并行（Tensor Parallelism）通信**

    TP 需要在每个 TP 组内对激活值做 AllReduce。

    每层激活大小：

    $$n_{\text{act}} = B \times S \times H \times 2 \text{ bytes (FP16)}$$

    $$= 61 \times 4096 \times 5120 \times 2 = 2.56 \text{ GB}$$

    每个 Transformer 层：4 次 AllReduce 操作（前向 2 次，反向 2 次）

    每次 AllReduce（环形，P=8，NVLink）：

    $$T_{\text{AR}} = 2(8-1) \times 10^{-6} + 2 \times \frac{7}{8} \times \frac{2.56 \times 10^9}{3 \times 10^{11}}$$

    $$= 14 \mu s + 14.9 \text{ ms} = 14.9 \text{ ms}$$

    假设约 40 层：

    $$T_{\text{TP}} = 40 \times 4 \times 14.9 = \boxed{2,384 \text{ ms}}$$

    **第 2 部分：数据并行（Data Parallelism）通信**

    DP 需要在各 DP 组之间对梯度做 AllReduce。

    梯度大小：13B × 2 字节（FP16）= 共 26 GB

    但在 TP=8 时，每个 TP 组只有 26/8 = 3.25 GB 的独有梯度需要同步。

    跨 DP=8 个组做 AllReduce（网络）：

    $$T_{\text{DP}} = 2(8-1) \times 5 \times 10^{-6} + 2 \times \frac{7}{8} \times \frac{3.25 \times 10^9}{5 \times 10^{10}}$$

    $$= 70 \mu s + 113.75 \text{ ms} = \boxed{114 \text{ ms}}$$

    **汇总：**

    | 并行方式 | 通信时间 | 占比 |
    |-------------|-------------------|------------|
    | 张量并行 | 2,384 ms | 95.4% |
    | 数据并行 | 114 ms | 4.6% |
    | **总计** | **2,498 ms** | 100% |

    $$\boxed{\text{Tensor Parallelism dominates by } 21\times}$$

    **分析：**

    TP 占主导的原因：
    1. 激活张量很大（每次 AllReduce 2.56 GB）
    2. 每层 4 次 AllReduce × 40 层 = 160 次操作
    3. 即使有高速 NVLink，总量依然巨大

    **可考虑的优化：**

    1. **序列并行（Sequence Parallelism）**：减小每块 GPU 上的激活大小
    2. **激活检查点（checkpoint）**：用计算换内存（不直接改善通信）
    3. **降低 TP 度**：TP=4 而非 TP=8 可以让 TP 通信减半，但会增加每 GPU 显存
    4. **将 TP AllReduce 与计算重叠**：使用异步集合通信

4. **层次化收益**：你需要在以下两者之间选择：

   - 64 块 GPU：8 节点 × 8 GPU
   - 64 块 GPU：16 节点 × 4 GPU

   对 4 GB AllReduce，NVLink 300 GB/s、网络 50 GB/s，哪个更快？

??? success "解答"
    **已知：**

    - n = 4 GB = 4 × 10⁹ 字节
    - β_NV = 300 GB/s（NVLink，节点内）
    - β_net = 50 GB/s（网络，节点间）
    - α_NV = 1 μs，α_net = 5 μs

    **配置 A：8 节点 × 8 GPU（G=8，N=8）**

    *阶段 1：节点内 ReduceScatter（G=8）：*
    $$T_1 = (8-1) \times 10^{-6} + \frac{7}{8} \times \frac{4 \times 10^9}{3 \times 10^{11}}$$

    $$= 7 \mu s + 11.67 \text{ ms} = 11.67 \text{ ms}$$

    *阶段 2：节点间 AllReduce（N=8，数据量 = 4GB/8 = 每 GPU 500 MB）：*
    $$T_2 = 2(8-1) \times 5 \times 10^{-6} + 2 \times \frac{7}{8} \times \frac{5 \times 10^8}{5 \times 10^{10}}$$

    $$= 70 \mu s + 17.5 \text{ ms} = 17.57 \text{ ms}$$

    *阶段 3：节点内 AllGather（G=8）：*
    $$T_3 = (8-1) \times 10^{-6} + \frac{7}{8} \times \frac{4 \times 10^9}{3 \times 10^{11}} = 11.67 \text{ ms}$$

    $$T_A = 11.67 + 17.57 + 11.67 = \boxed{40.91 \text{ ms}}$$

    **配置 B：16 节点 × 4 GPU（G=4，N=16）**

    *阶段 1：节点内 ReduceScatter（G=4）：*
    $$T_1 = (4-1) \times 10^{-6} + \frac{3}{4} \times \frac{4 \times 10^9}{3 \times 10^{11}}$$

    $$= 3 \mu s + 10 \text{ ms} = 10 \text{ ms}$$

    *阶段 2：节点间 AllReduce（N=16，数据量 = 4GB/4 = 每 GPU 1 GB）：*
    $$T_2 = 2(16-1) \times 5 \times 10^{-6} + 2 \times \frac{15}{16} \times \frac{10^9}{5 \times 10^{10}}$$

    $$= 150 \mu s + 37.5 \text{ ms} = 37.65 \text{ ms}$$

    *阶段 3：节点内 AllGather（G=4）：*
    $$T_3 = 10 \text{ ms}$$

    $$T_B = 10 + 37.65 + 10 = \boxed{57.65 \text{ ms}}$$

    **对比：**

    | 配置 | 阶段 1 | 阶段 2 | 阶段 3 | 总计 |
    |---------------|---------|---------|---------|-------|
    | 8×8（A） | 11.67 ms | 17.57 ms | 11.67 ms | **40.91 ms** |
    | 16×4（B） | 10 ms | 37.65 ms | 10 ms | 57.65 ms |

    $$\boxed{\text{8×8 is 1.41× faster than 16×4}}$$

    **为什么 8×8 胜出：**

    1. **每节点 GPU 更多** → 跨慢速网络的数据更少
       - 8×8：每 GPU 500 MB 跨网络
       - 16×4：每 GPU 1 GB 跨网络（多 2 倍！）

    2. **节点数更少** → 节点间 AllReduce 步数更少
       - 8×8：N=8 → 2×7 = 14 个延迟步
       - 16×4：N=16 → 2×15 = 30 个延迟步

    **关键洞见：**尽可能提高每节点 GPU 数，以最小化网络流量。网络是瓶颈，因此把更多计算留在节点内是划算的。

5. **重叠潜力**：计算耗时 2000 毫秒，通信耗时 600 毫秒。如果能重叠 80% 的通信，加速比是多少？

??? success "解答"
    **已知：**

    - $T_{\text{compute}} = 2000$ 毫秒
    - $T_{\text{comm}} = 600$ 毫秒
    - 重叠比例 = 80%

    **无重叠（顺序执行）：**

    $$T_{\text{sequential}} = T_{\text{compute}} + T_{\text{comm}} = 2000 + 600 = 2600 \text{ ms}$$

    **重叠 80%：**

    重叠部分与计算并发执行。只有未重叠部分计入总时间：

    $$T_{\text{overlapped}} = T_{\text{compute}} + (1 - 0.80) \times T_{\text{comm}}$$

    $$= 2000 + 0.20 \times 600 = 2000 + 120 = 2120 \text{ ms}$$

    **加速比：**

    $$\text{Speedup} = \frac{T_{\text{sequential}}}{T_{\text{overlapped}}} = \frac{2600}{2120} = \boxed{1.23\times}$$

    **另一种视角——节省的时间：**

    $$\text{Time saved} = 2600 - 2120 = 480 \text{ ms}$$

    $$\text{Reduction} = \frac{480}{2600} = 18.5\%$$

    **分析：**

    | 场景 | 总时间 | 加速比 |
    |----------|------------|---------|
    | 无重叠（0%） | 2600 ms | 1.00× |
    | 80% 重叠 | 2120 ms | 1.23× |
    | 100% 重叠 | 2000 ms | 1.30× |

    **理论最大加速比**（完全重叠）：

    $$\text{Speedup}_{\text{max}} = \frac{T_{\text{compute}} + T_{\text{comm}}}{T_{\text{compute}}} = \frac{2600}{2000} = 1.30\times$$

    我们实现了理论最大收益的 $\frac{1.23 - 1.00}{1.30 - 1.00} = 77\%$。

    **实践考量：**

    1. **重叠技术**：梯度分桶、异步 AllReduce、流水线
    2. **为什么做不到 100%？**：某些操作需要同步（例如首/末层、优化器步）
    3. **内存代价**：重叠需要在通信期间缓存梯度

6. **参数测量**：设计一个实验，分别测量：

   - GPU-GPU 延迟（同节点）
   - GPU-GPU 延迟（跨节点）
   - NVLink 带宽
   - 网络带宽

??? success "解答"
    **α-β 参数测量的实验设计**

    关键洞见：用**极小消息隔离延迟（α）**，用**大消息隔离带宽（β）**。

    **实验 1：GPU-GPU 延迟（同节点）——α_NV**

    ```python
    import torch
    import torch.distributed as dist
    import time

    def measure_intra_node_latency(iterations=10000):
        """Measure NVLink latency using tiny AllReduce."""
        # Tiny tensor - bandwidth term negligible
        tensor = torch.zeros(1, device='cuda')

        # Warmup
        for _ in range(100):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # For ring AllReduce: T ≈ 2(P-1)α when n→0
        P = dist.get_world_size()  # Should be GPUs on same node
        alpha_nv = elapsed / (iterations * 2 * (P - 1))

        return alpha_nv * 1e6  # Return in microseconds
    ```

    **运行配置**：单节点，全部 8 块 GPU，NCCL 仅使用 NVLink。

    **预期结果**：α_NV ≈ 1-5 μs

    ---

    **实验 2：GPU-GPU 延迟（跨节点）——α_net**

    ```python
    def measure_inter_node_latency(iterations=10000):
        """Measure network latency using tiny AllReduce across nodes."""
        # Use ONLY one GPU per node to isolate network latency
        tensor = torch.zeros(1, device='cuda')

        # Warmup
        for _ in range(100):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        N = dist.get_world_size()  # Number of nodes
        alpha_net = elapsed / (iterations * 2 * (N - 1))

        return alpha_net * 1e6  # microseconds
    ```

    **运行配置**：每节点 1 块 GPU，多节点，仅使用网络。

    **预期结果**：α_net ≈ 5-20 μs（取决于网络拓扑）

    ---

    **实验 3：NVLink 带宽——β_NV**

    ```python
    def measure_nvlink_bandwidth(size_gb=2.0, iterations=20):
        """Measure NVLink bandwidth using large AllReduce."""
        size = int(size_gb * 1e9 / 4)  # float32 elements
        tensor = torch.zeros(size, device='cuda')

        # Warmup
        for _ in range(3):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # For large n: T ≈ 2(P-1)/P × n/β (latency negligible)
        P = dist.get_world_size()
        n = size_gb * 1e9  # bytes
        factor = 2 * (P - 1) / P

        # β = factor × n × iterations / elapsed
        beta_nv = factor * n * iterations / elapsed

        return beta_nv / 1e9  # Return in GB/s
    ```

    **运行配置**：单节点，全部 8 块 GPU，仅使用 NVLink。

    **预期结果**：β_NV ≈ 200-300 GB/s 每 GPU（NVSwitch）

    ---

    **实验 4：网络带宽——β_net**

    ```python
    def measure_network_bandwidth(size_gb=4.0, iterations=10):
        """Measure network bandwidth using large AllReduce across nodes."""
        size = int(size_gb * 1e9 / 4)
        tensor = torch.zeros(size, device='cuda')

        # Warmup
        for _ in range(3):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        N = dist.get_world_size()
        n = size_gb * 1e9
        factor = 2 * (N - 1) / N

        beta_net = factor * n * iterations / elapsed

        return beta_net / 1e9  # GB/s
    ```

    **运行配置**：每节点 1 块 GPU（避免 NVLink 的贡献），多节点。

    **预期结果**：β_net ≈ 40-50 GB/s 每 GPU（400 GbE）

    ---

    **完整的测量规程**

    | 参数 | 配置 | 消息大小 | 迭代次数 |
    |-----------|---------------|--------------|------------|
    | α_NV | 8 GPU，1 节点 | 4 字节 | 10,000 |
    | α_net | 1 GPU/节点，N 节点 | 4 字节 | 10,000 |
    | β_NV | 8 GPU，1 节点 | 2 GB | 20 |
    | β_net | 1 GPU/节点，N 节点 | 4 GB | 10 |

    **验证步骤：**

    1. **一致性检查**：每个实验跑 5 次，报告均值 ± 标准差
    2. **尺寸扫描**：消息大小从 1KB 变化到 10GB，画出 T 与 n 的关系
    3. **拟合 α-β 模型**：对 T = α + n/β 做线性回归
    4. **与规格对比**：NVLink 带宽应约为网络的 6 倍

    **示例结果表：**

    | 参数 | 实测 | 预期 | 状态 |
    |-----------|----------|----------|--------|
    | α_NV | 2.3 μs | 1-5 μs | ✓ |
    | α_net | 8.7 μs | 5-20 μs | ✓ |
    | β_NV | 285 GB/s | 250-300 GB/s | ✓ |
    | β_net | 47 GB/s | 40-50 GB/s | ✓ |

## 关键要点

1. **所有集合通信都有对应公式**：使用 α-β 模型，配合与算法相关的因子。

2. **总线带宽让比较归一化**：它计入了算法层面的数据放大效应。

3. **层次化能大幅降低网络负载**：降低幅度为每节点 GPU 数 G。

4. **真实系统存在开销**：按理论峰值的 70%~80% 来规划。

5. **先预测，再测量**：在实际基础设施上验证你的模型。

6. **规模增大时通信往往占主导**：理解成本才能进行优化。