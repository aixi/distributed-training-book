---
title: "快速参考"
---

本书最常用公式、决策表和环境变量的汇总参考。请收藏此页。

## 核心方程

### 训练计算量

$$C = 6\Psi D \qquad T_{\text{train}} = \frac{C}{P \cdot F \cdot \text{MFU}}$$

### 内存（混合精度，AdamW）

| 组件 | 公式 | 70B 模型 |
|------|------|----------|
| 参数（FP16） | $2\Psi$ | 140 GB |
| 梯度（FP16） | $2\Psi$ | 140 GB |
| 优化器（FP32） | $12\Psi$ | 840 GB |
| **静态总量** | $16\Psi$ | **1.12 TB** |

### 激活内存（每层，无检查点）

$$M_{\text{act}}^{\text{layer}} \approx BSH \cdot \left(34 + 5\frac{AS}{H}\right) \text{ bytes}$$

每 $k$ 层设检查点（最优 $k^* = \sqrt{L}$）：

$$M_{\text{act}}^{\text{ckpt}} \approx \frac{L}{k} \cdot 2BSH + k \cdot M_{\text{act}}^{\text{layer}}$$

### Chinchilla 最优分配

$$\Psi^* \approx \sqrt{\frac{C}{120}} \qquad D^* \approx 20\,\Psi^* \qquad \frac{D^*}{\Psi^*} \approx 20$$

### 每秒 token 数

$$\text{tokens/s} = \frac{P \cdot F \cdot \text{MFU}}{6\Psi}$$

## 通信代价公式（α-β 模型）

所有公式使用：$P$ = 进程数，$n$ = 消息大小（字节），$\alpha$ = 延迟，$\beta$ = 带宽（字节/秒）。

| 集合操作 | 延迟项 | 带宽项 | 适用于 |
|----------|--------|--------|--------|
| **AllReduce**（环形） | $2(P-1)\alpha$ | $2\frac{P-1}{P}\frac{n}{\beta}$ | 大消息 |
| **AllReduce**（树形） | $2\log_2 P \cdot \alpha$ | $2\log_2 P \cdot \frac{n}{\beta}$ | 小消息 |
| **AllReduce**（RHD） | $2\log_2 P \cdot \alpha$ | $2\frac{P-1}{P}\frac{n}{\beta}$ | $P$ 为 2 的幂 |
| **ReduceScatter** | $(P-1)\alpha$ | $\frac{P-1}{P}\frac{n}{\beta}$ | ZeRO、FSDP |
| **AllGather** | $(P-1)\alpha$ | $\frac{P-1}{P}\frac{n}{\beta}$ | ZeRO-3 前向 |
| **All-to-All** | $(P-1)\alpha$ | $\frac{P-1}{P}\frac{n}{\beta}$ | MoE 路由 |
| **Broadcast**（树形） | $\log_2 P \cdot \alpha$ | $\log_2 P \cdot \frac{n}{\beta}$ | 权重初始化 |

### 分层 AllReduce（每节点 $G$ 块 GPU，$N$ 个节点）

$$T_{\text{hier}} = \underbrace{2(G{-}1)\alpha_{\text{NV}} + 2\tfrac{G{-}1}{G}\tfrac{n}{\beta_{\text{NV}}}}_{\text{intra-node}} + \underbrace{2(N{-}1)\alpha_{\text{net}} + 2\tfrac{N{-}1}{N}\tfrac{n/G}{\beta_{\text{net}}}}_{\text{inter-node (1/G data)}}$$

### 总线带宽修正因子

| 集合操作 | 修正因子 |
|----------|----------|
| AllReduce | $2(P-1)/P$ |
| AllGather / ReduceScatter / All-to-All (AlltoAll) | $(P-1)/P$ |
| Broadcast / Reduce | 1 |

$$\text{busbw} = \text{algbw} \times \text{correction factor} \qquad \text{algbw} = n / T$$

### 屋顶点（H100 SXM）

| 链路 | 带宽 | 屋顶点（$F/\beta$） |
|------|------|---------------------|
| NVLink 4.0 | 900 GB/s | ~1,099 FLOPs/字节 |
| InfiniBand NDR 400 | 50 GB/s | ~19,780 FLOPs/字节 |

## 各并行策略每步通信

| 策略 | 每 GPU 通信量 | 集合操作 | 链路 |
|------|---------------|----------|------|
| 数据并行 | $\approx 2\Psi \cdot s$ 字节 | AllReduce | 网络 |
| 张量并行 | $4L \times BSH \cdot 2$ 字节 | AllReduce | NVLink |
| 流水线并行 | $\text{micro-batches} \times BSH \cdot 2$ 字节 | Send/Recv | 网络 |
| ZeRO-3（每层） | $2 \times n_{\text{layer}}$ 字节 | AG + RS | 网络 |

（$s$ = 每元素字节数，例如 FP16 为 2）

## 流水线气泡比例

| 调度 | 气泡比例 | 内存（每 GPU） |
|------|----------|----------------|
| GPipe | $\frac{p-1}{m+p-1}$ | $O(m)$ 激活 |
| 1F1B | $\frac{p-1}{m+p-1}$ | $O(p)$ 激活 |
| 交错（$v$ 个虚拟阶段） | $\frac{p-1}{m \cdot v + p - 1}$ | $O(p)$ 激活 |
| 零气泡（ZB-H1） | $\approx 0$ | $O(p)$ 激活 |

$p$ = 流水线阶段数，$m$ = 每批微批数。

## ZeRO 内存节省

| 阶段 | 分片对象 | 每 GPU 内存 | 额外通信 |
|------|----------|-------------|----------|
| 基线（DDP） | 无 | $16\Psi$ | $2\Psi s$（AllReduce） |
| ZeRO-1 | 优化器状态 | $4\Psi + 12\Psi/P$ | 与 DDP 相同 |
| ZeRO-2 | + 梯度 | $2\Psi + 14\Psi/P$ | 与 DDP 相同 |
| ZeRO-3 | + 参数 | $16\Psi/P$ | $3 \times 2\Psi s$（AG + RS） |

## 并行放置决策流程

```
1. MEMORY FIT?
   Total static memory = 16Ψ/P_shard + activations
   If no → add ZeRO stages, activation checkpointing, or more PP/TP

2. TP DEGREE (within node, NVLink):
   TP = min(8, GPUs_per_node)  — typically 4 or 8
   Only increase if per-GPU params don't fit

3. PP DEGREE (across nodes, tolerates latency):
   PP = ceil(layers / max_layers_per_device)
   Use m ≥ 4×PP micro-batches to keep bubbles < 20%

4. DP DEGREE (outer dimension):
   DP = P / (TP × PP)
   Use gradient accumulation if DP × local_batch < B_crit

5. CP DEGREE (if sequence > 8K):
   CP = S / S_max_per_GPU
   Ring Attention or Ulysses depending on S and H

6. EP DEGREE (if MoE):
   EP = E / experts_per_GPU
   Ensure AlltoAll volume fits in network budget
```

## 关键 NCCL 环境变量

| 变量 | 用途 | 常用值 |
|------|------|--------|
| `NCCL_ALGO` | 强制集合算法 | `Ring`、`Tree`、`CollnetDirect` |
| `NCCL_PROTO` | 通信协议 | `Simple`、`LL`、`LL128` |
| `NCCL_TOPO_FILE` | 自定义拓扑文件 | XML 路径 |
| `NCCL_DEBUG` | 调试日志级别 | `INFO`、`WARN`、`TRACE` |
| `NCCL_IB_DISABLE` | 禁用 InfiniBand | `0`（默认）、`1` |
| `NCCL_P2P_LEVEL` | P2P 通信层级 | `NVL`、`PIX`、`PHB`、`SYS` |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA 层级 | `SYS`、`PHB`、`PIX`、`LOC` |
| `NCCL_SOCKET_IFNAME` | 网络接口 | `eth0`、`ib0` 等 |
| `NCCL_BUFFSIZE` | 每通道缓冲区大小 | 字节（默认 4 MB） |

## 常见健全性检查

| 检查项 | 公式 | 危险信号 |
|--------|------|----------|
| MFU | $\frac{6\Psi \cdot \text{tokens/s}}{P \cdot F}$ | < 30% |
| 通信占比 | $T_{\text{comm}} / T_{\text{step}}$ | 未重叠 > 40% |
| 流水线气泡 | $(p-1)/(m+p-1)$ | > 25% |
| 内存利用率 | $M_{\text{used}} / M_{\text{HBM}}$ | > 95%（碎片风险） |
| 数据饥饿 | tokens/s vs $PF \cdot \text{MFU} / (6\Psi)$ | 数据加载器供不上 |