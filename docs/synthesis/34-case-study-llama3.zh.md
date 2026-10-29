---
title: "案例研究：LLaMA 3"
subtitle: "解构 Meta 的训练基础设施"
---

<div class="chapter-opener" markdown>
LLaMA 3 代表了开放权重大语言模型的最先进水平。用本书的框架审视它的训练基础设施，我们就能理解为什么做出了特定的选择，以及分布式训练的数学如何塑造了最终的系统。
</div>

<div class="investigation-question" markdown>
**问题**：Meta 如何用 16,000 块 H100、在 15 万亿 token 上训练一个 405B 参数的模型？是什么约束驱动了他们的并行选择，这些选择又如何与我们已建立的理论框架相吻合？
</div>

!!! note "本章使用的证据标注"
    - **已报道（Reported）**：公开来源中明确陈述。
    - **推导估算（Derived estimate）**：用本书的公式、由报道的数字计算得出。
    - **推测（Speculative）**：合理但未获公开确认。
    请把推导值与推测值视为建模辅助，而非权威披露。

## 挑战的规模

LLaMA 3 405B 需要前所未有的规模：

| 指标 | 数值 |
|--------|-------|
| 参数 | 4050 亿 |
| 训练 token | 15.6 万亿 |
| GPU | 16,384 块 H100 |
| 训练时间 | 约 54 天 |
| 估算 FLOPs | 约 $3.8 \times 10^{25}$ |

表中混合了报道值（例如模型规模与 token 数）与推导估算（例如总 FLOPs）。让我们推导这些估算，并理解其中的约束。

### 计算需求

使用 Chinchilla 最优缩放（第 8 章）：

$$C = 6\Psi D$$

其中：

- $\Psi = 405 \times 10^9$ 个参数
- $D = 15.6 \times 10^{12}$ 个 token

$$C = 6 \times (405 \times 10^9) \times (15.6 \times 10^{12}) \approx 3.79 \times 10^{25} \text{ FLOPs}$$

### 硬件容量

每块 H100 SXM 提供：

- 峰值 FP16/BF16：989 TFLOPs
- 峰值 FP8：1,979 TFLOPs
- HBM3 内存：80 GB
- 内存带宽：3.35 TB/s

对 16,384 块 GPU、按现实的 40% MFU（超大模型的典型值）：

$$\text{Effective FLOPs} = 16384 \times 989 \times 10^{12} \times 0.40 \approx 6.5 \times 10^{18} \text{ FLOPs}$$

### 训练时间推导

$$\text{Training time} = \frac{C}{\text{Effective FLOPs}} = \frac{3.79 \times 10^{25}}{6.5 \times 10^{18}} \approx 5.8 \times 10^6 \text{ seconds} \approx 67 \text{ days}$$

实际训练用时约 54 天，说明：

- 有效 MFU 更接近约 50%（或有效计算预算更低）
- 故障与重启损失了时间
- 验证与检查点（checkpoint）带来开销

## 内存分析

### 模型内存需求

应用第 19 章的内存方程：

**模型参数（BF16）**：

$$M_{\text{params}} = 405 \times 10^9 \times 2 \text{ bytes} = 810 \text{ GB}$$

**优化器状态（AdamW）**：

$$M_{\text{opt}} = 405 \times 10^9 \times (4 + 4) \text{ bytes} = 3,240 \text{ GB}$$

**梯度（BF16）**：

$$M_{\text{grad}} = 405 \times 10^9 \times 2 \text{ bytes} = 810 \text{ GB}$$

**不切分总计**：4,860 GB（至少 61 块 GPU！）

### 为什么完全切分是必要的

每块 GPU 80 GB，我们至少需要：

$$\text{GPUs for parameters} = \frac{4860}{80} \approx 61$$

但这样就没有空间放激活值了。LLaMA 3 使用 **4D 并行**：

1. **张量并行（TP）**：节点内 8 路
2. **流水线并行（PP）**：16 个阶段
3. **数据并行（DP）**：其余 GPU
4. **上下文并行（CP）**：用于长序列

$$\text{Total GPUs} = TP \times PP \times DP \times CP = 8 \times 16 \times 128 \times 1 = 16,384$$

## 架构分析

LLaMA 3 405B 架构：

| 组成 | 数值 |
|-----------|-------|
| 层数 | 126 |
| 隐层维 | 16,384 |
| FFN 维 | 53,248 |
| 注意力头 | 128 |
| KV 头 | 8 |
| 词表 | 128,256 |
| 上下文长度 | 8,192（扩展到 128K） |

### 每层内存

**注意力块（仅 QKV）**：

$$M_{\text{attn}} = 4 \times d_{\text{model}}^2 = 4 \times 16384^2 \times 2 = 2.15 \text{ GB}$$

等等——这里用的是 8 个 KV 头的 GQA，所以：

$$M_{QKV} = d_{\text{model}} \times (d_{\text{model}} + 2 \times \frac{d_{\text{model}} \times n_{\text{kv}}}{n_{\text{heads}}}) \times 2$$

$$= 16384 \times (16384 + 2 \times \frac{16384 \times 8}{128}) \times 2 = 16384 \times (16384 + 2048) \times 2 \approx 0.61 \text{ GB}$$

**输出投影**：

$$M_{O} = d_{\text{model}}^2 \times 2 \approx 0.54 \text{ GB}$$

**注意力合计**（QKV + 输出）：约 1.15 GB

**FFN 块**（SwiGLU 有 3 个矩阵）：

$$M_{\text{ffn}} = 3 \times d_{\text{model}} \times d_{\text{ffn}} \times 2 = 3 \times 16384 \times 53248 \times 2 \approx 5.24 \text{ GB}$$

**每层合计**：约 6 GB

**全部 126 层**：约 756 GB（仅层权重！）

### 激活内存

对批大小为 $B$、序列长度为 $S$ 的单次前向：

$$M_{\text{act}} = L \times (2 \times B \times S \times d_{\text{model}} + B \times S \times d_{\text{ffn}})$$

取 $B=1$、$S=8192$、$L=126$：

$$M_{\text{act}} = 126 \times (2 \times 8192 \times 16384 + 8192 \times 53248) \times 2 \text{ bytes}$$

$$\approx 126 \times (268M + 436M) \times 2 \approx 177 \text{ GB per sequence}$$

这解释了为什么激活检查点必不可少。

## 并行配置

### 张量并行（Tensor Parallelism）：8 路

LLaMA 3 使用 TP=8，与每个节点上通过 NVLink 连接的 8 块 GPU 相匹配。

**为什么是 8？** 应用第 4 章的 alpha-beta 分析：

对跨 NVLink 的 TP（H100 上双向 900 GB/s）：

$$T_{\text{comm}} = \alpha + \frac{4 \times B \times S \times d_{\text{model}} \times 2}{8 \times \beta}$$

取 $B=1$、$S=8192$、$d=16384$：

$$\text{Data per AllReduce} = 4 \times 1 \times 8192 \times 16384 \times 2 \approx 1.07 \text{ GB}$$

使用环形 AllReduce：

$$T_{\text{NVLink}} = \frac{2 \times (8-1)/8 \times 537\text{MB}}{900 \text{ GB/s}} \approx 1.0 \text{ ms}$$

若把 TP 扩展到 16（跨节点经 IB）：

$$T_{\text{IB}} = \frac{2 \times (16-1)/16 \times 537\text{MB}}{50 \text{ GB/s}} \approx 20 \text{ ms}$$

跨越节点边界带来的 20 倍减速，让 TP>8 不可行。

### 流水线并行：16 个阶段

126 层、16 个流水线阶段：

$$\text{Layers per stage} = \lceil 126/16 \rceil = 8 \text{ layers}$$

**每阶段内存**：

$$M_{\text{stage}} = 8 \times 6\text{ GB} = 48 \text{ GB}$$

这留下约 32 GB 给激活值、优化器状态（已切分）与工作内存。

**流水线气泡（pipeline bubble）分析**（第 16 章）：

对微批数为 $m$ 的 1F1B 调度：

$$\text{Bubble fraction} = \frac{p-1}{m+p-1}$$

取 $p=16$、$m=64$ 个微批（micro-batch）：

$$\text{Bubble} = \frac{15}{79} \approx 19\%$$

这代表显著的效率损失，但受内存约束所迫。

### 数据并行：FSDP（ZeRO-3）

剩下的维度是切分式数据并行（Data Parallelism）：

$$DP = \frac{16384}{8 \times 16} = 128$$

**FSDP 切分**：

- 模型参数：810 GB / 128 = 每 rank 6.3 GB
- 优化器状态（optimizer states）：3,240 GB / 128 = 每 rank 25.3 GB
- 梯度：810 GB / 128 = 每 rank 6.3 GB

**切分状态合计**：每 rank 约 38 GB

**每层 AllGather 成本**（FSDP 在计算前重建）：

对一层（约 6 GB / TP）：

$$\text{Data per AllGather} = \frac{6\text{ GB}}{8} = 750 \text{ MB}$$

跨 128 个 DP rank（使用分层 NCCL）：

$$T_{\text{AllGather}} = \alpha \log_2(128) + \frac{127/128 \times 750\text{MB}}{50 \text{ GB/s}} \approx 15 \text{ ms}$$

126 层并重叠：

$$\text{Theoretical comm time} = 126 \times 15 \text{ ms} = 1.89 \text{ s}$$

这必须与计算重叠，才能达到合理的效率。

## 通信分析

### 通信量

计算每步的总通信量：

**张量并行**（每层 2 次 AllReduce）：

$$V_{\text{TP}} = 126 \times 2 \times 2 \times \frac{7}{8} \times B \times S \times d / TP$$

$$= 126 \times 2 \times 2 \times 0.875 \times B \times S \times 16384 / 8 \times 2 \text{ bytes}$$

对 $BS = 1M$ 个 token（分布式）：

$$V_{\text{TP}} = 126 \times 4 \times 0.875 \times \frac{1M}{128} \times 2048 \times 2 \approx 14.2 \text{ GB per rank}$$

**流水线并行**（点对点）：

$$V_{\text{PP}} = 2 \times \text{microbatches} \times B_{\mu} \times S \times d \times 2$$

$$= 2 \times 64 \times \text{tokens per } \mu\text{batch} \times 16384 \times 2$$

**数据并行**（每层 ReduceScatter + AllGather）：

$$V_{\text{DP}} = 2 \times \frac{127}{128} \times \frac{810\text{ GB}}{8} \approx 200 \text{ GB per rank}$$

### 带宽需求

对约 30 秒的步时间（超大模型批的典型值）：

$$\text{Required BW} = \frac{V_{\text{DP}}}{T_{\text{step}}} = \frac{200\text{ GB}}{30\text{ s}} \approx 6.7 \text{ GB/s}$$

这远在 50 GB/s 的 IB 能力之内，从而允许充分重叠。

## 效率分析

### 模型浮点利用率

计算 LLaMA 3 405B 的 MFU：

**每 token 前向 FLOPs**：

$$F_{\text{fwd}} \approx 2N = 2 \times 405 \times 10^9 = 8.1 \times 10^{11}$$

（反向约为 4N，故每 token 合计约 6N。）

**每步、100 万 token**：

$$F_{\text{step}} = 6 \times 405 \times 10^9 \times 10^6 = 2.43 \times 10^{18} \text{ FLOPs}$$

**硬件峰值**：

$$F_{\text{peak}} = 16384 \times 989 \times 10^{12} = 1.62 \times 10^{19} \text{ FLOPs}$$

若一步耗时 0.3 秒：

$$\text{MFU} = \frac{2.43 \times 10^{18}}{1.62 \times 10^{19} \times 0.3} \approx 0.50 = 50\%$$

### 效率分解

效率都去哪了？

| 因素 | 效率损失 |
|--------|----------------|
| 流水线气泡 | 19% |
| 通信开销 | 15% |
| 内存操作 | 10% |
| 负载不均衡 | 5% |
| 故障/重启 | 5% |

合并的分布式效率：$0.81 \times 0.85 \times 0.90 \times 0.95 \times 0.95 \approx 0.56$

单 GPU kernel 效率（内存带宽、非矩阵乘操作等）：约为峰值 FLOP/s 的 50%

**估算 MFU**：$0.50 \times 0.56 \approx 0.28$，与 LLaMA 3 405B 训练约 30% MFU 的报道一致。单 GPU 效率（约 50%）与端到端 MFU（约 30%）之间的差距，完全归因于上述分布式开销。

## 训练稳定性技术

### 批大小动态

LLaMA 3 从较小的批大小起步并逐步提升：

| 训练阶段 | 批大小（token） |
|---------------|---------------------|
| 0–8T token | 4M |
| 8T–15T token | 8M–16M |

**为什么？** 由第 10 章，临界批大小（critical batch size）在训练中会增大：

$$B_{\text{crit}} \propto \frac{\text{tr}(\Sigma)}{\text{tr}(H)}$$

早期：梯度噪声高（$\Sigma$ 大）→ $B_{\text{crit}}$ 小
后期：噪声低、损失平滑（$\Sigma$ 小）→ $B_{\text{crit}}$ 大

### 学习率调度

LLaMA 3 使用：
1. 线性预热：2,000 步
2. 余弦衰减至峰值的 10%

**峰值学习率**：$1.5 \times 10^{-4}$

使用 AdamW：

- $\beta_1 = 0.9$、$\beta_2 = 0.95$
- 权重衰减：$0.1$

### 数值稳定性

**混合精度（mixed precision）策略**：

- 权重：BF16
- 激活值：BF16
- 梯度：BF16
- 优化器状态：FP32
- 损失缩放（loss scaling）：动态

**为什么用 BF16 而非 FP16？** 更大的指数范围（8 位对 5 位）可防止以下情形中的溢出：

- 长序列下的注意力 softmax
- 训练早期较大的梯度幅度
- 大词表下的交叉熵损失

## 容错

### 故障率

在 16,384 块 GPU 上，故障是常态。Meta 报道：

| 故障类型 | MTBF（平均无故障时间） |
|--------------|-----------------------------------|
| GPU 硬件 | 约 1 次/天 |
| 网络 | 约 2–3 次/天 |
| 软件（OOM、超时） | 约 5–10 次/天 |

**每次训练预期的故障数**：

$$N_{\text{failures}} = 54 \text{ days} \times 10 \text{ failures/day} = 540 \text{ failures}$$

### 检查点策略

**检查点大小**：

$$C_{\text{ckpt}} = 810\text{ GB (params)} + 3240\text{ GB (opt)} = 4.05 \text{ TB}$$

**检查点频率**：每 1,000 步

**检查点耗时**（写入并行文件系统）：

假设文件系统的**聚合**带宽为 5 GB/s：

$$T_{\text{ckpt}} = \frac{4.05 \text{ TB}}{5 \text{ GB/s}} \approx 810 \text{ s} \approx 13.5 \text{ min}$$

用异步检查点，这会与训练重叠。

**恢复策略**：
1. 检测故障（NCCL 超时、心跳失败）
2. 终止受影响的节点
3. 用备用节点替换
4. 加载最近的检查点
5. 恢复训练

**每次故障损失的时间**：约 5–10 分钟

### 内存中检查点

为加快恢复，LLaMA 3 使用内存中复制的检查点：

```python
# Simplified in-memory checkpoint strategy
class ReplicatedCheckpoint:
    def __init__(self, model, optimizer, replicas=2):
        self.replicas = replicas
        # Each checkpoint is replicated across 2 nodes

    def save(self):
        # Save to local CPU memory
        state = {
            'model': gather_sharded_state(model),
            'optimizer': gather_sharded_state(optimizer),
            'step': current_step
        }

        # Replicate to partner node
        partner_rank = get_partner_rank(dist.get_rank())
        dist.send_object_list([state], partner_rank)

    def restore(self, failed_ranks):
        # If our checkpoint is valid, provide to failed ranks
        # If our checkpoint failed, request from partner
        pass
```

这让恢复时间从分钟级降到秒级。

## 序列长度扩展

LLaMA 3 先在 8K 上下文上训练，再扩展到 128K。

### 挑战

**内存随序列长度的缩放**：

$$M_{\text{attn}} = O(B \times L \times n_{\text{heads}} \times S^2)$$

对 $S = 128K$：

$$M_{\text{attn}} \approx 126 \times 128 \times 128000^2 \times 2 \text{ bytes} \approx 500 \text{ TB}$$

### 上下文并行

解决方案：把序列切分到多块 GPU 上。

取 CP=4：

$$S_{\text{local}} = \frac{128000}{4} = 32000$$

每块 GPU 处理 32K 个 token，跨段依赖用环形注意力（ring attention）处理。

**通信模式**（环形注意力）：

```
GPU 0: Q[0:32K] × K[0:32K] → send K to GPU 1
GPU 1: Q[32K:64K] × K[0:32K] → send K to GPU 2
...
```

**每层额外的通信**：

$$V_{\text{CP}} = 2 \times (CP-1) \times B \times S/CP \times d_k \times n_{kv}$$

对 8 个 KV 头的 GQA：

$$V_{\text{CP}} = 2 \times 3 \times B \times 32000 \times 128 \times 8 \times 2 \approx 0.4 \text{ GB}$$

这在节点内用 NVLink 完全可以应付。

### RoPE 扩展

LLaMA 3 使用 RoPE（旋转位置编码，Rotary Position Embeddings），需要针对更长的上下文做调整：

$$\text{RoPE}(x, m) = x \odot \cos(m\theta) + \text{rotate}(x) \odot \sin(m\theta)$$

对超出训练长度的位置，频率被调整：

$$\theta_i' = \theta_i \times \text{scale}^{-2i/d}$$

其中 8K→128K 扩展时 scale = 8。

## 经验与设计模式

### 模式 1：分层并行

```
Within node (NVLink): TP=8
Across nodes (IB): PP=16, DP=128
```

**原则**：高带宽的集合通信（TP 的 AllReduce）留在节点内；较低带宽的操作（PP 点对点、带重叠的 DP）跨节点。

### 模式 2：内存–计算–通信的平衡

配置平衡了：

- **内存**：FSDP 让切分状态保持在 80GB 上限内
- **计算**：流水线让所有阶段保持忙碌（气泡除外）
- **通信**：重叠隐藏了大部分 DP 通信

### 模式 3：渐进式缩放

训练参数不断演化：

- 批大小：训练中逐步增大
- 学习率：先预热再衰减
- 序列长度：最后阶段再扩展

### 模式 4：纵深防御保障可靠性

多层容错：
1. 硬件监控与主动更换
2. 内存中检查点实现快速恢复
3. 周期性磁盘检查点用于灾难恢复
4. 弹性训练支持节点替换

## 复现分析

```python
class LLaMA3Analyzer:
    """Analyze LLaMA 3 training configuration."""

    def __init__(self):
        # Model parameters
        self.params = 405e9
        self.layers = 126
        self.d_model = 16384
        self.d_ffn = 53248
        self.n_heads = 128
        self.n_kv_heads = 8

        # Hardware
        self.gpus = 16384
        self.gpu_memory = 80  # GB
        self.nvlink_bw = 900  # GB/s (aggregate; use lower for effective bw)
        self.ib_bw = 50  # GB/s
        self.peak_flops = 989e12  # FP16/BF16 dense

        # Parallelism
        self.tp = 8
        self.pp = 16
        self.dp = 128

    def model_memory(self):
        """Calculate model memory requirements."""
        params_bytes = self.params * 2  # BF16
        opt_bytes = self.params * 8  # FP32 moments
        grad_bytes = self.params * 2  # BF16

        return {
            'params_gb': params_bytes / 1e9,
            'optimizer_gb': opt_bytes / 1e9,
            'gradients_gb': grad_bytes / 1e9,
            'total_gb': (params_bytes + opt_bytes + grad_bytes) / 1e9
        }

    def sharded_memory_per_gpu(self):
        """Calculate memory per GPU with full sharding."""
        mem = self.model_memory()

        # Sharded across DP dimension
        params_per_gpu = mem['params_gb'] / self.dp
        opt_per_gpu = mem['optimizer_gb'] / self.dp
        grad_per_gpu = mem['gradients_gb'] / self.dp

        # But TP means each rank only has 1/TP of the layer weights
        params_per_gpu /= self.tp

        return {
            'params_gb': params_per_gpu,
            'optimizer_gb': opt_per_gpu,
            'gradients_gb': grad_per_gpu,
            'total_sharded_gb': params_per_gpu + opt_per_gpu + grad_per_gpu
        }

    def activation_memory(self, batch_size: int, seq_len: int):
        """Calculate activation memory with checkpointing."""
        # Per layer: attention + FFN activations
        attn_act = 2 * batch_size * seq_len * self.d_model  # Q, K, V intermediate
        ffn_act = batch_size * seq_len * self.d_ffn  # SwiGLU intermediate

        # With selective checkpointing, store ~1/3 of activations
        per_layer = (attn_act + ffn_act) * 2 * 0.33  # BF16, 1/3 stored

        # Distributed across TP
        per_layer /= self.tp

        # Only layers in this PP stage
        layers_per_stage = self.layers // self.pp

        total = layers_per_stage * per_layer

        return total / 1e9  # GB

    def tp_communication_time(self, batch_size: int, seq_len: int):
        """Time for TP AllReduce."""
        # 2 AllReduce per layer (attention, FFN)
        data_per_allreduce = batch_size * seq_len * self.d_model * 2  # BF16

        # Ring AllReduce within node
        ring_factor = 2 * (self.tp - 1) / self.tp
        time_per = ring_factor * data_per_allreduce / (self.nvlink_bw * 1e9)

        layers_per_stage = self.layers // self.pp
        return layers_per_stage * 2 * time_per

    def dp_communication_time(self):
        """Time for DP gradient sync (FSDP)."""
        mem = self.model_memory()

        # AllGather + ReduceScatter per layer
        params_per_layer = mem['params_gb'] * 1e9 / self.layers / self.tp

        ring_factor = 2 * (self.dp - 1) / self.dp
        time_per_layer = ring_factor * params_per_layer / (self.ib_bw * 1e9)

        layers_per_stage = self.layers // self.pp
        return layers_per_stage * time_per_layer

    def pipeline_bubble_fraction(self, microbatches: int):
        """Calculate pipeline bubble overhead."""
        return (self.pp - 1) / (microbatches + self.pp - 1)

    def estimate_mfu(self, tokens_per_step: int, step_time: float):
        """Estimate Model FLOP Utilization."""
        # FLOPs per token (forward + backward)
        flops_per_token = 6 * self.params

        total_flops = flops_per_token * tokens_per_step
        achieved_flops = total_flops / step_time

        peak_flops = self.gpus * self.peak_flops

        return achieved_flops / peak_flops

    def training_time_estimate(self, total_tokens: int, mfu: float):
        """Estimate total training time."""
        flops_per_token = 6 * self.params
        total_flops = flops_per_token * total_tokens

        effective_flops = self.gpus * self.peak_flops * mfu

        seconds = total_flops / effective_flops
        days = seconds / (24 * 3600)

        return {'seconds': seconds, 'days': days}

def analyze_llama3():
    """Run the analysis."""
    analyzer = LLaMA3Analyzer()

    print("=== LLaMA 3 405B Training Analysis ===\n")

    # Memory
    mem = analyzer.model_memory()
    print(f"Total model state: {mem['total_gb']:.0f} GB")

    sharded = analyzer.sharded_memory_per_gpu()
    print(f"Sharded per GPU: {sharded['total_sharded_gb']:.1f} GB")

    act_mem = analyzer.activation_memory(batch_size=1, seq_len=8192)
    print(f"Activation memory per GPU: {act_mem:.1f} GB")
    print(f"Total per GPU: {sharded['total_sharded_gb'] + act_mem:.1f} GB / 80 GB available\n")

    # Communication
    tp_time = analyzer.tp_communication_time(batch_size=1024, seq_len=8192)
    dp_time = analyzer.dp_communication_time()
    print(f"TP communication time: {tp_time*1000:.1f} ms")
    print(f"DP communication time: {dp_time:.1f} s\n")

    # Efficiency
    bubble = analyzer.pipeline_bubble_fraction(microbatches=64)
    print(f"Pipeline bubble fraction: {bubble*100:.1f}%")

    mfu = analyzer.estimate_mfu(tokens_per_step=1_000_000, step_time=30)
    print(f"Estimated MFU: {mfu*100:.1f}%\n")

    # Training time
    time_est = analyzer.training_time_estimate(
        total_tokens=15.6e12,
        mfu=0.30
    )
    print(f"Estimated training time: {time_est['days']:.0f} days")

    return analyzer

if __name__ == "__main__":
    analyze_llama3()
```

## 练习题

1. **并行权衡**：LLaMA 3 使用 PP=16。计算相对 PP=8 与 PP=32 的内存节省。各自的流水线气泡是多少？

??? success "解答"
    **练习 1：并行权衡（PP=8 vs PP=16 vs PP=32）**

    **流水线并行（Pipeline Parallelism）下每 GPU 的内存：**

    对 126 层的 LLaMA 3 405B：

    | 配置 | 每阶段层数 | 每 GPU 参数 | 相对 PP=16 的内存节省 |
    |--------|--------------|----------------|-------------------------|
    | PP=8 | 126/8 ≈ 16 | 405B/8 = 50.6B | -50%（用得更多） |
    | PP=16 | 126/16 ≈ 8 | 405B/16 = 25.3B | 基线 |
    | PP=32 | 126/32 ≈ 4 | 405B/32 = 12.7B | +50% 节省 |

    **流水线气泡计算：**

    使用 1F1B 调度、$m$ 个微批、$p$ 个阶段：

    $$\text{Bubble fraction} = \frac{p-1}{m+p-1}$$

    假设 $m = 64$ 个微批：

    | PP | 阶段数（p） | 气泡比例 | 效率损失 |
    |----|------------|-----------------|-----------------|
    | 8 | 8 | $\frac{7}{71} = 9.9\%$ | 9.9% |
    | 16 | 16 | $\frac{15}{79} = 19.0\%$ | 19.0% |
    | 32 | 32 | $\frac{31}{95} = 32.6\%$ | 32.6% |

    **权衡分析：**

    $$\boxed{\text{PP=16 balances memory (25.3B/GPU) with acceptable bubble (19\%)}}$$

    PP=32 省内存，但把 1/3 的计算浪费在气泡上。

2. **TP 扩展极限**：在多大的消息尺寸下，TP=16（跨节点边界）尽管带宽更低，反而比 TP=8 更快？（提示：考虑延迟项。）

??? success "解答"
    **练习 2：TP 扩展极限**

    **通信模型：**

    AllReduce 时间：$T = \alpha + \frac{M}{\beta}$

    其中：
    - $\alpha$ = 延迟（NVLink 1 μs，IB 5 μs）
    - $M$ = 消息大小
    - $\beta$ = 带宽（NVLink 900 GB/s，IB 50 GB/s）

    **TP=8（节点内，NVLink）：**
    $$T_8 = 1\mu s + \frac{M}{900 \text{ GB/s}}$$

    **TP=16（跨节点，跨节点用 IB）：**
    对 TP=16，使用分层 AllReduce：
    - 节点内（8 块 GPU）：NVLink
    - 跨节点（2 组）：IB

    $$T_{16} = 2 \times (5\mu s + \frac{M/2}{50 \text{ GB/s}}) + 1\mu s + \frac{M}{900 \text{ GB/s}}$$

    **收支平衡点：**

    令 $T_8 = T_{16}$（TP 中每层有 2 次 AllReduce 操作）：

    对 TP=8：2 次大小为 $M$ 的 AllReduce
    对 TP=16：2 次大小为 $M$ 的 AllReduce，但分摊到更多 GPU 上

    TP=16 时每 GPU 的计算量是 TP=8 的一半，故计算时间减半。

    当计算节省 > 通信开销时达到平衡：

    $$\frac{T_{compute}}{2} + T_{comm,16} < T_{compute} + T_{comm,8}$$

    对 H=16384 的典型 transformer 层求解：

    $$M = 2 \times B \times S \times H = 2 \times 1 \times 8192 \times 16384 = 268 \text{ MB}$$

    在这个尺寸下，NVLink 的优势占主导。只有当下式成立时 TP=16 才取胜：

    $$\boxed{M > 2 \text{ GB (rarely practical for single AllReduce)}}$$

    **结论：** 由于 NVLink 有 18 倍的带宽优势，TP=8（节点本地）几乎总是优于 TP=16。

3. **FSDP 重叠**：LLaMA 3 将 FSDP AllGather 与计算重叠。要以 50 GB/s 的带宽隐藏 200 GB/s 的 DP 通信，必须有多大比例的计算被重叠？

??? success "解答"
    **练习 3：FSDP 重叠**

    **已知：**
    - DP 通信量：200 GB（完整模型 gather）
    - 网络带宽：50 GB/s
    - 通信时间：$T_{comm} = \frac{200}{50} = 4$ 秒

    **重叠要求：**

    要完全隐藏通信，计算至少要耗时相同：

    $$T_{compute} \geq T_{comm} = 4 \text{ seconds}$$

    **LLaMA 3 405B 每次前向的计算：**

    每 token FLOPs：$2\Psi = 2 \times 405 \times 10^9 = 810$ GFLOPs

    对 100 万 token 的批：

    $$F_{fwd} = 810 \times 10^9 \times 10^6 = 8.1 \times 10^{17} \text{ FLOPs}$$

    在 H100 上 50% MFU（有效 990 TFLOP/s）：

    $$T_{compute} = \frac{8.1 \times 10^{17}}{990 \times 10^{12}} = 818 \text{ seconds (per GPU)}$$

    等等——这是完整模型。FSDP 下每个 DP 分片，计算是分布式进行的。

    **逐层分析（更实用）：**

    每层计算时间 ≈ $\frac{818}{126} \approx 6.5$ 秒

    每层 AllGather：$\frac{200/126}{50} \approx 32$ ms

    $$\boxed{\text{Overlap fraction needed} = \frac{32 \text{ ms}}{6.5 \text{ s}} \approx 0.5\%}$$

    每层的计算远超通信，所以极少的重叠就足够。

4. **上下文并行（Context Parallelism）**：对 128K 上下文、CP=8，计算每层额外的 KV 缓存通信。与标准注意力内存缩放作比较。

??? success "解答"
    **练习 4：上下文并行**

    **标准注意力内存缩放：**

    每层 KV 缓存：$2 \times B \times S \times H$（K 和 V，FP16）

    对 S=128K、H=16384、B=1：

    $$M_{KV} = 2 \times 2 \times 1 \times 128000 \times 16384 = 8.4 \text{ GB per layer}$$

    126 层合计：$8.4 \times 126 = 1,058$ GB（单 GPU 不可能！）

    **使用上下文并行（CP=8）：**

    每块 GPU 处理 $S/8 = 16K$ 个 token 的上下文。

    本地 KV 缓存：每 GPU 每层 $\frac{8.4}{8} = 1.05$ GB

    **每层额外的通信：**

    环形注意力需要在 CP rank 之间传递 KV 块：
    - 每块 GPU 把它的 KV 块发给环中的下一块
    - 消息大小：$2 \times 2 \times B \times (S/CP) \times H = 1.05$ GB
    - 每层发送次数：$CP - 1 = 7$

    每层的 KV 通信总量：

    $$\text{Comm} = 7 \times 1.05 \text{ GB} = 7.35 \text{ GB}$$

    在 50 GB/s（节点间 IB）：

    $$T_{KV} = \frac{7.35}{50} = 147 \text{ ms per layer}$$

    **对比：**

    | 方案 | 每 GPU 内存 | 通信 |
    |----------|------------|---------------|
    | 标准（无 CP） | 合计 1,058 GB | 0 |
    | CP=8 | 132 GB/GPU | 147 ms/层 |

    $$\boxed{\text{CP=8 trades 147 ms/layer communication for 8× memory reduction}}$$

    没有 CP，128K 上下文不可能；有了 CP，完整前向额外约 18.5s 通信即可行。

5. **故障分析**：若每天 10 次故障、每次恢复 5 分钟，训练时间损失多少比例？改用 30 秒的内存中恢复会怎样变化？

??? success "解答"
    **练习 5：故障分析**

    **场景 A：5 分钟恢复**

    - 每天故障数：10
    - 恢复时间：5 分钟 = 300 秒
    - 每日恢复开销：$10 \times 300 = 3000$ 秒 = 50 分钟

    每日训练时间：24 小时 = 1440 分钟

    $$\text{Time lost} = \frac{50}{1440} = 3.5\%$$

    **场景 B：30 秒内存中恢复**

    - 恢复时间：30 秒
    - 每日恢复开销：$10 \times 30 = 300$ 秒 = 5 分钟

    $$\text{Time lost} = \frac{5}{1440} = 0.35\%$$

    **整个训练（2T token 约 85 天）：**

    | 恢复方式 | 每日损失 | 总损失天数 | 有效训练 |
    |-----------------|------------|-----------------|-------------------|
    | 5 分钟检查点 | 3.5% | 3.0 天 | 82 天 |
    | 30 秒内存中 | 0.35% | 0.3 天 | 84.7 天 |

    $$\boxed{\text{In-memory recovery saves } 2.7 \text{ days (3.2\% improvement)}}$$

    **其他考量：**
    - 检查点之间的工作丢失（通常间隔 10–30 分钟）
    - 5 分钟恢复：每次故障平均损失 15 分钟工作 = 每天 150 分钟 = 额外 10%
    - 30 秒恢复：平均损失 1 分钟工作 = 每天 10 分钟 = 额外 0.7%

6. **批大小动态**：LLaMA 3 把批从 4M 增至 16M token。若训练中临界批大小从 2M 增至 10M，估算各阶段的计算效率。

??? success "解答"
    **练习 6：批大小动态**

    **临界批大小理论：**

    当 $B < B_{crit}$ 时的计算效率：

    $$\eta = \frac{B}{B + B_{crit}}$$

    当 $B > B_{crit}$ 时：

    $$\eta \approx 1 - \frac{B_{crit}}{2B}$$

    **阶段 1：早期训练（B=4M，B_crit=2M）**

    $$\eta_1 = 1 - \frac{2M}{2 \times 4M} = 1 - 0.25 = 75\%$$

    训练是高效的——批超过了临界批大小。

    **阶段 2：中期训练（B=8M，B_crit=5M）**

    $$\eta_2 = 1 - \frac{5M}{2 \times 8M} = 1 - 0.31 = 69\%$$

    仍然高效，但临界批大小正在追上。

    **阶段 3：后期训练（B=16M，B_crit=10M）**

    $$\eta_3 = 1 - \frac{10M}{2 \times 16M} = 1 - 0.31 = 69\%$$

    通过让批大小与临界批大小一起缩放，效率得以维持。

    **如果批大小固定在 4M 呢？**

    后期训练 B=4M、B_crit=10M：

    $$\eta_{fixed} = \frac{4M}{4M + 10M} = 28.6\%$$

    **小结：**

    | 阶段 | 批大小 | B_crit | 效率 | 对比固定 B=4M |
    |-------|------------|--------|------------|---------------|
    | 早期 | 4M | 2M | 75% | 75% |
    | 中期 | 8M | 5M | 69% | 44% |
    | 后期 | 16M | 10M | 69% | 29% |

    $$\boxed{\text{Dynamic batching maintains 69-75\% efficiency vs 29\% with fixed batch}}$$

    动态批大小带来的 2–3 倍效率增益，节省了数周的训练时间。

## 不变量小结

| 不变量 | 主要压力 | 应对 |
|---|---|---|
| 内存 | 模型状态与激活值 | ZeRO/FSDP 式切分 + 检查点 |
| 计算 | 每 token 的高 FLOPs | FP16/BF16 kernel 与大批 |
| 通信 | 梯度与流水线流量 | 分层并行 + 重叠 |

## 关键要点

1. **规模要求 4D+ 并行**：405B 参数跨越 16K 块 GPU，需要组合 TP、PP、DP 与 CP。

2. **节点边界很重要**：NVLink（H100 上 900 GB/s）与 InfiniBand（50 GB/s）决定了每个并行维度在哪里运行。

3. **内存约束一切**：80GB 的 GPU 上限迫使采用 FSDP 切分与激活检查点。

4. **规模上效率很难**：即便是 Meta 优化过的基础设施，也因流水线气泡、通信与开销只能达到约 30% MFU。

5. **容错是强制要求**：每次训练有数百次故障，恢复必须快速且自动化。

6. **动态训练参数**：批大小、学习率与序列长度都在训练中演化，以匹配不断变化的损失地形。