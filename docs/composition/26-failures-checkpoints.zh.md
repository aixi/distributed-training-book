---
title: "故障与检查点"
subtitle: "把容错看作数学不变量"
---

<div class="chapter-opener" markdown>
在大规模下，故障不是例外——而是常态。在一次刻意悲观的压测中（10,000 块 GPU、每块 GPU 的 MTBF 为 4 小时），你每 1.4 秒就会损失一块 GPU。在本章的基准模型中（GPU MTBF 20,000 小时，加上主机故障），系统仍大约每 96 分钟就经历一次故障。检查点的数学决定了训练是能完成，还是会崩溃。
</div>

<div class="investigation-question" markdown>
**问题**：你如何为一个分布在 8192 块 GPU 上的 1TB 模型做检查点，使得任何故障最多损失 5 分钟的工作，而检查点开销低于训练时间的 1%？检查点必须维护哪些数学不变量？
</div>

## 问题的规模

### 大规模下的故障统计

各组件的平均无故障时间（mean time between failures, MTBF）：

| 组件 | MTBF（小时） | 故障率 λ（每小时） |
|-----------|--------------|---------------------------|
| GPU | 10,000 - 50,000 | 0.0001 - 0.00002 |
| HBM | 100,000+ | 0.00001 |
| NVLink | 50,000+ | 0.00002 |
| 网络交换机 | 100,000+ | 0.00001 |
| 电源 | 50,000+ | 0.00002 |
| 主机 | 5,000 - 20,000 | 0.0002 - 0.00005 |

对于一个有 $N$ 个组件、每个组件 MTBF 为 $M$ 的系统：

$$\text{MTBF}_{\text{system}} = \frac{M}{N}$$

### 10,000 块 GPU 的计算

假设 10,000 块 GPU、1,250 台主机（每台 8 块 GPU）、有效 GPU MTBF 为 20,000 小时、主机 MTBF 为 10,000 小时：

$$\lambda_{\text{GPU}} = \frac{10,000}{20,000} = 0.5 \text{ failures/hour}$$

$$\lambda_{\text{host}} = \frac{1,250}{10,000} = 0.125 \text{ failures/hour}$$

$$\lambda_{\text{total}} \approx 0.625 \text{ failures/hour}$$

$$\text{MTBF}_{\text{system}} = \frac{1}{0.625} = 1.6 \text{ hours}$$

**平均每 96 分钟，就会有东西出故障。**

### 训练时间方程

没有检查点时，一个需要 $T$ 小时的任务的期望完成时间：

$$\mathbb{E}[\text{completion time}] = \frac{e^{\lambda T} - 1}{\lambda}$$

当 $T = 720$ 小时（30 天）、$\lambda = 0.625$ 时：

$$\mathbb{E}[\text{completion time}] = \frac{e^{450} - 1}{0.625} \approx \infty$$

没有检查点，这个任务将**永远无法完成**。

## 检查点基础

### 检查点不变量

有效的检查点必须满足**一致性不变量**：

<div class="definition" markdown>
**检查点一致性**：检查点 $C$ 是一致的，当且仅当从 $C$ 恢复训练所产生的模型状态序列，与从创建 $C$ 的位置开始不中断训练所本会产生的结果相同。
</div>

对于分布式训练，这要求：

1. **模型状态一致性**：所有参数分片代表同一个逻辑步
2. **优化器状态一致性**：动量、速度与参数对齐
3. **数据加载器状态**：从 epoch 中的正确位置恢复
4. **RNG 状态**：随机数生成器可复现

### 检查点的内容

一个完整的检查点包含：

```python
@dataclass
class DistributedCheckpoint:
    """Complete state for resumable distributed training."""

    # Training progress
    global_step: int
    tokens_seen: int
    epoch: int

    # Model state (sharded)
    model_state: Dict[str, ShardedTensor]

    # Optimizer state (sharded)
    optimizer_state: Dict[str, Dict[str, ShardedTensor]]

    # Learning rate scheduler
    scheduler_state: Dict[str, Any]

    # Data loading
    dataloader_state: DataLoaderState

    # Random number generators
    rng_states: RNGStates

    # Configuration for validation
    config: TrainingConfig

    # Metadata
    timestamp: float
    checkpoint_version: str
```

### 分片张量

在模型并行下，每个 rank 持有部分状态：

```python
@dataclass
class ShardedTensor:
    """Tensor distributed across multiple ranks."""

    # Local shard data
    local_tensor: torch.Tensor

    # Global tensor metadata
    global_shape: Tuple[int, ...]
    global_dtype: torch.dtype

    # Sharding specification
    sharding_spec: ShardingSpec

    # Placement
    rank: int
    world_size: int

    def global_offset(self) -> Tuple[int, ...]:
        """Compute where this shard fits in global tensor."""
        return self.sharding_spec.offset_for_rank(self.rank)

@dataclass
class ShardingSpec:
    """How a tensor is sharded across ranks."""

    dim: int                    # Dimension sharded along
    num_shards: int             # Number of shards
    shard_sizes: List[int]      # Size of each shard

    def offset_for_rank(self, rank: int) -> Tuple[int, ...]:
        offset = [0] * len(self.shard_sizes)
        offset[self.dim] = sum(self.shard_sizes[:rank])
        return tuple(offset)
```

## 分布式检查点策略

### 策略 1：先聚集再写入（简单但慢）

```
                 Gather to Rank 0
    ┌─────────────────────────────────┐
    │                                 │
    │    ┌───┐  ┌───┐  ┌───┐  ┌───┐  │    ┌───────────────┐
    │    │R0 │  │R1 │  │R2 │  │R3 │  ├───►│  Single File  │
    │    └───┘  └───┘  └───┘  └───┘  │    │    on Disk    │
    │       ▲      │      │      │   │    └───────────────┘
    │       └──────┴──────┴──────┘   │
    │           AllGather           │
    └─────────────────────────────────┘
```

**优点**：简单、单文件、易于加载
**缺点**：rank 0 处有内存瓶颈、I/O 串行、无法扩展

```python
def gather_checkpoint(model: DistributedModel) -> Dict[str, torch.Tensor]:
    """Gather all shards to rank 0 for checkpointing."""

    full_state = {}

    for name, param in model.named_parameters():
        if dist.get_rank() == 0:
            gathered = [torch.zeros_like(param)
                       for _ in range(dist.get_world_size())]
        else:
            gathered = None

        dist.gather(param, gathered, dst=0)

        if dist.get_rank() == 0:
            # Concatenate shards
            full_state[name] = torch.cat(gathered, dim=param.shard_dim)

    return full_state
```

rank 0 处的内存需求：$O(P \cdot \text{model size})$ —— 对大模型不可行。

### 策略 2：并行写入（可扩展）

```
           Each Rank Writes Independently
    ┌─────────────────────────────────────────┐
    │                                         │
    │   ┌───┐    ┌───┐    ┌───┐    ┌───┐     │
    │   │R0 │    │R1 │    │R2 │    │R3 │     │
    │   └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘     │
    │     │        │        │        │       │
    │     ▼        ▼        ▼        ▼       │
    │   ┌───┐    ┌───┐    ┌───┐    ┌───┐     │
    │   │.0 │    │.1 │    │.2 │    │.3 │     │  ◄── Parallel FS
    │   └───┘    └───┘    └───┘    └───┘     │
    │                                         │
    └─────────────────────────────────────────┘
```

**优点**：随 rank 数扩展、无内存瓶颈
**缺点**：文件很多、需要并行文件系统

```python
class ParallelCheckpointer:
    """Write checkpoints in parallel across all ranks."""

    def __init__(self, checkpoint_dir: str, world_size: int):
        self.checkpoint_dir = checkpoint_dir
        self.world_size = world_size

    def save(self,
             model: DistributedModel,
             optimizer: DistributedOptimizer,
             step: int) -> None:
        """Save checkpoint with each rank writing its own shard."""

        rank = dist.get_rank()

        # Create step directory
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step:08d}")
        os.makedirs(step_dir, exist_ok=True)

        # Each rank saves its shard
        shard_path = os.path.join(step_dir, f"shard_{rank:05d}.pt")

        shard_state = {
            'model': self._get_local_model_state(model),
            'optimizer': self._get_local_optimizer_state(optimizer),
            'step': step,
            'rank': rank,
            'world_size': self.world_size,
        }

        torch.save(shard_state, shard_path)

        # Barrier ensures all ranks complete
        dist.barrier()

        # Rank 0 writes metadata
        if rank == 0:
            self._write_metadata(step_dir, step)

    def _get_local_model_state(self,
                                model: DistributedModel) -> Dict[str, Any]:
        """Extract local shard state with sharding metadata."""

        state = {}
        for name, param in model.named_parameters():
            state[name] = {
                'data': param.data,
                'sharding_spec': param.sharding_spec,
                'global_shape': param.global_shape,
            }
        return state

    def _write_metadata(self, step_dir: str, step: int) -> None:
        """Write metadata file describing checkpoint structure."""

        metadata = {
            'step': step,
            'world_size': self.world_size,
            'timestamp': time.time(),
            'shards': [f"shard_{r:05d}.pt" for r in range(self.world_size)],
        }

        with open(os.path.join(step_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
```

### I/O 带宽分析

对于大小为 $C$ 字节、分布在 $P$ 个 rank 上的检查点，文件系统带宽为 $B$：

**先聚集再写入**：

$$T_{\text{gather}} = \frac{C \cdot (P-1)}{B_{\text{network}}} + \frac{C}{B_{\text{disk}}}$$

**并行写入**（有 $P$ 条并行路径）：

$$T_{\text{parallel}} = \frac{C/P}{B_{\text{disk}}/P} = \frac{C}{B_{\text{disk}}}$$

等等——如果总带宽固定，有 $P$ 条路径的并行写入并没有帮助。但像 Lustre/GPFS 这样的并行文件系统提供**聚合带宽扩展**：

$$B_{\text{aggregate}} = \min(P, N_{\text{OSTs}}) \cdot B_{\text{per-OST}}$$

在 OST（对象存储目标）足够多时：

$$T_{\text{parallel}} = \frac{C}{P \cdot B_{\text{per-OST}}}$$

这会随 rank 数线性扩展！

## 异步检查点

同步检查点会阻塞训练。异步检查点让 I/O 与计算重叠。

### 异步策略

```
Training:    ═══════════════════════════════════════════►
                   │            │            │
                   ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
Copy to      │  Buffer  │ │  Buffer  │ │  Buffer  │
pinned mem:  └────┬─────┘ └────┬─────┘ └────┬─────┘
                  │            │            │
                  ▼            ▼            ▼
             ┌──────────────────────────────────────┐
Async I/O:   │  Background thread pool writing      │
             └──────────────────────────────────────┘
```

```python
class AsyncCheckpointer:
    """Checkpoint asynchronously to minimize training disruption."""

    def __init__(self,
                 checkpoint_dir: str,
                 num_io_workers: int = 4,
                 pinned_buffer_size: int = 1 << 30):  # 1 GB

        self.checkpoint_dir = checkpoint_dir
        self.executor = ThreadPoolExecutor(max_workers=num_io_workers)

        # Pinned memory for fast GPU -> CPU transfer
        self.pinned_buffer = torch.empty(
            pinned_buffer_size, dtype=torch.uint8, pin_memory=True
        )

        # Track pending writes
        self.pending_futures: List[Future] = []

        # Double buffering
        self.buffer_a = {}
        self.buffer_b = {}
        self.active_buffer = 'a'

    def checkpoint_async(self,
                         model: DistributedModel,
                         optimizer: DistributedOptimizer,
                         step: int) -> Future:
        """Initiate asynchronous checkpoint."""

        # Wait for any pending checkpoint to complete
        self._wait_pending()

        # Swap buffers
        buffer = self.buffer_a if self.active_buffer == 'a' else self.buffer_b
        self.active_buffer = 'b' if self.active_buffer == 'a' else 'a'

        # Fast copy to CPU (async)
        self._copy_to_buffer(model, optimizer, step, buffer)

        # Submit write to thread pool
        future = self.executor.submit(
            self._write_buffer, buffer, step
        )
        self.pending_futures.append(future)

        return future

    def _copy_to_buffer(self,
                        model: DistributedModel,
                        optimizer: DistributedOptimizer,
                        step: int,
                        buffer: Dict) -> None:
        """Copy state to CPU buffer using CUDA streams."""

        copy_stream = torch.cuda.Stream()

        with torch.cuda.stream(copy_stream):
            buffer['step'] = step
            buffer['model'] = {}

            for name, param in model.named_parameters():
                # Non-blocking copy to pinned memory
                cpu_tensor = torch.empty_like(param, device='cpu',
                                              pin_memory=True)
                cpu_tensor.copy_(param, non_blocking=True)
                buffer['model'][name] = cpu_tensor

            buffer['optimizer'] = {}
            for key, state in optimizer.state.items():
                buffer['optimizer'][key] = {}
                for name, tensor in state.items():
                    if isinstance(tensor, torch.Tensor):
                        cpu_tensor = torch.empty_like(tensor, device='cpu',
                                                     pin_memory=True)
                        cpu_tensor.copy_(tensor, non_blocking=True)
                        buffer['optimizer'][key][name] = cpu_tensor

        # Synchronize copy stream
        copy_stream.synchronize()

    def _write_buffer(self, buffer: Dict, step: int) -> str:
        """Write buffer to disk (runs in thread pool)."""

        rank = dist.get_rank()
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step:08d}")
        os.makedirs(step_dir, exist_ok=True)

        shard_path = os.path.join(step_dir, f"shard_{rank:05d}.pt")
        torch.save(buffer, shard_path)

        return shard_path

    def _wait_pending(self) -> None:
        """Wait for all pending checkpoints to complete."""

        for future in self.pending_futures:
            future.result()
        self.pending_futures.clear()
```

### 开销分析

**同步检查点时间**：

$$T_{\text{sync}} = T_{\text{copy}} + T_{\text{write}}$$

**异步开销**（训练被阻塞的时间）：

$$T_{\text{async}} = T_{\text{copy}}$$

使用锁页内存与 CUDA 流：

$$T_{\text{copy}} = \frac{C}{B_{\text{PCIe}}} \approx \frac{C}{25 \text{ GB/s}}$$

对于 100GB 的检查点分片：

- $T_{\text{copy}} \approx 4$ 秒
- $T_{\text{write}} \approx 10$ 秒（100 GB/s 并行文件系统）

**每次检查点节省 10 秒！**

## 检查点频率优化

### 代价模型

设：

- $T_{\text{ckpt}}$：做一次检查点的时间（开销）
- $\lambda$：故障率（每秒故障数）
- $I$：检查点间隔（两次检查点之间的秒数）
- $T_{\text{total}}$：总墙钟训练时间

**单位时间的开销**：

$$\text{overhead} = \frac{T_{\text{ckpt}}}{I}$$

**每次故障的期望工作损失**：

$$W_{\text{lost}} = \frac{I}{2}$$

（平均而言，故障发生在两次检查点之间的一半处。）

**期望重启次数**：

$$\mathbb{E}[\text{restarts}] = \lambda \cdot T_{\text{total}}$$

### 最优检查点频率

最小化含故障的总训练时间（Young/Daly）：

$$\text{overhead factor} = 1 + \frac{T_{\text{ckpt}}}{I} + \lambda \cdot \frac{I}{2}$$

对 $I$ 求导并令其为零：

$$\frac{d}{dI}\left(\frac{T_{\text{ckpt}}}{I} + \lambda \cdot \frac{I}{2}\right) = 0$$

解得：

$$I^* = \sqrt{\frac{2 T_{\text{ckpt}}}{\lambda}}$$

### 数值示例

给定：

- $T_{\text{step}} = 1$ 秒
- $T_{\text{ckpt}} = 60$ 秒（含开销）
- $\lambda = 0.625$ 故障/小时 = $1.74 \times 10^{-4}$ 故障/秒

$$I^* = \sqrt{\frac{2 \cdot 60}{1.74 \times 10^{-4}}} \approx 830 \text{ s} \approx 14 \text{ min}$$

每 $1/f^* \approx 830$ 步做一次检查点。

按每秒一步计算，**约每 14 分钟**做一次检查点。

## 分布式系统中的检查点一致性

### 一致性问题

有多个并行维度时，保证一致性并非易事：

```
              Time ──────────────────────────►

Rank 0:    ──────┬─────────────┬─────────────
                 │ step 100    │ step 101
                 ▼             ▼
Rank 1:    ────────┬───────────┬─────────────
                   │ step 100  │ step 101
                   ▼           ▼

DANGER: If Rank 0 saves at step 100
        but Rank 1 saves at step 101
        ──► Inconsistent checkpoint!
```

### 解决方案：协同检查点

```python
class CoordinatedCheckpointer:
    """Ensure all ranks checkpoint at the same logical step."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_step = None

    def maybe_checkpoint(self,
                         model: DistributedModel,
                         optimizer: DistributedOptimizer,
                         step: int,
                         checkpoint_interval: int) -> bool:
        """Checkpoint with distributed coordination."""

        should_checkpoint = (step % checkpoint_interval == 0)

        # All-reduce to ensure agreement
        should_checkpoint_tensor = torch.tensor(
            [1 if should_checkpoint else 0],
            device='cuda'
        )
        dist.all_reduce(should_checkpoint_tensor, op=dist.ReduceOp.MIN)

        if should_checkpoint_tensor.item() == 0:
            return False

        # Barrier before checkpoint
        dist.barrier()

        # Now all ranks are at the same step
        self._save_checkpoint(model, optimizer, step)

        # Barrier after checkpoint
        dist.barrier()

        return True

    def _save_checkpoint(self,
                         model: DistributedModel,
                         optimizer: DistributedOptimizer,
                         step: int) -> None:
        """Save local shard with step verification."""

        rank = dist.get_rank()
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step:08d}")

        # Create directory only on rank 0
        if rank == 0:
            os.makedirs(step_dir, exist_ok=True)
        dist.barrier()

        # Each rank saves its shard
        shard_path = os.path.join(step_dir, f"shard_{rank:05d}.pt")

        state = {
            'model': {name: param.data.cpu()
                     for name, param in model.named_parameters()},
            'optimizer': optimizer.state_dict(),
            'step': step,
            'rank': rank,
        }

        torch.save(state, shard_path)
```

### 处理流水线并行

在流水线并行下，不同阶段可能在处理不同的微批：

```
                  Micro-batch timeline
Stage 0:    [MB0][MB1][MB2][MB3]     ← Ahead
Stage 1:        [MB0][MB1][MB2][MB3] ← Behind
```

**解决方案**：在流水线气泡处（所有阶段同步时）做检查点：

```python
class PipelineCheckpointer:
    """Checkpoint at pipeline synchronization points."""

    def __init__(self, pp_group: dist.ProcessGroup):
        self.pp_group = pp_group
        self.pp_rank = dist.get_rank(pp_group)
        self.pp_size = dist.get_world_size(pp_group)

    def checkpoint_at_bubble(self,
                             model: nn.Module,
                             optimizer: torch.optim.Optimizer,
                             step: int) -> None:
        """Checkpoint when pipeline is drained."""

        # Signal checkpoint intent from last stage
        if self.pp_rank == self.pp_size - 1:
            checkpoint_signal = torch.ones(1, device='cuda')
        else:
            checkpoint_signal = torch.zeros(1, device='cuda')

        # Broadcast from last stage (map group rank to global rank)
        src_global = dist.get_global_rank(self.pp_group, self.pp_size - 1)
        dist.broadcast(checkpoint_signal,
                      src=src_global,
                      group=self.pp_group)

        # All stages wait for pipeline to drain
        # (This happens naturally at the end of each step in 1F1B)

        # Now all stages are synchronized
        self._save_stage_checkpoint(model, optimizer, step)

    def _save_stage_checkpoint(self,
                               model: nn.Module,
                               optimizer: torch.optim.Optimizer,
                               step: int) -> None:
        """Save this pipeline stage's checkpoint."""

        stage_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'pp_rank': self.pp_rank,
        }

        path = f"checkpoint_step{step}_stage{self.pp_rank}.pt"
        torch.save(stage_state, path)
```

## 检查点加载与恢复

### 弹性加载

```python
class ResilientLoader:
    """Load checkpoints with validation and recovery."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest valid checkpoint."""

        # Find all checkpoint directories
        checkpoints = self._find_checkpoints()

        # Try from newest to oldest
        for ckpt_dir in reversed(checkpoints):
            try:
                state = self._load_checkpoint(ckpt_dir)
                if self._validate_checkpoint(state):
                    return state
                else:
                    print(f"Checkpoint {ckpt_dir} failed validation")
            except Exception as e:
                print(f"Failed to load {ckpt_dir}: {e}")

        return None

    def _find_checkpoints(self) -> List[str]:
        """Find checkpoint directories sorted by step."""

        pattern = os.path.join(self.checkpoint_dir, "step_*")
        dirs = glob.glob(pattern)

        # Sort by step number
        def extract_step(path):
            match = re.search(r'step_(\d+)', path)
            return int(match.group(1)) if match else 0

        return sorted(dirs, key=extract_step)

    def _load_checkpoint(self, ckpt_dir: str) -> Dict[str, Any]:
        """Load all shards of a checkpoint."""

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Read metadata
        with open(os.path.join(ckpt_dir, 'metadata.json')) as f:
            metadata = json.load(f)

        # Verify world size matches
        if metadata['world_size'] != world_size:
            raise ValueError(
                f"Checkpoint world_size {metadata['world_size']} "
                f"!= current {world_size}"
            )

        # Load this rank's shard
        shard_path = os.path.join(ckpt_dir, f"shard_{rank:05d}.pt")
        state = torch.load(shard_path, map_location='cuda')

        return state

    def _validate_checkpoint(self, state: Dict[str, Any]) -> bool:
        """Validate checkpoint integrity."""

        # Check required keys
        required = ['model', 'optimizer', 'step']
        local_valid = all(key in state for key in required)

        # Check model state
        if local_valid:
            for param in state['model'].values():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    local_valid = False
                    break

        # All ranks must agree checkpoint is valid
        valid_tensor = torch.tensor(1 if local_valid else 0, device='cuda')
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)

        return valid_tensor.item() == 1
```

### 针对不同并行度的重分片

当以不同的并行配置加载检查点时：

```python
class ReshardingLoader:
    """Load checkpoint with different sharding configuration."""

    def __init__(self,
                 ckpt_world_size: int,
                 current_world_size: int,
                 ckpt_tp_size: int,
                 current_tp_size: int):
        self.ckpt_world_size = ckpt_world_size
        self.current_world_size = current_world_size
        self.ckpt_tp_size = ckpt_tp_size
        self.current_tp_size = current_tp_size

    def load_and_reshard(self,
                         ckpt_dir: str,
                         model: DistributedModel) -> None:
        """Load checkpoint and reshard for current configuration."""

        rank = dist.get_rank()

        if self.ckpt_tp_size == self.current_tp_size:
            # Simple case: same TP, just load
            self._load_direct(ckpt_dir, model, rank)
        elif self.current_tp_size > self.ckpt_tp_size:
            # More TP shards: split each checkpoint shard
            self._load_and_split(ckpt_dir, model, rank)
        else:
            # Fewer TP shards: merge checkpoint shards
            self._load_and_merge(ckpt_dir, model, rank)

    def _load_and_split(self,
                        ckpt_dir: str,
                        model: DistributedModel,
                        rank: int) -> None:
        """Split checkpoint shards for higher TP degree."""

        # Calculate which checkpoint shard this rank needs
        split_factor = self.current_tp_size // self.ckpt_tp_size
        ckpt_rank = rank // split_factor
        split_idx = rank % split_factor

        # Load the checkpoint shard
        shard_path = os.path.join(ckpt_dir, f"shard_{ckpt_rank:05d}.pt")
        state = torch.load(shard_path, map_location='cuda')

        # Split each parameter
        for name, param in model.named_parameters():
            ckpt_tensor = state['model'][name]['data']
            shard_dim = param.sharding_spec.dim

            # Split along shard dimension
            chunks = torch.chunk(ckpt_tensor, split_factor, dim=shard_dim)
            param.data.copy_(chunks[split_idx])

    def _load_and_merge(self,
                        ckpt_dir: str,
                        model: DistributedModel,
                        rank: int) -> None:
        """Merge checkpoint shards for lower TP degree."""

        merge_factor = self.ckpt_tp_size // self.current_tp_size

        # Load multiple checkpoint shards
        tensors_to_merge = []
        for i in range(merge_factor):
            ckpt_rank = rank * merge_factor + i
            shard_path = os.path.join(ckpt_dir, f"shard_{ckpt_rank:05d}.pt")
            state = torch.load(shard_path, map_location='cuda')
            tensors_to_merge.append(state)

        # Merge each parameter
        for name, param in model.named_parameters():
            shard_dim = param.sharding_spec.dim

            chunks = [t['model'][name]['data'] for t in tensors_to_merge]
            merged = torch.cat(chunks, dim=shard_dim)
            param.data.copy_(merged)
```

## 弹性训练

弹性训练允许集群规模在训练过程中变化。

### 弹性的挑战

```
Initial: 64 GPUs, DP=64, TP=1

         ┌───┐ ┌───┐ ┌───┐ ... ┌───┐
         │G0 │ │G1 │ │G2 │     │G63│
         └───┘ └───┘ └───┘     └───┘
                    │
                    ▼  8 GPUs fail

After:   56 GPUs, DP=56, TP=1

         ┌───┐ ┌───┐ ┌───┐ ... ┌───┐
         │G0 │ │G1 │ │G2 │     │G55│
         └───┘ └───┘ └───┘     └───┘

         Batch size changes!
         Learning rate must adjust!
```

### 弹性数据并行

```python
class ElasticDataParallel:
    """Data parallelism that handles changing cluster size."""

    def __init__(self,
                 model: nn.Module,
                 base_batch_size: int,
                 base_lr: float):

        self.model = model
        self.base_batch_size = base_batch_size
        self.base_lr = base_lr

        # Track cluster state
        self.initial_world_size = dist.get_world_size()
        self.current_world_size = self.initial_world_size

    def handle_membership_change(self,
                                  new_world_size: int,
                                  optimizer: torch.optim.Optimizer) -> None:
        """Adjust for new cluster size."""

        old_world_size = self.current_world_size
        self.current_world_size = new_world_size

        # Adjust learning rate (linear scaling)
        lr_scale = new_world_size / old_world_size
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_scale

        # Effective batch size changes automatically
        # (each rank still processes base_batch_size)

        print(f"Cluster resized: {old_world_size} → {new_world_size}")
        print(f"New effective batch: {new_world_size * self.base_batch_size}")
        print(f"New learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    def forward_backward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward and backward with proper gradient scaling."""

        # Forward
        output = self.model(batch)
        loss = self.compute_loss(output)

        # Backward
        loss.backward()

        # All-reduce gradients
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.current_world_size

        return loss
```

### 弹性检查点

对于弹性训练，检查点必须能以任意集群规模恢复：

```python
class ElasticCheckpoint:
    """Checkpoint format that supports elastic loading."""

    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             step: int,
             path: str) -> None:
        """Save elastic-compatible checkpoint."""

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank saves with its position info
        state = {
            'model_shard': model.state_dict(),
            'optimizer_shard': optimizer.state_dict(),
            'step': step,
            'rank': rank,
            'world_size': world_size,

            # For resharding
            'sharding_info': self._get_sharding_info(model),
        }

        shard_path = os.path.join(path, f"shard_{rank:05d}_of_{world_size}.pt")
        torch.save(state, shard_path)

        # Also save combined model for easy single-GPU loading
        if rank == 0:
            full_model = self._gather_full_model(model)
            torch.save(full_model, os.path.join(path, "model_full.pt"))

    def load_elastic(self,
                     model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     path: str) -> int:
        """Load checkpoint into potentially different cluster size."""

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Find checkpoint metadata
        shards = glob.glob(os.path.join(path, "shard_*_of_*.pt"))
        ckpt_world_size = self._extract_world_size(shards[0])

        if world_size == ckpt_world_size:
            # Easy case: same size
            state = torch.load(
                os.path.join(path, f"shard_{rank:05d}_of_{world_size}.pt"),
                map_location='cuda'
            )
            model.load_state_dict(state['model_shard'])
            optimizer.load_state_dict(state['optimizer_shard'])

        elif world_size < ckpt_world_size:
            # Fewer GPUs: each rank loads multiple shards
            self._load_merged(path, model, optimizer,
                            ckpt_world_size, world_size, rank)

        else:
            # More GPUs: distribute shards across ranks
            self._load_split(path, model, optimizer,
                           ckpt_world_size, world_size, rank)

        return state['step']

    def _load_merged(self,
                     path: str,
                     model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     ckpt_size: int,
                     current_size: int,
                     rank: int) -> None:
        """Load when current cluster is smaller than checkpoint."""

        # Calculate which shards this rank handles
        shards_per_rank = ckpt_size // current_size
        remainder = ckpt_size % current_size

        start_shard = rank * shards_per_rank + min(rank, remainder)
        end_shard = start_shard + shards_per_rank + (1 if rank < remainder else 0)

        # Load and merge shards
        merged_model_state = {}
        for shard_idx in range(start_shard, end_shard):
            shard_path = os.path.join(path,
                                      f"shard_{shard_idx:05d}_of_{ckpt_size}.pt")
            state = torch.load(shard_path, map_location='cuda')

            for name, param in state['model_shard'].items():
                if name not in merged_model_state:
                    merged_model_state[name] = []
                merged_model_state[name].append(param)

        # Concatenate sharded parameters
        for name, params in merged_model_state.items():
            merged_model_state[name] = torch.cat(params, dim=0)

        model.load_state_dict(merged_model_state)
```

## 增量检查点与差分检查点

对于非常大的模型，完整检查点代价高昂。增量检查点只保存变化。

### 差分检查点

```python
class DeltaCheckpointer:
    """Save only parameter changes between checkpoints."""

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.base_checkpoint = None
        self.base_step = None

    def checkpoint(self,
                   model: nn.Module,
                   step: int,
                   path: str) -> None:
        """Save delta or full checkpoint."""

        current_state = {name: param.data.clone()
                        for name, param in model.named_parameters()}

        if self.base_checkpoint is None:
            # First checkpoint: save full
            self._save_full(current_state, step, path)
            self.base_checkpoint = current_state
            self.base_step = step
        else:
            # Compute and save delta
            delta = self._compute_delta(current_state)
            self._save_delta(delta, step, self.base_step, path)

    def _compute_delta(self,
                       current: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute sparse delta from base checkpoint."""

        delta = {}

        for name, current_param in current.items():
            base_param = self.base_checkpoint[name]
            diff = current_param - base_param

            # Sparsify: only keep significant changes
            mask = torch.abs(diff) > self.threshold

            if mask.any():
                delta[name] = {
                    'indices': mask.nonzero(as_tuple=True),
                    'values': diff[mask],
                    'shape': diff.shape,
                }

        return delta

    def _save_delta(self,
                    delta: Dict[str, Any],
                    step: int,
                    base_step: int,
                    path: str) -> None:
        """Save delta checkpoint."""

        state = {
            'type': 'delta',
            'step': step,
            'base_step': base_step,
            'delta': delta,
        }

        delta_path = os.path.join(path, f"delta_{base_step}_to_{step}.pt")
        torch.save(state, delta_path)

    def load_with_deltas(self,
                         base_path: str,
                         delta_paths: List[str],
                         model: nn.Module) -> None:
        """Load base checkpoint and apply deltas."""

        # Load base
        base_state = torch.load(base_path)
        model.load_state_dict(base_state['model'])

        # Apply deltas in order
        for delta_path in delta_paths:
            delta_state = torch.load(delta_path)
            self._apply_delta(model, delta_state['delta'])

    def _apply_delta(self,
                     model: nn.Module,
                     delta: Dict[str, Any]) -> None:
        """Apply delta to model parameters."""

        for name, param in model.named_parameters():
            if name in delta:
                d = delta[name]
                param.data[d['indices']] += d['values']
```

### 检查点的压缩

```python
class CompressedCheckpointer:
    """Checkpoint with compression for storage efficiency."""

    def __init__(self, compression: str = 'lz4'):
        self.compression = compression

        if compression == 'lz4':
            import lz4.frame
            self.compress = lz4.frame.compress
            self.decompress = lz4.frame.decompress
        elif compression == 'zstd':
            import zstandard
            self.compressor = zstandard.ZstdCompressor(level=3)
            self.decompressor = zstandard.ZstdDecompressor()
            self.compress = self.compressor.compress
            self.decompress = self.decompressor.decompress

    def save(self, state: Dict[str, Any], path: str) -> None:
        """Save compressed checkpoint."""

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(state, buffer)
        data = buffer.getvalue()

        # Compress
        compressed = self.compress(data)

        # Write
        with open(path, 'wb') as f:
            f.write(compressed)

        ratio = len(data) / len(compressed)
        print(f"Compression ratio: {ratio:.2f}x")

    def load(self, path: str) -> Dict[str, Any]:
        """Load compressed checkpoint."""

        with open(path, 'rb') as f:
            compressed = f.read()

        data = self.decompress(compressed)
        buffer = io.BytesIO(data)

        return torch.load(buffer, map_location='cuda')
```

## 完整的容错训练循环

```python
class FaultTolerantTrainer:
    """Complete training loop with fault tolerance."""

    def __init__(self,
                 model: DistributedModel,
                 optimizer: torch.optim.Optimizer,
                 dataloader: DataLoader,
                 checkpoint_dir: str,
                 checkpoint_interval: int = 1000,
                 max_failures: int = 100):

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.max_failures = max_failures

        self.checkpointer = AsyncCheckpointer(checkpoint_dir)
        self.loader = ResilientLoader(checkpoint_dir)

        # Failure tracking
        self.failure_count = 0
        self.last_checkpoint_step = 0

    def train(self, total_steps: int) -> None:
        """Main training loop with fault tolerance."""

        # Try to resume from checkpoint
        start_step = self._maybe_resume()

        step = start_step

        try:
            for batch in self.dataloader:
                if step >= total_steps:
                    break

                # Training step
                loss = self._train_step(batch)

                step += 1

                # Checkpoint
                if step % self.checkpoint_interval == 0:
                    self._checkpoint(step)

                # Logging
                if step % 100 == 0:
                    print(f"Step {step}, Loss: {loss:.4f}")

        except Exception as e:
            self._handle_failure(e, step)

            if self.failure_count < self.max_failures:
                # Retry from last checkpoint
                self.train(total_steps)
            else:
                raise RuntimeError(f"Exceeded max failures: {self.max_failures}")

        # Final checkpoint
        self._checkpoint(step, force=True)

    def _maybe_resume(self) -> int:
        """Resume from checkpoint if available."""

        state = self.loader.load_latest()

        if state is None:
            print("No checkpoint found, starting fresh")
            return 0

        # Restore state
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

        step = state['step']
        self.last_checkpoint_step = step

        print(f"Resumed from checkpoint at step {step}")

        return step

    def _train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""

        self.optimizer.zero_grad()

        output = self.model(batch)
        loss = output.loss

        loss.backward()

        # Gradient synchronization
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

        self.optimizer.step()

        return loss.item()

    def _checkpoint(self, step: int, force: bool = False) -> None:
        """Save checkpoint asynchronously."""

        # Coordinate across ranks
        should_checkpoint = torch.tensor([1], device='cuda')
        dist.all_reduce(should_checkpoint, op=dist.ReduceOp.MIN)

        if should_checkpoint.item() == 0 and not force:
            return

        dist.barrier()

        self.checkpointer.checkpoint_async(
            self.model, self.optimizer, step
        )

        self.last_checkpoint_step = step

    def _handle_failure(self, error: Exception, step: int) -> None:
        """Handle training failure."""

        self.failure_count += 1

        work_lost = step - self.last_checkpoint_step

        print(f"Failure at step {step}: {error}")
        print(f"Work lost: {work_lost} steps")
        print(f"Total failures: {self.failure_count}")

        # Wait for any pending checkpoints
        self.checkpointer._wait_pending()

        # Reset distributed state
        dist.barrier()
```

## 练习题

1. **MTBF 计算**：一个集群有 4,096 块 GPU（MTBF 25,000 小时）、512 台主机（MTBF 8,000 小时）和 32 台网络交换机（MTBF 100,000 小时）。计算系统 MTBF。

??? success "解答"
    **系统 MTBF 公式：**

    对于相互独立的组件，故障率相加：

    $$\lambda_{system} = \sum_i n_i \cdot \lambda_i = \sum_i \frac{n_i}{MTBF_i}$$

    $$MTBF_{system} = \frac{1}{\lambda_{system}}$$

    **各组件故障率：**

    | 组件 | 数量 | MTBF（小时） | 故障率（每小时） |
    |-----------|-------|------------|----------------------|
    | GPU | 4,096 | 25,000 | $\frac{4096}{25000} = 0.164$ |
    | 主机 | 512 | 8,000 | $\frac{512}{8000} = 0.064$ |
    | 交换机 | 32 | 100,000 | $\frac{32}{100000} = 0.00032$ |

    **系统故障率：**

    $$\lambda_{system} = 0.164 + 0.064 + 0.00032 = 0.228 \text{ failures/hour}$$

    **系统 MTBF：**

    $$MTBF_{system} = \frac{1}{0.228} = \boxed{4.38 \text{ hours}}$$

    **分析：**

    | 组件 | 对故障率的贡献 |
    |-----------|------------------------------|
    | GPU | 72% |
    | 主机 | 28% |
    | 交换机 | 0.14% |

    尽管单个 GPU 的可靠性比主机更好，但 GPU 数量众多，因此在故障率中占主导。

    **含义：**

    - 大约每 4.4 小时会有一次故障
    - 检查点间隔应远小于 4.4 小时
    - 投资于 GPU 可靠性的回报最高

2. **最优检查点间隔**：给定 MTBF 为 2 小时、检查点时间 30 秒、步时间 500ms，最优检查点间隔是多少？

??? success "解答"
    **给定：**

    - MTBF = 2 小时 = 7,200 秒
    - 检查点时间：$C$ = 30 秒
    - 步时间：$T_{step}$ = 0.5 秒

    **最优检查点间隔公式（Young/Daly）：**

    $$I^* = \sqrt{2 \cdot C \cdot MTBF}$$

    **计算：**

    $$I^* = \sqrt{2 \times 30 \times 7200} = \sqrt{432,000} = \boxed{657 \text{ seconds}} \approx 11 \text{ minutes}$$

    **换算成步数：**

    $$\text{Steps between checkpoints} = \frac{657}{0.5} = \boxed{1,314 \text{ steps}}$$

    **验证最优性：**

    带检查点的总时间 = 训练时间 + 检查点开销 + 期望浪费的工作

    $$T_{total} = T_{train} \times \left(1 + \frac{C}{I} + \frac{I}{2 \times MTBF}\right)$$

    | 间隔 | 检查点开销 | 期望浪费 | 总开销 |
    |----------|--------------------:|---------------:|---------------:|
    | 200s | 15.0% | 1.4% | 16.4% |
    | 400s | 7.5% | 2.8% | 10.3% |
    | **657s** | 4.6% | 4.6% | **9.1%** |
    | 1000s | 3.0% | 6.9% | 9.9% |
    | 2000s | 1.5% | 13.9% | 15.4% |

    在最优间隔处，检查点开销等于期望浪费的工作（都约 4.6%）。

    **实践考量：**

    - 取整到好记的步数：1,300 或 1,500 步
    - 考虑异步检查点会降低有效 $C$
    - 监控实际故障率并据此调整

    **小结：**

    | 指标 | 值 |
    |--------|-------|
    | 最优间隔 | 657 秒 |
    | 检查点间隔步数 | 1,314 |
    | 最优下的总开销 | 9.1% |

3. **异步开销**：每块 GPU 的模型大小为 50GB。PCIe 带宽为 32 GB/s。每个 rank 的并行文件系统带宽为 2 GB/s。计算同步与异步检查点的开销。

??? success "解答"
    **给定：**

    - 每 rank 的模型大小：50 GB
    - PCIe 带宽（GPU→CPU）：32 GB/s
    - 并行文件系统带宽（CPU→存储）：2 GB/s

    **同步检查点：**

    所有操作都在关键路径上：

    $$T_{sync} = T_{GPU \to CPU} + T_{CPU \to storage}$$

    $$T_{sync} = \frac{50}{32} + \frac{50}{2} = 1.56 + 25 = \boxed{26.56 \text{ seconds}}$$

    训练在整个过程中被阻塞。

    **异步检查点：**

    只有 GPU→CPU 的拷贝在关键路径上（之后训练即恢复）：

    $$T_{async}^{blocking} = T_{GPU \to CPU} = \frac{50}{32} = \boxed{1.56 \text{ seconds}}$$

    存储写入在后台进行。

    **开销对比：**

    | 方案 | 阻塞时间 | 加速比 |
    |----------|---------------|---------|
    | 同步 | 26.56s | 1× |
    | 异步 | 1.56s | **17×** |

    **有效检查点时间的缩减：**

    $$\text{Reduction} = \frac{26.56 - 1.56}{26.56} = \boxed{94\%}$$

    **后台写入的考量：**

    25 秒的后台写入必须在下次检查点之前完成：

    $$\text{Min checkpoint interval} > 25 \text{ seconds}$$

    对于约 657s 的最优间隔（来自练习 2），这很容易满足。

    **对最优检查点的影响：**

    使用异步后，有效检查点时间变为 1.56s 而非 26.56s：

    $$I_{async}^* = \sqrt{2 \times 1.56 \times 7200} = 150 \text{ seconds}$$

    更频繁的检查点会减少浪费的工作！

    **内存开销：**

    异步需要在 CPU 内存中做双缓冲：

    $$M_{overhead} = 50 \text{ GB/rank}$$

    | 指标 | 同步 | 异步 |
    |--------|------|-------|
    | 阻塞时间 | 26.56s | 1.56s |
    | 最优间隔 | 657s | 150s |
    | 期望浪费的工作 | 4.6% | 1.0% |
    | CPU 内存开销 | 0 | 50 GB |

4. **重分片的数学**：一个检查点以 TP=8 保存。以 TP=4 加载。每个参数沿维度 0 分片。写出重分片公式。

??? success "解答"
    **场景：**

    - 保存时：TP=8（沿维度 0 有 8 个分片）
    - 加载时：TP=4（沿维度 0 需要 4 个分片）

    **原始分片（TP=8）：**

    对于形状为 $[D_0, D_1, ...]$ 的参数：

    每个 rank $r \in [0,7]$ 持有：

    $$\text{shard}_r = \text{param}\left[\frac{r \cdot D_0}{8} : \frac{(r+1) \cdot D_0}{8}, :, ...\right]$$

    **目标分片（TP=4）：**

    每个 rank $r' \in [0,3]$ 需要：

    $$\text{shard}_{r'} = \text{param}\left[\frac{r' \cdot D_0}{4} : \frac{(r'+1) \cdot D_0}{4}, :, ...\right]$$

    **重分片公式：**

    由于 $8/4 = 2$，每个新分片由 2 个旧分片拼接而成：

    $$\boxed{\text{shard}_{r'}^{new} = \text{concat}\left(\text{shard}_{2r'}^{old}, \text{shard}_{2r'+1}^{old}\right)}$$

    **映射：**

    | 新 rank（TP=4） | 旧 rank（TP=8） | 公式 |
    |-----------------|------------------|---------|
    | 0 | 0, 1 | concat(shard_0, shard_1) |
    | 1 | 2, 3 | concat(shard_2, shard_3) |
    | 2 | 4, 5 | concat(shard_4, shard_5) |
    | 3 | 6, 7 | concat(shard_6, shard_7) |

    **通用重分片公式：**

    对于 TP_old → TP_new 且 TP_old > TP_new（合并分片）：

    $$k = \frac{TP_{old}}{TP_{new}}$$

    $$\text{shard}_{r'}^{new} = \text{concat}\left(\text{shard}_{k \cdot r'}^{old}, \text{shard}_{k \cdot r'+1}^{old}, ..., \text{shard}_{k \cdot r'+(k-1)}^{old}\right)$$

    对于 TP_old < TP_new（切分分片）：

    $$k = \frac{TP_{new}}{TP_{old}}$$

    $$\text{shard}_{r'}^{new} = \text{split}_k\left(\text{shard}_{\lfloor r'/k \rfloor}^{old}\right)[r' \mod k]$$

    **Python 实现：**

    ```python
    def reshard_tp(old_shards, old_tp, new_tp, dim=0):
        """Reshard from old_tp to new_tp along specified dimension."""
        # Reconstruct full tensor
        full = torch.cat(old_shards, dim=dim)

        # Create new shards
        chunk_size = full.shape[dim] // new_tp
        new_shards = torch.split(full, chunk_size, dim=dim)

        return list(new_shards)

    # Example: TP=8 → TP=4
    # old_shards: list of 8 tensors
    # new_shards: list of 4 tensors (each 2× larger in dim 0)
    new_shards = reshard_tp(old_shards, old_tp=8, new_tp=4)
    ```

5. **压缩权衡**：检查点大小 100GB。LZ4 压缩比 2.5×、吞吐 4 GB/s。不压缩写入 10 GB/s。保存时哪个更快？加载时呢？

??? success "解答"
    **给定：**

    - 检查点大小：100 GB（未压缩）
    - LZ4 压缩比：2.5× → 压缩后大小 = 40 GB
    - LZ4 吞吐：4 GB/s（压缩与解压）
    - 未压缩 I/O：10 GB/s

    **保存时间分析：**

    **不压缩：**
    $$T_{save}^{uncomp} = \frac{100}{10} = 10 \text{ seconds}$$

    **压缩：**
    $$T_{compress} = \frac{100}{4} = 25 \text{ seconds}$$

    $$T_{write} = \frac{40}{10} = 4 \text{ seconds}$$

    若串行：$T_{save}^{comp} = 25 + 4 = 29$ 秒

    若流水化（边写边压缩）：$T_{save}^{comp} = \max(25, 4) = 25$ 秒

    **保存的结论：**

    $$\boxed{\text{Uncompressed is faster for save: 10s vs 25s}}$$

    **加载时间分析：**

    **不压缩：**
    $$T_{load}^{uncomp} = \frac{100}{10} = 10 \text{ seconds}$$

    **压缩：**
    $$T_{read} = \frac{40}{10} = 4 \text{ seconds}$$

    $$T_{decompress} = \frac{100}{4} = 25 \text{ seconds}$$

    若串行：$T_{load}^{comp} = 4 + 25 = 29$ 秒

    若流水化：$T_{load}^{comp} = \max(4, 25) = 25$ 秒

    **加载的结论：**

    $$\boxed{\text{Uncompressed is faster for load: 10s vs 25s}}$$

    **小结：**

    | 操作 | 不压缩 | 压缩 | 胜者 |
    |-----------|--------------|------------|--------|
    | 保存 | 10s | 25s | 不压缩 |
    | 加载 | 10s | 25s | 不压缩 |
    | 存储 | 100 GB | 40 GB | 压缩（2.5×） |

    **压缩何时占优：**

    当 I/O 是瓶颈时，压缩变得更快：

    $$T_{uncomp} > T_{comp}$$

    $$\frac{S}{BW_{io}} > \frac{S}{BW_{comp}}$$

    解得：$BW_{io} < \frac{BW_{comp}}{ratio} = \frac{4}{2.5} = 1.6$ GB/s

    **如果存储带宽为 1.5 GB/s：**

    | 操作 | 不压缩 | 压缩 |
    |-----------|--------------|------------|
    | 保存 | 66.7s | max(25, 26.7) = 26.7s |
    | 加载 | 66.7s | max(26.7, 25) = 26.7s |

    在低 I/O 带宽下，压缩占优！

    **建议：** 当存储带宽 < 1.6 GB/s 时使用压缩。

6. **弹性批大小**：在 64 块 GPU 上以批 2048、学习率 0.001 训练。集群缩容到 48 块 GPU。在线性缩放规则下，新的批大小与学习率应为多少？

??? success "解答"
    **给定：**

    - 原始：64 块 GPU、batch=2048、LR=0.001
    - 新的：48 块 GPU

    **每 GPU 的批大小：**

    $$B_{gpu} = \frac{2048}{64} = 32 \text{ samples/GPU}$$

    **方案 1：保持每 GPU 的批不变**

    新的全局批：

    $$B_{new} = 32 \times 48 = \boxed{1536}$$

    **学习率的线性缩放规则：**

    $$\frac{LR_{new}}{LR_{old}} = \frac{B_{new}}{B_{old}}$$

    $$LR_{new} = 0.001 \times \frac{1536}{2048} = \boxed{0.00075}$$

    **小结（方案 1）：**

    | 参数 | 原始（64 GPU） | 新的（48 GPU） |
    |-----------|-------------------|--------------|
    | GPU | 64 | 48 |
    | 每 GPU 批 | 32 | 32 |
    | 全局批 | 2048 | 1536 |
    | 学习率 | 0.001 | 0.00075 |

    **方案 2：用梯度累积保持全局批**

    通过累积梯度保持 B=2048：

    $$\text{accum steps} = \frac{2048}{48 \times 32} = \frac{2048}{1536} = 1.33$$

    不是整数！调整每 GPU 批：

    - 选项 A：$B_{gpu}=43$、accum=1 → 全局批 = 2064（接近）
    - 选项 B：$B_{gpu}=32$、accum=2 → 全局批 = 3072（太高）
    - 选项 C：$B_{gpu}=21$、accum=2 → 全局批 = 2016（接近）

    **方案 2A 的配置：**

    | 参数 | 值 |
    |-----------|-------|
    | GPU | 48 |
    | 每 GPU 批 | 43 |
    | 全局批 | 2064 |
    | 学习率 | 0.001 × (2064/2048) = 0.001008 |

    **实践建议：**

    方案 1（减小批并缩放学习率）更简洁，且能保持训练动态。

    **最终答案：**

    $$\boxed{B = 1536, \quad LR = 0.00075}$$

    **吞吐影响：**

    每步 token 数：$1536 \times S$ 对 $2048 \times S$（原来的 75%）

    所需步数：要处理同样的数据需要多 1.33× 的步数

    每步时间：相近（受计算限制）

    **总训练时间增加：** 约慢 33%

## 关键要点

1. **大规模下故障是统计上的必然**：要为它做规划，而不是指望避开它。

2. **检查点频率可以优化**：存在一个数学最优值，在开销与工作损失之间取得平衡。

3. **异步检查点不可或缺**：让 I/O 与计算重叠，以最小化训练中断。

4. **一致性需要协同**：分布式检查点需要屏障与验证。

5. **重分片使弹性成为可能**：检查点应能以不同的并行配置加载。

6. **压缩与差分法减少存储**：但需要考虑 I/O 带宽的权衡。

7. **容错是一个系统属性**：需要检查点、故障检测与恢复协同工作。