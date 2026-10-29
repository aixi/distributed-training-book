---
title: "调查协议"
subtitle: "系统化排查分布式训练问题的方法"
---

<div class="chapter-opener" markdown>
当训练在凌晨三点、横跨 256 块 GPU 时崩溃，你需要的是方法论，而不是运气。本章提供一套系统化的协议，用于调查分布式训练问题——从最初的症状，到定位、隔离、根因，再到修复。
</div>

<div class="investigation-question" markdown>
**问题**：你的训练任务卡住了。损失已经好几个小时没有下降。一些 GPU 显示 100% 利用率，另一些只有 20%。一个节点无法访问。你从哪里开始？按什么顺序调查？如何隔离问题？
</div>

!!! tip "推荐阅读路径"
    先读五阶段流程，建立诊断的心智模型，然后把较长的代码块当作实现参考。
    这样即便你不逐字采纳每个脚本，这份协议依然可操作。

## 调查心态

分布式训练问题与单机 bug 有着本质区别：

1. **不可复现性**：依赖时序的问题可能不会再次出现
2. **部分可观测性**：你无法同时看到所有东西
3. **级联故障**：一个故障会触发其他故障
4. **规模放大**：罕见事件在规模之上会变得常见

调查协议正是为了系统化地应对这些挑战而设计的。

## 五个阶段

```
┌─────────────────────────────────────────────────────────────┐
│                    INVESTIGATION PROTOCOL                    │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: TRIAGE         → Is this urgent? What's broken?   │
│  Phase 2: LOCALIZATION   → Where is the problem?            │
│  Phase 3: ISOLATION      → What exactly is failing?         │
│  Phase 4: ROOT CAUSE     → Why is it failing?               │
│  Phase 5: RESOLUTION     → How do we fix it and prevent it? │
└─────────────────────────────────────────────────────────────┘
```

每个阶段都有特定的问题、工具与产出。

## 阶段 1：分诊

**目标**：评估严重程度并对问题分类。

### 最初的五分钟

```
1. Is training still running?
   ├── Yes → Proceed to performance investigation
   └── No → Proceed to crash investigation

2. If crashed, when did it crash?
   ├── Immediately → Configuration or environment issue
   ├── After warmup → Numerical instability or resource exhaustion
   └── After hours/days → Hardware failure or rare race condition

3. What are the symptoms?
   ├── All ranks crashed → Collective failure or shared resource issue
   ├── One rank crashed → Local hardware or OOM
   ├── Training hangs → Deadlock or straggler
   └── Slow training → Performance regression
```

### 快速健康检查

```python
import time

def triage_check(heartbeat_store):
    """First-response triage for distributed training."""
    checks = {
        'processes_alive': check_all_ranks_alive(heartbeat_store),
        'gpus_visible': check_gpu_visibility(),
        'nccl_initialized': check_nccl_init(),
        'memory_available': check_memory_headroom(),
        'network_reachable': check_inter_node_connectivity(),
        'recent_checkpoints': check_checkpoint_age(),
    }

    critical = []
    warning = []

    for check, status in checks.items():
        if status == 'FAIL':
            critical.append(check)
        elif status == 'WARN':
            warning.append(check)

    return {
        'severity': 'CRITICAL' if critical else ('WARNING' if warning else 'OK'),
        'critical': critical,
        'warning': warning,
    }

def check_all_ranks_alive(heartbeat_store) -> str:
    """Non-blocking liveness check using an external heartbeat store."""
    now = time.time()
    heartbeat_store.write(dist.get_rank(), now)

    if dist.get_rank() == 0:
        timestamps = heartbeat_store.read_all()
        stale = any(now - t > 30 for t in timestamps.values())
        return 'FAIL' if stale else 'OK'
    return 'OK'

def check_memory_headroom():
    """Check for sufficient GPU memory."""
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    usage = allocated / total

    if usage > 0.95:
        return 'FAIL'
    elif usage > 0.85:
        return 'WARN'
    return 'OK'
```

### 分诊决策树

```
START
  │
  ▼
Is training running? ──No──► Did it ever start?
  │                              │
  │Yes                          │No──► Environment issue (Phase 2A)
  │                              │
  │                             │Yes──► Crash analysis (Phase 2B)
  ▼
Is loss decreasing? ──No──► Has loss ever decreased?
  │                              │
  │Yes                          │No──► Initialization issue
  │                              │
  │                             │Yes──► Training stalled (Phase 2C)
  ▼
Is step time stable? ──No──► Performance regression (Phase 2D)
  │
  │Yes
  ▼
Proceed with monitoring
```

## 阶段 2：定位

**目标**：确定问题发生在哪里。

### 2A：环境问题

阻止训练启动的问题。

```python
class EnvironmentChecker:
    """Check for environment configuration issues."""

    def check_all(self):
        results = {}

        # CUDA
        results['cuda_available'] = torch.cuda.is_available()
        results['cuda_device_count'] = torch.cuda.device_count()

        # NCCL
        results['nccl_version'] = self._get_nccl_version()

        # Network
        results['master_addr'] = os.environ.get('MASTER_ADDR', 'NOT SET')
        results['master_port'] = os.environ.get('MASTER_PORT', 'NOT SET')
        results['world_size'] = os.environ.get('WORLD_SIZE', 'NOT SET')
        results['rank'] = os.environ.get('RANK', 'NOT SET')

        # File system
        results['checkpoint_dir_writable'] = self._check_checkpoint_dir()

        return results

    def diagnose(self, results):
        """Provide diagnosis based on check results."""
        issues = []

        if not results['cuda_available']:
            issues.append("CUDA not available. Check driver installation.")

        if results['cuda_device_count'] == 0:
            issues.append("No CUDA devices found. Check GPU visibility.")

        if results['master_addr'] == 'NOT SET':
            issues.append("MASTER_ADDR not set. Required for distributed.")

        if not results['checkpoint_dir_writable']:
            issues.append("Cannot write to checkpoint directory.")

        return issues

    def _get_nccl_version(self):
        try:
            return torch.cuda.nccl.version()
        except:
            return "Unknown"

    def _check_checkpoint_dir(self):
        ckpt_dir = os.environ.get('CHECKPOINT_DIR', './checkpoints')
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            test_file = os.path.join(ckpt_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except:
            return False
```

### 2B：崩溃分析

针对已启动但崩溃的训练。

```python
class CrashAnalyzer:
    """Analyze crash logs and stack traces."""

    KNOWN_PATTERNS = {
        r'CUDA out of memory': {
            'category': 'OOM',
            'likely_cause': 'Batch size too large or memory leak',
            'next_steps': ['Reduce batch size', 'Enable gradient checkpointing',
                          'Check for memory leaks with torch.cuda.memory_snapshot()']
        },
        r'NCCL error|ncclSystemError': {
            'category': 'NETWORK',
            'likely_cause': 'Network timeout or hardware failure',
            'next_steps': ['Check network connectivity', 'Increase NCCL timeout',
                          'Check for hardware errors with nvidia-smi']
        },
        r'RuntimeError: Expected all tensors to be on the same device': {
            'category': 'DEVICE_MISMATCH',
            'likely_cause': 'Tensor on wrong device',
            'next_steps': ['Check .to(device) calls', 'Verify data loading pipeline']
        },
        r'loss.*nan|NaN': {
            'category': 'NUMERICAL',
            'likely_cause': 'Numerical instability',
            'next_steps': ['Check learning rate', 'Enable gradient clipping',
                          'Use loss scaling for mixed precision']
        },
        r'Timeout|deadline exceeded': {
            'category': 'TIMEOUT',
            'likely_cause': 'Collective operation timed out',
            'next_steps': ['Check for stragglers', 'Increase timeout',
                          'Check network bandwidth']
        }
    }

    def analyze_log(self, log_content: str) -> dict:
        """Analyze a crash log for known patterns."""
        import re

        findings = []
        for pattern, info in self.KNOWN_PATTERNS.items():
            if re.search(pattern, log_content, re.IGNORECASE):
                findings.append({
                    'pattern': pattern,
                    **info
                })

        return {
            'findings': findings,
            'unknown': len(findings) == 0,
            'log_snippet': log_content[-1000:]  # Last 1000 chars
        }

    def aggregate_rank_logs(self, log_dir: str) -> dict:
        """Aggregate crash info from all rank logs."""
        from pathlib import Path

        rank_findings = {}
        for log_file in Path(log_dir).glob("rank_*.log"):
            rank = int(log_file.stem.split('_')[1])
            with open(log_file) as f:
                content = f.read()
            rank_findings[rank] = self.analyze_log(content)

        # Find common patterns
        categories = {}
        for rank, findings in rank_findings.items():
            for f in findings['findings']:
                cat = f['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(rank)

        return {
            'per_rank': rank_findings,
            'categories': categories,
            'all_same': len(categories) == 1,
        }
```

### 2C：训练停滞

损失不下降或训练卡住。

```python
class StallDetector:
    """Detect and diagnose training stalls."""

    def __init__(self, window_size: int = 100):
        self.loss_history = []
        self.window_size = window_size
        self.last_progress_time = time.time()

    def record_loss(self, loss: float):
        """Record a loss value."""
        self.loss_history.append((time.time(), loss))

        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]

    def is_stalled(self) -> tuple:
        """Check if training has stalled."""
        if len(self.loss_history) < self.window_size:
            return False, "Insufficient data"

        recent = self.loss_history[-self.window_size:]
        older = self.loss_history[-2*self.window_size:-self.window_size]

        if not older:
            return False, "Insufficient data"

        recent_avg = sum(l for _, l in recent) / len(recent)
        older_avg = sum(l for _, l in older) / len(older)

        # Check for improvement
        improvement = (older_avg - recent_avg) / older_avg

        if improvement < 0.001:  # Less than 0.1% improvement
            return True, f"Loss plateau: {older_avg:.4f} → {recent_avg:.4f}"

        if improvement < 0:  # Loss increasing
            return True, f"Loss increasing: {older_avg:.4f} → {recent_avg:.4f}"

        return False, f"Normal progress: {improvement:.2%} improvement"

    def diagnose_stall(self) -> list:
        """Provide potential causes for a stall."""
        causes = [
            "Learning rate too low or decayed too much",
            "Gradient clipping too aggressive",
            "Batch size changed without LR adjustment",
            "Model capacity insufficient for task",
            "Data loader returning same data (shuffle issue)",
            "Optimizer state corrupted after checkpoint load",
        ]
        return causes
```

### 2D：性能退化

训练能跑，但比预期慢。

```python
class PerformanceRegression:
    """Detect and diagnose performance regressions."""

    def __init__(self, baseline_step_time: float):
        self.baseline = baseline_step_time
        self.step_times = []

    def record_step(self, step_time: float):
        self.step_times.append(step_time)

    def detect_regression(self, threshold: float = 1.1) -> tuple:
        """Detect if current performance is regressed."""
        if len(self.step_times) < 10:
            return False, "Insufficient data"

        current_avg = sum(self.step_times[-10:]) / 10
        ratio = current_avg / self.baseline

        if ratio > threshold:
            return True, f"Step time {ratio:.1%} of baseline ({current_avg:.2f}s vs {self.baseline:.2f}s)"

        return False, f"Normal: {ratio:.1%} of baseline"

    def diagnose_slowdown(self) -> dict:
        """Categorize potential causes of slowdown."""
        return {
            'compute': [
                "Thermal throttling (check GPU temperature)",
                "Power capping (check nvidia-smi)",
                "Memory bandwidth saturation",
                "Kernel launch overhead increase",
            ],
            'communication': [
                "Network congestion from other jobs",
                "Straggler node",
                "NCCL algorithm changed",
                "Bucket size misconfiguration",
            ],
            'memory': [
                "Increased fragmentation",
                "Swap/paging due to CPU memory pressure",
                "Checkpoint saving blocking training",
            ],
            'data': [
                "Data loader bottleneck",
                "Disk I/O contention",
                "Network storage latency",
            ]
        }
```

## 阶段 3：隔离

**目标**：隔离出导致问题的具体组件或交互。

### 二分法

分而治之，找出有问题的组件。

```python
from typing import Tuple

class BisectionDebugger:
    """Systematic isolation through bisection."""

    def __init__(self, input_shape: Tuple[int, ...]):
        self.input_shape = input_shape

    def bisect_scale(self, working_scale: int, failing_scale: int):
        """
        Find the scale at which training breaks.

        Args:
            working_scale: Known working GPU count
            failing_scale: Known failing GPU count
        """
        print(f"Bisecting between {working_scale} and {failing_scale} GPUs")

        while failing_scale - working_scale > 1:
            mid = (working_scale + failing_scale) // 2
            print(f"  Testing {mid} GPUs...")

            success = self._test_at_scale(mid)

            if success:
                working_scale = mid
                print(f"    → Works at {mid}")
            else:
                failing_scale = mid
                print(f"    → Fails at {mid}")

        print(f"\nBreakpoint: between {working_scale} and {failing_scale} GPUs")
        return working_scale, failing_scale

    def bisect_batch_size(self, working_batch: int, failing_batch: int):
        """Find the batch size at which OOM occurs."""
        print(f"Bisecting batch size between {working_batch} and {failing_batch}")

        while failing_batch - working_batch > 1:
            mid = (working_batch + failing_batch) // 2
            print(f"  Testing batch size {mid}...")

            success = self._test_batch_size(mid)

            if success:
                working_batch = mid
            else:
                failing_batch = mid

        print(f"\nMax batch size: {working_batch}")
        return working_batch

    def bisect_layers(self, model, num_layers: int):
        """Find which layer causes the issue."""
        print(f"Bisecting {num_layers} layers")

        working = 0
        failing = num_layers

        while failing - working > 1:
            mid = (working + failing) // 2
            print(f"  Testing first {mid} layers...")

            # Create partial model
            success = self._test_partial_model(model, mid)

            if success:
                working = mid
            else:
                failing = mid

        print(f"\nProblematic layer: {failing}")
        return failing

    def _test_at_scale(self, num_gpus: int) -> bool:
        """Test if training works at given scale."""
        # Implementation depends on your infrastructure
        pass

    def _test_batch_size(self, batch_size: int) -> bool:
        """Test if batch size fits in memory."""
        try:
            # Allocate test tensor
            dummy = torch.zeros(batch_size, *self.input_shape, device='cuda')
            del dummy
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            return False

    def _test_partial_model(self, model, num_layers: int) -> bool:
        """Test forward pass with subset of layers."""
        pass
```

### 组件隔离测试

```python
class ComponentIsolator:
    """Test individual components in isolation."""

    def test_compute_only(self, model, batch):
        """Test forward/backward without communication."""
        # Disable distributed
        model_copy = copy.deepcopy(model)
        for param in model_copy.parameters():
            param.grad = None

        with torch.no_grad():
            output = model_copy(batch)

        return {'success': True, 'output_shape': output.shape}

    def test_communication_only(self, size_bytes: int, num_iterations: int = 10):
        """Test collective communication in isolation."""
        tensor = torch.zeros(size_bytes // 4, dtype=torch.float32, device='cuda')

        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return {
            'success': True,
            'avg_time_ms': sum(times) / len(times) * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
        }

    def test_memory_only(self, allocation_gb: float):
        """Test if GPU can allocate specified memory."""
        try:
            size = int(allocation_gb * 1e9 / 4)  # float32
            tensor = torch.zeros(size, dtype=torch.float32, device='cuda')
            del tensor
            torch.cuda.empty_cache()
            return {'success': True, 'allocated_gb': allocation_gb}
        except RuntimeError as e:
            return {'success': False, 'error': str(e)}

    def test_data_loading(self, dataloader, num_batches: int = 10):
        """Test data loading in isolation."""
        times = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            start = time.perf_counter()
            if isinstance(batch, torch.Tensor):
                batch = batch.cuda()
            times.append(time.perf_counter() - start)

        return {
            'success': True,
            'avg_time_ms': sum(times) / len(times) * 1000,
            'batches_loaded': len(times),
        }
```

### 网络拓扑分析

```python
class TopologyAnalyzer:
    """Analyze network topology for issues."""

    def map_topology(self):
        """Map the current network topology."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Get hostname
        import socket
        hostname = socket.gethostname()

        # Gather all hostnames
        hostnames = [None] * world_size
        dist.all_gather_object(hostnames, hostname)

        # Group by host
        host_to_ranks = {}
        for r, h in enumerate(hostnames):
            if h not in host_to_ranks:
                host_to_ranks[h] = []
            host_to_ranks[h].append(r)

        return {
            'world_size': world_size,
            'num_hosts': len(host_to_ranks),
            'ranks_per_host': host_to_ranks,
            'local_ranks': host_to_ranks.get(hostname, []),
        }

    def test_pairwise_bandwidth(self, sample_ranks: list = None):
        """Test bandwidth between pairs of ranks."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if sample_ranks is None:
            # Test a few representative pairs
            sample_ranks = [(0, i) for i in range(1, min(4, world_size))]

        results = {}
        tensor_size = 100 * 1024 * 1024  # 100MB

        for src, dst in sample_ranks:
            # Ensure all ranks synchronize for each measurement
            dist.barrier()

            if rank == src or rank == dst:
                tensor = torch.zeros(tensor_size // 4, dtype=torch.float32, device='cuda')

                if rank == src:
                    tensor.fill_(1.0)

                start = time.perf_counter()

                if rank == src:
                    dist.send(tensor, dst)
                elif rank == dst:
                    dist.recv(tensor, src)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                bandwidth = tensor_size / elapsed / 1e9  # GB/s

                if rank == 0 or rank == src:
                    results[(src, dst)] = bandwidth

        # Gather results to rank 0
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

        if rank == 0:
            combined = {}
            for r in all_results:
                if r:
                    combined.update(r)
            return combined

        return None
```

## 阶段 4：根因分析

**目标**：准确理解问题为什么会发生。

### 五个为什么

```python
class FiveWhysAnalyzer:
    """Structured root cause analysis using Five Whys."""

    def analyze(self, symptom: str) -> list:
        """
        Guide through Five Whys analysis.

        Returns a template chain of causes for common symptoms.
        """
        templates = {
            'OOM': [
                "Memory allocation exceeded GPU capacity",
                "Activation memory larger than expected",
                "Batch size and sequence length combination too large",
                "No gradient checkpointing enabled for deep model",
                "Root: Memory budget not calculated before training"
            ],
            'TIMEOUT': [
                "Collective operation timed out",
                "One or more ranks were slow to participate",
                "Work distribution was uneven across ranks",
                "Data loading was slow on some ranks",
                "Root: No data prefetching or imbalanced dataset sharding"
            ],
            'NUMERICAL': [
                "Loss became NaN or Inf",
                "Gradient explosion during backward pass",
                "Learning rate too high for current loss landscape",
                "No gradient clipping with unstable initialization",
                "Root: Hyperparameters not tuned for model scale"
            ],
        }

        return templates

    def interactive_analysis(self, symptom: str):
        """Interactive Five Whys session."""
        print(f"\n=== Five Whys Analysis ===")
        print(f"Symptom: {symptom}")

        chain = []
        current = symptom

        for i in range(5):
            print(f"\nWhy #{i+1}: Why did '{current}' happen?")
            # In practice, this would be interactive
            # Here we show the pattern

        return chain
```

### 假设检验

```python
class HypothesisTester:
    """Test hypotheses about failure causes."""

    def __init__(self):
        self.hypotheses = []
        self.results = []

    def add_hypothesis(self, description: str, test_fn, expected_if_true: str):
        """Add a hypothesis to test."""
        self.hypotheses.append({
            'description': description,
            'test': test_fn,
            'expected': expected_if_true,
            'status': 'pending'
        })

    def run_all(self):
        """Run all hypothesis tests."""
        print("\n=== Hypothesis Testing ===\n")

        for h in self.hypotheses:
            print(f"Testing: {h['description']}")
            print(f"  Expected if true: {h['expected']}")

            try:
                result = h['test']()
                h['status'] = 'passed' if result else 'failed'
                h['result'] = result
            except Exception as e:
                h['status'] = 'error'
                h['result'] = str(e)

            print(f"  Result: {h['status']}")
            if h['status'] == 'passed':
                print(f"  → This hypothesis is SUPPORTED")
            print()

        # Summary
        supported = [h for h in self.hypotheses if h['status'] == 'passed']
        print(f"Supported hypotheses: {len(supported)}/{len(self.hypotheses)}")
        for h in supported:
            print(f"  ✓ {h['description']}")

# Example usage
def example_hypothesis_testing():
    tester = HypothesisTester()

    # Hypothesis: OOM is due to batch size
    tester.add_hypothesis(
        "OOM caused by batch size exceeding memory",
        lambda: test_batch_size(current_batch // 2),  # Test with half batch
        "Training succeeds with smaller batch"
    )

    # Hypothesis: Timeout due to straggler
    tester.add_hypothesis(
        "Timeout caused by straggler node",
        lambda: max(get_rank_times()) > 1.5 * mean(get_rank_times()),
        "One rank consistently slower"
    )

    tester.run_all()
```

### 相关性分析

```python
class CorrelationAnalyzer:
    """Find correlations between events and failures."""

    def __init__(self):
        self.events = []  # List of (timestamp, event_type, details)
        self.failures = []  # List of (timestamp, failure_type, details)

    def record_event(self, event_type: str, details: dict = None):
        self.events.append((time.time(), event_type, details or {}))

    def record_failure(self, failure_type: str, details: dict = None):
        self.failures.append((time.time(), failure_type, details or {}))

    def find_correlations(self, window_seconds: float = 60) -> list:
        """Find events that frequently precede failures."""
        correlations = {}

        for f_time, f_type, f_details in self.failures:
            # Find events in window before failure
            preceding = [
                (e_type, e_details)
                for e_time, e_type, e_details in self.events
                if f_time - window_seconds < e_time < f_time
            ]

            for e_type, e_details in preceding:
                key = (e_type, f_type)
                if key not in correlations:
                    correlations[key] = 0
                correlations[key] += 1

        # Sort by frequency
        sorted_corr = sorted(correlations.items(), key=lambda x: -x[1])

        return [
            {'event': e, 'failure': f, 'count': c}
            for (e, f), c in sorted_corr
        ]

    def analyze_temporal_patterns(self) -> dict:
        """Analyze temporal patterns in failures."""
        if not self.failures:
            return {'pattern': 'none', 'details': 'No failures recorded'}

        times = [f[0] for f in self.failures]

        # Check for periodicity
        if len(times) >= 3:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((i - avg_interval)**2 for i in intervals) / len(intervals)

            if variance < avg_interval * 0.1:  # Low variance = periodic
                return {
                    'pattern': 'periodic',
                    'interval_seconds': avg_interval,
                    'details': f'Failures occur every ~{avg_interval:.0f}s'
                }

        # Check for clustering
        # ... additional analysis

        return {'pattern': 'irregular', 'details': 'No clear pattern'}
```

## 阶段 5：解决

**目标**：修复问题并防止复发。

### 修复分类

```python
class FixRegistry:
    """Registry of known fixes for common problems."""

    FIXES = {
        'OOM': {
            'immediate': [
                ('Reduce batch size', 'batch_size //= 2'),
                ('Enable gradient checkpointing', 'model.gradient_checkpointing_enable()'),
                ('Use mixed precision', 'scaler = GradScaler(); autocast()'),
            ],
            'structural': [
                ('Implement ZeRO-3', 'Use FSDP or DeepSpeed ZeRO-3'),
                ('Add activation offloading', 'Offload activations to CPU'),
                ('Model parallelism', 'Split model across GPUs'),
            ]
        },
        'TIMEOUT': {
            'immediate': [
                ('Increase timeout', 'NCCL_TIMEOUT=1800'),
                ('Add barrier before collective', 'dist.barrier()'),
            ],
            'structural': [
                ('Balance workload', 'Ensure equal batch distribution'),
                ('Add straggler detection', 'Monitor per-rank timing'),
                ('Async data loading', 'Use multiple workers with prefetch'),
            ]
        },
        'NUMERICAL': {
            'immediate': [
                ('Reduce learning rate', 'lr *= 0.1'),
                ('Add gradient clipping', 'clip_grad_norm_(model.parameters(), 1.0)'),
                ('Use dynamic loss scaling', 'GradScaler with backoff'),
            ],
            'structural': [
                ('Review initialization', 'Use appropriate init for depth'),
                ('Add layer normalization', 'Normalize between layers'),
                ('Use stable attention', 'Implement attention with numerical guards'),
            ]
        },
        'PERFORMANCE': {
            'immediate': [
                ('Enable overlap', 'bucket_cap_mb=25 for DDP'),
                ('Check thermal throttling', 'Monitor GPU temperature'),
            ],
            'structural': [
                ('Profile and optimize', 'Use Nsight to find bottlenecks'),
                ('Tune bucket size', 'Experiment with different sizes'),
                ('Enable TF32/FP16', 'Use mixed precision training'),
            ]
        }
    }

    @classmethod
    def get_fixes(cls, problem_category: str) -> dict:
        return cls.FIXES.get(problem_category, {})

    @classmethod
    def apply_fix(cls, fix_name: str, context: dict):
        """Apply a known fix."""
        # Implementation depends on fix
        pass
```

### 验证

```python
class FixVerifier:
    """Verify that fixes actually resolve the problem."""

    def __init__(self):
        self.baseline_metrics = None
        self.post_fix_metrics = None

    def capture_baseline(self, train_fn, num_steps: int = 10):
        """Capture metrics before fix."""
        self.baseline_metrics = self._run_and_measure(train_fn, num_steps)

    def verify_fix(self, train_fn, num_steps: int = 10) -> dict:
        """Verify fix by comparing to baseline."""
        self.post_fix_metrics = self._run_and_measure(train_fn, num_steps)

        comparison = {
            'problem_resolved': self._check_resolution(),
            'performance_impact': self._compare_performance(),
            'side_effects': self._check_side_effects(),
        }

        return comparison

    def _run_and_measure(self, train_fn, num_steps):
        metrics = {
            'success': True,
            'step_times': [],
            'memory_peaks': [],
            'losses': [],
        }

        try:
            for i in range(num_steps):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()

                loss = train_fn()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak_mem = torch.cuda.max_memory_allocated()

                metrics['step_times'].append(elapsed)
                metrics['memory_peaks'].append(peak_mem)
                metrics['losses'].append(loss)

        except Exception as e:
            metrics['success'] = False
            metrics['error'] = str(e)

        return metrics

    def _check_resolution(self) -> bool:
        """Check if the original problem is resolved."""
        if not self.baseline_metrics['success'] and self.post_fix_metrics['success']:
            return True

        # Additional checks based on problem type
        return self.post_fix_metrics['success']

    def _compare_performance(self) -> dict:
        """Compare performance before and after."""
        if not (self.baseline_metrics['success'] and self.post_fix_metrics['success']):
            return {'comparable': False}

        baseline_avg = sum(self.baseline_metrics['step_times']) / len(self.baseline_metrics['step_times'])
        postfix_avg = sum(self.post_fix_metrics['step_times']) / len(self.post_fix_metrics['step_times'])

        return {
            'comparable': True,
            'baseline_step_time': baseline_avg,
            'postfix_step_time': postfix_avg,
            'change_percent': (postfix_avg - baseline_avg) / baseline_avg * 100
        }

    def _check_side_effects(self) -> list:
        """Check for unexpected side effects of the fix."""
        effects = []

        # Memory increase
        if self.post_fix_metrics['success'] and self.baseline_metrics['success']:
            baseline_mem = sum(self.baseline_metrics['memory_peaks']) / len(self.baseline_metrics['memory_peaks'])
            postfix_mem = sum(self.post_fix_metrics['memory_peaks']) / len(self.post_fix_metrics['memory_peaks'])

            if postfix_mem > baseline_mem * 1.1:
                effects.append(f"Memory increased by {(postfix_mem/baseline_mem - 1)*100:.1f}%")

        return effects
```

### 预防

```python
class PreventionChecklist:
    """Checklist to prevent common issues."""

    CHECKLISTS = {
        'pre_training': [
            "Calculate memory budget: params + grads + optimizer + activations",
            "Test at small scale first (1-2 GPUs)",
            "Verify data loading doesn't bottleneck",
            "Check network bandwidth between nodes",
            "Ensure checkpointing works before long run",
            "Set appropriate NCCL timeout",
            "Enable NCCL debug logging initially",
        ],
        'scaling_up': [
            "Adjust learning rate for new batch size",
            "Extend warmup proportionally",
            "Monitor for stragglers at new scale",
            "Profile communication overhead",
            "Verify reproducibility at small scale first",
        ],
        'long_runs': [
            "Enable automatic checkpointing",
            "Set up monitoring and alerting",
            "Configure automatic restart on failure",
            "Plan checkpoint storage capacity",
            "Enable gradient/loss monitoring",
        ]
    }

    @classmethod
    def get_checklist(cls, phase: str) -> list:
        return cls.CHECKLISTS.get(phase, [])

    @classmethod
    def run_checks(cls, phase: str) -> dict:
        """Run automated checks from checklist."""
        checklist = cls.get_checklist(phase)
        results = {}

        for item in checklist:
            # Some items can be automated
            if 'memory budget' in item.lower():
                results[item] = cls._check_memory_budget()
            elif 'network bandwidth' in item.lower():
                results[item] = cls._check_network()
            else:
                results[item] = 'manual_check_required'

        return results

    @classmethod
    def _check_memory_budget(cls) -> str:
        available = torch.cuda.get_device_properties(0).total_memory
        return f"Available: {available/1e9:.1f}GB"

    @classmethod
    def _check_network(cls) -> str:
        # Simplified network check
        return "Run bandwidth test with NCCL_DEBUG=INFO"
```

## 快速参考：决策树

### 崩溃决策树

```
TRAINING CRASHED
      │
      ▼
Check error message
      │
      ├── "CUDA out of memory" ──────────────────────────────────┐
      │                                                          │
      ├── "NCCL" / "Timeout" ────────────────────────────────────┤
      │                                                          │
      ├── "NaN" / "Inf" ─────────────────────────────────────────┤
      │                                                          │
      └── Other ─────────────────────────────────────────────────┤
                                                                 │
                                                                 ▼
                                               ┌─────────────────────────────┐
      ┌────────────────────────────────────────│     DIAGNOSTIC ACTION       │
      │                                        └─────────────────────────────┘
      │
      │  OOM:
      │  1. Check peak memory: torch.cuda.max_memory_allocated()
      │  2. Profile memory breakdown
      │  3. Calculate theoretical requirement
      │  4. Reduce batch or enable checkpointing
      │
      │  NCCL/Timeout:
      │  1. Check which ranks failed: NCCL_DEBUG=INFO
      │  2. Test network: ping, iperf
      │  3. Check for stragglers
      │  4. Increase timeout or fix slow rank
      │
      │  Numerical:
      │  1. Find when NaN first appeared
      │  2. Check gradient norms before NaN
      │  3. Reduce LR or add gradient clipping
      │  4. Check loss scaling settings
      │
      └──────────────────────────────────────────────────────────────────────
```

### 性能决策树

```
SLOW TRAINING
      │
      ▼
Profile step breakdown
      │
      ├── Forward slow ─────────► Check GPU utilization
      │                                   │
      │                          ├── Low ──► Data loading bottleneck
      │                          └── High ─► Memory bandwidth limit
      │
      ├── Backward slow ────────► Check for extra synchronization
      │                                   │
      │                          ├── Yes ──► Remove unnecessary syncs
      │                          └── No ───► Check gradient computation
      │
      ├── Communication slow ───► Check overlap efficiency
      │                                   │
      │                          ├── No overlap ──► Enable bucketing
      │                          └── Has overlap ──► Check bandwidth
      │
      └── Optimizer slow ───────► Check optimizer state size
                                          │
                                 ├── Large ──► Consider ZeRO-1
                                 └── Small ──► Check for Python overhead
```

## 完整案例复盘：凌晨三点的吞吐坍塌

本节端到端地复盘一次真实的事故，展示在规模之上调试分布式训练时的混乱现实。这个场景是合成的——由常见的故障模式拼装而成——但每一个元素都来自真实的生产经验。

### 事故报告

**系统**：70B 稠密 transformer，256 块 H100（32 节点 × 8 GPU），TP=8，DP=32，ZeRO-1。

**症状**：在为期 21 天的训练的第 9 天凌晨 02:47，吞吐从 142K tokens/s 跌至 89K tokens/s（37% 的退化）。损失曲线继续下降——模型仍在训练，只是变慢了。

**告警**：监控系统因 MFU 从 41% 跌至 26% 而触发。

### 阶段 1：分诊（5 分钟）

**第一步**：检查任务是否仍在推进。

```
[02:52] Loss at step 48,201: 2.134 (still decreasing)
[02:52] Throughput: 89,412 tokens/s (was 142,000)
[02:52] Per-step time: 11.2s (was 7.0s)
[02:52] No OOM errors in any rank's stderr
```

**分类**：性能退化（不是崩溃，不是发散）。优先级：中——训练仍在继续，但浪费了约 37% 的 GPU 小时。

**初始假设清单**：

1. 另一个任务造成的网络争用
2. 硬件退化（热降频、坏网卡）
3. 数据流水线停顿
4. 软件变更（不太可能——夜间没有部署）

### 阶段 2：定位（15 分钟）

**逐 rank 时间分解**（从不同节点采样 4 个 rank）：

| Rank | 节点 | 计算（ms） | AllReduce（ms） | 合计（ms） |
|------|------|-------------|----------------|-----------|
| 0 | node-00 | 4,100 | 2,900 | 7,000 |
| 8 | node-01 | 4,100 | 2,900 | 7,000 |
| 72 | node-09 | 4,100 | 7,100 | 11,200 |
| 200 | node-25 | 4,100 | 7,100 | 11,200 |

**关键观察**：所有 rank 的计算时间完全相同（4.1s）。某些节点的 AllReduce 时间为 2.9s，另一些为 7.1s。步时间由最慢的 AllReduce 决定（BSP 同步）。

**定位到**：网络通信。并非所有节点都同等受影响——排除了全局争用。

### 阶段 3：隔离（20 分钟）

**在受影响与未受影响的节点上运行 `nccl-tests`**：

```
# Unaffected nodes (node-00 to node-07):
all_reduce_perf -b 1G -e 1G -g 8: busbw = 42.3 GB/s

# Affected nodes (node-09, node-25, others):
all_reduce_perf -b 1G -e 1G -g 8: busbw = 17.1 GB/s
```

受影响节点在跨节点 AllReduce 上看到**带宽下降 59%**。两组节点的节点内（NVLink）带宽都正常。

**进一步隔离**：在特定节点对之间运行点对点带宽测试。慢路径总是经过某个特定的骨干交换机（switch-04）。

### 阶段 4：根因（10 分钟）

**switch-04 诊断**（通过网络监控）：

```
switch-04 port 17: CRC errors 14,302 (rising)
switch-04 port 17: link flaps: 3 in last hour
switch-04 temperature: 72°C (normal)
```

**根因**：switch-04 端口 17 上的一个正在退化的光模块。CRC 错误迫使数据包重传，降低了有效带宽。该链路尚未完全失效（网络监控没有告警），但错误率足以让受影响路径上的吞吐减半。

**五个为什么**：

1. 为什么吞吐低？→ 某些节点上的 AllReduce 很慢
2. 为什么 AllReduce 慢？→ 跨节点带宽退化
3. 为什么带宽退化？→ 数据包正在被重传
4. 为什么数据包被重传？→ switch-04 端口 17 上的 CRC 错误
5. 为什么会有 CRC 错误？→ 光模块正在退化

### 阶段 5：解决（30 分钟）

**即时修复**（02:47 + 50 分钟 = 03:37）：

1. 在下一个检查点暂停训练（step 48,250）
2. 请求网络团队禁用 switch-04 端口 17 并重新路由
3. 网络经冗余路径重新路由（胖树提供此能力）
4. 从检查点恢复训练

**结果**：吞吐恢复到 139K tokens/s（基线的 98%——更长路径带来少量开销）。

**预防**：

- 把 NCCL 总线带宽纳入逐步监控（若 busbw 相比基线下降 >20% 则告警）
- 网络团队在 CRC 错误率 > 1000/小时时启用主动光模块更换
- 把 `nccl-tests` 加入周期性健康检查（每 6 小时）

### 经验教训

| 阶段 | 耗时 | 关键动作 |
|-------|-----------|------------|
| 分诊 | 5 分钟 | 归类为性能退化，而非崩溃 |
| 定位 | 15 分钟 | 逐 rank 时序揭示出不对称的 AllReduce |
| 隔离 | 20 分钟 | 在受影响与未受影响节点上运行 `nccl-tests` |
| 根因 | 10 分钟 | 交换机诊断 → 光模块 |
| 解决 | 30 分钟 | 重新路由，从检查点恢复 |
| **合计** | **80 分钟** | **损失：约 80 分钟的低吞吐 + 30 分钟暂停** |

**事故成本**：256 块 H100 上 80 分钟 37% 的浪费 + 30 分钟停机 ≈ 浪费 360 GPU 小时（按 4 美元/小时计为 1,440 美元）。

**若未检测到的成本**：以 89K 而非 142K tokens/s 跑完剩余 12 天，将浪费约 8.6 万美元的 GPU 小时。早期检测挽回了约 8.5 万美元。

!!! tip "元教训"
    调查协议之所以有效，是因为它*系统性地缩小了搜索空间*。凌晨三点时的诱惑是重启一切，指望它自己修好。而协议在 50 分钟内就找到了那个光模块，而不是靠盲目重启浪费一整天的功夫。

## 练习题

1. **构建分诊系统**：为你的训练基础设施实现一套完整的分诊系统。它应把故障归类为 OOM、网络、数值及其他。通过人为制造每种故障类型来测试它。

??? success "解答"
    **练习 1：分诊系统**

    ```python
    import re
    import traceback
    from enum import Enum
    from dataclasses import dataclass
    from typing import Optional, List, Dict
    import torch.distributed as dist

    class FailureCategory(Enum):
        OOM = "out_of_memory"
        NETWORK = "network"
        NUMERICAL = "numerical"
        HARDWARE = "hardware"
        SOFTWARE = "software"
        UNKNOWN = "unknown"

    @dataclass
    class TriageResult:
        category: FailureCategory
        confidence: float  # 0.0 to 1.0
        details: str
        suggested_actions: List[str]
        affected_ranks: List[int]

    class TriageSystem:
        """Automatic failure categorization for distributed training."""

        OOM_PATTERNS = [
            r"CUDA out of memory",
            r"RuntimeError: CUDA error: out of memory",
            r"torch.cuda.OutOfMemoryError",
            r"Tried to allocate .* GiB",
            r"OOM",
        ]

        NETWORK_PATTERNS = [
            r"NCCL error",
            r"Connection refused",
            r"Timeout",
            r"Socket",
            r"ETIMEDOUT",
            r"Connection reset",
            r"NCCL WARN",
        ]

        NUMERICAL_PATTERNS = [
            r"NaN",
            r"Inf",
            r"overflow",
            r"underflow",
            r"Loss is nan",
            r"Gradient overflow",
        ]

        HARDWARE_PATTERNS = [
            r"ECC error",
            r"GPU has fallen off the bus",
            r"Xid",
            r"hardware exception",
            r"NVLink error",
        ]

        def triage(self, error: Exception, logs: str = "") -> TriageResult:
            """Categorize a failure based on exception and logs."""
            error_str = str(error) + "\n" + traceback.format_exc() + "\n" + logs

            # Check patterns in order of specificity
            checks = [
                (self.OOM_PATTERNS, FailureCategory.OOM, self._oom_actions),
                (self.HARDWARE_PATTERNS, FailureCategory.HARDWARE, self._hardware_actions),
                (self.NETWORK_PATTERNS, FailureCategory.NETWORK, self._network_actions),
                (self.NUMERICAL_PATTERNS, FailureCategory.NUMERICAL, self._numerical_actions),
            ]

            for patterns, category, action_fn in checks:
                matches = self._count_matches(error_str, patterns)
                if matches > 0:
                    confidence = min(0.5 + 0.1 * matches, 0.95)
                    return TriageResult(
                        category=category,
                        confidence=confidence,
                        details=self._extract_details(error_str, patterns),
                        suggested_actions=action_fn(),
                        affected_ranks=self._find_affected_ranks(error_str),
                    )

            return TriageResult(
                category=FailureCategory.UNKNOWN,
                confidence=0.3,
                details=str(error)[:500],
                suggested_actions=["Collect full logs", "Check system health", "Review recent changes"],
                affected_ranks=[],
            )

        def _count_matches(self, text: str, patterns: List[str]) -> int:
            return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

        def _extract_details(self, text: str, patterns: List[str]) -> str:
            for pattern in patterns:
                match = re.search(f".*{pattern}.*", text, re.IGNORECASE)
                if match:
                    return match.group(0)[:200]
            return ""

        def _find_affected_ranks(self, text: str) -> List[int]:
            ranks = set()
            for match in re.finditer(r"rank[:\s]+(\d+)", text, re.IGNORECASE):
                ranks.add(int(match.group(1)))
            return sorted(ranks)

        def _oom_actions(self) -> List[str]:
            return [
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Increase tensor parallelism",
                "Enable ZeRO-3 if using ZeRO-1/2",
                "Check for memory leaks in custom code",
            ]

        def _network_actions(self) -> List[str]:
            return [
                "Check network connectivity between nodes",
                "Verify NCCL_IB_DISABLE setting",
                "Increase NCCL timeout (NCCL_TIMEOUT)",
                "Check for stragglers causing timeouts",
                "Verify firewall rules",
            ]

        def _numerical_actions(self) -> List[str]:
            return [
                "Reduce learning rate",
                "Enable loss scaling if using FP16",
                "Check for data corruption",
                "Add gradient clipping",
                "Verify model initialization",
            ]

        def _hardware_actions(self) -> List[str]:
            return [
                "Run nvidia-smi to check GPU health",
                "Check ECC error counts",
                "Verify GPU temperatures",
                "Consider excluding problematic GPU",
                "Contact infrastructure team",
            ]

    # Test harness to induce failures
    def test_triage_system():
        triage = TriageSystem()

        # Test OOM
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
        except Exception as e:
            result = triage.triage(e)
            assert result.category == FailureCategory.OOM
            print(f"✓ OOM detected: {result.confidence:.0%} confidence")

        # Test Network
        try:
            raise RuntimeError("NCCL error: unhandled system error, ETIMEDOUT")
        except Exception as e:
            result = triage.triage(e)
            assert result.category == FailureCategory.NETWORK
            print(f"✓ Network error detected: {result.confidence:.0%} confidence")

        # Test Numerical
        try:
            raise ValueError("Loss is nan at step 1000")
        except Exception as e:
            result = triage.triage(e)
            assert result.category == FailureCategory.NUMERICAL
            print(f"✓ Numerical error detected: {result.confidence:.0%} confidence")

        print("\nAll triage tests passed!")
    ```

2. **二分调试**：一个模型在 64 块 GPU 上失败，但在 8 块上正常。用二分法找出它开始失败的确切规模。记录在那个规模上发生了什么变化。

??? success "解答"
    **练习 2：二分调试**

    ```python
    from dataclasses import dataclass
    from typing import Callable, List, Tuple
    import subprocess

    @dataclass
    class BisectionResult:
        breaking_point: int
        last_working: int
        observations: List[str]

    def bisection_debug(
        test_fn: Callable[[int], bool],
        min_scale: int = 8,
        max_scale: int = 64,
    ) -> BisectionResult:
        """
        Find the exact scale where training breaks.

        Args:
            test_fn: Function that returns True if training works at given scale
            min_scale: Minimum GPU count (known to work)
            max_scale: Maximum GPU count (known to fail)

        Returns:
            BisectionResult with breaking point and observations
        """
        observations = []
        low, high = min_scale, max_scale

        # Verify boundaries
        assert test_fn(low), f"Test should pass at min_scale={low}"
        assert not test_fn(high), f"Test should fail at max_scale={high}"

        observations.append(f"Confirmed: works at {low}, fails at {high}")

        while high - low > 1:
            mid = (low + high) // 2
            # Round to power of 2 if close
            for p2 in [8, 16, 32, 64, 128]:
                if abs(mid - p2) <= 2:
                    mid = p2
                    break

            print(f"Testing scale: {mid} GPUs...")
            if test_fn(mid):
                observations.append(f"✓ Scale {mid}: PASS")
                low = mid
            else:
                observations.append(f"✗ Scale {mid}: FAIL")
                high = mid

        observations.append(f"Breaking point: {low} → {high}")
        return BisectionResult(
            breaking_point=high,
            last_working=low,
            observations=observations,
        )

    def analyze_scale_change(working: int, breaking: int) -> List[str]:
        """Analyze what changes at the breaking scale."""
        changes = []

        # Network topology changes
        if working <= 8 and breaking > 8:
            changes.append("Crossed single-node boundary (NVLink → network)")

        # NCCL algorithm changes
        if working < 16 and breaking >= 16:
            changes.append("NCCL may switch from tree to ring algorithm")

        # Memory per GPU
        changes.append(f"Per-GPU batch size: {1024//breaking} vs {1024//working}")

        # Communication volume
        changes.append(f"AllReduce participants: {working} → {breaking}")
        changes.append(f"Ring allreduce steps: {working-1} → {breaking-1}")

        # Synchronization
        changes.append(f"Barrier sync time increases with more GPUs")

        return changes

    # Example usage
    """
    def run_training_test(num_gpus: int) -> bool:
        result = subprocess.run(
            ["torchrun", f"--nproc_per_node={num_gpus}", "train.py", "--test"],
            capture_output=True,
            timeout=300,
        )
        return result.returncode == 0

    result = bisection_debug(run_training_test, min_scale=8, max_scale=64)
    print(f"Breaking point: {result.breaking_point} GPUs")
    for obs in result.observations:
        print(f"  {obs}")

    changes = analyze_scale_change(result.last_working, result.breaking_point)
    print("\nChanges at breaking point:")
    for change in changes:
        print(f"  - {change}")
    """
    ```

    **从 8 → 64 GPU 的典型发现：**

    | 规模 | 变化 | 影响 |
    |-------|--------|--------|
    | 8 → 16 | 跨越节点边界 | NVLink → InfiniBand |
    | 16 → 32 | AllReduce 延迟 | 参与者翻倍 |
    | 32 → 64 | 内存压力 | 每 GPU 批更小 |

3. **五个为什么练习**：取一次最近的训练失败，应用五个为什么分析。至少深入五层。你找到的根因是什么？

??? success "解答"
    **练习 3：五个为什么练习**

    ```markdown
    # Five Whys Analysis Template

    ## Incident: Training NaN at step 10,000

    **Why 1: Why did training produce NaN?**
    → Loss exploded to infinity before becoming NaN

    **Why 2: Why did loss explode?**
    → Gradients grew exponentially over ~50 steps

    **Why 3: Why did gradients grow exponentially?**
    → Learning rate warmup completed, full LR was too high for model state

    **Why 4: Why was the learning rate too high?**
    → LR was copied from a paper using different batch size without adjustment

    **Why 5: Why wasn't LR adjusted for batch size?**
    → No documented procedure for scaling hyperparameters with batch size

    ## Root Cause
    Missing documentation/checklist for adapting hyperparameters when
    changing training configuration.

    ## Corrective Actions
    1. Immediate: Reduce LR by sqrt(batch_ratio)
    2. Short-term: Add LR scaling formula to training checklist
    3. Long-term: Implement automatic LR scaling based on batch size
    ```

    ```python
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class WhyLevel:
        question: str
        answer: str
        evidence: Optional[str] = None

    class FiveWhysAnalysis:
        """Structured Five Whys analysis for training failures."""

        def __init__(self, incident: str):
            self.incident = incident
            self.levels: List[WhyLevel] = []

        def add_why(self, question: str, answer: str, evidence: str = None):
            self.levels.append(WhyLevel(question, answer, evidence))
            return self

        def get_root_cause(self) -> str:
            if len(self.levels) >= 5:
                return self.levels[-1].answer
            return "Analysis incomplete - need at least 5 levels"

        def generate_report(self) -> str:
            report = [f"# Five Whys Analysis\n", f"**Incident**: {self.incident}\n"]

            for i, level in enumerate(self.levels, 1):
                report.append(f"\n**Why {i}**: {level.question}")
                report.append(f"→ {level.answer}")
                if level.evidence:
                    report.append(f"  *Evidence*: {level.evidence}")

            if len(self.levels) >= 5:
                report.append(f"\n## Root Cause\n{self.get_root_cause()}")

            return "\n".join(report)

    # Example
    analysis = FiveWhysAnalysis("GPU 3 consistently 20% slower")
    analysis.add_why(
        "Why is GPU 3 slower?",
        "GPU 3 thermals hit 85°C, triggering throttling",
        "nvidia-smi shows temp at 85°C vs 72°C for others"
    ).add_why(
        "Why is GPU 3 hotter?",
        "Airflow to GPU 3 slot is restricted",
        "Thermal camera shows hot spot"
    ).add_why(
        "Why is airflow restricted?",
        "Adjacent cable bundle blocks intake",
        "Visual inspection confirmed"
    ).add_why(
        "Why is cable bundle there?",
        "Storage expansion installed without cable management",
        "Change log shows storage added 2 weeks ago"
    ).add_why(
        "Why wasn't cable management done?",
        "No thermal verification step in hardware change procedure",
        "Procedure document review"
    )

    print(analysis.generate_report())
    ```

4. **相关性分析**：对训练插桩记录事件（检查点保存、LR 变化、批大小变化）与失败。一周后分析相关性。哪些事件能预测失败？

??? success "解答"
    **练习 4：相关性分析**

    ```python
    import time
    import json
    from collections import defaultdict
    from dataclasses import dataclass, asdict
    from typing import List, Dict, Optional
    from datetime import datetime
    import numpy as np

    @dataclass
    class TrainingEvent:
        timestamp: float
        event_type: str  # "checkpoint", "lr_change", "batch_size_change", "failure"
        details: Dict
        step: int

    class EventLogger:
        """Log training events for correlation analysis."""

        def __init__(self, log_file: str = "training_events.jsonl"):
            self.log_file = log_file
            self.events: List[TrainingEvent] = []

        def log(self, event_type: str, details: Dict, step: int):
            event = TrainingEvent(
                timestamp=time.time(),
                event_type=event_type,
                details=details,
                step=step,
            )
            self.events.append(event)

            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")

        def log_checkpoint(self, step: int, path: str):
            self.log("checkpoint", {"path": path}, step)

        def log_lr_change(self, step: int, old_lr: float, new_lr: float):
            self.log("lr_change", {"old": old_lr, "new": new_lr}, step)

        def log_failure(self, step: int, error: str, category: str):
            self.log("failure", {"error": error, "category": category}, step)

    class CorrelationAnalyzer:
        """Analyze correlations between events and failures."""

        def __init__(self, events: List[TrainingEvent]):
            self.events = sorted(events, key=lambda e: e.timestamp)
            self.failures = [e for e in events if e.event_type == "failure"]

        def analyze_precursors(self, window_seconds: float = 300) -> Dict[str, float]:
            """Find events that commonly precede failures."""
            precursor_counts = defaultdict(int)
            total_failures = len(self.failures)

            for failure in self.failures:
                # Look back in time window
                for event in self.events:
                    if event.event_type == "failure":
                        continue
                    time_diff = failure.timestamp - event.timestamp
                    if 0 < time_diff <= window_seconds:
                        precursor_counts[event.event_type] += 1

            # Normalize by failure count
            return {
                event_type: count / total_failures
                for event_type, count in precursor_counts.items()
            }

        def calculate_risk_ratios(self) -> Dict[str, float]:
            """
            Calculate risk ratio: P(failure | event) / P(failure | no event)
            """
            # Time windows after each event type
            WINDOW = 600  # 10 minutes

            event_types = set(e.event_type for e in self.events) - {"failure"}
            risk_ratios = {}

            total_time = self.events[-1].timestamp - self.events[0].timestamp
            base_failure_rate = len(self.failures) / total_time

            for event_type in event_types:
                events_of_type = [e for e in self.events if e.event_type == event_type]

                failures_after_event = 0
                total_window_time = 0

                for event in events_of_type:
                    window_end = event.timestamp + WINDOW
                    total_window_time += WINDOW

                    for failure in self.failures:
                        if event.timestamp < failure.timestamp <= window_end:
                            failures_after_event += 1

                if total_window_time > 0:
                    failure_rate_after = failures_after_event / total_window_time
                    risk_ratios[event_type] = failure_rate_after / base_failure_rate

            return risk_ratios

        def generate_report(self) -> str:
            precursors = self.analyze_precursors()
            risk_ratios = self.calculate_risk_ratios()

            report = ["# Event-Failure Correlation Analysis\n"]
            report.append(f"Total events: {len(self.events)}")
            report.append(f"Total failures: {len(self.failures)}\n")

            report.append("## Precursor Analysis (5-min window)")
            report.append("| Event Type | Preceded Failures |")
            report.append("|------------|-------------------|")
            for event_type, rate in sorted(precursors.items(), key=lambda x: -x[1]):
                report.append(f"| {event_type} | {rate:.0%} |")

            report.append("\n## Risk Ratios")
            report.append("| Event Type | Risk Ratio |")
            report.append("|------------|------------|")
            for event_type, ratio in sorted(risk_ratios.items(), key=lambda x: -x[1]):
                flag = "⚠️" if ratio > 2.0 else ""
                report.append(f"| {event_type} | {ratio:.2f}x {flag} |")

            return "\n".join(report)
    ```

    **一周后的示例输出：**

    | 事件类型 | 风险比 | 解读 |
    |------------|------------|----------------|
    | checkpoint | 3.2x ⚠️ | 检查点可能导致 OOM 压力 |
    | lr_change | 2.1x ⚠️ | LR 提高会使训练失稳 |
    | batch_size_change | 1.8x | 批变化给内存带来压力 |
    | eval_start | 0.9x | 评估是安全的 |

5. **预防清单**：为你的具体模型与基础设施创建一份定制的预防清单。在下一次大规模训练之前逐项检查。

??? success "解答"
    **练习 5：预防清单**

    ```python
    from dataclasses import dataclass
    from typing import List, Callable, Optional
    from enum import Enum

    class CheckResult(Enum):
        PASS = "✓"
        FAIL = "✗"
        WARN = "⚠"
        SKIP = "○"

    @dataclass
    class CheckItem:
        name: str
        description: str
        check_fn: Optional[Callable[[], CheckResult]] = None
        category: str = "general"
        critical: bool = False

    class PreventionChecklist:
        """Custom prevention checklist for training runs."""

        def __init__(self, model_name: str):
            self.model_name = model_name
            self.items: List[CheckItem] = []
            self.results: dict = {}

        def add_check(self, name: str, description: str,
                      check_fn: Callable = None, category: str = "general",
                      critical: bool = False):
            self.items.append(CheckItem(name, description, check_fn, category, critical))
            return self

        def run_all(self) -> bool:
            """Run all checks, return True if no critical failures."""
            all_passed = True

            for item in self.items:
                if item.check_fn:
                    try:
                        result = item.check_fn()
                    except Exception as e:
                        result = CheckResult.FAIL
                        print(f"Check '{item.name}' raised: {e}")
                else:
                    result = CheckResult.SKIP

                self.results[item.name] = result

                if result == CheckResult.FAIL and item.critical:
                    all_passed = False

            return all_passed

        def generate_report(self) -> str:
            categories = {}
            for item in self.items:
                if item.category not in categories:
                    categories[item.category] = []
                categories[item.category].append(item)

            report = [f"# Pre-Training Checklist: {self.model_name}\n"]

            for category, items in categories.items():
                report.append(f"\n## {category.title()}")
                for item in items:
                    result = self.results.get(item.name, CheckResult.SKIP)
                    critical = " [CRITICAL]" if item.critical else ""
                    report.append(f"- {result.value} {item.name}{critical}")
                    report.append(f"  {item.description}")

            return "\n".join(report)

    # Example checklist for LLaMA-style training
    def create_llama_checklist() -> PreventionChecklist:
        import torch

        checklist = PreventionChecklist("LLaMA-70B")

        # Memory checks
        def check_gpu_memory():
            free = torch.cuda.mem_get_info()[0] / 1e9
            return CheckResult.PASS if free > 70 else CheckResult.FAIL

        checklist.add_check(
            "GPU Memory Available",
            "Verify >70GB free per GPU for 70B model",
            check_gpu_memory, "memory", critical=True
        )

        # Network checks
        checklist.add_check(
            "NCCL Version",
            "Verify NCCL >= 2.18 for optimal performance",
            category="network"
        )

        checklist.add_check(
            "InfiniBand Status",
            "Verify IB links are up: ibstat shows Active",
            category="network", critical=True
        )

        # Numerical checks
        checklist.add_check(
            "Loss Scaling Configured",
            "Verify dynamic loss scaling is enabled for FP16",
            category="numerical"
        )

        checklist.add_check(
            "Gradient Clipping",
            "Verify grad clip norm = 1.0",
            category="numerical"
        )

        # Checkpoint checks
        checklist.add_check(
            "Checkpoint Storage",
            "Verify checkpoint directory has >5TB free",
            category="checkpoint", critical=True
        )

        return checklist

    # Run checklist
    # checklist = create_llama_checklist()
    # if not checklist.run_all():
    #     print("CRITICAL CHECKS FAILED - DO NOT PROCEED")
    # print(checklist.generate_report())
    ```

6. **修复验证**：对一个已知问题应用修复。使用 FixVerifier 确认问题已解决且未引入回归。

??? success "解答"
    **练习 6：修复验证**

    ```python
    import time
    import statistics
    from dataclasses import dataclass
    from typing import List, Dict, Callable, Optional
    from enum import Enum

    class VerificationStatus(Enum):
        FIXED = "fixed"
        NOT_FIXED = "not_fixed"
        REGRESSION = "regression"
        INCONCLUSIVE = "inconclusive"

    @dataclass
    class Metric:
        name: str
        value: float
        baseline: float
        tolerance: float  # Acceptable deviation from baseline

        @property
        def is_regression(self) -> bool:
            # Assuming higher is better for throughput, lower for loss
            if "throughput" in self.name.lower() or "speed" in self.name.lower():
                return self.value < self.baseline * (1 - self.tolerance)
            else:
                return self.value > self.baseline * (1 + self.tolerance)

    @dataclass
    class VerificationResult:
        status: VerificationStatus
        problem_resolved: bool
        regressions: List[str]
        metrics: Dict[str, Metric]
        notes: str

    class FixVerifier:
        """Verify that a fix resolves the problem without regressions."""

        def __init__(self, baseline_metrics: Dict[str, float]):
            self.baseline = baseline_metrics
            self.tolerance = 0.05  # 5% tolerance

        def verify(
            self,
            problem_test: Callable[[], bool],  # Returns True if problem exists
            metric_collector: Callable[[], Dict[str, float]],
            num_trials: int = 3,
        ) -> VerificationResult:
            """
            Verify a fix by:
            1. Checking if the original problem is resolved
            2. Collecting metrics to check for regressions
            3. Running multiple trials for statistical significance
            """
            # Test if problem is resolved
            problem_trials = [problem_test() for _ in range(num_trials)]
            problem_resolved = not any(problem_trials)

            # Collect metrics across trials
            all_metrics = [metric_collector() for _ in range(num_trials)]

            # Aggregate metrics
            metrics = {}
            regressions = []

            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                avg_value = statistics.mean(values)
                baseline_value = self.baseline.get(metric_name, avg_value)

                metric = Metric(
                    name=metric_name,
                    value=avg_value,
                    baseline=baseline_value,
                    tolerance=self.tolerance,
                )
                metrics[metric_name] = metric

                if metric.is_regression:
                    regressions.append(
                        f"{metric_name}: {baseline_value:.2f} → {avg_value:.2f}"
                    )

            # Determine overall status
            if not problem_resolved:
                status = VerificationStatus.NOT_FIXED
            elif regressions:
                status = VerificationStatus.REGRESSION
            else:
                status = VerificationStatus.FIXED

            return VerificationResult(
                status=status,
                problem_resolved=problem_resolved,
                regressions=regressions,
                metrics=metrics,
                notes=f"Ran {num_trials} trials",
            )

        def generate_report(self, result: VerificationResult) -> str:
            report = ["# Fix Verification Report\n"]

            status_emoji = {
                VerificationStatus.FIXED: "✅",
                VerificationStatus.NOT_FIXED: "❌",
                VerificationStatus.REGRESSION: "⚠️",
                VerificationStatus.INCONCLUSIVE: "❓",
            }

            report.append(f"**Status**: {status_emoji[result.status]} {result.status.value.upper()}")
            report.append(f"**Problem Resolved**: {'Yes' if result.problem_resolved else 'No'}")

            if result.regressions:
                report.append("\n## Regressions Detected")
                for reg in result.regressions:
                    report.append(f"- ⚠️ {reg}")

            report.append("\n## Metrics Comparison")
            report.append("| Metric | Baseline | Current | Status |")
            report.append("|--------|----------|---------|--------|")

            for name, metric in result.metrics.items():
                status = "🔴" if metric.is_regression else "🟢"
                report.append(f"| {name} | {metric.baseline:.2f} | {metric.value:.2f} | {status} |")

            return "\n".join(report)

    # Example usage
    """
    # Baseline before attempting fix
    baseline = {
        "throughput_tokens_sec": 150000,
        "memory_gb": 72,
        "step_time_ms": 450,
    }

    verifier = FixVerifier(baseline)

    def check_oom_problem():
        # Returns True if OOM still occurs
        try:
            run_training_step()
            return False
        except RuntimeError as e:
            return "out of memory" in str(e).lower()

    def collect_metrics():
        return {
            "throughput_tokens_sec": measure_throughput(),
            "memory_gb": get_peak_memory(),
            "step_time_ms": measure_step_time(),
        }

    result = verifier.verify(check_oom_problem, collect_metrics)
    print(verifier.generate_report(result))
    """
    ```

    **验证工作流小结：**

    | 步骤 | 动作 | 目的 |
    |------|--------|---------|
    | 1 | 记录基线指标 | 确立性能参照 |
    | 2 | 应用修复 | 实施提议的解决方案 |
    | 3 | 测试问题是否解决 | 确认原始问题已修复 |
    | 4 | 收集新指标 | 测量当前性能 |
    | 5 | 与基线比较 | 检测任何回归 |
    | 6 | 生成报告 | 记录验证结果 |

## 故障模式快速参考

上面的调查协议适用于任何故障。本节列出五种最常见的故障模式，方便你直接跳转到正确的不变量。

### 不变量分诊

每一次故障都违反三条不变量之一：

1. **内存**：OOM、分配器碎片化、激活值尖峰
2. **计算**：低 MFU、kernel 占用不足、空闲间隙
3. **通信**：暴露的集合通信、高延迟敏感性、掉队者

### 故障模式 1：内存耗尽（OOM）

**症状**：立即崩溃，或跑若干步后后期 OOM。

**可能原因**：

- 更大的批／序列导致的激活值增长
- 动态形状造成的碎片化
- 过多的 AllGather 缓冲区（ZeRO-3）

**快速检查**：

- 比较峰值已分配与已保留内存（`torch.cuda.memory_stats()`）
- 禁用动态形状；固定序列长度
- 启用激活检查点或减小批

### 故障模式 2：通信停滞

**症状**：NCCL／集合通信时间在步中占主导。

**可能原因**：

- 通信强度低（$I_{\text{net}}$ 低于脊点）
- 与计算缺少重叠
- 拓扑放置欠佳（TP 跨节点）

**快速检查**：

- 剖析时间线，查看暴露的 AllReduce
- 增大本地批或梯度累积步数
- 用 `nccl-tests` 验证网卡饱和与预期带宽

### 故障模式 3：流水线气泡

**症状**：流水线阶段出现周期性空闲间隙。

**可能原因**：

- 微批太少（$m < 4p$）
- 阶段分配不均衡（各阶段 FLOPs 不等）

**快速检查**：

- 计算气泡比例：$(p-1)/(m+p-1)$
- 增加微批，或跨阶段重新平衡层

### 故障模式 4：规模上的发散

**症状**：扩展 GPU 或批大小后损失爆炸。

**可能原因**：

- 学习率对新的有效批来说过高
- 梯度噪声降低，越过了 $B_{\text{crit}}$

**快速检查**：

- 降低 LR 或使用更长的预热
- 应用梯度裁剪
- 在 $B_{\text{crit}}$ 以上改用 sqrt LR 缩放（第 10 章）

### 故障模式 5：掉队者主导

**症状**：步时间等于最慢 rank；rank 之间方差很大。

**可能原因**：

- 某节点的 CPU 数据流水线瓶颈
- 专家路由不均衡（MoE 模型）
- 主机侧开销（日志、检查点）

**快速检查**：

- 测量逐 rank 步时间，找出离群者
- 检查数据加载器队列占用
- 添加专家负载均衡损失（第 18 章）

### 诊断查找表

| 症状 | 不变量 | 首个测试 | 典型修复 |
|---------|-----------|------------|-------------|
| OOM | 内存 | 峰值已分配 vs 已保留 | 检查点、ZeRO、卸载 |
| 低 MFU | 计算 | Kernel 占用 | 精度、融合、更好的批处理 |
| 暴露的 NCCL | 通信 | 时间线重叠 | 分桶、重叠、拓扑 |
| 气泡间隙 | 计算+通信 | 气泡比例 | 更多微批、重新平衡 |
| 发散 | 计算 | LR 敏感性扫描 | 预热、LR 缩放规则 |
| 掉队者 | 通信 | 逐 rank 步时间 | 数据流水线、负载均衡 |

!!! tip "经验法则"
    先把症状映射到不变量，再应用最小修复。一处恰当的改变（批大小、重叠、拓扑）往往能挽回大部分损失的性能。

## 关键要点

1. **系统化胜过临时应付**：五阶段协议确保不漏掉任何东西。

2. **快速分诊**：最初的五分钟决定了你的调查将有多高效。

3. **先隔离再修复**：不要猜——证明哪个组件在失败。

4. **找根因，而非症状**：五个为什么揭示真正的问题。

5. **验证修复**：引入新问题的修复不算修复。

6. **预防，而非仅仅反应**：清单与监控能预防已知问题。

7. **记录一切**：你今天的调查会帮到明天的下一个人。

8. **了解五种故障模式**：OOM、通信停滞、流水线气泡、发散与掉队者覆盖了绝大多数问题。