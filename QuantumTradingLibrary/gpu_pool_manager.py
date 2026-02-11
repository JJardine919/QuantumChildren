"""
GPU Pool Manager - Signal Farm Infrastructure
==============================================
Manages GPU resources for parallel prop farm simulation.
Distributes tensor workloads across available DirectML devices.

Usage:
    from gpu_pool_manager import GPUPool, get_device

    pool = GPUPool()
    device = pool.acquire()   # Get a GPU slot
    # ... do work on device ...
    pool.release(device)      # Return slot
"""

import torch
import threading
import time
from contextlib import contextmanager

# DirectML setup
try:
    import torch_directml
    _DML_AVAILABLE = True
    _DML_DEVICE = torch_directml.device()
    _DML_NAME = torch_directml.device_name(0)
except ImportError:
    _DML_AVAILABLE = False
    _DML_DEVICE = torch.device("cpu")
    _DML_NAME = "CPU (DirectML not available)"


def get_device():
    """Get the best available device. Always prefer DirectML GPU."""
    if _DML_AVAILABLE:
        return _DML_DEVICE
    return torch.device("cpu")


def get_device_name():
    """Get human-readable device name."""
    return _DML_NAME


class GPUPool:
    """
    Manages concurrent access to GPU for parallel simulation workers.

    Since we have 1 AMD RX 6800 XT (16GB VRAM), this pool manages
    concurrent tensor operations by limiting simultaneous GPU users
    to prevent OOM errors.

    For multi-GPU setups, extend this to distribute across devices.
    """

    def __init__(self, max_concurrent=8, vram_budget_mb=14000):
        self.device = get_device()
        self.device_name = get_device_name()
        self.max_concurrent = max_concurrent
        self.vram_budget_mb = vram_budget_mb

        # Semaphore controls concurrent GPU access
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._active_workers = 0
        self._total_allocations = 0
        self._peak_concurrent = 0

    def acquire(self, timeout=30):
        """Acquire a GPU slot. Blocks if pool is full."""
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"GPU pool full ({self.max_concurrent} slots). Waited {timeout}s.")

        with self._lock:
            self._active_workers += 1
            self._total_allocations += 1
            if self._active_workers > self._peak_concurrent:
                self._peak_concurrent = self._active_workers

        return self.device

    def release(self, device=None):
        """Release a GPU slot back to the pool."""
        with self._lock:
            self._active_workers = max(0, self._active_workers - 1)
        self._semaphore.release()

    @contextmanager
    def slot(self):
        """Context manager for GPU slot acquisition."""
        device = self.acquire()
        try:
            yield device
        finally:
            self.release(device)

    def status(self):
        """Get current pool status."""
        with self._lock:
            return {
                "device": self.device_name,
                "gpu_available": _DML_AVAILABLE,
                "max_concurrent": self.max_concurrent,
                "active_workers": self._active_workers,
                "available_slots": self.max_concurrent - self._active_workers,
                "total_allocations": self._total_allocations,
                "peak_concurrent": self._peak_concurrent,
                "vram_budget_mb": self.vram_budget_mb,
            }

    def __repr__(self):
        s = self.status()
        return (
            f"GPUPool({s['device']}, "
            f"active={s['active_workers']}/{s['max_concurrent']}, "
            f"peak={s['peak_concurrent']})"
        )


class BatchScheduler:
    """
    Schedules tensor batches across simulation accounts.

    Splits 100 accounts into batches that fit within GPU VRAM,
    runs them sequentially on GPU, and collects results.
    """

    def __init__(self, pool: GPUPool, batch_size=25):
        self.pool = pool
        self.batch_size = batch_size

    def schedule_evaluations(self, population, features_tensor, prices, evaluate_fn):
        """
        Evaluate a population of individuals against features.
        Batches the work to fit GPU memory.

        Args:
            population: list of TradingIndividual
            features_tensor: torch.Tensor on CPU (will be moved to GPU per batch)
            prices: numpy array of prices
            evaluate_fn: callable(individual, features_tensor, prices) -> (win_rate, trades)

        Returns:
            list of (win_rate, trades) tuples
        """
        results = []

        for batch_start in range(0, len(population), self.batch_size):
            batch = population[batch_start:batch_start + self.batch_size]

            with self.pool.slot() as device:
                # Move features to GPU for this batch
                gpu_features = features_tensor.to(device)

                for individual in batch:
                    wr, trades = evaluate_fn(individual, gpu_features, prices)
                    results.append((wr, trades))

                # Free GPU memory after batch
                del gpu_features
                if _DML_AVAILABLE:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    def parallel_evaluate(self, accounts, evaluate_fn, num_threads=4):
        """
        Evaluate multiple simulation accounts in parallel threads.
        Each thread acquires a GPU slot from the pool.

        Args:
            accounts: list of account simulation states
            evaluate_fn: callable(account, device) -> results
            num_threads: number of parallel threads

        Returns:
            list of results per account
        """
        results = [None] * len(accounts)
        errors = []

        def _worker(idx, account):
            try:
                with self.pool.slot() as device:
                    results[idx] = evaluate_fn(account, device)
            except Exception as e:
                errors.append((idx, str(e)))

        threads = []
        for i, account in enumerate(accounts):
            t = threading.Thread(target=_worker, args=(i, account), daemon=True)
            threads.append(t)

            # Limit concurrent thread launches
            if len(threads) >= num_threads:
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(timeout=120)
                threads = []

        # Launch remaining
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        if errors:
            print(f"[GPU POOL] {len(errors)} worker errors:")
            for idx, err in errors[:5]:
                print(f"  Account {idx}: {err}")

        return results


# Module-level singleton
_DEFAULT_POOL = None

def get_pool(max_concurrent=8):
    """Get or create the default GPU pool singleton."""
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = GPUPool(max_concurrent=max_concurrent)
    return _DEFAULT_POOL


if __name__ == "__main__":
    print("=" * 60)
    print("  GPU POOL MANAGER - Status Check")
    print("=" * 60)

    pool = GPUPool()
    print(f"\n  {pool}")
    status = pool.status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    # Quick benchmark
    print(f"\n  Running quick benchmark...")
    device = get_device()
    t0 = time.time()
    for _ in range(100):
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
    elapsed = time.time() - t0
    print(f"  100x matmul(1000x1000): {elapsed:.2f}s ({elapsed/100*1000:.1f}ms each)")

    # Test pool concurrency
    print(f"\n  Testing pool concurrency (3 workers)...")
    import concurrent.futures

    def gpu_work(worker_id):
        with pool.slot() as dev:
            x = torch.randn(500, 500, device=dev)
            y = torch.matmul(x, x)
            return f"Worker {worker_id}: done ({y.shape})"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(gpu_work, i) for i in range(6)]
        for f in concurrent.futures.as_completed(futures):
            print(f"    {f.result()}")

    print(f"\n  Final pool state: {pool}")
    print("=" * 60)
