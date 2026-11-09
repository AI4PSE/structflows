import jax
import jax.numpy as jnp
import time
import numpy as np
from structflow.bayes.metrics import compute_swd_jax, compute_sinkhorn_jax

def benchmark_swd_vs_sinkhorn():
    print("Standardized Benchmarking SWD vs Sinkhorn...")
    print("Both use Root Mean Squared Distance normalized by sqrt(N*D).")
    key = jax.random.PRNGKey(0)
    
    dims = [2, 10, 50]
    samples = [100, 500, 1000]
    num_projections = 200
    epsilon = 0.1
    num_runs = 5

    print(f"{'N':>5} | {'D':>5} | {'SWD (ms)':>12} | {'Sinkhorn (ms)':>15} | {'Ratio':>10} | {'Abs Diff':>12} | {'Error %':>10}")
    print("-" * 100)

    for n in samples:
        for d in dims:
            k1, k2, k3 = jax.random.split(key, 3)
            X = jax.random.normal(k1, (n, d))
            Y = jax.random.normal(k2, (n, d))
            
            # Warmup and get values
            swd_val = compute_swd_jax(k3, X, Y, num_projections=num_projections).block_until_ready()
            sinkhorn_val = compute_sinkhorn_jax(X, Y, epsilon=epsilon).block_until_ready()
            
            # SWD Benchmark
            start = time.time()
            for _ in range(num_runs):
                _ = compute_swd_jax(k3, X, Y, num_projections=num_projections).block_until_ready()
            end = time.time()
            swd_time = (end - start) / num_runs * 1000
            
            # Sinkhorn Benchmark
            start = time.time()
            for _ in range(num_runs):
                _ = compute_sinkhorn_jax(X, Y, epsilon=epsilon).block_until_ready()
            end = time.time()
            sinkhorn_time = (end - start) / num_runs * 1000
            
            ratio = sinkhorn_time / swd_time
            abs_diff = jnp.abs(swd_val - sinkhorn_val)
            error_pct = (abs_diff / (sinkhorn_val + 1e-10)) * 100
            
            print(f"{n:5d} | {d:5d} | {swd_time:12.4f} | {sinkhorn_time:15.4f} | {ratio:10.1f}x | {abs_diff:12.6f} | {error_pct:9.2f}%")

if __name__ == "__main__":
    benchmark_swd_vs_sinkhorn()
