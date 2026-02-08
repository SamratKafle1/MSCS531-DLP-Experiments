import time
import numpy as np

# Keep this moderate so the scalar loop finishes in reasonable time.
N = 3_000_000

A = np.random.rand(N).astype(np.float32)
B = np.random.rand(N).astype(np.float32)

# Scalar loop
start = time.perf_counter()
C = np.empty_like(A)
for i in range(N):
    C[i] = A[i] + B[i]
t_scalar = time.perf_counter() - start

# Vectorized (NumPy uses optimized native kernels, often SIMD)
start = time.perf_counter()
C2 = A + B
t_vec = time.perf_counter() - start

print(f"N = {N:,}")
print(f"Scalar loop time: {t_scalar:.6f} s")
print(f"Vectorized time : {t_vec:.6f} s")
print(f"Speedup         : {t_scalar / t_vec:.2f}x")
print("Max error       :", float(np.max(np.abs(C - C2))))
