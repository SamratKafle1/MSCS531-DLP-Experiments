import time
import numpy as np

N = 5_000_000
X = np.random.rand(N).astype(np.float32)
Y = np.random.rand(N).astype(np.float32)
alpha = np.float32(1.25)

Y_base = Y.copy()
start = time.perf_counter()
for i in range(N):
    Y_base[i] = alpha * X[i] + Y_base[i]
t_base = time.perf_counter() - start

Y_unroll = Y.copy()
start = time.perf_counter()
i = 0
while i <= N - 4:
    Y_unroll[i]   = alpha * X[i]   + Y_unroll[i]
    Y_unroll[i+1] = alpha * X[i+1] + Y_unroll[i+1]
    Y_unroll[i+2] = alpha * X[i+2] + Y_unroll[i+2]
    Y_unroll[i+3] = alpha * X[i+3] + Y_unroll[i+3]
    i += 4
while i < N:
    Y_unroll[i] = alpha * X[i] + Y_unroll[i]
    i += 1
t_unroll = time.perf_counter() - start

print(f"N = {N:,}")
print(f"Baseline loop time: {t_base:.6f} s")
print(f"Unrolled loop time : {t_unroll:.6f} s")
print(f"Speedup            : {t_base / t_unroll:.2f}x")
print("Max error          :", float(np.max(np.abs(Y_base - Y_unroll))))
