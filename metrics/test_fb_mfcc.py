import numpy as np
import time

Xfft = np.random.rand(512).astype(np.float32)
fb = np.random.rand(512, 32).astype(np.float32)
Xb = np.empty(32, dtype=np.float32)

num_runs = 1000
times = []

for _ in range(num_runs):
    start_time = time.perf_counter()
    # Xb_prealloc = np.log(np.dot(Xfft, fb ) + 1.0)
    Xb = np.log1p(Xfft @ fb)
    end_time = time.perf_counter()
    times.append((end_time - start_time) * 1000)

print(f"Min Xb time: {min(times):.6f} ms")
print(f"Max Xb time: {max(times):.6f} ms")
print(f"Average Xb time: {sum(times) / num_runs:.6f} ms")