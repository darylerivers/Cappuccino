#!/usr/bin/env python3
"""Quick test to verify GPU training setup"""
import torch
import torch.nn as nn
import time

print("="*60)
print("GPU Training Readiness Test")
print("="*60)

# Check PyTorch
print(f"\n1. PyTorch: {torch.__version__}")
print(f"   ROCm: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

# Check GPU
if not torch.cuda.is_available():
    print("\n✗ ERROR: GPU not available!")
    exit(1)

print(f"\n2. GPU: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"   VRAM: {props.total_memory / 1e9:.2f} GB")
print(f"   Compute: {props.major}.{props.minor}")

# Memory test
print(f"\n3. Memory Test:")
allocated_start = torch.cuda.memory_allocated(0) / 1e9
x = torch.randn(10000, 10000, device='cuda')
y = torch.randn(10000, 10000, device='cuda')
allocated_end = torch.cuda.memory_allocated(0) / 1e9
print(f"   Allocated: {allocated_end:.2f} GB")

# Performance test
print(f"\n4. Performance Test:")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    z = torch.matmul(x, y)
torch.cuda.synchronize()
elapsed = time.time() - start
gflops = (2 * 10000**3 * 100) / elapsed / 1e9
print(f"   100x 10000x10000 matmul: {elapsed:.2f}s ({gflops:.1f} GFLOPS)")

# Simulate DRL training
print(f"\n5. DRL Training Simulation:")
class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.net(x)

model = FakeActor().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
batch_size = 32768

states = torch.randn(batch_size, 100, device='cuda')

torch.cuda.synchronize()
start = time.time()
for i in range(100):
    actions = model(states)
    loss = actions.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
elapsed = time.time() - start

throughput = batch_size * 100 / elapsed
print(f"   100 training steps: {elapsed:.2f}s")
print(f"   Throughput: {throughput:.0f} samples/sec")

print(f"\n{'='*60}")
print("✓ GPU Training Ready!")
print(f"{'='*60}")
print(f"\nYou can now run training with:")
print(f"  source activate_rocm_env.sh")
print(f"  python scripts/training/1_optimize_unified.py --help")
