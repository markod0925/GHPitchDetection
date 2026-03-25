import torch

path = "model/model-2000.pt"

print("Testing TorchScript load...")

try:
    m = torch.jit.load(path, map_location="cpu")
    print("SUCCESS: TorchScript model detected")
    print(type(m))
    print("Forward schema:", m.forward.schema)
except Exception as e:
    print("FAIL: Not TorchScript")
    print(type(e).__name__, e)

print("\nTesting torch.load...")

try:
    obj = torch.load(path, map_location="cpu")
    print("SUCCESS: torch.load worked")
    print("Type:", type(obj))
except Exception as e:
    print("FAIL: torch.load failed")
    print(type(e).__name__, e)