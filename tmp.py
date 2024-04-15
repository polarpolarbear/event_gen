import torch

# Example tensors
b, m, n, k = 3, 4, 5, 2
A = torch.randn(b, m, n)
B = torch.randint(0, m, (b, k))

# Create indices tensor
indices = B.unsqueeze(2).expand(-1, -1, n)  # Expand indices to match shape of A along dim 2

# Create mask tensor
mask = torch.ones_like(A, dtype=torch.bool)
mask.scatter_(1, B, 0)

# Gather values from A
C = torch.masked_select(A, mask).view(b, m - k, n)

print(C.shape)  # Output: torch.Size([3, 2, 5])
