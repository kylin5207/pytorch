"""
tensor expand
"""
import torch

x = torch.tensor([[1], [2], [3]])
print(f"x = {x}")

print(f"size = {x.size()}")

expand_1 = x.expand(3, 4)
print(f"expand with (3, 4) :\n {expand_1}")

# -1表示不更改该维度的大小，拓展维度时不能为-1。
expand_2 = x.expand(-1, 4)
print(f"expand with (-1, 4) :\n {expand_2}")