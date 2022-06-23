import torch
"""
arange(start=0, end, step=1)返回[start, end)的一维tensor
"""

a = torch.arange(9, dtype=torch.float) - 4

# 非整数step会出现浮点舍入错误。end为避免不一致，我们建议end在这种情况下添加一个小的epsilon。
b = torch.arange(1, 2.5, 0.3)
print(a)
print(b)

