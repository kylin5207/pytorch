"""
模型加载
"""
import time

import torch
from check_model import Model

# import sys
# print(sys.path)
#
# 定义tensor_vector和tensor_matrix
m, n = 10000, 2048
torch.manual_seed(2)
tensor_1 = torch.rand(n).reshape(1,-1)
tensor_2 = torch.rand((m, n))

# 使用torch.load("xxx.pt")，注意，这种方式需要引入源model
model = torch.load("face_model.pt")
similar = model(tensor_1, tensor_2)
print(similar)

# 使用torch.jit.load()
# 加载之前使用 torch.jit.save 保存的 ScriptModule 或 ScriptFunction
# 所有以前保存的模块，无论它们的设备如何，都首先加载到CPU上，然后移动到保存它们的设备。如果失败(例如，因为运行时系统没有某些设备)，则会引发异常。
model = torch.jit.load("face_model.pt")
t1 = time.time()
similar = model(tensor_1, tensor_2)
t2 = time.time()
print(similar)
print(t2-t1)


