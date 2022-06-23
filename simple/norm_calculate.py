"""
tensor的范数计算
torch.linalg.norm(tensor, ord=2, dim=None)
dim=None时，如果tensor为多维，则会被flatten
"""
from math import inf

import torch
from torch import linalg as LA
import numpy as np

def calculate_vector_sqaure_norm(tensor, dim=None):
    """
    计算vector的范数
    :param tensor:
    :param dim:
    :return:
    """
    # 计算无穷范数
    # inf_norm = LA.norm(tensor, ord=inf)
    # 计算二范数
    sqaure_norm = LA.vector_norm(tensor, ord=2, dim=dim)
    return sqaure_norm

def calculate_matrix_sqaure_norm(matrix, dim=(-2,-1)):
    """
    计算matrix的范数
    :param tensor:
    :param dim:
    :return:
    """
    # 计算无穷范数
    # inf_norm = LA.matrix_norm(matrix, ord=inf)
    # 计算二范数
    sqaure_norm = LA.matrix_norm(matrix,dim=dim)
    return sqaure_norm

if __name__ == "__main__":
    # 向量的范数计算
    # 创建一个tensor向量
    t = torch.arange(9, dtype=torch.float) - 4
    t_array = np.arange(9, dtype=float) - 4
    print(f"tensor向量 = {t}")
    print(f"array = {t_array}")

    # 计算向量的平方范数
    vector_square_norm = calculate_vector_sqaure_norm(t)
    array_square_norm = np.linalg.norm(t_array)
    # 与numpy计算结果对比，基本一致
    print(f"square_norm_tensor = {vector_square_norm}")
    print(f"square_norm_numpy = {array_square_norm}")
    print(f"loss = {abs(array_square_norm - vector_square_norm)}")

    # 计算二维矩阵的范数
    t_matrix = t.reshape((3,3))
    t_array2 = t_array.reshape((3,3))
    print(f"tensor matrix = {t_matrix}")
    print(f"matrix shape = {t_matrix.size()}")
    matrix_square_norm = calculate_matrix_sqaure_norm(t_matrix)
    print(f"square_norm_matrix = {matrix_square_norm}")

    # 对原始矩阵垂直扩充一份
    expand_matrix = t_matrix.expand(2, -1, -1)
    print(f"expand matrix = {expand_matrix}")
    matrix_expand_square_norm = calculate_matrix_sqaure_norm(expand_matrix, dim=(0,2))
    print(f"expand matrix square norm = {matrix_expand_square_norm}")
