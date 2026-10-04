#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
BatchMatmulMaxSum算子golden实现
以 numpy bmm + amax + sum 的 FP64 结果为 golden
"""
import numpy as np


def impl(x1, x2, transposeX1=False, transposeX2=False):
    """BatchMatmulMaxSum算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    x1: float16/bfloat16 numpy数组，逻辑形状 (B, M, K)；
        transposeX1=True 时物理形状为 (B, K, M)。
    x2: 与 x1 同数据类型的 numpy数组，逻辑形状 (B, K, N)；
        transposeX2=True 时物理形状为 (B, N, K)。
    transposeX1: bool，是否交换 x1 最后两个维度，默认 False。
    transposeX2: bool，是否交换 x2 最后两个维度，默认 False。

    返回:
    y: (B,) float32 numpy数组，每个 batch 的相关性分数。
    """
    x1_np = np.asarray(x1)
    x2_np = np.asarray(x2)

    if x1_np.ndim != 3 or x2_np.ndim != 3:
        raise ValueError("x1 and x2 must both be rank-3 tensors")

    # 使用实际存储值执行 FP64 golden 计算
    x1_f = x1_np.astype(np.float64)
    x2_f = x2_np.astype(np.float64)

    x1_logical = np.swapaxes(x1_f, -1, -2) if transposeX1 else x1_f
    x2_logical = np.swapaxes(x2_f, -1, -2) if transposeX2 else x2_f

    if x1_logical.shape[0] != x2_logical.shape[0]:
        raise ValueError("x1 and x2 must have the same batch dimension; no broadcast")
    if x1_logical.shape[2] != x2_logical.shape[1]:
        raise ValueError("the logical K dimensions of x1 and x2 must match")

    similarity = np.matmul(x1_logical, x2_logical)   # (B, M, N)
    max_sim = np.max(similarity, axis=-1)            # (B, M)
    y = np.sum(max_sim, axis=-1)                     # (B,)

    return y.astype(np.float32)
