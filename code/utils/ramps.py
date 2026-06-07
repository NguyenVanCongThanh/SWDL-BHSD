# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np

# 用于实现一个指数型的渐变增加（ramp-up）机制，
# 这种机制通常用于深度学习中的学习率调整或权重调整等场景，以便在训练初期逐渐引入某些策略或调整，以避免训练过程的不稳定。
# current：当前的时间步或迭代次数。
# rampup_length：渐变增加的总长度或总时间步/迭代次数。
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0  # 没有渐变增加的过程，立即达到最大值。
    else:
        # 将 current 限制在0到 rampup_length 的范围内，确保 current 不会超过 rampup_length。
        # 这一步是为了防止 current 大于 rampup_length 时导致的除以零错误或超出预期的计算结果。
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length  # 表示当前处于渐变增加过程的哪个阶段。phase 的值从1（开始阶段）逐渐减少到0（结束阶段）
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
