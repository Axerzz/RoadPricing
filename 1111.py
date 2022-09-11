#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from pylab import *
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# region 【功能函数】生成截断对数正态分布，要求对数正态在[log_lower,log_upper]
# def get_trunc_lognorm(mu, sigma, log_lower, log_upper=np.inf, data_num=):
#     norm_lower = np.log(log_lower)
#     norm_upper = np.log(log_upper)
#     X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
#     norm_data = X.rvs(data_num)
#     log_data = np.exp(norm_data)
#     return norm_data, log_data
#
#
# # endregion
#
# mu, sigma = 0, 1
# norm_data, log_data = get_trunc_lognorm(mu, sigma, 0, 1)
# plt.plot(np.arange(0,36), log_data)
# plt.show()
# print(norm_data)
# print(log_data)

s = 0.01
a = st.lognorm.rvs(s,size=36)
print(a)


