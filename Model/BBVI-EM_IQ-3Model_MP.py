#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:52:56 2022

@author: aixuexi
"""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib import rcParams

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam

import pickle
import time
import copy
import math
import multiprocessing
import prettytable as pt
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
from scipy.stats import spearmanr, pearsonr, ttest_rel

from MyQPModel import utils_Mcoop_split

from MyQPModel.bbvi_em_coop_org_country import *  # BBVI-EM_IQ-3Model.py


def split_data(data, row_col_list, mp_num):
    # 总数据
    x_obs, i_obs, j_obs, M_coop = data['x_obs'], data['i_obs'], data['j_obs'], data['M_coop']
    # 切割成 len(row_col_list) 块
    loop_num = math.ceil(len(row_col_list) / mp_num) # 循环数目
    datas    = dict()
    for l in range(loop_num):
        for m in range(mp_num):
            # 第 mp_i 块数据
            mp_i = l * mp_num + m
            if mp_i >= len(row_col_list):
                continue
            
            data_mp_i = dict()                     # 该进程负责使用的观测数据
            #
            row_mp_i, col_mp_i = row_col_list[mp_i]# row_mp_i(该进程所需文章编号); col_mp_i(该进程所需作者编号). M_coop
            x_obs_mp_i  = x_obs[row_mp_i]          # 文章的引用
            M_coop_mp_i = M_coop[row_mp_i]
            M_coop_mp_i = M_coop_mp_i[:, col_mp_i] # 合作矩阵 
            M_coop_mp_i = M_coop_mp_i / np.sum(M_coop_mp_i, axis=-1, keepdims=True)  # 对M_coop子块重新归一化
            i_obs_mp_i  = i_obs[col_mp_i]          # 作者隶属国家编号
            j_obs_mp_i  = j_obs[col_mp_i]          # 作者隶属机构编号
            data_mp_i['x_obs']  = x_obs_mp_i
            data_mp_i['i_obs']  = i_obs_mp_i
            data_mp_i['j_obs']  = j_obs_mp_i
            data_mp_i['M_coop'] = M_coop_mp_i

            datas[mp_i] = data_mp_i
    
    pid_num, aid_num = M_coop.shape
    datas["i_obs"]   = i_obs        # 学者隶属国家信息, 加权更新国家模型参数需使用
    datas["j_obs"]   = j_obs        # 学者隶属机构信息, 加权更新机构变分参数需使用
    datas["pid_num"] = pid_num      # 总发文量数目,     加权更新引用模型参数需使用

    return datas

# 多进程函数
def Estep_MP(datas, row_col_list, model_params, var_params, num_samples, step_size, num_iters):
    # 完整变分参数
    var_params_q2, var_params_q3 = var_params
    i_obs   = datas["i_obs"]
    j_obs   = datas["j_obs"]
    pid_num = datas["pid_num"]
    
    # 由于实证研究分析中数据内存限制: 采用训练 + 多进程方式 (2022-12-30)
    mp_num   = 8                                     # 多进程数目
    loop_num = math.ceil(len(row_col_list) / mp_num) # 循环数目
    results  = list()                                # 存放更新结果
    for l in range(loop_num):
        # 创建进程池
        print("{} / {}".format(l, loop_num))
        pool = multiprocessing.Pool(processes=len(row_col_list))
        for m in range(mp_num):
            # 第 mp_i 块数据
            mp_i = l * mp_num + m
            if mp_i >= len(row_col_list):
                continue

            data_mp_i = datas[mp_i]                # 该进程负责使用的观测数据
            var_params_mp_i = tuple()              # 该进程负责更新的变分参数
            # 
            row_mp_i, col_mp_i = row_col_list[mp_i]# row_mp_i(该进程所需文章编号); col_mp_i(该进程所需作者编号). M_coop
            var_params_q2_mp_i = var_params_q2            # 机构科研能力 
            var_params_q3_mp_i = var_params_q3[col_mp_i]  # 人员研究能力
            var_params_mp_i    = (var_params_q2_mp_i, var_params_q3_mp_i)
            #
            results.append(pool.apply_async(Estep, (data_mp_i, model_params, var_params_mp_i,
                                                    num_samples, step_size, num_iters, )))
            
        pool.close()
        pool.join()

    
    # 合并多进程的变分参数更新结果, 赋值操作
    org_num = len(set(list(j_obs)))
    def count_j_noa(org_num, j_obs):
        # 数每个机构人数: 当j_obs时, 为全数据每个机构人数
        #                 当j_obs_mp_i时, 为该进程数据每个机构人数
        # j_noa_rt_i = j_noa_mp_i / j_noa 被用来加权更新 q2变分参数
        j_noa = np.zeros(org_num)
        for j in sorted(set(list(j_obs))):
            noa_in_j = np.sum(np.array(j_obs == j, dtype=np.int32))  # j机构的人数
            j_noa[int(j)] = noa_in_j
        j_noa = j_noa[:, np.newaxis, np.newaxis]
        return j_noa
    j_noa = count_j_noa(org_num, j_obs)
    
    var_params_q2_next = np.zeros(var_params_q2.shape)
    var_params_q3_next = copy.copy(var_params_q3)
    for res, (row_mp_i, col_mp_i) in zip(results, row_col_list):
        var_params_mp_i = res.get()
        var_params_q2_mp_i, var_params_q3_mp_i = var_params_mp_i
        var_params_q3_next[col_mp_i] = var_params_q3_mp_i # 赋值: 人员变分参数只有特定进程更新
        # 机构变分参数, 所有进程都有更新, 取加权平均
        j_obs_mp_i = j_obs[col_mp_i]
        j_noa_mp_i = count_j_noa(org_num, j_obs_mp_i)
        j_noa_rt_i = j_noa_mp_i / j_noa
        var_params_q2_next += j_noa_rt_i * var_params_q2_mp_i
        
    var_params_next = (var_params_q2_next, var_params_q3_next)
    return var_params_next


def Mstep_MP(datas, row_col_list, model_params, var_params, num_samples, step_size, num_iters):
    # 完整变分参数
    var_params_q2, var_params_q3 = var_params
    i_obs   = datas["i_obs"]
    j_obs   = datas["j_obs"]
    pid_num = datas["pid_num"]

    # 由于实证研究分析中数据内存限制: 采用训练 + 多进程方式 (2022-12-30)
    mp_num   = 8                                     # 多进程数目
    loop_num = math.ceil(len(row_col_list) / mp_num) # 循环数目
    results  = list()                                # 存放更新结果
    for l in range(loop_num):
        # 创建进程池
        print("{} / {}".format(l, loop_num))
        pool = multiprocessing.Pool(processes=len(row_col_list))
        for m in range(mp_num):
            # 第 mp_i 块数据
            mp_i = l * mp_num + m
            if mp_i >= len(row_col_list):
                continue

            data_mp_i       = datas[mp_i]          # 该进程负责使用的观测数据
            var_params_mp_i = tuple()              # 该进程负责更新的变分参数
            # 
            row_mp_i, col_mp_i = row_col_list[mp_i]# row_mp_i(该进程所需文章编号); col_mp_i(该进程所需作者编号). M_coop
            var_params_q2_mp_i = var_params_q2            # 机构科研能力 
            var_params_q3_mp_i = var_params_q3[col_mp_i]  # 人员研究能力
            var_params_mp_i    = (var_params_q2_mp_i, var_params_q3_mp_i)
            #
            results.append(pool.apply_async(Mstep, (data_mp_i, model_params, var_params_mp_i,
                                                    num_samples, step_size, num_iters, )))
        pool.close()
        pool.join()
    
    
    # 合并多进程的模型参数更新结果, 赋值操作
    # pid_num, aid_num = M_coop.shape
    cou_num = len(set(list(i_obs)))
    def count_i_noa(cou_num, i_obs):
        # 数每个国家人数: 当i_obs时, 为全数据每个国家人数
        #                 当i_obs_mp_i时, 为该进程数据每个国家人数
        # i_noa_rt_i = i_noa_mp_i / i_noa 被用来加权更新国家模型参数
        i_noa = np.zeros(cou_num)
        for i in sorted(set(list(i_obs))):
            noa_in_i = np.sum(np.array(i_obs == i, dtype=np.int32))  # i国家的人数
            i_noa[int(i)] = noa_in_i
        i_noa = i_noa[:, np.newaxis, np.newaxis]
        return i_noa
    i_noa = count_i_noa(cou_num, i_obs)
    
    model_params_next = np.zeros(model_params.shape)
    for res, (row_mp_i, col_mp_i) in zip(results, row_col_list):
        model_params_mp_i = res.get()
        # 加权更新引用参数 (根据进程i中, 论文数目比率)
        nop_rt_i   = np.sum(row_mp_i) / pid_num
        model_params_next[-1:] += nop_rt_i * model_params_mp_i[-1:]
        # 加权更新国家参数 (根据每个进程的国家i人数比率)
        i_obs_mp_i = i_obs[col_mp_i]
        i_noa_mp_i = count_i_noa(cou_num, i_obs_mp_i)
        i_noa_rt_i = i_noa_mp_i / i_noa              # 进程i中, 国家i的人员比例
        model_params_next[:-1] += i_noa_rt_i * model_params_mp_i[:-1]
        
    return model_params_next


#%%
np.set_printoptions(precision=6, suppress=True)
def main():
    # 生成模拟数据
    mu_0, log_sig_0 = 0., -1.
    mu_1, log_sig_1 = -1., -1.
    mu_2, log_sig_2 = 0, -1.      
    mu_3, log_sig_3 = -1, -1.
    
    # 采样参数 
    cou_num = 10   # 国家数目
    org_num = 10   # 机构数目
    aid_num = 10   # 人员数目
    nop_num = 10  # 服从泊松分布 poisson(nop_num) 每位作者的平均发文量数目
    coa_num = 3.0 # 服从泊松分布 poisson(coa_num) 每篇文章的平均合著作者数目
    group_num = 8 
    sampling_params = [cou_num, org_num, aid_num, nop_num, coa_num, group_num]
    
    # 国家模型参数: 
    mu1_cou_list      = sampling_normal(mu_0, log_sig_0, cou_num)   # 国家研究能力的均值: 描述国家下机构研究能力的平均大小情况 —— 其值越大, 该国家下机构内人员能力越大
    log_sig1_cou_list = sampling_normal(mu_1, log_sig_1, cou_num)   # 国家研究能力的均值的方差: 描述国家下机构研究能力的平均大小的浮动情况
    mu2_cou_list      = sampling_normal(mu_2, log_sig_2, cou_num)   # 国家研究能力的方差: 描述国家下机构研究能力的平均浮动情况 —— 其值越大, 该国家下机构内人员能力差异大
    log_sig2_cou_list = sampling_normal(mu_3, log_sig_3, cou_num)   # 国家研究能力的方差的方差: 描述国家下机构研究能力的平均浮动的浮动情况
    model_params_real_q1 = np.concatenate([np.concatenate([mu1_cou_list, log_sig1_cou_list], axis=-1),
                                           np.concatenate([mu2_cou_list, log_sig2_cou_list], axis=-1)], axis=-1)
    model_params_real_q1 = model_params_real_q1.reshape((cou_num, 2, 2))
    # 随机波动模型参数 & 补两个无意义的参数, 方便拼接
    mu_P_real      = 0.0
    log_sig_P_real = 0.5 
    model_params_real_P = np.array([[mu_P_real, log_sig_P_real], [-1, -1]])
    model_params_real = np.concatenate([model_params_real_q1, [model_params_real_P]], axis=0)
          
    # 采样模拟数据
    data = create_simulation_data(model_params_real, sampling_params)
    # 极大似然估计
    var_params_est, model_params_est = max_likelihoood(data)
    evaluate_on_simulation_data(data, model_params_real, var_params_est, model_params_est)
    
    # 贝叶斯后验估计
    mp_num      = 8
    Epochs      = 10
    step_size   = 1e-1
    num_iters   = 100
    num_samples = 1
    var_params, model_params = var_params_est, model_params_est
    #
    row_col_list = utils_Mcoop_split.utils_mp_split(data, mp_num)
    datas        = split_data(data, row_col_list, mp_num)
    del data['M_coop']
    #
    for e in range(Epochs):
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))           
        E_start_time = time.perf_counter()
        var_params_next = Estep_MP(datas, row_col_list, model_params, var_params, num_samples, step_size, num_iters)
        E_end_time = time.perf_counter()                      
        print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
        #
        var_params = var_params_next
        # M-Step
        print("({}) Optimizing model parameters...".format(e))
        M_start_time = time.perf_counter()
        model_params_next = Mstep_MP(datas, row_col_list, model_params, var_params, num_samples, step_size, num_iters)
        M_end_time = time.perf_counter()
        print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time))
        # 
        model_params = model_params_next
     # 变分参数估计 - BBVI-EM
    var_params_bbvi, model_params_bbvi = var_params, model_params
    
    # 相关性检查
    var_params_q2_OUR, var_params_q3_OUR = var_params_bbvi
    var_params_q2_WSB, var_params_q3_WSB = var_params_est
    M_coop = data['M_coop']
    M_coop = np.array(M_coop > 0)
    x_obs  = data['x_obs']
    tcc    = np.sum(np.multiply(M_coop, x_obs.reshape((-1, 1))), axis=0).reshape((-1, 1))
    acc    = tcc / np.sum(M_coop, axis=0).reshape((-1, 1))
    q2s    = np.concatenate([var_params_q3_OUR[:, :1], var_params_q3_WSB[:, :1], tcc, acc], axis=-1)
    q2s    = pd.DataFrame(q2s)
    matrix = q2s.corr()
    
    # 绘图检查
    evaluate_on_simulation_data(data, model_params_real, var_params_est,  model_params_est) 
    evaluate_on_simulation_data(data, model_params_real, var_params_bbvi, model_params_bbvi)
