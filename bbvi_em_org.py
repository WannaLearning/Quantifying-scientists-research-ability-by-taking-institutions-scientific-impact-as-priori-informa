# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 11:15:03 2022

@author: ShengzhiHuang
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
import os
import time
import math
import multiprocessing
import prettytable as pt
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
from scipy.stats import spearmanr, pearsonr, ttest_rel



def save_pkl_file(file, file_name):
    with open("./tmp/{}.pkl".format(file_name), 'wb') as f:
        pickle.dump(file, f)
   

#%%
def sampling_normal(mean, log_std, num_samples):
    # 抽取正态分布 ---> 引用量
    rs = npr.RandomState()
    samples = rs.randn(num_samples, 1) * np.exp(log_std) + mean
    return samples


def sampling_poisson(lam, num_samples):
    # 抽取泊松分布 ---> 发文量
    rs = npr.RandomState()
    samples = rs.poisson(lam, num_samples)
    return samples


def log_p_zx_density(z_i_arr, x_i_arr, model_params_i_P, mask_i_arr):
    # x: 观测, z: 隐变量
    # model_params: 模型参数
    # model_params_i: 机构先验参数
    # model_params_P: 领域引用范式参数(luck)
    model_params_i, model_params_P = model_params_i_P
    # 
    log_Q2_density = norm.logpdf(z_i_arr, model_params_i[0], np.exp(model_params_i[1]))
    log_P_density  = norm.logpdf(x_i_arr-z_i_arr, 0, np.exp(model_params_P[1]))
    # mask: 遮挡补全为0的数据
    log_P_density  = log_P_density * mask_i_arr
    log_P_density  = np.sum(log_P_density, axis=-1, keepdims=True)
    #
    logpq = log_Q2_density + log_P_density
    return logpq


#%%
def split_xobs_varparams_chunks(var_params, x_obs, mp_num):
    ''' 将变分参数 var_params 和 观测数据 x_obs 划分为多块 mp_num '''

    total_num = len(x_obs)                           # 机构总数目
    each_mp_num = math.ceil(total_num / mp_num)      # 每个进程负责 * 个机构的变分参数更新
    idx_chunks = list()                              # 每个进程负责 * 个机构的变分参数更新
    start_idx, end_idx = 0, 0
    for i in range(mp_num):
        end_idx = min(start_idx + each_mp_num, total_num)
        if start_idx != end_idx:
            idx_chunks.append((start_idx, end_idx))
        start_idx = end_idx
    
    # 数据划分 & 变分参数划分 ---> 多块数据
    x_obs_chunks = list()
    var_params_chunks = list()
    for start_idx, end_idx in idx_chunks:
        x_obs_mp_i = dict()
        var_params_mp_i = dict() 
        for i in range(start_idx, end_idx):
            x_obs_mp_i[i] = x_obs[i]                 # 第i个进程需使用的观测数据
            var_params_mp_i[i] = var_params[i]       # 第i个进程需要更新变分参数
        x_obs_chunks.append(x_obs_mp_i)
        var_params_chunks.append(var_params_mp_i)
        
    return x_obs_chunks, var_params_chunks, idx_chunks


def EStep_MP_func(var_params_mp_i, x_obs_mp_i, model_params, 
                  step_size, num_iters, num_samples, mp_i):
    ''' 第i个进程变分参数更新 '''
    
    def variational_objective(var_params_list, t):
        """Provides a stochastic estimate of the variational lower bound.
           循环版本 --- 速度会慢
        """
        lower_bound_total = 0
        for var_params_k, x_k in zip(var_params_list, x_list):
            # 从q(z)中抽取样本
            z_q2 = sampling_normal(var_params_k[0], var_params_k[1], num_samples)
            x_i = np.ones((num_samples, 1)) * x_k
            mask_i = np.ones(x_i.shape)  # 相当于未mask
            
            # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
            log_p_zx = log_p_zx_density(z_q2, x_i, [model_params_i, model_params_P], mask_i)
            part1 = np.mean(log_p_zx)
            # 
            q_var_z2 = norm.logpdf(z_q2, var_params_k[0], np.exp(var_params_k[1]))
            part2 = np.mean(q_var_z2)
            
            # 求ELBO最大, 所以这里加个负号即minimize
            lower_bound = part1 - part2
            lower_bound_total += lower_bound
        # 所有人员的ELBO平均
        lower_bound = lower_bound_total / len(var_params_list)
        
        return -lower_bound
    
    def variational_objective_fast(var_params_list, t):
        """Provides a stochastic estimate of the variational lower bound.
           矩阵操作版本 --- 速度更快
        """
        # 求x_list内最大发文量数目 (机构i内最大发文量数目) --- padding mask
        nop_i_j_list = list()
        for x_i_j in x_list:
           nop_i_j = len(x_i_j)
           nop_i_j_list.append(nop_i_j)
        max_nop = max(nop_i_j_list)
        
        # padding 操作
        x_i_padding = list()  # 将一个机构人员的发文量补齐为最大值
        mask_i = list()       # 标记那些位置是补齐
        for x_i_j in x_list:
            x_i_j = np.ones((num_samples, 1)) * x_i_j
            num_samples_i_j, nop_i_j = x_i_j.shape
            x_i_j_zeros = np.zeros((num_samples_i_j, max_nop-nop_i_j))
            x_i_j_padding = np.concatenate([x_i_j, x_i_j_zeros], axis=-1)
            x_i_padding.append(x_i_j_padding)
            #
            mask_i_j = np.concatenate([np.ones(nop_i_j), np.zeros(max_nop-nop_i_j)])
            mask_i_j = np.ones((num_samples_i_j, 1)) * mask_i_j
            mask_i.append(mask_i_j)
        
        # 人员数目 * 样本数目 * 最大文章数目
        x_i_arr = np.array(x_i_padding)
        mask_i_arr = np.array(mask_i)
        # 人员数目 * 样本数目 * 1  --- 采样隐变量z~q(z), 标准高斯采样, sig * sample + mu
        noa = len(var_params_list)
        q2_z = sampling_normal(0, 0, num_samples * noa)
        q2_z = q2_z.reshape((noa, num_samples, 1))
        # sig * sample + mu 
        mean = var_params_list[:, 0].reshape((noa, 1, 1))
        log_std = var_params_list[:, 1].reshape((noa, 1, 1))
        q2_z = q2_z * np.exp(log_std) + mean
        
        x_i_arr = np.concatenate(x_i_arr, axis=0)
        mask_i_arr = np.concatenate(mask_i_arr, axis=0)
        z_i_arr = np.concatenate(q2_z, axis=0)
        
        # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
        logp_zx = log_p_zx_density(z_i_arr, x_i_arr, [model_params_i, model_params_P], mask_i_arr)
        part1 = np.mean(logp_zx)
        logp_q2_z = norm.logpdf(q2_z, mean, np.exp(log_std))  # 注意维度
        part2 = np.mean(logp_q2_z)
        
        lower_bound = part1 - part2
        return -lower_bound
    
    
    # 逐机构内人员的变分参数更新
    for i in x_obs_mp_i:                                  
        x_obs_i = x_obs_mp_i[i]                 # i机构的观测数据
        var_params_i = var_params_mp_i[i]       # i机构下人员的变分参数
        model_params_i = model_params[i]['q1']  # i机构的模型参数        --- 模型参数
        model_params_P = model_params['P']      # 领域引用范式参数(luck) --- 模型参数
        
        # 将 机构i内 变分参数 和 观测数据 提取
        var_params_list = list()                # 该机构的所有作者的变分参数
        x_list = list()                         # 该机构的所有作者的观测数据
        for j in x_obs_i:                       # 人员编号 j
            if j == 'q1':                       # q1是机构真实参数
                continue
            else:
               var_params_q2_i_j = var_params_i[j]['q2']
               x_i_j = x_obs_i[j]['x']
               var_params_list.append(var_params_q2_i_j)
               x_list.append(x_i_j)
        var_params_list = np.array(var_params_list)  # 人员数目 x 2(mu 和 sig)
        # x_list = np.array(x_list)                  # 人员数目 x 文章数目 (当前相同)
        
        # 梯度下降更新变分参数
        gradient = grad(variational_objective_fast)
        var_params_list_next = adam(gradient, var_params_list, step_size=step_size, num_iters=num_iters)
 
        # 参数更新
        for j in x_obs_i:                       # 人员编号 j
            if j == 'q1':                       # q1是机构真实参数
                continue
            else:
               var_params_mp_i[i][j]['q2'] = var_params_list_next[j]

    return var_params_mp_i


def EStep_MP(var_params, x_obs, model_params,
             step_size, num_iters, num_samples, mp_num):
    '''启用多进程进行变分参数更新'''
    
    # 将观测数据 和 变分参数 划分成多块
    x_obs_chunks, var_params_chunks, idx_chunks = split_xobs_varparams_chunks(var_params, x_obs, mp_num) 
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=len(idx_chunks))
    results = list()  # 存放更新结果
    for mp_i in range(len(idx_chunks)):
        x_obs_mp_i      = x_obs_chunks[mp_i]
        var_params_mp_i = var_params_chunks[mp_i]
        results.append(pool.apply_async(EStep_MP_func, (var_params_mp_i, x_obs_mp_i, model_params,
                                                        step_size, num_iters, num_samples, mp_i, )))
    pool.close()
    pool.join()
    
    # 合并多进程的变分参数更新结果, 赋值操作
    for res in results:
        var_params_mp_i = res.get()
        for i in var_params_mp_i:
            for j in var_params_mp_i[i]:
                var_params[i][j]['q2'] = var_params_mp_i[i][j]['q2']
    return var_params


#%%
def sampling_z(var_params_mp_i, x_obs_mp_i, num_samples):
    '''
    进程采样: 
        var_params_mp_i: 进程使用的变分参数
        x_obs_mp_i: 进程使用的观测数据
    '''
    x_mp_i_list = list()
    z_mp_i_list = list()
    org_mp_i_list = list()
    
    for i in var_params_mp_i:
        x_i_list  = list()        # 机构i内所有作者j的观测数据 
        z_q2_i_list = list()      # 变分参数j抽取样本
        org_id_i_list = list()    # 标记人员所来自的机构, 从而确定机构模型参数
        # 第 i 个机构
        x_obs_i = x_obs_mp_i[i]
        var_params_i = var_params_mp_i[i]
        for j in x_obs_i:
            if j == "q1":         # 真实的机构参数
                continue
            # 第 j 个人员
            var_params_q2_i_j = var_params_i[j]['q2']
            x_i_j = x_obs_i[j]['x']
            #             
            z_q2 = sampling_normal(var_params_q2_i_j[0], var_params_q2_i_j[1], num_samples)
            x = np.ones((num_samples, 1)) * x_i_j
            # 
            x_i_list.append(x)
            z_q2_i_list.append(z_q2)
            org_id_i_list.append(i)
        #
        x_mp_i_list.append(x_i_list)
        z_mp_i_list.append(z_q2_i_list)
        org_mp_i_list.append(org_id_i_list)

    return x_mp_i_list, z_mp_i_list, org_mp_i_list


def sampling_z_MP(var_params, x_obs, model_params, num_samples, mp_num):
    '''
    多进程采样, 调用 sampling_z
    '''
    # 将观测数据x 和 变分参数var 划分
    x_obs_chunks, var_params_chunks, idx_chunks = split_xobs_varparams_chunks(var_params, x_obs, mp_num) 
        
    # 多进程采样
    pool = multiprocessing.Pool(processes=len(idx_chunks))
    results = list()
    for mp_i in range(len(idx_chunks)):
        x_obs_mp_i      = x_obs_chunks[mp_i]
        var_params_mp_i = var_params_chunks[mp_i]
        results.append(pool.apply_async(sampling_z, (var_params_mp_i, x_obs_mp_i, num_samples, )))                                                        
    pool.close()
    pool.join()
        
    # 合并多进程的抽样结果
    x_list = list()     # 人员数目 * 抽样数目 * 文章数目
    z_list = list()     # 人员数目 * 抽样数目 (num_samples) * 1
    org_id_list = list()
    for res in results:
        x_mp_i_list, z_mp_i_list, org_mp_i_list = res.get()
        
        x_list.append(x_mp_i_list)
        z_list.append(z_mp_i_list)
        org_id_list.append(org_mp_i_list)
     
    return x_list, z_list, org_id_list, idx_chunks
    

def MStep_MP_func(z_mp_i_list, x_mp_i_list, org_mp_i_list,   # 数据: 抽样数据, 观测数据, 机构编号
                  model_params,                              # 待更新模型参数 
                  step_size, num_iters, num_samples, mp_i):  # 训练配置

    # 第i个进程模型参数更新
    def variational_objective(model_params_list, t):
        """Provides a stochastic estimate of the variational lower bound."""
        # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
        # 数组广播版本 (padding 操作) 
        
        # 确定该机构个人的最大发文量
        nop_i_j_list = list()
        for x_i_j in x_i:
            _, nop_i_j = x_i_j.shape
            nop_i_j_list.append(nop_i_j)
        max_nop = max(nop_i_j_list)
        
        x_i_padding = list()  # 将一个机构人员的发文量补齐为最大值
        mask_i = list()       # 标记那些位置是补齐
        for x_i_j in x_i:
            num_samples_i_j, nop_i_j = x_i_j.shape
            x_i_j_zeros = np.zeros((num_samples_i_j, max_nop-nop_i_j))
            x_i_j_padding = np.concatenate([x_i_j, x_i_j_zeros], axis=-1)
            x_i_padding.append(x_i_j_padding)
            #
            mask_i_j = np.concatenate([np.ones(nop_i_j), np.zeros(max_nop-nop_i_j)])
            mask_i_j = np.ones((num_samples_i_j, 1)) * mask_i_j
            mask_i.append(mask_i_j)
        
        # 人员数目 * 样本数目 * 最大文章数目
        x_i_arr = np.array(x_i_padding)
        mask_i_arr = np.array(mask_i)
        # 人员数目 * 样本数目 * 1
        z_i_arr = np.array(z_i)
        org_i_j = org_i[0]
        
        x_i_arr = np.concatenate(x_i_arr, axis=0)
        mask_i_arr = np.concatenate(mask_i_arr, axis=0)
        z_i_arr = np.concatenate(z_i_arr, axis=0)
        model_params_i = model_params_list[org_i_j]
        model_params_P = model_params_list[-1]
        
        log_p_zx = log_p_zx_density(z_i_arr, x_i_arr, [model_params_i, model_params_P], mask_i_arr)
        part1 = np.mean(log_p_zx)
        lower_bound = part1
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound

    # 模型参数字典转成为数组
    org_num = 0
    model_params_list = list()
    for i in model_params:
        if i == "P":
            model_params_P_ = model_params[i]             
        else:
            model_params_list.append(model_params[i]['q1'])
            org_num += 1                                    # 机构数目
    model_params_list.append(model_params_P_) 
    model_params_list = np.array(model_params_list)
    
    # 逐个机构更新机构模型参数
    for i in range(len(z_mp_i_list)):
        z_i = z_mp_i_list[i]
        x_i = x_mp_i_list[i]
        org_i = org_mp_i_list[i]  # 待更新机构参数
        # 梯度更新模型参数
        gradient = grad(variational_objective)        
        model_params_list_next = adam(gradient, model_params_list, step_size=step_size, num_iters=num_iters)
        model_params_list = model_params_list_next
        
    # 模型参数数组还原为字典
    model_params_next = dict() 
    for i, _ in enumerate(model_params_list):
        if i >= org_num:
            continue
        model_params_next[i] = dict()
        model_params_next[i]['q1'] = model_params_list[i]
    model_params_next['P'] = model_params_list[-1]
    
    return model_params_next


def MStep_MP_func2(x_list, z_list, org_id_list,               # 数据: 抽样数据, 观测数据, 机构编号
                   model_params,                              # 待更新模型参数 
                   step_size, num_iters, num_samples):        # 训练配置
    ''' 只更新引用范式luck参数'''
    z, x, org = list(), list(), list()
    for x_i, z_i, org_i in zip(x_list, z_list, org_id_list):
        x += x_i
        z += z_i
        org += org_i
        
    def variational_objective2(model_params_P, t):
        """Provides a stochastic estimate of the variational lower bound."""
        
        lower_bound_total = 0
        for i in range(len(z)):
            # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
            z_i = z[i]  
            x_i = x[i]
            org_i = org[i] 
            
            # 数组广播版本 (padding 操作) 
            # 确定该机构个人的最大发文量
            nop_i_j_list = list()
            for x_i_j in x_i:
                _, nop_i_j = x_i_j.shape
                nop_i_j_list.append(nop_i_j)
            max_nop = max(nop_i_j_list)
            
            x_i_padding = list()  # 将一个机构人员的发文量补齐为最大值
            mask_i = list()       # 标记那些位置是补齐
            for x_i_j in x_i:
                num_samples_i_j, nop_i_j = x_i_j.shape
                x_i_j_zeros = np.zeros((num_samples_i_j, max_nop-nop_i_j))
                x_i_j_padding = np.concatenate([x_i_j, x_i_j_zeros], axis=-1)
                x_i_padding.append(x_i_j_padding)
                #
                mask_i_j = np.concatenate([np.ones(nop_i_j), np.zeros(max_nop-nop_i_j)])
                mask_i_j = np.ones((num_samples_i_j, 1)) * mask_i_j
                mask_i.append(mask_i_j)
            
            # 人员数目 * 样本数目 * 最大文章数目
            x_i_arr = np.array(x_i_padding)
            mask_i_arr = np.array(mask_i)
            # 人员数目 * 样本数目 * 1
            z_i_arr = np.array(z_i)
            org_i_j = org_i[0]
            
            x_i_arr = np.concatenate(x_i_arr, axis=0)
            mask_i_arr = np.concatenate(mask_i_arr, axis=0)
            z_i_arr = np.concatenate(z_i_arr, axis=0)
            model_params_i = model_params_list[org_i_j]
            # model_params_P = model_params_list[-1]
            
            log_p_zx = log_p_zx_density(z_i_arr, x_i_arr, [model_params_i, model_params_P], mask_i_arr)
            part1 = np.mean(log_p_zx)
            lower_bound = part1
            # 累计ELBO
            lower_bound_total += lower_bound
            
        lower_bound_total = lower_bound_total / len(z)
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound_total

    # 模型参数字典转成为数组
    org_num = 0
    model_params_list = list()
    for i in model_params:
        if i == "P":
            model_params_P = np.array(model_params[i])
        else:
            model_params_list.append(model_params[i]['q1'])
            org_num += 1
    model_params_list.append(model_params_P) 
    model_params_list = np.array(model_params_list)
    
    # 梯度更新模型参数
    gradient = grad(variational_objective2)
    model_params_P_next = adam(gradient, model_params_P, step_size=step_size, num_iters=num_iters)
        
    # 模型参数数组还原为字典
    model_params_next = dict() 
    for i, _ in enumerate(model_params_list):
        if i >= org_num:
            continue
        model_params_next[i] = dict()
        model_params_next[i]['q1'] = model_params_list[i]
    model_params_next['P'] = model_params_P_next
    
    return model_params_next


def MStep_MP(var_params, x_obs, model_params,
             step_size, num_iters, num_samples, mp_num):
    
    # 多进程采样
    # 将所有机构下人员的z采样出来
    sampling_start_time = time.perf_counter()
    x_list, z_list, org_id_list, idx_chunks = sampling_z_MP(var_params, x_obs, model_params, num_samples, mp_num)
    sampling_end_time = time.perf_counter()
    print("MStep抽样耗时: {:.4f}".format(sampling_end_time - sampling_start_time))
    
    '''多进程更新机构参数'''
    M_start_time = time.perf_counter()
    # 创建进程池
    pool = multiprocessing.Pool(processes=len(idx_chunks))
    results = list()   # 存放更新结果
    for mp_i in range(len(idx_chunks)):
        z_mp_i_list = z_list[mp_i]
        x_mp_i_list = x_list[mp_i]
        org_mp_i_list = org_id_list[mp_i]
        results.append(pool.apply_async(MStep_MP_func, (z_mp_i_list, x_mp_i_list, org_mp_i_list, model_params,
                                                        step_size, num_iters, num_samples, mp_i, )))
    pool.close()
    pool.join()
    
    # (1) 更新所有机构参数
    model_params_next1 = dict()
    P_params_list = list()
    for mp_i in range(len(idx_chunks)):
        res = results[mp_i]
        model_params_next_mp_i = res.get()
        P_params = model_params_next_mp_i["P"]
        start_idx, end_idx = idx_chunks[mp_i]
        for i in range(start_idx, end_idx):
            model_params_next1[i] = dict()
            model_params_next1[i]['q1'] = model_params_next_mp_i[i]['q1']
        P_params_list.append(P_params)
        
    # 所有P_params的均值, 简单更新P_params
    model_params_next1["P"] = np.mean(np.array(P_params_list), axis=0)
    M_end_time = time.perf_counter()
    print("MStep更新机构参数耗时: {:.4f}".format(M_end_time-M_start_time))
    model_params_next = model_params_next1
    
    # # (2) 单独更新公共机构参数
    # M_start_time2 = time.perf_counter()
    # model_params_next2 = MStep_MP_func2(x_list, z_list, org_id_list, model_params_next, step_size, num_iters, num_samples)
    # M_end_time2 = time.perf_counter()
    # print("MStep更新公共参数耗时: {:.4f}".format(M_end_time2-M_start_time2))
    # model_params_next = model_params_next2
    
    return model_params_next


#%%
'''
    模拟数据分析
    Quantification performance on the simulation data analysis
'''
def create_simulation_data(model_params_real, 
                           sampling_params):
    # 采样参数
    org_num, aid_num, nop_num = sampling_params
    # 模型参数
    mu_org_list = list()
    log_sig_org_list = list()
    for i in model_params_real:
        if i == "P":
            mu_P_real, log_sig_P_real = model_params_real[i] 
        else:
            mu_org_list.append(model_params_real[i]['q1'][0])
            log_sig_org_list.append(model_params_real[i]['q1'][1])
 
    # 开始抽样
    x_obs = dict()
    org_id = -1
    for mu_org_i, log_sig_org_i in zip(mu_org_list, log_sig_org_list):
        org_id += 1
        x_obs[org_id] = dict()
        x_obs[org_id]['q1'] = [mu_org_i, log_sig_org_i]
        aid_id = -1
        # 第 i 个机构中抽取人员能力
        q_i_list = sampling_normal(mu_org_i, log_sig_org_i, aid_num).squeeze()
        for q_i_j in q_i_list:
            aid_id += 1
            x_obs[org_id][aid_id] = dict()
            x_obs[org_id][aid_id]['q2'] = q_i_j
            # 抽取人员的发文量 ~ 泊松分布 (至少一篇文章)
            nop_num_i = max(sampling_poisson(nop_num, 1), [1])[0]
            # 第 i 个机构中第j个人抽样文章引用数目 ~ 正态分布
            p_i_j_list = sampling_normal(mu_P_real, log_sig_P_real, nop_num_i).squeeze()
            
            x = q_i_j + p_i_j_list
            if nop_num_i > 1:
                x_obs[org_id][aid_id]['x'] = x
            else:
                x_obs[org_id][aid_id]['x'] = np.array([x])
    return x_obs


def evaluate_real2pred(Y, X):
    # 评价指标: 
    # 相关性评价: Pearsonr
    # 距离评价: MAE, MSE
    cor, pvalue = pearsonr(Y, X)
    rmse = np.sqrt(mean_squared_error(Y, X))
    mae = mean_absolute_error(Y, X)
    r2  = r2_score(Y, X)
    return cor, rmse, mae, r2


def evaluate_on_simulation_data(x_obs,
                                est_params,
                                normarlized):
    
    q1_real_mu_list  = list()
    q1_real_std_list = list()
    q1_var_mu_list   = list()
    q1_var_std_list  = list() 
    
    q2_real_mu_list  = list()
    q2_var_mu_list   = list()
    q2_var_std_list  = list()
    
    for i in x_obs:
        q1_real_mu_list.append(x_obs[i]['q1'][0])     # 均值和方差模型参数
        q1_real_std_list.append(x_obs[i]['q1'][1])
        if "q1" in est_params[i]:
            q1_var_mu_list.append(est_params[i]['q1'][0])
            q1_var_std_list.append(est_params[i]['q1'][1])
        else:
            rs = npr.RandomState()
            q1_var_mu_list.append(rs.randn(1, 1)[0][0])
            q1_var_std_list.append(1)
        for k in x_obs[i]:
            if k == 'q1':
                continue 
            q2_real_mu_list.append(x_obs[i][k]['q2']) # 隐变量没有方差
            q2_var_mu_list.append(est_params[i][k]['q2'][0])
            q2_var_std_list.append(est_params[i][k]['q2'][1])
            
    q1_real_mu_list = np.array(q1_real_mu_list)
    q1_real_std_list = np.array(q1_real_std_list)
    q1_var_mu_list = np.array(q1_var_mu_list)
    q1_var_std_list = np.array(q1_var_std_list)
    
    q2_real_mu_list = np.array(q2_real_mu_list)
    q2_var_mu_list = np.array(q2_var_mu_list)
    q2_var_std_list = np.array(q2_var_std_list)

    if normarlized:
        def nomralized_func(x):
            return (x - np.mean(x)) / np.std(x)    
    
    def plot_q(q_real, q_var, q_err, xlabel, ylabel, legend_1, legend_2):
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        config = {
                  "font.family" : "Times New Roman",
                  "font.size" : 25
                  }
        rcParams.update(config)
    
        plt.plot(np.arange(len(q_real)), q_real, 
                 label=legend_1,
                 c='red', marker='s', alpha=0.75, linewidth=1)
        plt.errorbar(np.arange(len(q_real)), q_var, yerr=q_err, # np.zeros(len(q_real))
                      label=legend_2,
                      fmt="o:", color='blue', ecolor='black', capsize=5, markersize=5, elinewidth=0.5)
        plt.ylabel(ylabel, fontsize=25)
        plt.xlabel(xlabel, fontsize=25)
        plt.legend(frameon=False, loc='upper right', fontsize=25)
        
        # plt.yticks(np.arange(-15, 16, 5))
        # plt.ylim(-15, 15)
        # plt.xticks(np.arange(0, 350, 50))
        # plt.yticks(np.arange(-6, 7, 2))
        # plt.ylim(-6, 6)
        # plt.xticks(np.arange(0, 14, 1))
        
        
    plot_q(q1_real_mu_list, q1_var_mu_list, np.exp(q1_var_std_list), "Institutions", r'$\mu_\lambda$', "Real", "Estimated")
    plot_q(q2_real_mu_list, q2_var_mu_list, np.exp(q2_var_std_list), "Scientists", r"$\hat{Q}$", "Real", "Estimated")
    
    cor_q1, rmse_q1, mae_q1, r2_q1 = evaluate_real2pred(q1_real_mu_list, q1_var_mu_list)
    cor_q2, rmse_q2, mae_q2, r2_q2 = evaluate_real2pred(q2_real_mu_list, q2_var_mu_list)
    
    tb = pt.PrettyTable()
    tb.field_names = ["", "cor", "r2", "rmse", 'mae']
    tb.add_row(["机构", "{:.4f}".format(cor_q1), "{:.4f}".format(r2_q1), "{:.4f}".format(rmse_q1), "{:.4f}".format(mae_q1)])
    tb.add_row(["人员", "{:.4f}".format(cor_q2), "{:.4f}".format(r2_q2), "{:.4f}".format(rmse_q2), "{:.4f}".format(mae_q2)])
    print(tb)
    
    return (cor_q1, rmse_q1, mae_q1, r2_q1), (cor_q2, rmse_q2, mae_q2, r2_q2)


def max_likelihoood(x_obs):
    # 极大似然估计 - 均值估计
    # 忽略先验分布, 是我们的Baseline
    x_total = list()
    for i in x_obs:
        x_obs_i = x_obs[i]
        q1_est = list()   # 第 i 个机构的均值和方差估计
        for j in x_obs_i:
           if j == "q1":  # 第 j 个人的真实能力 q
               continue
           x_obs_i_j = x_obs_i[j]
           x = x_obs_i_j['x']
           x_total.append(x)
    x_total = np.concatenate(x_total, axis=0)
    sig_total = np.std(x_total)
    
    est_params = dict()
    for i in x_obs:
        est_params[i] = dict()
        x_obs_i = x_obs[i]
        q1_est_list = list()
        for j in x_obs_i:
           if j == "q1":  # 第 j 个人的真实能力 q
               continue
           x_obs_i_j = x_obs_i[j]
           est_params[i][j] = dict()
           x = x_obs_i_j['x']
           # 第j个人员能力的均值和方差估计 (变分参数)
           q2_mu_est = np.mean(x)
           est_params[i][j]['q2'] = [q2_mu_est, np.log(max(sig_total - np.std(x), 1e-2))]
           q1_est_list.append(q2_mu_est)
        # 第i个机构的均值和方差估计 (模型参数)
        est_params[i]['q1'] = [np.mean(q1_est_list), np.log(max(np.std(q1_est_list), 1e-2))]
    return est_params


def init_var_params(est_params):
    ''' 利用均值估计的参数初始化变分参数和模型参数'''
    # 待估计模型参数初始化 & 待估计变分参数初始化: 
    model_params_init = dict()
    var_params_init = dict()
    log_sig_P_est_list = list()
    for i in est_params:
        model_params_init[i] = dict()
        var_params_init[i] = dict()
        for j in est_params[i]:
            if j == "q1":
                mu_org_est, log_sig_org_est = est_params[i][j]
                model_params_init[i][j] = [mu_org_est, log_sig_org_est]
            else:
                var_params_init[i][j] = dict()
                mu_q2_est, log_sig_q2_est = est_params[i][j]['q2']
                var_params_init[i][j]['q2'] = [mu_q2_est, log_sig_q2_est]
                #
                log_sig_P_est_list.append(log_sig_q2_est)       
    model_params_init["P"] = [0, np.mean(log_sig_P_est_list)]
    return model_params_init, var_params_init


# 模拟数据分析...
np.set_printoptions(precision=6, suppress=True)
def BBVI_Algorithm(): 
    '''
    模型结构:
        多个机构i, 每个机构有模型参数 (mu_org_i, log_sig_org_i), 即 q1 = mu_org_i.
        采样人员j: q2_j ~ normal(mu_org_i, exp(log_sig_org_i))
        
        多个人员j, 每个人员有隐变量 (q2_j)
        采样文章cc: p_l ~ normal(0, exp(log_sig_p)), p_l是隐变量, 但利用cc_jl - p_l替换
        cc_jl = q2_j + p_l 

    # 2022-8-31 在模拟数据上能够取得满意的效果
    ''' 
    # 生成模拟数据
    mu_0, sig_0 = 0., 0.
    mu_1, sig_1 = 0., 0.
    
    results = dict()
    for t in range(20):
        # 模型参数
        # 机构数目 & 机构模型参数: 
        org_num = 14
        mu_org_list = list(sampling_normal(mu_0, sig_0, org_num).squeeze())
        log_sig_org_list = list(sampling_normal(mu_1, sig_1, org_num).squeeze())
        model_params_real = dict()
        for i in range(org_num): 
            model_params_real[i] = dict()
            model_params_real[i]['q1'] = [mu_org_list[i], log_sig_org_list[i]]
        # 人员数目
        aid_num = 20
        # 人均文章数目 & 随机波动模型参数
        nop_num = 15   # 服从泊松分布 poisson(nop_num)
        mu_P_real = 0
        log_sig_P_real = 2
        model_params_real["P"] = [mu_P_real, log_sig_P_real]
    
        # 采样模拟数据
        sampling_params = [org_num, aid_num, nop_num]
        x_obs = create_simulation_data(model_params_real, sampling_params)
        
        # Baseline 均值估计 (mu_1, sig_1) 和 log_sig_P_real 控制
        est_params = max_likelihoood(x_obs)
        # evaluate_on_simulation_data(x_obs, est_params, True)
    
        # 待估计模型参数初始化 & 待估计变分参数初始化:
        model_params_init, var_params_init = init_var_params(est_params)
        # model_params = model_params_real
    
        # 变分估计
        mp_num = 7
        Epochs = 20
        step_size = 1e-2
        num_iters = 100
        num_samples = 100
        model_params, var_params = model_params_init, var_params_init
        # 
        for e in range(Epochs):
            # E-Step
            print("({}) Optimizing variational parameters...".format(e))           
            E_start_time = time.perf_counter()
            var_params_next = EStep_MP(var_params, x_obs, model_params, step_size, num_iters, num_samples, mp_num)
            E_end_time = time.perf_counter()                      
            print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
            #
            var_params = var_params_next
            # M-Step
            print("({}) Optimizing model parameters...".format(e))
            M_start_time = time.perf_counter()
            # model_params_next = MStep_non_MP(var_params, x_obs, model_params, step_size, num_iters, mp_num)
            model_params_next = MStep_MP(var_params, x_obs, model_params, step_size, num_iters, num_samples, mp_num)
            M_end_time = time.perf_counter()
            print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time))
            # 
            model_params = model_params_next
        # 模型参数和变分参数合并储存
        var_params_final, model_params_final = var_params, model_params
        for i in model_params_final:
            if i == "P":
                continue
            else:
                var_params_final[i]["q1"] = model_params_final[i]["q1"]
        
        # 模型结果评价
        q1_WSB, q2_WSB = evaluate_on_simulation_data(x_obs, est_params, True)
        q1_our, q2_our = evaluate_on_simulation_data(x_obs, var_params_final, False)
    
        results[t] = dict()
        results[t]["wsb"] = (q1_WSB, q2_WSB)
        results[t]["our"] = (q1_our, q2_our)
    # 存储 20 次结果
    with open("./tmp/results_org_simulation_{}.pkl".format((org_num, aid_num, nop_num, log_sig_P_real)), 'wb') as f:
        pickle.dump(results, f)
    
    
    '''表格评价结果'''
    # 读取 20 次结果
    with open("./tmp/results_org_simulation_{}.pkl".format((org_num, aid_num, nop_num, log_sig_P_real)), 'rb') as f:
        results = pickle.load(f)
    # 计算每个指标的均值; t检验
    q1_WSB_list = list()
    q1_our_list = list()
    q2_WSB_list = list()
    q2_our_list = list()
    for t in results:
        q1_WSB, q2_WSB = results[t]["wsb"]
        q1_our, q2_our = results[t]["our"]
        q1_WSB_list.append(list(q1_WSB))
        q1_our_list.append(list(q1_our))
        q2_WSB_list.append(list(q2_WSB))
        q2_our_list.append(list(q2_our))
    
    # 检查q1的估计水平
    q1_WSB_list = np.array(q1_WSB_list)  # cor_q1, rmse_q1, mae_q1, r2_q1
    q1_our_list = np.array(q1_our_list)
    q1_WSB_list = np.maximum(q1_WSB_list, 0)
    q1_our_list = np.maximum(q1_our_list, 0)
    q1_WSB_mean = np.mean(q1_WSB_list, axis=0)
    q1_our_mean = np.mean(q1_our_list, axis=0)
    print("Pearsonr: {:.4f}, R2: {:.4f}, RMES: {:.4f}, MAE: {:.4f}".format(q1_WSB_mean[0], q1_WSB_mean[-1], q1_WSB_mean[1], q1_WSB_mean[2]))
    print("Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(q1_our_mean[0], q1_our_mean[-1], q1_our_mean[1], q1_our_mean[2]))
    for i in range(4):
        p1 = ttest_rel(q1_WSB_list[:, i], q1_our_list[:, i])
        print(p1)
    
    # 检查q2的估计水平
    q2_WSB_list = np.array(q2_WSB_list)
    q2_our_list = np.array(q2_our_list)
    q2_WSB_list = np.maximum(q2_WSB_list, 0)
    q2_our_list = np.maximum(q2_our_list, 0)
    q2_WSB_mean = np.mean(q2_WSB_list, axis=0)
    q2_our_mean = np.mean(q2_our_list, axis=0)
    print("Pearsonr: {:.4f}, R2: {:.4f}, RMES: {:.4f}, MAE: {:.4f}".format(q2_WSB_mean[0], q2_WSB_mean[-1], q2_WSB_mean[1], q2_WSB_mean[2]))
    print("Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(q2_our_mean[0], q2_our_mean[-1], q2_our_mean[1], q2_our_mean[2]))
    for i in range(4):
        p1 = ttest_rel(q2_WSB_list[:, i], q2_our_list[:, i])
        print(p1)
    
    
#%%
'''实证数据分析
    Prediction performance on the Empirical data analysis
'''
def evaluate_on_empirical_data(est_params,
                               normarlized):
    # 针对经验数据进行分析的时候, 不再具备真实的Q值
    q1_var_mu_list  = list()
    q1_var_std_list = list() 
    
    q2_var_mu_list  = list()
    q2_var_std_list = list()
    
    for i in est_params:
        if "q1" in est_params[i]:
            q1_var_mu_list.append(est_params[i]['q1'][0])
            q1_var_std_list.append(est_params[i]['q1'][1])
        else:
            rs = npr.RandomState()
            q1_var_mu_list.append(rs.randn(1, 1)[0][0])
            q1_var_std_list.append(1)
        for k in est_params[i]:
            if k == 'q1':
                continue 
            q2_var_mu_list.append(est_params[i][k]['q2'][0])
            q2_var_std_list.append(est_params[i][k]['q2'][1])
            
    q1_var_mu_list = np.array(q1_var_mu_list)
    q1_var_std_list = np.array(q1_var_std_list)
    
    q2_var_mu_list = np.array(q2_var_mu_list)
    q2_var_std_list = np.array(q2_var_std_list)

    if normarlized:
        def nomralized_func(x):
            return (x - np.mean(x)) / np.std(x)    
    
    def plot_q(q_var, q_err):
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        config = {
                  "font.family" : "Times New Roman",
                  "font.size" : 30
                  }
        rcParams.update(config)
    
        plt.errorbar(np.arange(len(q_var)), q_var, yerr=q_err, # np.zeros(len(q_real))
                      label=r'$\mu_\lambda$',
                      fmt="o:", color='blue', ecolor='black', capsize=3, markersize=5, elinewidth=0.25)
          
        plt.ylabel("Q value")
        plt.xlabel("Scientists")
        plt.legend(frameon=False, loc='upper right', fontsize=25)

    plot_q(q1_var_mu_list, np.exp(q1_var_std_list))
    plot_q(q2_var_mu_list, np.exp(q2_var_std_list))


def sort_aid_cc(yearlynop2cc, beforeyear):
    yearlist = sorted(yearlynop2cc.keys()) # 职业生涯跨度 
    startyear = min(yearlist)              # 发表第一篇文章的时间
    endyear = max(yearlist)                # 发表最后一篇时间
    
    # startyear -> cutyear 计算Q值
    cclist = list()
    for year in yearlist:
        if year <= beforeyear:
            for pid, cc in yearlynop2cc[year]:
                cclist.append(cc)
    return np.array(cclist)
    

# 数据挂载盘
abs_path = "/mnt/disk2/"

# 预处理数据存放路径
file_name = "cs"
process_data_path = abs_path + "EmpiricalData/StatisticalData_cs"

# file_name = "physics"
# process_data_path = abs_path + "EmpiricalData/StatisticalData_physics"

def BBVI_Algorithm_empirical_analysis():
    
    '''(1) 实证数据读取 '''
    # 读取实证数据 --- 由mag_aid.py生成
    with open(os.path.join(process_data_path, "aid_empirical.pkl"), 'rb') as f:
        targeted_aid = pickle.load(f)
    print("待分析作者数目: {}".format(len(targeted_aid)))
    
    # 训练集确定: 机构信息, 作者每篇论文累计引用数目列表
    beforeyear = 1997
    # 确定机构数; 由此确定机构模型参数数目
    org_id2aid = dict()
    for aid in targeted_aid:
        org_id = targeted_aid[aid]['org_id']
        cclist = sort_aid_cc(targeted_aid[aid]['x'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
        if len(cclist) == 0:
            continue
        if org_id not in org_id2aid:
            org_id2aid[org_id] = list()
        org_id2aid[org_id].append(aid)      # 机构下包含那些作者
   
    # 转换成所需要的x_obs格式 (见上述模拟数据生成的x_obs)
    x_obs = dict()
    for i, org_id in enumerate(org_id2aid):
        x_obs[i] = dict()         # 第i个机构
        x_obs[i]['q1'] = ["", ""]
        for j, aid in enumerate(org_id2aid[org_id]):
            cclist = sort_aid_cc(targeted_aid[aid]['x'], beforeyear)
            if len(cclist) == 0:
                continue
            x_obs[i][j] = dict()  # 第j个人
            x_obs[i][j]['q2'] = ""
            x_obs[i][j]['x'] = np.log(cclist + 1)
    # 人数统计 - 经过N1筛选
    count = 0
    for i in x_obs:
        for j in x_obs[i]:
            if j == 'q1':
                continue
            else:
                count += 1
    print("待分析机构数目为: {} \n待分析作者数目: {}".format(len(org_id2aid), count))
    

    '''(2) 估计Q值'''
    # Baseline 均值估计 (mu_1, sig_1) 和 log_sig_P_real 控制
    est_params = max_likelihoood(x_obs)
    evaluate_on_empirical_data(est_params, True)
    
    # 待估计模型参数初始化 & 待估计变分参数初始化:
    model_params_init, var_params_init = init_var_params(est_params)
    # 变分估计
    mp_num = 8
    Epochs = 20
    step_size = 1e-2
    num_iters = 100
    num_samples = 100
    model_params, var_params = model_params_init, var_params_init
    # 
    for e in range(Epochs):
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))           
        E_start_time = time.perf_counter()
        var_params_next = EStep_MP(var_params, x_obs, model_params, step_size, num_iters, num_samples, mp_num)
        E_end_time = time.perf_counter()                      
        print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
        #
        var_params = var_params_next
        
        # M-Step
        print("({}) Optimizing model parameters...".format(e))
        M_start_time = time.perf_counter()
        # model_params_next = MStep_non_MP(var_params, x_obs, model_params, step_size, num_iters, mp_num)
        model_params_next = MStep_MP(var_params, x_obs, model_params, step_size, num_iters, num_samples, mp_num)
        M_end_time = time.perf_counter()
        print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time))
        # 
        model_params = model_params_next
    
    # Instituion Q model 评价结果整理
    model_params_final, var_params_final = model_params, var_params
    for i in model_params_final:
        if i == "P":
            continue
        else:
            var_params_final[i]["q1"] = model_params_final[i]["q1"]
    evaluate_on_empirical_data(var_params_final, True)
    # Q model 评价结果整理
    model_params_est, var_params_est = init_var_params(est_params)
    
    ''' (3) 结果储存 '''
    mu_P = 0               # 假定未0 不影响结果  
    log_sig_P = 0          # 引用范式待估参数
    aid2Q = dict()         # 提出的基于机构能力先验 
    orgid2Q = dict()       # 提出的基于机构能力先验 
    mu_P_WSB = 0           # 假定为0 不影响结果
    log_sig_P_WSB = 0      # 引用范式待估参数
    aid2Q_WSB = dict()     # 王大顺的平均估计(极大似然)
    orgid2Q_WSB = dict()   # 王大顺的平均估计(极大似然) ---> 我们简单将其扩展到机构
    for i, org_id in enumerate(org_id2aid):
        orgid2Q[org_id]     = model_params_final[i]['q1']
        orgid2Q_WSB[org_id] = model_params_est[i]['q1']
        for j, aid in enumerate(org_id2aid[org_id]):
            aid2Q[aid]      = var_params_final[i][j]['q2']
            aid2Q_WSB[aid]  = var_params_est[i][j]['q2']
    log_sig_P = model_params_final["P"][1]
    log_sig_P_WSB = model_params_est["P"][1]
    
    # 依次是随机参数, 作者q2, 机构q1
    results_org = ([mu_P, log_sig_P], aid2Q, orgid2Q)
    results_org_WSB = ([mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB)

    with open("./tmp/results_org_{}.pkl".format(beforeyear), 'wb') as f:
        pickle.dump(results_org, f)
    with open("./tmp/results_org_WSB_{}.pkl".format(beforeyear), 'wb') as f:
        pickle.dump(results_org_WSB, f)