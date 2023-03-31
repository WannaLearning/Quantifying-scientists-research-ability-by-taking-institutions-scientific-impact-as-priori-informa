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
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score 
from scipy.stats import spearmanr, pearsonr, ttest_rel
   


#%%
def sampling_normal(mean, log_std, num_samples):
    rs = npr.RandomState()
    samples = rs.randn(num_samples, 1) * np.exp(log_std) + mean
    return samples


def sampling_poisson(lam, num_samples):
    rs = npr.RandomState()
    samples = rs.poisson(lam, num_samples)
    return samples


def log_p_zx_density(data_tuple, z_tuple, model_params_tuple):
    x_arr, mask_arr                  = data_tuple
    q3_z_arr                         = z_tuple
    model_params_arr, model_params_P = model_params_tuple
    
    mu  = model_params_arr[:, 0][:, np.newaxis, np.newaxis]
    std = np.exp(model_params_arr[:, 1][:, np.newaxis, np.newaxis])
    log_q3_density = norm.logpdf(q3_z_arr, mu, std)
    
    log_P_density = norm.logpdf(x_arr-q3_z_arr, 0, np.exp(model_params_P[1]))
    log_P_density = log_P_density * mask_arr # mask: 遮挡补全为0的数据
    log_P_density = np.sum(log_P_density, axis=-1, keepdims=True)
    
    logpq = np.sum(log_q3_density, axis=0) + np.sum(log_P_density, axis=0)
    return logpq


#%%
def split_data(data, var_params, mp_num):
    ''' 将变分参数 var_params 和 观测数据 x_obs 划分为多块 mp_num '''

    total_org_num = len(data)                          # 机构总数目
    batch_size    = math.ceil(total_org_num / mp_num)  # 每个进程负责 * 个机构的变分参数更新
    idx_chunks    = list()                             # 每个进程负责 * 个机构的变分参数更新
    start         = 0
    end           = 0
    for i in range(mp_num):
        end = min(start + batch_size, total_org_num)
        if start != end:
            idx_chunks.append((start, end))
        start = end
    
    # 数据划分 & 变分参数划分 ---> 多块数据
    data_chunks       = list()
    var_params_chunks = list()
    for start, end in idx_chunks:
        data_mp_i       = dict()
        var_params_mp_i = dict() 
        for i in range(start, end):
            data_mp_i[i]       = data[i]            # 第i个进程需使用的观测数据
            var_params_mp_i[i] = var_params[i]      # 第i个进程需要更新变分参数
        data_chunks.append(data_mp_i)
        var_params_chunks.append(var_params_mp_i)
        
    return data_chunks, var_params_chunks, idx_chunks


def EStep_MP_func(data_mp_i, var_params_mp_i, model_params, 
                  batch_size_org, step_size, num_iters, num_samples, mp_i):
    ''' 第i个进程变分参数更新 '''
    
    
    def variational_objective(q3_var_params_arr, t):
        """Provides a stochastic estimate of the variational lower bound.
           矩阵操作版本
        """
        lower_bound    = 0                                            # 累计的ELBO
        i_org_num      = len(x_list)                                  # 国家i下机构数目
        # batch_size_org = 8                                          # 每次处理的机构数
        loop_num_org   = int(np.ceil(i_org_num / batch_size_org))     # 需要循环的次数
        start          = 0
        end            = 0
        for l in range(loop_num_org):
            # 第l个batch使用的机构观测数据 (论文引用量和论文策略执行情况)
            end        = min(start + batch_size_org, len(x_list))
            x_list_l   = x_list[start:  end]
            
            # 确定该batch_l下机构内最大发文量 - padding mask
            max_nop_list = list()
            for j in range(len(x_list_l)):
                nop_list_j = list()
                for x in x_list_l[j]:
                    nop = len(x)
                    nop_list_j.append(nop)
                max_nop_j  = np.max(nop_list_j)
                max_nop_list.append(max_nop_j)
            max_nop = np.max(max_nop_list)
            
            
            x_arr             = list()
            mask_arr          = list()
            q3_z_arr          = list()
            model_params_arrj = list()
            #
            q3_mu_var         = list()
            q3_log_std_var    = list()
            for j in range(len(x_list_l)):
                # 第j个机构下学者的观测数据
                x_list_j  = x_list_l[j]
                j_        = j + l * batch_size_org              # 算上batchsize后的第j_个机构
                noa_j_beg = int(np.sum(noa_list[: j_]))
                noa_j_end = int(np.sum(noa_list[: j_+1]))
                noa_j     = noa_j_end - noa_j_beg
                # 第j个机构下学者的变分参数
                q3_var_j  = q3_var_params_arr[noa_j_beg: noa_j_end, :]
                # 第j个机构的模型参数
                model_params_j     = np.array(model_params_list[j_])
                model_params_arr_j = np.ones((noa_j, 1)) * model_params_j
                
                 # padding 操作 - 人员数目 * 样本数目 * 最大文章数目
                x_list_pad_j = list()                           # 将机构j人员的发文量补齐为最大值max_nop
                mask_j       = list()                           # 标记那些位置是补齐
                for x in x_list_j:
                    nop = len(x)
                    x         = np.ones((num_samples, 1)) * x
                    x_zeros   = np.zeros((num_samples, max_nop-nop))
                    x_padding = np.concatenate([x, x_zeros], axis=-1)
                    x_list_pad_j.append(x_padding)
                    #
                    mask  = np.concatenate([np.ones(nop), np.zeros(max_nop-nop)])
                    mask  = np.ones((num_samples, 1)) * mask
                    mask_j.append(mask)
                x_arr_j    = np.array(x_list_pad_j)             # 人员数目 * 样本数目 * 最大文章数目
                mask_arr_j = np.array(mask_j)                   # 人员数目 * 样本数目 * 最大文章数目
    
                # 采样隐变量q3, 标准高斯采样, sig * sample + mu 
                q3_z_j           = sampling_normal(0, 0, num_samples * noa_j).reshape((noa_j, num_samples, 1))
                q3_mu_var_j      = q3_var_j[:, 0].reshape((noa_j, 1, 1))
                q3_log_std_var_j = q3_var_j[:, 1].reshape((noa_j, 1, 1))
                q3_z_arr_j       = q3_z_j * np.exp(q3_log_std_var_j) + q3_mu_var_j  # 人员数目 * 样本数目 * 1
                
                x_arr.append(x_arr_j)
                mask_arr.append(mask_arr_j)
                q3_z_arr.append(q3_z_arr_j)
                model_params_arrj.append(model_params_arr_j)
                #
                q3_mu_var.append(q3_mu_var_j)
                q3_log_std_var.append(q3_log_std_var_j)
                
            # 计算ELBO的part1
            x_arr              = np.concatenate(x_arr,              axis=0)   
            mask_arr           = np.concatenate(mask_arr,           axis=0)
            q3_z_arr           = np.concatenate(q3_z_arr,           axis=0)
            model_params_arrj  = np.concatenate(model_params_arrj,  axis=0)
            # 计算ELBO的part2
            q3_mu_var          = np.concatenate(q3_mu_var,          axis=0)
            q3_log_std_var     = np.concatenate(q3_log_std_var,     axis=0)
        
            # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
            data_tuple         = (x_arr, mask_arr)
            z_tuple            = (q3_z_arr)
            model_params_tuple = (model_params_arrj, model_params_P)   
            logp_zx            = log_p_zx_density(data_tuple, z_tuple, model_params_tuple)
            part1              = np.mean(logp_zx)  
            
            logp_q2_z          = norm.logpdf(q3_z_arr, q3_mu_var, np.exp(q3_log_std_var))
            part2              = np.mean(np.sum(logp_q2_z, axis=0))
            
            # 求ELBO最大, 所以这里加个负号即minimize
            lower_bound_j      = part1 - part2
            lower_bound       += lower_bound_j / len(x_list_l)
            start = end
            
        return -lower_bound 
    
    
    # 机构内人员变分参数更新
    q3_var_params_list = list()
    x_list             = list()
    noa_list           = list()                 # 每个机构下人数
    model_params_list  = list()
    model_params_P     = model_params['P']      # 领域引用范式参数(luck) - 模型参数
    for j in data_mp_i:                                  
        data_j         = data_mp_i[j]           # 机构j的观测数据
        var_params_j   = var_params_mp_i[j]     # 机构j下人员的变分参数
        model_params_j = model_params[j]['q2']  # 机构j的模型参数        - 模型参数
        
        # 将 机构i内 变分参数 和 观测数据 提取
        q3_var_params_list_j = list()           # 机构j的所有作者的变分参数
        x_list_j             = list()           # 机构j的所有作者的观测数据
        for k in data_j:                       
            if k == 'q2':
                continue
            else:
               var_params_q3_j_k = var_params_j[k]['q3']
               x_j_k             = data_j[k]['x']
               q3_var_params_list_j.append(var_params_q3_j_k)
               x_list_j.append(x_j_k)
        
        q3_var_params_list.append(np.array(q3_var_params_list_j))
        x_list.append(x_list_j)
        noa_list.append(len(x_list_j))
        model_params_list.append(model_params_j)
        
    # 更新变分参数 q3
    q3_var_params_arr = np.concatenate(q3_var_params_list, axis=0)   
    # 梯度下降更新变分参数
    gradient               = grad(variational_objective)
    q3_var_params_arr_next = adam(gradient, q3_var_params_arr, step_size=step_size, num_iters=num_iters)
 
    # 变分参数更新
    for j_, j in enumerate(data_mp_i):                   
        data_j = data_mp_i[j]
        
        noa_j_beg = int(np.sum(noa_list[: j_]))
        noa_j_end = int(np.sum(noa_list[: j_+1]))
        noa_j     = noa_j_end - noa_j_beg
        q3_var_params_next_j = q3_var_params_arr_next[noa_j_beg: noa_j_end, :]
        
        for k in data_j:                       
            if k == 'q2':
                continue
            else:  
                var_params_mp_i[j][k]['q3'] = q3_var_params_next_j[k, :]

    return var_params_mp_i


def EStep_MP(data, var_params, model_params,
             batch_size_org, step_size, num_iters, num_samples, mp_num):
    '''启用多进程进行变分参数更新'''
    
    # 将观测数据 和 变分参数 划分成多块
    data_chunks, var_params_chunks, idx_chunks = split_data(data, var_params, mp_num) 
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=len(idx_chunks))
    results = list()
    for mp_i in range(len(idx_chunks)):
        data_mp_i       = data_chunks[mp_i]
        var_params_mp_i = var_params_chunks[mp_i]
        results.append(pool.apply_async(EStep_MP_func, (data_mp_i, var_params_mp_i, model_params,
                                                        batch_size_org, step_size, num_iters, num_samples, mp_i, )))
    pool.close()
    pool.join()
    
    # 合并多进程的变分参数更新结果, 赋值操作
    for res in results:
        var_params_mp_i = res.get()
        for j in var_params_mp_i:
            for k in var_params_mp_i[j]:
                if k == 'q2':
                    continue
                var_params[j][k]['q3'] = var_params_mp_i[j][k]['q3']
                
    return var_params


#%%
def MStep_MP_func(data_mp_i, var_params_mp_i, model_params, 
                  batch_size_org, step_size, num_iters, num_samples, mp_i):
    ''' 第i个进程变分参数更新 '''
    
    def variational_objective(model_params_arr, t):
        """Provides a stochastic estimate of the variational lower bound.
           矩阵操作版本
        """
        model_params_P = model_params_arr[-1, :]
        
        lower_bound    = 0                                            # 累计的ELBO
        i_org_num      = len(x_list)                                  # 国家i下机构数目
        # batch_size_org = 8                                           # 每次处理的机构数
        loop_num_org   = int(np.ceil(i_org_num / batch_size_org))     # 需要循环的次数
        start          = 0
        end            = 0
        for l in range(loop_num_org):
            # 第l个batch使用的机构观测数据 (论文引用量和论文策略执行情况)
            end        = min(start + batch_size_org, len(x_list))
            x_list_l   = x_list[start:  end]
            
            # 确定该batch_l下机构内最大发文量 - padding mask
            max_nop_list = list()
            for j in range(len(x_list_l)):
                nop_list_j = list()
                for x in x_list_l[j]:
                    nop = len(x)
                    nop_list_j.append(nop)
                max_nop_j  = np.max(nop_list_j)
                max_nop_list.append(max_nop_j)
            max_nop = np.max(max_nop_list)
            
            x_arr             = list()
            mask_arr          = list()
            q3_z_arr          = list()
            model_params_arrj = list()
            q3_mu_var         = list()
            q3_log_std_var    = list()
            for j in range(len(x_list_l)):
                # 第j个机构下学者的观测数据
                x_list_j  = x_list_l[j]
                j_        = j + l * batch_size_org               # 算上batchsize后的第j_个机构
                noa_j_beg = int(np.sum(noa_list[: j_]))
                noa_j_end = int(np.sum(noa_list[: j_+1]))
                noa_j     = noa_j_end - noa_j_beg
                # 第j个机构下学者的变分参数
                q3_var_j  = q3_var_params_arr[noa_j_beg: noa_j_end, :]
                # 第j个机构的模型参数
                model_params_j     = model_params_arr[j_, :]
                model_params_arr_j = np.ones((noa_j, 1)) * model_params_j
                
                 # padding 操作 - 人员数目 * 样本数目 * 最大文章数目
                x_list_pad_j = list()                           # 将机构j人员的发文量补齐为最大值max_nop
                mask_j       = list()                           # 标记那些位置是补齐
                for x in x_list_j:
                    nop = len(x)
                    x         = np.ones((num_samples, 1)) * x
                    x_zeros   = np.zeros((num_samples, max_nop-nop))
                    x_padding = np.concatenate([x, x_zeros], axis=-1)
                    x_list_pad_j.append(x_padding)
                    #
                    mask  = np.concatenate([np.ones(nop), np.zeros(max_nop-nop)])
                    mask  = np.ones((num_samples, 1)) * mask
                    mask_j.append(mask)
                x_arr_j    = np.array(x_list_pad_j)             # 人员数目 * 样本数目 * 最大文章数目
                mask_arr_j = np.array(mask_j)                   # 人员数目 * 样本数目 * 最大文章数目
    
                # 采样隐变量q3, 标准高斯采样, sig * sample + mu 
                q3_z_j           = sampling_normal(0, 0, num_samples * noa_j).reshape((noa_j, num_samples, 1))
                q3_mu_var_j      = q3_var_j[:, 0].reshape((noa_j, 1, 1))
                q3_log_std_var_j = q3_var_j[:, 1].reshape((noa_j, 1, 1))
                q3_z_arr_j       = q3_z_j * np.exp(q3_log_std_var_j) + q3_mu_var_j  # 人员数目 * 样本数目 * 1
                
                x_arr.append(x_arr_j)
                mask_arr.append(mask_arr_j)
                q3_z_arr.append(q3_z_arr_j)
                model_params_arrj.append(model_params_arr_j)
                #
                q3_mu_var.append(q3_mu_var_j)
                q3_log_std_var.append(q3_log_std_var_j)
                
            # 计算ELBO的part1
            x_arr              = np.concatenate(x_arr,              axis=0)   
            mask_arr           = np.concatenate(mask_arr,           axis=0)
            q3_z_arr           = np.concatenate(q3_z_arr,           axis=0)
            model_params_arrj  = np.concatenate(model_params_arrj,   axis=0)
            # 计算ELBO的part2
            q3_mu_var          = np.concatenate(q3_mu_var,          axis=0)
            q3_log_std_var     = np.concatenate(q3_log_std_var,     axis=0)
        
            # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
            data_tuple         = (x_arr, mask_arr)
            z_tuple            = (q3_z_arr)
            model_params_tuple = (model_params_arrj, model_params_P)   
            logp_zx            = log_p_zx_density(data_tuple, z_tuple, model_params_tuple)
            part1              = np.mean(logp_zx)  
            
            logp_q2_z          = norm.logpdf(q3_z_arr, q3_mu_var, np.exp(q3_log_std_var))
            part2              = np.mean(np.sum(logp_q2_z, axis=0))
            
            # 求ELBO最大, 所以这里加个负号即minimize
            lower_bound_j      = part1 - part2
            lower_bound       += lower_bound_j / len(x_list_l)
            start = end
            
        return -lower_bound 
    

    # 机构内人员变分参数更新
    q3_var_params_list = list()
    x_list             = list()
    noa_list           = list()                 # 每个机构下人数
    model_params_list  = list()
    model_params_P     = model_params['P']      # 领域引用范式参数(luck) - 模型参数
    for j in data_mp_i:                                  
        data_j         = data_mp_i[j]           # 机构j的观测数据
        var_params_j   = var_params_mp_i[j]     # 机构j下人员的变分参数
        model_params_j = model_params[j]['q2']  # 机构j的模型参数        - 模型参数
        
        # 将 机构i内 变分参数 和 观测数据 提取
        q3_var_params_list_j = list()           # 机构j的所有作者的变分参数
        x_list_j             = list()           # 机构j的所有作者的观测数据
        for k in data_j:                       
            if k == 'q2':
                continue
            else:
               var_params_q3_j_k = var_params_j[k]['q3']
               x_j_k             = data_j[k]['x']
               q3_var_params_list_j.append(var_params_q3_j_k)
               x_list_j.append(x_j_k)
        
        q3_var_params_list.append(np.array(q3_var_params_list_j))
        x_list.append(x_list_j)
        noa_list.append(len(x_list_j))
        model_params_list.append(model_params_j)
        
    # 更新变分参数 q3
    q3_var_params_arr = np.concatenate(q3_var_params_list, axis=0)   
    
    model_params_arr  = np.array(model_params_list)
    model_params_P    = np.array([model_params_P])
    model_params_arr  = np.concatenate([model_params_arr, model_params_P], axis=0)
    
    # 梯度下降更新模型参数
    gradient              = grad(variational_objective)
    model_params_arr_next = adam(gradient, model_params_arr, step_size=step_size, num_iters=num_iters)
 
    # 模型参数更新
    model_params_P_next   = model_params_arr_next[-1, :]
    model_params['P']     = model_params_P_next
    for j_, j in enumerate(data_mp_i):                   
        model_params[j]['q2'] = model_params_arr_next[j_, :]
      
    return model_params


def MStep_MP(data, var_params, model_params,
             batch_size_org, step_size, num_iters, num_samples, mp_num):
    '''启用多进程进行变分参数更新'''
    
    # 将观测数据 和 变分参数 划分成多块
    data_chunks, var_params_chunks, idx_chunks = split_data(data, var_params, mp_num) 
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=len(idx_chunks))
    results = list()
    for mp_i in range(len(idx_chunks)):
        data_mp_i       = data_chunks[mp_i]
        var_params_mp_i = var_params_chunks[mp_i]
        results.append(pool.apply_async(MStep_MP_func, (data_mp_i, var_params_mp_i, model_params,
                                                        batch_size_org, step_size, num_iters, num_samples, mp_i, )))
    pool.close()
    pool.join()
    
    # 模型参数更新
    total_aid_num_mp_i   = list()
    model_params_P_mp_i  = list()
    for idx, res in zip(idx_chunks, results):
        beg, end = idx
        model_params_mp_i = res.get()
        for j in range(beg, end):
            # 统计机构j内人数
            noa_j = 0
            for k in data[j]:
                if k != "q2":
                    noa_j += 1
            # 直接更新机构模型参数
            model_params[j] = model_params_mp_i[j]
            total_aid_num_mp_i.append(noa_j)
            model_params_P_mp_i.append(model_params_mp_i['P'])
    
    ratio               = np.array(total_aid_num_mp_i) / np.sum(total_aid_num_mp_i)
    ratio               = ratio.reshape((-1, 1))
    model_params_P_mp_i = np.array(model_params_P_mp_i)
    model_params_P_next = np.sum(ratio * model_params_P_mp_i, axis=0) 
    model_params['P']   = model_params_P_next
                
    return model_params


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
    mu_org_list      = list()
    log_sig_org_list = list()
    for i in model_params_real:
        if i == "P":
            mu_P_real, log_sig_P_real = model_params_real[i] 
        else:
            mu_org_list.append(model_params_real[i]['q2'][0])
            log_sig_org_list.append(model_params_real[i]['q2'][1])
 
    # 开始抽样
    data   = dict()
    org_id = -1
    for mu_org_i, log_sig_org_i in zip(mu_org_list, log_sig_org_list):
        org_id             += 1
        data[org_id]       = dict()
        data[org_id]['q2'] = np.array([mu_org_i, log_sig_org_i])
        aid_id = -1
        # 第 i 个机构中抽取人员能力
        aid_num_i = max(sampling_poisson(aid_num, 1), [1])[0]
        q3_i_list  = sampling_normal(mu_org_i, log_sig_org_i, aid_num_i).squeeze()
        for q_i_j in q3_i_list:
            aid_id                     += 1
            data[org_id][aid_id]       = dict()
            data[org_id][aid_id]['q3'] = q_i_j
            # 抽取人员的发文量 ~ 泊松分布 (至少一篇文章)
            nop_num_i = max(sampling_poisson(nop_num, 1), [1])[0]
            # 第 i 个机构中第j个人抽样文章引用数目 ~ 正态分布
            p_i_j_list = sampling_normal(mu_P_real, log_sig_P_real, nop_num_i).squeeze()
            x          = q_i_j + p_i_j_list
            if nop_num_i > 1:
                data[org_id][aid_id]['x'] = x
            else:
                data[org_id][aid_id]['x'] = np.array([x])
    return data


def max_likelihoood(data):
    # 极大似然估计 - 均值估计
    # 忽略先验分布, 是我们的Baseline
    x_obs = list()
    for i in data:
        for j in data[i]:
           if j == "q2":
               continue
           x  = data[i][j]['x']
           x_obs.append(x)
    x_obs = np.concatenate(x_obs, axis=0)
    std   = np.std(x_obs)
    
    model_params_init  = dict()
    var_params_init    = dict()
    log_sig_P_est_list = list()
    for i in data:
        model_params_init[i] = dict()
        var_params_init[i]   = dict()
        q2_est_list          = list()
        for j in data[i]:
           if j == "q2":
               continue
           var_params_init[i][j] = dict()
           x                     = data[i][j]['x']
           q3_mu_est             = np.mean(x)
           q3_std_est            = np.log(max(std - np.std(x), 1e-2))
           var_params_init[i][j]['q3'] = np.array([q3_mu_est, q3_std_est])
           q2_est_list.append(q3_mu_est)
           log_sig_P_est_list.append(q3_std_est)
        
        model_params_init[i]['q2'] = [np.mean(q2_est_list), np.log(max(np.std(q2_est_list), 1e-2))]
        
    model_params_init["P"] = np.array([0, np.mean(log_sig_P_est_list)])
    return var_params_init, model_params_init


def evaluate_on_simulation_data(data, model_params_real, var_params_init, model_params_init):
    
    def nomralized_func(x):
        return (x - np.mean(x)) / np.std(x)    
    
    def evaluate_real2pred(Y, X):
        # 评价指标: 
        cor, pvalue = pearsonr(Y, X)
        rmse = np.sqrt(mean_squared_error(Y, X))
        mae = mean_absolute_error(Y, X)
        r2  = r2_score(Y, X)
        return cor, rmse, mae, r2
        
    def plot_q(q_real, q_var, q_err, xlabel, ylabel, legend_1, legend_2, title):
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        config = {
                  "font.family" : 'SimHei', # Times New Roman  #SimHei
                  "font.size" : 20
                  }
        rcParams.update(config)
        plt.rcParams['axes.unicode_minus'] = False # SimHei 字体符号不正常显示
        
        fontsize = 22
        
        plt.plot(np.arange(len(q_real)) +1, q_real, 
                 label=legend_1, c='red', marker='s', alpha=0.5, linewidth=1)
        plt.errorbar(np.arange(len(q_real))+1, q_var, yerr=q_err, 
                     label=legend_2, fmt="o:", color='blue', ecolor='dimgray', capsize=5, markersize=5, elinewidth=0.8)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.legend(frameon=False, loc='upper right', fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.yticks(np.arange(-6, 7, 2))
        # plt.ylim(-15, 15)
    
    q1_real_mu_list, q1_real_std_list = list(), list()
    q1_var_mu_list,  q1_var_std_list  = list(), list()

    q2_real_mu_list                   = list()
    q2_var_mu_list,  q2_var_std_list  = list(), list()
    
    for j in data:
        q1_real_mu_list.append(model_params_real[j]['q2'][0])     # 均值和方差模型参数
        q1_real_std_list.append(model_params_real[j]['q2'][1])
        
        q1_var_mu_list.append(model_params_init[j]['q2'][0])
        q1_var_std_list.append(model_params_init[j]['q2'][1])
         
        for k in data[j]:
            if k == 'q2':
                continue 
            q2_real_mu_list.append(data[j][k]['q3']) 
            
            q2_var_mu_list.append(var_params_init[j][k]['q3'][0])
            q2_var_std_list.append(var_params_init[j][k]['q3'][1])
            
    q1_real_mu_list  = np.array(q1_real_mu_list)
    q1_real_std_list = np.array(q1_real_std_list)
    q1_var_mu_list   = np.array(q1_var_mu_list)
    q1_var_std_list  = np.array(q1_var_std_list)
    q2_real_mu_list  = np.array(q2_real_mu_list)
    q2_var_mu_list   = np.array(q2_var_mu_list)
    q2_var_std_list  = np.array(q2_var_std_list)

    # plot_q(q1_real_mu_list, q1_var_mu_list, np.exp(q1_var_std_list), "Institutions", r'$\mu_\lambda$', "Real", "Estimated")
    # plot_q(q2_real_mu_list, q2_var_mu_list, np.exp(q2_var_std_list), "Scientists",   r"$\hat{Q}$",     "Real", "Estimated")
    plot_q(q1_real_mu_list, q1_var_mu_list, np.exp(q1_var_std_list), "机构", "机构科研能力", "真实值", "估计值", "融合机构信息的科研能力量化模型 (模拟配置1)")
    plot_q(q2_real_mu_list, q2_var_mu_list, np.exp(q2_var_std_list), "学者", "学者科研能力", "真实值", "估计值", "融合机构信息的科研能力量化模型 (模拟配置1)")
    
    
    cor_q1, rmse_q1, mae_q1, r2_q1 = evaluate_real2pred(q1_real_mu_list, q1_var_mu_list)
    cor_q2, rmse_q2, mae_q2, r2_q2 = evaluate_real2pred(q2_real_mu_list, q2_var_mu_list)
    
    tb = pt.PrettyTable()
    tb.field_names = ["", "Pearsonr", "R2", "RMSE", "MAE"]
    tb.add_row(["机构", "{:.4f}".format(cor_q1), "{:.4f}".format(r2_q1), "{:.4f}".format(rmse_q1), "{:.4f}".format(mae_q1)])
    tb.add_row(["学者", "{:.4f}".format(cor_q2), "{:.4f}".format(r2_q2), "{:.4f}".format(rmse_q2), "{:.4f}".format(mae_q2)])
    print(tb)
    
    return (cor_q1, r2_q1, rmse_q1, mae_q1), (cor_q2, r2_q2, rmse_q2, mae_q2)


#%%
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
    mu_1, sig_1 = 0., -1.
    
    # 机构模型参数: 
    org_num           = 25            # 机构数目
    aid_num           = 20            # 人员数目
    nop_num           = 10            # 人均文章数目, 服从泊松分布
    mu_org_list       = list(sampling_normal(mu_0, sig_0, org_num).squeeze())
    log_sig_org_list  = list(sampling_normal(mu_1, sig_1, org_num).squeeze())
    model_params_real = dict()
    for i in range(org_num): 
        model_params_real[i] = dict()
        model_params_real[i]['q2'] = [mu_org_list[i], log_sig_org_list[i]]
    mu_P_real         = 0            # 随机波动模型参数
    log_sig_P_real    = 0.5
    model_params_real["P"] = [mu_P_real, log_sig_P_real]
    # 采样模拟数据
    sampling_params = [org_num, aid_num, nop_num]
    data            = create_simulation_data(model_params_real, sampling_params)
    # 极大似然估计 (基准模型)
    var_params_init, model_params_init = max_likelihoood(data)

    # 变分估计
    mp_num         = 5
    Epochs         = 10
    step_size      = 1e-1
    batch_size_org = 512
    num_iters      = 100
    num_samples    = 1
    model_params   = model_params_init
    var_params     = var_params_init
    for e in range(Epochs):
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))           
        E_start_time    = time.perf_counter()
        var_params_next = EStep_MP(data, var_params, model_params, batch_size_org, step_size, num_iters, num_samples, mp_num)
        E_end_time      = time.perf_counter()                      
        var_params      = var_params_next
        print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
        
        # M-Step
        print("({}) Optimizing model parameters...".format(e))
        M_start_time      = time.perf_counter()
        model_params_next = MStep_MP(data, var_params, model_params, batch_size_org, step_size, num_iters, num_samples, mp_num)
        M_end_time        = time.perf_counter()
        model_params      = model_params_next
        print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time)) 
    # 变分参数估计 - BBVI-EM
    var_params_bbvi, model_params_bbvi = var_params, model_params
    var_params_init, model_params_init = max_likelihoood(data)
    
    # 模型结果评价
    q1_WSB, q2_WSB = evaluate_on_simulation_data(data, model_params_real, var_params_init, model_params_init)
    q1_ORG, q2_ORG = evaluate_on_simulation_data(data, model_params_real, var_params_bbvi, model_params_bbvi)
    
    return (q1_WSB, q2_WSB), (q1_ORG, q2_ORG)
        
        
def main():
    # 将模拟数据数据运行10次, 比较结果
    times   = 10
    number  = 1
    results = dict()
    for t in range(times):
        (q1_WSB, q2_WSB), (q1_ORG, q2_ORG) = BBVI_Algorithm()
        results[t] = dict()
        results[t]["wsb"] = (q1_WSB, q2_WSB)
        results[t]["org"] = (q1_ORG, q2_ORG)

    # 计算每个指标的均值; t检验
    def get_list(results, Key):
        q1_eval_list = list()
        q2_eval_list = list()
        for t in results:
            q1_eval, q2_eval, q3_eval = results[t][Key]
            q1_eval_list.append(list(q1_eval))
            q2_eval_list.append(list(q2_eval))
        q1_eval_list = np.maximum(np.array(q1_eval_list), 0)  # R2可为负, 置0
        q2_eval_list = np.maximum(np.array(q2_eval_list), 0)
        return q1_eval_list, q2_eval_list
    
    q1_WSB_list, q2_WSB_list = get_list(results, "wsb")
    q1_ORG_list, q2_ORG_list = get_list(results, "org")
    
    # 机构科研能力评价
    q1_ORG_mean = np.mean(q1_ORG_list, axis=0)
    q1_ORG_std  = np.std(q1_ORG_list,  axis=0)
    print("OUR: Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q1_ORG_mean))
    
    # 学者科研能力评价
    q2_WSB_mean = np.mean(q2_WSB_list, axis=0)
    q2_ORG_mean = np.mean(q2_ORG_list, axis=0)
    print("WSB: Pearsonr: {:.4f}, R2: {:.4f}, RMES: {:.4f}, MAE: {:.4f}".format(*q2_WSB_mean))
    print("OUR: Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q2_ORG_mean))
    for i in range(4):
        _, pvalue = ttest_rel(q2_WSB_list[:, i], q2_ORG_list[:, i])
        print("{:.6f}".format(pvalue))
