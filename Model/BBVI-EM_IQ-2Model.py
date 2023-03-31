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
import time
import math
import multiprocessing
import prettytable as pt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr, ttest_rel

from MyQPModel import bbvi_em_org


#%%
def sampling_normal(mean, log_std, num_samples):
    # 抽取正态分布
    rs = npr.RandomState()
    samples = rs.randn(num_samples, 1) * np.exp(log_std) + mean
    return samples


def sampling_poisson(lam, num_samples):
    # 抽取泊松分布
    rs = npr.RandomState()
    samples = rs.poisson(lam, num_samples)
    return samples


def logp_zx_density(data_tuple, z_tuple, model_params_tuple):
    # x: 观测, z: 隐变量, model_params: 模型参数
    x_arr_j, mask_arr_1_j, mask_arr_2_j           = data_tuple
    q2_mu_z_arr_j, q2_log_sig_z_arr_j, q3_z_arr_j = z_tuple
    model_params_i, model_params_P                = model_params_tuple
    
    # P(Q2_mu|Q1); P(Q2_log_sig|Q1) - 国家抽取机构  (需要mask, 因为机构只生成一次)
    log_q2_density1 = norm.logpdf(q2_mu_z_arr_j,      model_params_i[0, 0], np.exp(model_params_i[0, 1]))
    log_q2_density2 = norm.logpdf(q2_log_sig_z_arr_j, model_params_i[1, 0], np.exp(model_params_i[1, 1]))
    log_q2_density1 = log_q2_density1 * mask_arr_1_j
    log_q2_density2 = log_q2_density2 * mask_arr_1_j
    
    # P(Q3|Q2_mu, Q2_log_sig) - 机构抽取人员
    log_q3_density  = norm.logpdf(q3_z_arr_j, q2_mu_z_arr_j, np.exp(q2_log_sig_z_arr_j))
    
    # P(X|Q3) - 人员抽取文章 (需要mask, 因为文章数目被补齐为max_nop)
    log_P_density   = norm.logpdf(x_arr_j-q3_z_arr_j, 0, np.exp(model_params_P[1]))
    log_P_density   = log_P_density * mask_arr_2_j
    log_P_density   = np.sum(log_P_density, axis=-1, keepdims=True)
    
    logpq = np.sum(log_q2_density1, axis=0) + np.sum(log_q2_density2, axis=0) +\
            np.sum(log_q3_density, axis=0)  + np.sum(log_P_density, axis=0)
    return logpq


#%%
def split_data(data, var_params, mp_num):
    '''
    将变分参数和观测数据划分为多块
    '''
    total_num   = len(data)                        # 国家数目
    batch_size  = math.ceil(total_num / mp_num)    # 每个进程负责 * 个国家的变分参数更新
    idx_chunks  = list()                        
    start, end  = 0, 0
    
    for i in range(mp_num):
        end = min(start + batch_size, total_num)
        if start != end:
            idx_chunks.append((start, end))
        start = end
    
    # 数据划分 & 变分参数划分 -> 多块数据
    data_chunks       = list()
    var_params_chunks = list()
    for start, end in idx_chunks:
        data_mp_i       = dict()
        var_params_mp_i = dict() 
        for i in range(start, end):
            data_mp_i[i]       = data[i]             # 第i个进程需使用的观测数据
            var_params_mp_i[i] = var_params[i]       # 第i个进程需要更新变分参数
            
        data_chunks.append(data_mp_i)
        var_params_chunks.append(var_params_mp_i)
        
    return data_chunks, var_params_chunks, idx_chunks


def EStep_MP_func(data_mp_i, var_params_mp_i, model_params, 
                  batch_size_org, step_size, num_iters, num_samples, mp_i):
    # 第i个进程变分参数更新
    
    def variational_objective(q_var_params_arr, t):
        """Provides a stochastic estimate of the variational lower bound."""
        q2_var_params_arr = q_var_params_arr[:q2_len].reshape(q2_shape)
        q3_var_params_arr = q_var_params_arr[q2_len:].reshape(q3_shape)
        
        lower_bound    = 0                                            # 累计的ELBO
        i_org_num      = len(x_list)                                  # 国家i下机构数目
        # batch_size_org = 64                                         # 每次处理的机构数
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
            
            x_arr              = list()
            mask_arr_1         = list()
            mask_arr_2         = list()
            #
            q2_mu_z_arr        = list() 
            q2_log_sig_z_arr   = list()
            q3_z_arr           = list()
            #
            q2_mu_z            = list()
            q2_log_sig_z       = list()
            q2_var             = list()
            q3_mu_var_arr      = list()
            q3_log_std_var_arr = list()
            for j in range(len(x_list_l)):
                # 第j个机构的观测数目
                x_list_j  = x_list_l[j]
                j_         = j + l * batch_size_org               # 算上batchsize后的第j_个机构
                noa_j_beg  = int(np.sum(noa_list[: j_]))
                noa_j_end  = int(np.sum(noa_list[: j_+1]))
                noa_j      = noa_j_end - noa_j_beg
                # 第j个机构下的变分参数 (机构变分参数 和 机构下学者的变法参数)
                q2_var_j  = q2_var_params_arr[j, :, :]
                q3_var_j  = q3_var_params_arr[noa_j_beg: noa_j_end, :]
            
                # padding 操作 - 人员数目 * 样本数目 * 最大文章数目
                x_list_pad_j = list()  # 将机构j人员的发文量补齐为最大值max_nop
                mask_2_j     = list()  # 标记那些位置是补齐
                for x in x_list_j:
                    nop = len(x)
                    x         = np.ones((num_samples, 1)) * x
                    x_zeros   = np.zeros((num_samples, max_nop-nop))
                    x_padding = np.concatenate([x, x_zeros], axis=-1)
                    x_list_pad_j.append(x_padding)
                    #
                    mask  = np.concatenate([np.ones(nop), np.zeros(max_nop-nop)])
                    mask  = np.ones((num_samples, 1)) * mask
                    mask_2_j.append(mask)
                x_arr_j      = np.array(x_list_pad_j)
                mask_arr_2_j = np.array(mask_2_j)
            
                # 采样机构隐变量q2_z
                q2_mu_z_j      = sampling_normal(q2_var_j[0][0], q2_var_j[0][1], num_samples)
                q2_log_sig_z_j = sampling_normal(q2_var_j[1][0], q2_var_j[1][1], num_samples)
                
                # 采样人员隐变量q3_z ~ q(z) - 人员数目(noa) x 样本数(number of samples) x 1
                q3_z_j           = sampling_normal(0, 0, num_samples * noa_j).reshape((noa_j, num_samples, 1))
                q3_mu_var_j      = q3_var_j[:, 0].reshape((noa_j, 1, 1))
                q3_log_std_var_j = q3_var_j[:, 1].reshape((noa_j, 1, 1))
                q3_z_arr_j       = q3_z_j * np.exp(q3_log_std_var_j) + q3_mu_var_j
                
                # 针对noa个人员, 重复使用q2_z样本
                q2_mu_z_arr_j      = np.ones((noa_j, 1, 1)) * q2_mu_z_j
                q2_log_sig_z_arr_j = np.ones((noa_j, 1, 1)) * q2_log_sig_z_j
                mask_arr_1_j       = np.concatenate([np.ones((1, num_samples, 1)), np.zeros((noa_j-1, num_samples, 1))], axis=0)
                
                x_arr.append(x_arr_j) 
                mask_arr_1.append(mask_arr_1_j)
                mask_arr_2.append(mask_arr_2_j)
                #
                q2_mu_z_arr.append(q2_mu_z_arr_j)
                q2_log_sig_z_arr.append(q2_log_sig_z_arr_j)
                q3_z_arr.append(q3_z_arr_j)
                # 
                q2_mu_z.append(q2_mu_z_j)
                q2_log_sig_z.append(q2_log_sig_z_j)
                q2_var.append(q2_var_j[np.newaxis, :, :])
                q3_mu_var_arr.append(q3_mu_var_j)
                q3_log_std_var_arr.append(q3_log_std_var_j)
            
            # 计算ELBO的part1
            x_arr              = np.concatenate(x_arr,              axis=0)
            mask_arr_1         = np.concatenate(mask_arr_1,         axis=0)
            mask_arr_2         = np.concatenate(mask_arr_2,         axis=0)
            # 
            q2_mu_z_arr        = np.concatenate(q2_mu_z_arr,        axis=0)
            q2_log_sig_z_arr   = np.concatenate(q2_log_sig_z_arr,   axis=0)
            q3_z_arr           = np.concatenate(q3_z_arr,           axis=0)
            # 计算ELBO的part2
            q2_mu_z            = np.concatenate(q2_mu_z,            axis=1)
            q2_log_sig_z       = np.concatenate(q2_log_sig_z,       axis=1)
            q2_var             = np.concatenate(q2_var,             axis=0)
            q3_mu_var_arr      = np.concatenate(q3_mu_var_arr,      axis=0)
            q3_log_std_var_arr = np.concatenate(q3_log_std_var_arr, axis=0)

            # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
            data_tuple         = (x_arr, mask_arr_1, mask_arr_2)
            z_tuple            = (q2_mu_z_arr, q2_log_sig_z_arr, q3_z_arr)
            model_params_tuple = (model_params_i, model_params_P)
            logp_zx            = logp_zx_density(data_tuple, z_tuple, model_params_tuple)
            part1              = np.mean(logp_zx)
            
            logp_q2_mu_z       = norm.logpdf(q2_mu_z,      q2_var[:, 0, 0], np.exp(q2_var[:, 0, 1]))
            logp_q2_log_sig_z  = norm.logpdf(q2_log_sig_z, q2_var[:, 1, 0], np.exp(q2_var[:, 1, 1]))
            logp_q3_z          = norm.logpdf(q3_z_arr,     q3_mu_var_arr,   np.exp(q3_log_std_var_arr))
            part2              = np.mean(np.sum(logp_q2_mu_z,      axis=1, keepdims=True) +\
                                         np.sum(logp_q2_log_sig_z, axis=1, keepdims=True) +\
                                         np.sum(logp_q3_z,         axis=0))
            # 求ELBO最大, 所以这里加个负号即minimize
            lower_bound_j = part1 - part2
            lower_bound  += lower_bound_j / len(x_list_l)
            start = end
                
        return -lower_bound 
    
    
    ''' 国家i循环 '''
    for i in data_mp_i:                                  
        # 国家
        data_i         = data_mp_i[i]           # 国家i下所有观测数据
        var_params_i   = var_params_mp_i[i]      # 国家i下所有变分参数
        model_params_i = model_params[i]["q1"]   # 国家i的模型参数
        model_params_P = model_params['P']       # 统一随机性模型参数
        
        # 逐个机构更新q3人员变分参数 - q2机构变分参数随后更新
        q2_var_params_list = list()
        q3_var_params_list = list()
        x_list             = list()
        noa_list           = list()              # 每个机构下人数
        for j in data_i:
            # 机构
            if j == "q1":
                continue
            else:
                data_i_j       = data_i[j]       # 机构j下所有观测数据
                var_params_i_j = var_params_i[j] # 机构j下所有变分参数
                q2_i_j = var_params_i_j["q2"]    # 机构变分参数
                q2_var_params_list.append(q2_i_j)
                
            # 逐个机构更新q3人员变分参数
            q3_var_params_list_j = list()
            x_list_j             = list()        # 机构下人员的观测数据x
            for k in data_i_j:
                if k == "q2":
                    continue
                else:
                    x        = data_i_j[k]["x"]         # 人员k的观测数据
                    q3_i_j_k = var_params_i_j[k]["q3"]  # 人员k的变分参数
                    q3_var_params_list_j.append(q3_i_j_k)
                    x_list_j.append(x)
            q3_var_params_list.append(np.array(q3_var_params_list_j))
            x_list.append(x_list_j)
            noa_list.append(len(x_list_j))
            
        # 同时更新q2和q3变分参数
        q2_var_params_arr = np.array(q2_var_params_list)
        q3_var_params_arr = np.concatenate(q3_var_params_list, axis=0)
        
        q2_shape          = q2_var_params_arr.shape
        q3_shape          = q3_var_params_arr.shape
        q2_var_params_arr = q2_var_params_arr.flatten()
        q3_var_params_arr = q3_var_params_arr.flatten()
        q2_len            = len(q2_var_params_arr)
        q3_len            = len(q3_var_params_arr)
        q_var_params_arr  = np.concatenate([q2_var_params_arr, q3_var_params_arr], axis=0)
        
        # 梯度下降
        gradient              = grad(variational_objective)
        q_var_params_arr_next = adam(gradient, q_var_params_arr, step_size=step_size, num_iters=num_iters)
        q2_var_params_next    = q_var_params_arr_next[:q2_len].reshape(q2_shape)
        q3_var_params_next    = q_var_params_arr_next[q2_len:].reshape(q3_shape)
        
        for j in data_i: 
            if j == "q1":
                continue
            else:
                # 更新机构j变分参数
                var_params_mp_i[i][j]["q2"] = q2_var_params_next[j]
                # 机构j内所有人员的变分参数
                noa_j_beg            = int(np.sum(noa_list[: j]))
                noa_j_end            = int(np.sum(noa_list[: j+1]))
                q3_var_params_next_j = q3_var_params_next[noa_j_beg: noa_j_end, :]
                
                for k in data_i[j]:
                    if k == 'q2':
                        continue
                    else:
                        # 更新人员k变分参数
                        var_params_mp_i[i][j][k]['q3'] = q3_var_params_next_j[k, :]
    
    return var_params_mp_i


def EStep_MP(data, var_params, model_params,
             batch_size_org, step_size, num_iters, num_samples, mp_num):
    '''启用多进程进行变分参数更新'''
    
    # 将观测数据 和 变分参数 划分成多块
    data_chunks, var_params_chunks, idx_chunks = split_data(data, var_params, mp_num) 
    
    # 创建进程池
    pool    = multiprocessing.Pool(processes=len(idx_chunks))
    results = list() # 存放更新结果
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
        for i in var_params_mp_i:
            for j in var_params_mp_i[i]: 
                # 更新机构j变分参数
                var_params[i][j]['q2'] = var_params_mp_i[i][j]['q2']
                for k in var_params_mp_i[i][j]:
                    if k == "q2":
                        continue
                    # 更新人员k变分参数
                    var_params[i][j][k]["q3"] = var_params_mp_i[i][j][k]["q3"]
                
    return var_params


#%%
def MStep_MP_func(data_mp_i, var_params_mp_i, model_params, 
                  batch_size_org, step_size, num_iters, num_samples, mp_i):
    # 第i个进程变分参数更新
    
    def variational_objective(model_params_arr, t):
        """Provides a stochastic estimate of the variational lower bound."""
        q2_var_params_arr = q_var_params_arr[:q2_len].reshape(q2_shape)
        q3_var_params_arr = q_var_params_arr[q2_len:].reshape(q3_shape)
        
        model_params_i = model_params_arr[: mi_len].reshape(mi_shape)
        model_params_P = model_params_arr[mi_len: ].reshape(mp_shape)
        
        lower_bound    = 0                                            # 累计的ELBO
        i_org_num      = len(x_list)                                  # 国家i下机构数目
        batch_size_org = 64                                           # 每次处理的机构数
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
            
            x_arr              = list()
            mask_arr_1         = list()
            mask_arr_2         = list()
            #
            q2_mu_z_arr        = list() 
            q2_log_sig_z_arr   = list()
            q3_z_arr           = list()
            #
            q2_mu_z            = list()
            q2_log_sig_z       = list()
            q2_var             = list()
            q3_mu_var_arr      = list()
            q3_log_std_var_arr = list()
            for j in range(len(x_list_l)):
                # 第j个机构的观测数目
                x_list_j  = x_list_l[j]
                j_         = j + l * batch_size_org               # 算上batchsize后的第j_个机构
                noa_j_beg  = int(np.sum(noa_list[: j_]))
                noa_j_end  = int(np.sum(noa_list[: j_+1]))
                noa_j      = noa_j_end - noa_j_beg
                # 第j个机构下的变分参数 (机构变分参数 和 机构下学者的变法参数)
                q2_var_j  = q2_var_params_arr[j, :, :]
                q3_var_j  = q3_var_params_arr[noa_j_beg: noa_j_end, :]
            
                # padding 操作 - 人员数目 * 样本数目 * 最大文章数目
                x_list_pad_j = list()  # 将机构j人员的发文量补齐为最大值max_nop
                mask_2_j     = list()  # 标记那些位置是补齐
                for x in x_list_j:
                    nop = len(x)
                    x         = np.ones((num_samples, 1)) * x
                    x_zeros   = np.zeros((num_samples, max_nop-nop))
                    x_padding = np.concatenate([x, x_zeros], axis=-1)
                    x_list_pad_j.append(x_padding)
                    #
                    mask  = np.concatenate([np.ones(nop), np.zeros(max_nop-nop)])
                    mask  = np.ones((num_samples, 1)) * mask
                    mask_2_j.append(mask)
                x_arr_j      = np.array(x_list_pad_j)
                mask_arr_2_j = np.array(mask_2_j)
            
                # 采样机构隐变量q2_z
                q2_mu_z_j      = sampling_normal(q2_var_j[0][0], q2_var_j[0][1], num_samples)
                q2_log_sig_z_j = sampling_normal(q2_var_j[1][0], q2_var_j[1][1], num_samples)
                
                # 采样人员隐变量q3_z ~ q(z) - 人员数目(noa) x 样本数(number of samples) x 1
                q3_z_j           = sampling_normal(0, 0, num_samples * noa_j).reshape((noa_j, num_samples, 1))
                q3_mu_var_j      = q3_var_j[:, 0].reshape((noa_j, 1, 1))
                q3_log_std_var_j = q3_var_j[:, 1].reshape((noa_j, 1, 1))
                q3_z_arr_j       = q3_z_j * np.exp(q3_log_std_var_j) + q3_mu_var_j
                
                # 针对noa个人员, 重复使用q2_z样本
                q2_mu_z_arr_j      = np.ones((noa_j, 1, 1)) * q2_mu_z_j
                q2_log_sig_z_arr_j = np.ones((noa_j, 1, 1)) * q2_log_sig_z_j
                mask_arr_1_j       = np.concatenate([np.ones((1, num_samples, 1)), np.zeros((noa_j-1, num_samples, 1))], axis=0)
                
                x_arr.append(x_arr_j) 
                mask_arr_1.append(mask_arr_1_j)
                mask_arr_2.append(mask_arr_2_j)
                #
                q2_mu_z_arr.append(q2_mu_z_arr_j)
                q2_log_sig_z_arr.append(q2_log_sig_z_arr_j)
                q3_z_arr.append(q3_z_arr_j)
                # 
                q2_mu_z.append(q2_mu_z_j)
                q2_log_sig_z.append(q2_log_sig_z_j)
                q2_var.append(q2_var_j[np.newaxis, :, :])
                q3_mu_var_arr.append(q3_mu_var_j)
                q3_log_std_var_arr.append(q3_log_std_var_j)
            
             # 计算ELBO的part1
            x_arr              = np.concatenate(x_arr,              axis=0)
            mask_arr_1         = np.concatenate(mask_arr_1,         axis=0)
            mask_arr_2         = np.concatenate(mask_arr_2,         axis=0)
            # 
            q2_mu_z_arr        = np.concatenate(q2_mu_z_arr,        axis=0)
            q2_log_sig_z_arr   = np.concatenate(q2_log_sig_z_arr,   axis=0)
            q3_z_arr           = np.concatenate(q3_z_arr,           axis=0)
            # 计算ELBO的part2
            q2_mu_z            = np.concatenate(q2_mu_z,            axis=1)
            q2_log_sig_z       = np.concatenate(q2_log_sig_z,       axis=1)
            q2_var             = np.concatenate(q2_var,             axis=0)
            q3_mu_var_arr      = np.concatenate(q3_mu_var_arr,      axis=0)
            q3_log_std_var_arr = np.concatenate(q3_log_std_var_arr, axis=0)

            # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
            data_tuple         = (x_arr, mask_arr_1, mask_arr_2)
            z_tuple            = (q2_mu_z_arr, q2_log_sig_z_arr, q3_z_arr)
            model_params_tuple = (model_params_i, model_params_P)
            logp_zx            = logp_zx_density(data_tuple, z_tuple, model_params_tuple)
            part1              = np.mean(logp_zx)
            
            logp_q2_mu_z       = norm.logpdf(q2_mu_z,      q2_var[:, 0, 0], np.exp(q2_var[:, 0, 1]))
            logp_q2_log_sig_z  = norm.logpdf(q2_log_sig_z, q2_var[:, 1, 0], np.exp(q2_var[:, 1, 1]))
            logp_q3_z          = norm.logpdf(q3_z_arr,     q3_mu_var_arr,   np.exp(q3_log_std_var_arr))
            part2              = np.mean(np.sum(logp_q2_mu_z,      axis=1, keepdims=True) +\
                                         np.sum(logp_q2_log_sig_z, axis=1, keepdims=True) +\
                                         np.sum(logp_q3_z,         axis=0))
            # 求ELBO最大, 所以这里加个负号即minimize
            lower_bound_j = part1 - part2
            lower_bound  += lower_bound_j / len(x_list_l)
            start = end
                
        return -lower_bound 
    

    ''' 国家i循环 '''
    total_aid_num_mp_i   = list()
    model_params_P_mp_i  = list()
    for i in data_mp_i:                                  
        # 国家
        data_i         = data_mp_i[i]            # 国家i下所有观测数据
        var_params_i   = var_params_mp_i[i]      # 国家i下所有变分参数
        model_params_i = model_params[i]["q1"]   # 国家i的模型参数
        model_params_P = model_params['P']       # 统一随机性模型参数
        model_params_P = np.array(model_params_P)
        
        # 逐个机构更新q3人员变分参数 - q2机构变分参数随后更新
        q2_var_params_list = list()
        q3_var_params_list = list()
        x_list             = list()
        noa_list           = list()              # 每个机构下人数
        for j in data_i:
            # 机构
            if j == "q1":
                continue
            else:
                data_i_j       = data_i[j]       # 机构j下所有观测数据
                var_params_i_j = var_params_i[j] # 机构j下所有变分参数
                q2_i_j = var_params_i_j["q2"]    # 机构变分参数
                q2_var_params_list.append(q2_i_j)
                
            # 逐个机构更新q3人员变分参数
            q3_var_params_list_j = list()
            x_list_j             = list()        # 机构下人员的观测数据x
            for k in data_i_j:
                if k == "q2":
                    continue
                else:
                    x           = data_i_j[k]["x"]        # 人员k的观测数据
                    q3_i_j_k    = var_params_i_j[k]["q3"]  # 人员k的变分参数
                    q3_var_params_list_j.append(q3_i_j_k)
                    x_list_j.append(x)
            q3_var_params_list.append(np.array(q3_var_params_list_j))
            x_list.append(x_list_j)
            noa_list.append(len(x_list_j))
            
        # q2和q3变分参数
        q2_var_params_arr = np.array(q2_var_params_list)
        q3_var_params_arr = np.concatenate(q3_var_params_list, axis=0)
        q2_shape          = q2_var_params_arr.shape
        q3_shape          = q3_var_params_arr.shape
        q2_var_params_arr = q2_var_params_arr.flatten()
        q3_var_params_arr = q3_var_params_arr.flatten()
        q2_len            = len(q2_var_params_arr)
        q3_len            = len(q3_var_params_arr)
        q_var_params_arr  = np.concatenate([q2_var_params_arr, q3_var_params_arr], axis=0)
        
        # 模型参数更新
        mi_shape          = model_params_i.shape
        mp_shape          = model_params_P.shape
        model_params_i_arr= model_params_i.flatten()
        model_params_P_arr= model_params_P.flatten()
        mi_len            = len(model_params_i_arr)
        mp_len            = len(model_params_P_arr)
        model_params_arr  = np.concatenate([model_params_i_arr, model_params_P_arr], axis=0)
        
        # 梯度下降
        gradient              = grad(variational_objective)
        model_params_arr_next = adam(gradient, model_params_arr, step_size=step_size, num_iters=num_iters)
        model_params_i_next   = model_params_arr_next[:mi_len].reshape(mi_shape)
        model_params_P_next   = model_params_arr_next[mi_len:].reshape(mp_shape)
        # 直接更新国家模型参数
        model_params[i]["q1"] = model_params_i_next     # 国家i的模型参数
        total_aid_num_mp_i.append(np.sum(noa_list))     # 国家i的总人口数
        model_params_P_mp_i.append(model_params_P_next) # 国家i更新的引用随机参数
        
    # 统一随机性模型参数 (加权平均)
    ratio               = np.array(total_aid_num_mp_i) / np.sum(total_aid_num_mp_i)
    ratio               = ratio.reshape((-1, 1))
    model_params_P_mp_i = np.array(model_params_P_mp_i)
    model_params_P_next = np.sum(ratio * model_params_P_mp_i, axis=0) 
    model_params['P']   = model_params_P_next
    
    return model_params


def MStep_MP(data, var_params, model_params,
             batch_size_org, step_size, num_iters, num_samples, mp_num):
    '''启用多进程进行变分参数更新'''
    
    # 将观测数据 和 变分参数 划分成多块
    data_chunks, var_params_chunks, idx_chunks = split_data(data, var_params, mp_num) 
    
    # 创建进程池
    pool    = multiprocessing.Pool(processes=len(idx_chunks))
    results = list() # 存放更新结果
    for mp_i in range(len(idx_chunks)):
        data_mp_i      = data_chunks[mp_i]
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
        for i in range(beg, end):
            # 统计国家i内人数
            noa_i = 0
            for j in data[i]:
                if j != "q1":
                    noa_i +=  len(data[i][j]) - 1
            # 直接更新国家模型参数
            model_params[i] = model_params_mp_i[i]
            total_aid_num_mp_i.append(noa_i)
            model_params_P_mp_i.append(model_params_mp_i['P'])
            
    # 统一随机性模型参数 (加权平均)    
    ratio               = np.array(total_aid_num_mp_i) / np.sum(total_aid_num_mp_i)
    ratio               = ratio.reshape((-1, 1))
    model_params_P_mp_i = np.array(model_params_P_mp_i)
    model_params_P_next = np.sum(ratio * model_params_P_mp_i, axis=0) 
    model_params['P']   = model_params_P_next
    
    return model_params


#%%
'''
模拟数据评价
'''
def create_simulation_data(model_params_real, 
                           sampling_params):
    # 采样参数
    cou_num, org_num, aid_num, nop_num = sampling_params
    # 模型参数
    mu1_cou_list      = list()
    log_sig1_cou_list = list()
    mu2_cou_list      = list()
    log_sig2_cou_list = list()
    for i in model_params_real:
        if i == "P":
            mu_P_real, log_sig_P_real = model_params_real[i] 
        else:
            mu1_cou, log_sig1_cou = model_params_real[i]['q1'][0]
            mu1_cou_list.append(mu1_cou)
            log_sig1_cou_list.append(log_sig1_cou)
            # 
            mu2_cou, log_sig2_cou = model_params_real[i]['q1'][1]
            mu2_cou_list.append(mu2_cou)
            log_sig2_cou_list.append(log_sig2_cou)
 
    # 开始抽样
    data  = dict()
    cou_id = -1
    for mu1_cou, log_sig1_cou, mu2_cou, log_sig2_cou in zip(mu1_cou_list, log_sig1_cou_list, mu2_cou_list, log_sig2_cou_list):
        # 国家的四个参数: normal(mu1_cou, exp(log_sig1_cou)); normal(mu2_cou, exp(log_sig2_cou))
        cou_id += 1
        data[cou_id] = dict()
        data[cou_id]['q1'] = np.array([[mu1_cou, log_sig1_cou], [mu2_cou, log_sig2_cou]])
       
        # 抽取机构能力 ~ 隐变量
        mu_org_list      = sampling_normal(mu1_cou, log_sig1_cou, org_num).squeeze() # 机构隐变量
        log_sig_org_list = sampling_normal(mu2_cou, log_sig2_cou, org_num).squeeze() # 机构隐变量
        org_id           = -1
        for mu_org, log_sig_org in zip(mu_org_list, log_sig_org_list):
            org_id += 1
            data[cou_id][org_id] = dict()
            data[cou_id][org_id]['q2'] = np.array([mu_org, log_sig_org])
            
            # 抽取人员能力 ~ 隐变量
            aid_num_j = max(sampling_poisson(aid_num, 1), [1])[0]
            q3_list   = sampling_normal(mu_org, log_sig_org, aid_num_j).squeeze()
            aid_id    = -1
            for q3 in q3_list:
                aid_id += 1
                data[cou_id][org_id][aid_id] = dict()
                data[cou_id][org_id][aid_id]['q3'] = q3                              # 人员隐变量
            
                # 第j个机构中第k个人抽样nop_num_1篇文章的引用数目
                nop_num_k = max(sampling_poisson(nop_num, 1), [1])[0]                # 抽取人员发文量 ~ 泊松分布 (至少一篇文章)
                p = sampling_normal(mu_P_real, log_sig_P_real, nop_num_k).squeeze()  # 随机效应参数
                x = q3 + p                                                           # 观测数据
                if nop_num_k == 1:
                    x = np.array([x])
                data[cou_id][org_id][aid_id]['x'] = x   
    return data


def max_likelihoood(data):
    # Q-model 均值估计
    # 忽略先验分布, 是Baseline
    x_obs = list()
    for i in data:
        for j in data[i]:
           if j == "q1":                   # 国家真实参数
               continue
           for k in data[i][j]:
               if k == "q2":               # 机构能力真实隐变量
                   continue
               x = data[i][j][k]['x']
               x_obs.append(x)
    x_obs = np.concatenate(x_obs, axis=0)
    std   = np.std(x_obs)            # 融合了所有方差信息
    
    model_params_init  = dict()
    var_params_init    = dict()
    log_sig_P_list     = list()
    for i in data:
        model_params_init[i] = dict()
        var_params_init[i]   = dict()
        q1_mu1_est_list      = list()           # 估计国家平均能力
        q1_mu2_est_list      = list()           # 估计国家平均波动性
        
        for j in data[i]:
           if j == "q1":                        # 国家真实参数
               continue
           var_params_init[i][j] = dict()
           q3_mu_est_list        = list()
           q3_log_sig_est_list   = list()
           
           for k in data[i][j]:
               if k == "q2":                    # 机构能力真实隐变量
                   continue
               var_params_init[i][j][k] = dict()
               x                        = data[i][j][k]['x']
               q3_mu_est                = np.mean(x)                           # 估计人员能力均值
               q3_log_sig_est           = np.log(max(std - np.std(x), 1e-2))   # 估计人员能力方差 (所有方差信息 - sig_P)
               var_params_init[i][j][k]['q3'] = np.array([q3_mu_est, q3_log_sig_est])
               q3_mu_est_list.append(q3_mu_est)
               q3_log_sig_est_list.append(q3_log_sig_est)
               log_sig_P_list.append(q3_log_sig_est)
           # 机构科研能力估计
           q2_mu_est      = np.mean(q3_mu_est_list)                                  # 人员能力均值的均值估计机构能力均值
           q2_log_sig_est = np.log(max(np.std(q3_mu_est_list), 1e-2))
           var_params_init[i][j]['q2'] = np.array([[q2_mu_est, 0], [q2_log_sig_est, 0]])
           q1_mu1_est_list.append(q2_mu_est)
           q1_mu2_est_list.append(q2_log_sig_est)
           
        # 国家科研能力估计
        model_params_init[i]['q1'] = np.array([[np.mean(q1_mu1_est_list), np.log(max(np.std(q1_mu1_est_list), 1e-2))],
                                               [np.mean(q1_mu2_est_list), np.log(max(np.std(q1_mu2_est_list), 1e-2))]])
    
    model_params_init["P"] = [0, np.mean(log_sig_P_list)]
    
    return var_params_init, model_params_init


def evaluate_on_simulation_data(data, model_params_real, var_params_init, model_params_init):
    # var_params_init, model_params_init = var_params_bbvi,  model_params_bbvi
    
    def nomralized_func(x):
        return (x - np.mean(x)) / np.std(x)    
    
    def evaluate_real2pred(Y, X):
        # 评价指标
        cor, pvalue = pearsonr(Y, X)
        rmse        = np.sqrt(mean_squared_error(Y, X))
        mae         = mean_absolute_error(Y, X)
        r2          = r2_score(Y, X)
        return cor, rmse, mae, r2
        
    def plot_q(q_real, q_var, q_err, xlabel, ylabel, legend_1, legend_2, title):
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        config = {
                  "font.family" : "SimHei",
                  "font.size" : 20
                  }
        rcParams.update(config)
        plt.rcParams['axes.unicode_minus'] = False # SimHei 字体符号不正常显示
        
        fontsize = 22
        
        plt.plot(np.arange(len(q_real)) +1, q_real, 
                 label=legend_1, c='red', marker='s', alpha=0.5, linewidth=1)
        plt.errorbar(np.arange(len(q_real)) +1, q_var, yerr=q_err, # np.zeros(len(q_real)) 
                     label=legend_2, fmt="o:", color='blue', ecolor='dimgray', capsize=5, markersize=5, elinewidth=0.8)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.legend(frameon=False, loc='upper right', fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        # plt.xticks(np.arange(0, 11, 1))
        plt.yticks(np.arange(-6, 8, 2))
        plt.ylim(-6, 6)
            

    q1_real_mu1_list, q1_real_std1_list = list(), list()  # 国家模型参数真实值
    q1_real_mu2_list, q1_real_std2_list = list(), list()
    q1_var_mu1_list,  q1_var_std1_list  = list(), list()  # 国家科研能力模型参数估计值
    q1_var_mu2_list,  q1_var_std2_list  = list(), list()
    
    q2_real_mu_list,  q2_real_std_list  = list(), list()  # 机构隐变量真实值
    q2_var_mu_list,   q2_var_std_list   = list(), list()  # 机构科研能力隐变量估计值
    
    q3_real_mu_list                     = list()          # 学者科研能力隐变量真实值
    q3_var_mu_list,   q3_var_std_list   = list(), list()  # 学者科研能力隐变量估计值
    
    for i in data: 
        # 国家参数真实值
        # mu1, log_std1 = data[i]['q1'][0]
        # mu2, log_std2 = data[i]['q1'][1]
        mu1, log_std1 = model_params_real[i]['q1'][0]
        mu2, log_std2 = model_params_real[i]['q1'][1]
        q1_real_mu1_list.append(mu1)
        q1_real_std1_list.append(log_std1)
        q1_real_mu2_list.append(mu2)
        q1_real_std2_list.append(log_std2)
        
        # 国家参数估计值
        mu1_est, log_std1_est = model_params_init[i]["q1"][0]
        mu2_est, log_std2_est = model_params_init[i]["q1"][1]
        q1_var_mu1_list.append(mu1_est)
        q1_var_std1_list.append(log_std1_est)
        q1_var_mu2_list.append(mu2_est)
        q1_var_std2_list.append(log_std2_est)    

        for j in data[i]:
            if j == "q1":
                continue
            else:
                # 机构隐变量真实值
                q2_mu, q2_std = data[i][j]["q2"]
                q2_real_mu_list.append(q2_mu)
                q2_real_std_list.append(q2_std)
                
                # 机构隐变量变分参数估计值
                q2_mu_est, q2_std_est = var_params_init[i][j]["q2"]
                q2_var_mu_list.append(q2_mu_est[0])
                q2_var_std_list.append(q2_std_est[0])
                
            for k in data[i][j]:
                if k == "q2":
                    continue
                else:
                    # 人员隐变量真实值
                    q3 = data[i][j][k]["q3"]
                    q3_real_mu_list.append(q3)
                    
                    # 人员隐变量变分参数估计值
                    q3_mu_est, q3_std_est = var_params_init[i][j][k]["q3"]
                    q3_var_mu_list.append(q3_mu_est)
                    q3_var_std_list.append(q3_std_est)
    # 列表变成数组
    q1_real_mu1_list, q1_real_std1_list = np.array(q1_real_mu1_list), np.array(q1_real_std1_list)
    q1_real_mu2_list, q1_real_std2_list = np.array(q1_real_mu2_list), np.array(q1_real_std2_list)
    q1_var_mu1_list,  q1_var_std1_list  = np.array(q1_var_mu1_list),  np.array(q1_var_std1_list)
    q1_var_mu2_list,  q1_var_std2_list  = np.array(q1_var_mu2_list),  np.array(q1_var_std2_list)
    q2_real_mu_list,  q2_real_std_list  = np.array(q2_real_mu_list),  np.array(q2_real_std_list) 
    q2_var_mu_list,   q2_var_std_list   = np.array(q2_var_mu_list),   np.array(q2_var_std_list)
    q3_real_mu_list                     = np.array(q3_real_mu_list)
    q3_var_mu_list,   q3_var_std_list   = np.array(q3_var_mu_list),   np.array(q3_var_std_list)
    
    # plot_q(q1_real_mu1_list, q1_var_mu1_list, np.exp(q1_var_std1_list), "Countries",    r"$\hat{Q_1}$")
    # plot_q(q2_real_mu_list,  q2_var_mu_list,  np.exp(q2_var_std_list),  "Institutions", r"$\hat{Q_2}$") 
    # plot_q(q3_real_mu_list,  q3_var_mu_list,  np.exp(q3_var_std_list),  "Scientists",   r"$\hat{Q_3}$")
    
    plot_q(q1_real_mu1_list, q1_var_mu1_list, np.exp(q1_var_std1_list), "国家", "国家科研能力", "真实值", "估计值", "融合国家信息的科研能力量化模型 (模拟配置4)")
    plot_q(q2_real_mu_list,  q2_var_mu_list,  np.exp(q2_var_std_list),  "机构", "机构科研能力", "真实值", "估计值", "融合国家信息的科研能力量化模型 (模拟配置4)") 
    plot_q(q3_real_mu_list,  q3_var_mu_list,  np.exp(q3_var_std_list),  "学者", "学者科研能力", "真实值", "估计值", "融合国家信息的科研能力量化模型 (模拟配置4)")  
    
    
    cor_q1, rmse_q1, mae_q1, r2_q1 = evaluate_real2pred(q1_real_mu1_list, q1_var_mu1_list)
    cor_q2, rmse_q2, mae_q2, r2_q2 = evaluate_real2pred(q2_real_mu_list,  q2_var_mu_list)
    cor_q3, rmse_q3, mae_q3, r2_q3 = evaluate_real2pred(q3_real_mu_list,  q3_var_mu_list)
    # cor_q1, rmse_q1, mae_q1, r2_q1 = evaluate_real2pred(q1_real_mu2_list, q1_var_mu2_list) # 国家能力方差的估计效果 
    # cor_q1, rmse_q1, mae_q1, r2_q1 = evaluate_real2pred(q2_real_std_list, q2_var_std_list) # 机构能力方差的估计效果

    tb = pt.PrettyTable()
    tb.field_names = ["*", "Pearsonr", "r2", "rmse", 'mae']
    tb.add_row(["国家", "{:.4f}".format(cor_q1), "{:.4f}".format(r2_q1), "{:.4f}".format(rmse_q1), "{:.4f}".format(mae_q1)])
    tb.add_row(["机构", "{:.4f}".format(cor_q2), "{:.4f}".format(r2_q2), "{:.4f}".format(rmse_q2), "{:.4f}".format(mae_q2)])
    tb.add_row(["学者", "{:.4f}".format(cor_q3), "{:.4f}".format(r2_q3), "{:.4f}".format(rmse_q3), "{:.4f}".format(mae_q3)])
    print(tb)
    
    return (cor_q1, r2_q1, rmse_q1, mae_q1), (cor_q2, r2_q2, rmse_q2, mae_q2), (cor_q3, r2_q3, rmse_q3, mae_q3)


def compared_with_bbvi_em_org(data):
    ''' 
    将bbvi_em_org_country 与 bbvi_em_org 在模拟数据上比较
    '''
    
    def data_exchange(data):
        '''
        将bbvi_em_org_country生成的模拟数据退化成bbvi_em_org生成的模拟数据格式
        '''
        model_params_real2 = dict()
        j_    = 0 
        data2 = dict()
        for i in data:
            # 国家i
            for j in data[i]:
                if j == "q1":
                    continue
                # 机构j
                data2[j_]             = dict()
                model_params_real2[j_] = dict()
                for k in data[i][j]:
                    if k == "q2":
                        model_params_real2[j_]['q2'] =  data[i][j][k]
                        continue
                    # 学者k
                    data2[j_][k]       = dict()
                    data2[j_][k]['x']  = data[i][j][k]['x']
                    data2[j_][k]['q3'] = data[i][j][k]['q3']
                j_ += 1
        return data2, model_params_real2

    # 与bbvi_em_org 比较
    data2, model_params_real2            = data_exchange(data)
    var_params_init2, model_params_init2 = bbvi_em_org.max_likelihoood(data2)
    var_params,       model_params,      = var_params_init2, model_params_init2
    # 
    mp_num         = 8
    Epochs         = 10
    step_size      = 1e-1
    num_iters      = 100
    batch_size_org = 512
    num_samples    = 1
    for e in range(Epochs):
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))           
        E_start_time    = time.perf_counter()
        var_params_next = bbvi_em_org.EStep_MP(data2, var_params, model_params, batch_size_org, step_size, num_iters, num_samples, mp_num)
        E_end_time      = time.perf_counter()                    
        var_params      = var_params_next
        print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
        
        # M-Step
        print("({}) Optimizing model parameters...".format(e))
        M_start_time      = time.perf_counter()
        model_params_next = bbvi_em_org.MStep_MP(data2, var_params, model_params, batch_size_org, step_size, num_iters, num_samples, mp_num)
        M_end_time        = time.perf_counter()
        model_params      = model_params_next
        print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time))
    # 变分参数估计 - BBVI-EM
    var_params_bbvi2, model_params_bbvi2 = var_params, model_params
    q2_ORG, q3_ORG = bbvi_em_org.evaluate_on_simulation_data(data2, model_params_real2, var_params_bbvi2, model_params_bbvi2)
       
    return (q2_ORG, q3_ORG)
    

#%%
# 模拟数据分析...
np.set_printoptions(precision=6, suppress=True)
def BBVI_Algorithm(): 
    '''
    模型结构:
        多个国家i, 每个国家有模型参数 (mu1_cou_i, log_sig1_cou_i), (mu2_cou_i, log_sig2_cou_i)
        采样机构: mu_org_j      ~ normal(mu1_cou_i, exp(log_sig1_cou_i)), 即 q2 = mu_org_j
                  log_sig_org_j ~ normal(mu2_cou_i, exp(log_sig2_cou_i))
        
        多个机构j, 每个机构有隐变量 (mu_org_j, log_sig_org_j)
        采样人员: q3_k ~ normal(mu_org_j, exp(log_sig_org_j))
        
        多个人员k, 每个人员有隐变量 (q3_k)
        采样文章cc: p_l ~ normal(0, exp(log_sig_p))
        cc_kl = q3_k + p_l 
        
    ''' 
    # 生成模拟数据
    mu_0, log_sig_0 = 0., -1.
    mu_1, log_sig_1 = -1, -1.
    mu_2, log_sig_2 = 0, -1.      
    mu_3, log_sig_3 = -1, -1.
    
    # 模型参数
    cou_num = 10   # 国家数目
    org_num = 10   # 机构数目
    aid_num = 10   # 人员数目
    nop_num = 10   # 人均文章数-发文量服从泊松分布 poisson(nop_num)
    
    mu1_cou_list      = list(sampling_normal(mu_0, log_sig_0, cou_num).squeeze())   # 国家下机构能力的均值
    log_sig1_cou_list = list(sampling_normal(mu_1, log_sig_1, cou_num).squeeze())   # 国家下机构能力的均值的方差
    mu2_cou_list      = list(sampling_normal(mu_2, log_sig_2, cou_num).squeeze())   # 国家下机构能力的方差
    log_sig2_cou_list = list(sampling_normal(mu_3, log_sig_3, cou_num).squeeze())   # 国家下机构能力的方差的方差
    # 国家模型参数
    model_params_real = dict()
    for i in range(cou_num): 
        model_params_real[i]       = dict()
        model_params_real[i]['q1'] = np.array([[mu1_cou_list[i], log_sig1_cou_list[i]], [mu2_cou_list[i], log_sig2_cou_list[i]]])
    # 随机波动模型参数
    mu_P_real      = 0 
    log_sig_P_real = 0.5 
    model_params_real["P"] = np.array([mu_P_real, log_sig_P_real])

    # 采样模拟数据
    sampling_params = [cou_num, org_num, aid_num, nop_num]
    data = create_simulation_data(model_params_real, sampling_params)
    # 极大似然估计 (基准模型)
    var_params_init, model_params_init = max_likelihoood(data)

    # 变分估计
    mp_num         = 8
    Epochs         = 10
    step_size      = 1e-1
    num_iters      = 100
    batch_size_org = 512
    num_samples    = 1
    model_params, var_params = model_params_init, var_params_init
    # 
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
    
    # 评价模型: 融合机构信息的科研能力模型
    q2_ORG, q3_ORG = compared_with_bbvi_em_org(data)
    # 评价模型: 基准模型
    q1_WSB, q2_WSB, q3_WSB = evaluate_on_simulation_data(data, model_params_real, var_params_init,  model_params_init)
    # 评价模型: 融合国家信息的科研能力模型
    q1_COU, q2_COU, q3_COU = evaluate_on_simulation_data(data, model_params_real, var_params_bbvi,  model_params_bbvi)
    
    return (q1_WSB, q2_WSB, q3_WSB), (q1_WSB, q2_ORG, q3_ORG), (q1_COU, q2_COU, q3_COU)


def main():
    
    # 计算每个指标的均值; t检验
    def get_list(results, Key):
        q1_eval_list = list()
        q2_eval_list = list()
        q3_eval_list = list()
        for t in results:
            q1_eval, q2_eval, q3_eval = results[t][Key]
            q1_eval_list.append(list(q1_eval))
            q2_eval_list.append(list(q2_eval))
            q3_eval_list.append(list(q3_eval))
        q1_eval_list = np.maximum(np.array(q1_eval_list), 0)  # R2可为负, 置0
        q2_eval_list = np.maximum(np.array(q2_eval_list), 0)
        q3_eval_list = np.maximum(np.array(q3_eval_list), 0)
        return q1_eval_list, q2_eval_list, q3_eval_list
    
    # 将模拟数据数据运行10次, 比较结果
    times   = 10
    number  = 4  # 记录模拟实验数据编号  
    results = dict()
    for t in range(times):
        (q1_WSB, q2_WSB, q3_WSB), (q1_ORG, q2_ORG, q3_ORG), (q1_COU, q2_COU, q3_COU) = BBVI_Algorithm()
        results[t] = dict()
        results[t]["wsb"] = (q1_WSB, q2_WSB, q3_WSB)
        results[t]["org"] = (q1_ORG, q2_ORG, q3_ORG)
        results[t]["cou"] = (q1_COU, q2_COU, q3_COU)
    # 储存
    with open("./Results_org_country/simulation_{}.pkl".format(number), 'wb') as f:
        pickle.dump(results, f)
    # 读取
    with open("./Results_org_country/simulation_{}.pkl".format(number), 'rb') as f:
        results = pickle.load(f)
    
    q1_WSB_list, q2_WSB_list, q3_WSB_list = get_list(results, "wsb")
    _,           q2_ORG_list, q3_ORG_list = get_list(results, "org")
    q1_COU_list, q2_COU_list, q3_COU_list = get_list(results, "cou")
    
    # 国家科研能力评价
    q1_COU_mean = np.mean(q1_COU_list, axis=0)
    q1_COU_std  = np.std(q1_COU_list,  axis=0)
    print("COU(q1): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q1_COU_mean))
    
    # 机构科研能力评价
    q2_ORG_mean = np.mean(q2_ORG_list, axis=0)
    q2_COU_mean = np.mean(q2_COU_list, axis=0)
    print("ORG(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q2_ORG_mean))
    print("COU(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q2_COU_mean))
    pvalue_list = list()
    for i in range(4):
        _, pvalue = ttest_rel(q2_ORG_list[:, i], q2_COU_list[:, i])
        pvalue_list.append(pvalue)
    print("***(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*pvalue_list))
    
    # 学者科研能力评价
    q3_WSB_mean = np.mean(q3_WSB_list, axis=0)
    q3_ORG_mean = np.mean(q3_ORG_list, axis=0)
    q3_COU_mean = np.mean(q3_COU_list, axis=0)
    print("WSB(q3): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q3_WSB_mean))
    print("ORG(q3): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q3_ORG_mean))
    print("COU(q3): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q3_COU_mean))
    pvalue_list = list()
    for i in range(4):
        _, pvalue = ttest_rel(q3_ORG_list[:, i], q3_COU_list[:, i])
        pvalue_list.append(pvalue)
    print("***(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*pvalue_list))
    
    
