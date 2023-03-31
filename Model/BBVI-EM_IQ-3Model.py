#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:51:53 2022

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
import math
import random
import multiprocessing
import prettytable as pt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 
from scipy.stats import spearmanr, pearsonr, ttest_rel

from MyQPModel import bbvi_em_org_country  # BBVI-EM_IQ-2Model.py


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


def log_p_zx_density(z, x_obs, i_obs, j_mask, M_coop, model_params):
    # z: 隐变量, x: 观测引用量, M: 观测合作矩阵
    
    # model_params: 模型参数 
    model_params_q1 = model_params[:-1]
    model_params_P  = model_params[-1][0]
    mu_P, log_sig_P = model_params_P
    # 隐变量
    z_q3, z_q2_mu_arr, z_q2_log_sig_arr = z
    
    # 隐变量生成概率 P(Z_q2)
    log_q2_mu_density  = norm.logpdf(z_q2_mu_arr,      model_params_q1[:, 0, 0][i_obs], np.exp(model_params_q1[:, 0, 1][i_obs]))
    log_q2_sig_density = norm.logpdf(z_q2_log_sig_arr, model_params_q1[:, 1, 0][i_obs], np.exp(model_params_q1[:, 1, 1][i_obs]))
    log_q2_mu_density  = np.multiply(j_mask, log_q2_mu_density)  # mask遮挡重复计算部分
    log_q2_sig_density = np.multiply(j_mask, log_q2_sig_density)
    
    # 隐变量生成概率 P(Z_q3)
    log_q3_density = norm.logpdf(z_q3, z_q2_mu_arr, np.exp(z_q2_log_sig_arr))
    
    # 观测x生成概率 P(X|Z; M)
    p_obs = x_obs - np.matmul(M_coop, np.transpose(z_q3))
    log_P_density = norm.logpdf(p_obs, 0, np.exp(log_sig_P))
    log_P_density = np.transpose(log_P_density)
    # P(X, Z) = P(Z) * P(X|Z; M) --- 机构能力生成概率 + 人员能力生成概率 + 文章引用生成概率
    logpq = np.sum(log_q2_mu_density, axis=-1) + np.sum(log_q2_sig_density, axis=-1) +\
            np.sum(log_q3_density,    axis=-1) +\
            np.sum(log_P_density,     axis=-1)
    return logpq


#%%
def Estep(data, model_params, var_params, num_samples, step_size, num_iters):
    
    def callback(params, t, g):
        # print(t)
        pass
    
    def variational_objective(var_params_, t):
        """Provides a stochastic estimate of the variational lower bound."""
        var_params_q2_ = var_params_[: len_q2]
        var_params_q3_ = var_params_[-len_q3:]
        var_params_q2  = var_params_q2_.reshape(shape_q2)
        var_params_q3  = var_params_q3_.reshape(shape_q3)
        
        # 从q3(z)中抽取样本 - 每个作者的变分参数 (num_samples x 人员数目)
        z_q3         = sampling_normal(var_params_q3[:, 0], var_params_q3[:, 1], num_samples)
        # 从q2(z)中抽取样本 - 每个机构的变分参数
        z_q2_mu      = sampling_normal(var_params_q2[:, 0, 0], var_params_q2[:, 0, 1], num_samples)
        z_q2_log_sig = sampling_normal(var_params_q2[:, 1, 0], var_params_q2[:, 1, 1], num_samples)
        z_q2_mu_arr      = z_q2_mu[:, j_obs]         # 针对每个机构下的人员, 重复使用q2_z样本
        z_q2_log_sig_arr = z_q2_log_sig[:, j_obs]    # 针对每个机构下的人员, 重复使用q2_z样本 
        
        z = (z_q3, z_q2_mu_arr, z_q2_log_sig_arr)
        # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
        logpq = log_p_zx_density(z, x_obs, i_obs, j_mask, M_coop, model_params)
        part1 = np.mean(logpq)
        
        log_q2_mu_density  = norm.logpdf(z_q2_mu,      var_params_q2[:, 0, 0], np.exp(var_params_q2[:, 0, 1]))
        log_q2_sig_density = norm.logpdf(z_q2_log_sig, var_params_q2[:, 1, 0], np.exp(var_params_q2[:, 1, 1]))
        log_q3_density     = norm.logpdf(z_q3, var_params_q3[:, 0], np.exp(var_params_q3[:, 1]))
        part2 = np.sum(log_q2_mu_density, axis=-1) + np.sum(log_q2_sig_density, axis=-1) +\
                np.sum(log_q3_density,    axis=-1)
        part2 = np.mean(part2)
        
        lower_bound = part1 - part2 
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound
    
    # 变分参数
    var_params_q2, var_params_q3 = var_params
    # 变分参数 - reshape
    shape_q2       = var_params_q2.shape
    shape_q3       = var_params_q3.shape
    var_params_q2_ = var_params_q2.flatten()
    var_params_q3_ = var_params_q3.flatten()
    len_q2         = len(var_params_q2_)
    len_q3         = len(var_params_q3_)
    var_params_    = np.concatenate([var_params_q2_, var_params_q3_])
    
    # 观测数据: 引用数目 & 人员隶属国家 & 人员隶属机构 & 合作矩阵
    x_obs, i_obs, j_obs, M_coop = data['x_obs'], data['i_obs'], data['j_obs'], data['M_coop']
    i_obs = np.array(i_obs, dtype=np.int32)
    j_obs = np.array(j_obs, dtype=np.int32)
    j_mask = np.zeros(len(j_obs))  # 避免 log_q2_mu(sig)_density每个人重复计算
    j_id = set(j_obs)              # 机构编号
    for v, j in enumerate(j_obs):
        if j in j_id:
            j_mask[v] = 1          # 只提取首次遇到j机构, 这样每各机构q2_mu, q2_sig只生成一次概率
            j_id.remove(j)
    j_mask = np.ones((num_samples, 1)) * j_mask       
    
    # 梯度下降更新变分参数
    gradient            = grad(variational_objective)
    var_params_next_    = adam(gradient, var_params_, step_size=step_size, num_iters=num_iters, callback=callback)
    var_params_q2_next  = var_params_next_[: len_q2].reshape(shape_q2)
    var_params_q3_next  = var_params_next_[-len_q3:].reshape(shape_q3)
    
    return (var_params_q2_next, var_params_q3_next)


def Mstep(data, model_params, var_params, num_samples, step_size, num_iters):
                                                                               
    def variational_objective(model_params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        # 从q3(z)中抽取样本 - 每个作者的变分参数 (num_samples x 人员数目)
        z_q3 = sampling_normal(var_params_q3[:, 0], var_params_q3[:, 1], num_samples)
        # 从q2(z)中抽取样本 - 每个机构的变分参数
        z_q2_mu      = sampling_normal(var_params_q2[:, 0, 0], var_params_q2[:, 0, 1], num_samples)
        z_q2_log_sig = sampling_normal(var_params_q2[:, 1, 0], var_params_q2[:, 1, 1], num_samples)
        z_q2_mu_arr      = z_q2_mu[:, j_obs]         # 针对每个机构下的人员, 重复使用q2_z样本
        z_q2_log_sig_arr = z_q2_log_sig[:, j_obs]    # 针对每个机构下的人员, 重复使用q2_z样本 
        
        z = (z_q3, z_q2_mu_arr, z_q2_log_sig_arr)
        # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
        logpq = log_p_zx_density(z, x_obs, i_obs, j_mask, M_coop, model_params)
        part1 = np.mean(logpq)
        
        lower_bound = part1
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound
    
    # 变分参数
    var_params_q2, var_params_q3 = var_params
    # 观测数据: 引用数目 & 人员隶属国家 & 人员隶属机构 & 合作矩阵
    x_obs, i_obs, j_obs, M_coop = data['x_obs'], data['i_obs'], data['j_obs'], data['M_coop']
    i_obs  = np.array(i_obs, dtype=np.int16)
    j_obs  = np.array(j_obs, dtype=np.int16)
    j_mask = np.zeros(len(j_obs))  # 避免 log_q2_mu(sig)_density每个人重复计算
    j_id   = set(j_obs)            # 机构编号
    for v, j in enumerate(j_obs):
        if j in j_id:
            j_mask[v] = 1          # 只提取首次遇到j机构, 这样每各机构q2_mu, q2_sig只生成一次概率
            j_id.remove(j)
    j_mask = np.ones((num_samples, 1)) * j_mask 
    
    # 梯度下降更新模型参数
    gradient          = grad(variational_objective)
    model_params_next = adam(gradient, model_params, step_size=step_size, num_iters=num_iters)
    return model_params_next


#%%
def create_simulation_data(model_params, sampling_params):
    ''' 生成模拟数据 '''
    
    # 抽样参数
    cou_num, org_num, aid_num, nop_num, coa_num, group_num = sampling_params
    # 模型参数
    model_params_real_q1 = model_params[:-1]
    model_params_real_P  = model_params[-1]
    mu_P, log_sig_P      = model_params_real_P[0]
    
    # 抽取每个国家i下每个机构j下每位作者k的Q值
    j_count = 0
    i_obs = list()    # 标记人员所属国家编号
    j_obs = list()    # 标记人员所属机构编号
    q2_real = list()  # 所有机构隐变量
    q3_real = list()  # 所有人员隐变量
    for i, _ in enumerate(model_params_real_q1):
        mu1_cou_i, log_sig1_cou_i = model_params_real_q1[i][0]  # 国家参数
        mu2_cou_i, log_sig2_cou_i = model_params_real_q1[i][1]  # 国家参数
        
        mu_org_list      = sampling_normal(mu1_cou_i, log_sig1_cou_i, org_num)   # 机构隐变量
        log_sig_org_list = sampling_normal(mu2_cou_i, log_sig2_cou_i, org_num)   # 机构隐变量
        q2_real_i        = np.concatenate([mu_org_list, log_sig_org_list], axis=-1)
        q2_real.append(q2_real_i)  # 机构真实隐变量
        
        q3_real_i = list()
        i_obs_i   = list()
        j_obs_i   = list()
        for j, _ in enumerate(q2_real_i):
            mu_org_j, log_sig_org_j = q2_real_i[j]
        
            q3_real_j    = sampling_normal(mu_org_j, log_sig_org_j, aid_num)     # 人员真实隐变量
            q3_real_i.append(q3_real_j.squeeze())
            
            i_obs_i.append(np.ones(aid_num) * i)
            j_obs_i.append(np.ones(aid_num) * j_count)
            j_count += 1                               # 所有国家的机构统一计数      
        q3_real.append(q3_real_i)    
        i_obs.append(i_obs_i)
        j_obs.append(j_obs_i)
        
    q2_real = np.array(q2_real)  # cou_num x org_num x 2        所有机构隐变量
    q3_real = np.array(q3_real)  # cou_num x org_num x aid_num  所有人员隐变量
    i_obs   = np.array(i_obs)    # cou_num x org_num x aid_num  所有人员所属国家
    j_obs   = np.array(j_obs)    # cou_num x org_num x aid_num  所有人员所属机构
    q2_real = q2_real.reshape((-1, 2))         # 总机构数 x 2
    q3_real = q3_real.flatten()[:, np.newaxis] # 总人员数 x 1 
    i_obs   = i_obs.flatten()
    j_obs   = j_obs.flatten()
    
    # 抽取每位作者的(第一作者)发文量 - unique papers
    nop_obs       = np.maximum(sampling_poisson(nop_num, len(q3_real)), 1)      # 每个人的发文量
    total_nop_num = np.sum(nop_obs)
    # 抽取每篇文章的P值
    p_obs = sampling_normal(mu_P, log_sig_P, total_nop_num)
    # 抽取合作矩阵
    total_aid_num = len(q3_real)
    coa_obs       = np.maximum(sampling_poisson(coa_num,  total_nop_num), 1)    # 每篇文章的作者数目 = 第一作者 + 合作者数目
    M_coop        = np.zeros((total_nop_num, total_aid_num))                    # 合作矩阵: 总发文量数目 x 总作者数目  
 
    # 均匀分布多组
    group_nop_ratio = np.ones(group_num) * 1 / group_num
    group_nop_num   = list()  # 每组人员数目
    for i, nop_ratio in enumerate(group_nop_ratio):
        if i < group_num - 1:
            group_nop_num.append(int(nop_ratio * total_aid_num))
        else:
            group_nop_num.append(total_aid_num - sum(group_nop_num))
    # 每组包含人员编号 (合作闭集)    
    group_aid_idx = dict()    
    aid_idx       = np.arange(0, total_aid_num, 1)
    random.shuffle(aid_idx)
    for i in range(group_num):
        group_aid_idx[i] = dict()
        for j in aid_idx[: group_nop_num[i]]:
            group_aid_idx[i][j] = ''
        aid_idx = aid_idx[group_nop_num[i]: ]
        
    # 每篇文章的第一作者. coa_obs_i - 1 才是合作者数目
    first_aid = list()
    for i, nop in enumerate(nop_obs):
        first_aid += list(np.ones(nop) * i)
    first_aid = np.array(first_aid)
    # 生成合作矩阵 M_coop
    for j, coa_obs_j in enumerate(coa_obs):
        # 第一作者 (第j篇文章首作者的编号)
        first_aid_j = int(first_aid[j])
        M_coop[j, first_aid_j] = 1 
        # 合著者 - 允许重复抽到第一作者, 将合著者数目视作-1 即可
        coa_obs_j_ = coa_obs_j - 1
        if coa_obs_j_ > 0:
            # 确定first_aid_j的隶属小组
            for i in group_aid_idx:
                if first_aid_j in group_aid_idx[i]:
                    break
            # 从该小组抽取first_aid_j的合作者 (第j篇文章的合作者)
            coa_obs_j_ = min(len(group_aid_idx[i].keys()),      coa_obs_j_)
            co_aid_j   = random.sample(group_aid_idx[i].keys(), coa_obs_j_)
            for k in co_aid_j:
                M_coop[j, k] = 1

    # 观测数据 C = Q * P
    M_coop = M_coop / np.sum(M_coop, axis=-1, keepdims=True)
    x_obs  = np.matmul(M_coop, q3_real) + p_obs
     
    data = dict()
    data['x_obs'] = x_obs     # 观测数据 - logcc
    data['i_obs'] = i_obs     # 观测数据 - 所属国家
    data['j_obs'] = j_obs     # 观测数据 - 所属机构
    data['M_coop'] = M_coop   # 观测数据 - 合作情况矩阵
    data['q2_obs'] = q2_real  # 隐变量数据 - 机构
    data['q3_obs'] = q3_real  # 隐变量数据 - 人员
    return data


def max_likelihoood(data):
    # 
    x_obs, i_obs, j_obs, M_coop = data['x_obs'], data['i_obs'], data['j_obs'], data['M_coop'] 
    
    # 极大似然估计 - 估计隐变量q2, q3 (极大似然估计当作变分参数的初始值)
    sig_total = np.std(x_obs)
    
    # 人员变分参数
    var_params_q3_est    = list()
    M_coop               = np.array(M_coop > 0)
    nop_total, noa_total = M_coop.shape
    for k in tqdm(range(noa_total)):
        x                   = x_obs[M_coop[:, k]]     # 第k位学者的论文引用数据
        q3_mu               = np.mean(x)              # 
        q3_log_std          = np.log(max(sig_total - np.std(x), 1e-2)) # 总体波动性 - luck造成的波动性 约= Q自身波动性
        var_params_q3_est_k = [q3_mu, q3_log_std]     # 第k个人的变分参数
        var_params_q3_est.append(var_params_q3_est_k) 
    var_params_q3_est = np.array(var_params_q3_est)
    
    # 机构变分参数
    org_num              = len(set(j_obs))
    var_params_q2_est    = list()
    for j in range(org_num):
        var_params_q3_est_j = var_params_q3_est[j_obs == j]
        q2_mu               = np.mean(var_params_q3_est_j[:, 0])         
        q2_log_std          = np.log(max(np.std(var_params_q3_est_j[:, 0]), 1e-2))
        var_params_q2_est.append([[q2_mu, 0],         # 第j个机构的变分参数
                                  [q2_log_std, 0]])
    var_params_q2_est = np.array(var_params_q2_est)  
    
    # 国家模型参数
    cou_num             = len(set(i_obs))
    model_params_est_q1 = list() 
    for i in range(cou_num):
        q1_mu1_est_list = list()
        q1_mu2_est_list = list()
        
        for j in range(org_num):
            # 属于国家i的机构j
            i_j = np.array(i_obs == i, dtype=np.int32) + np.array(j_obs == j, dtype=np.int32) > 1
            if i_j.any():
                var_params_q3_est_i_j = var_params_q3_est[i_j]
                q2_mu                 = np.mean(var_params_q3_est_i_j[:, 0])         
                q2_log_std            = np.log(max(np.std(var_params_q3_est_i_j[:, 0]), 1e-2))
                q1_mu1_est_list.append(q2_mu)
                q1_mu2_est_list.append(q2_log_std)
        model_params_est_q1.append(np.array([[np.mean(q1_mu1_est_list), np.log(max(np.std(q1_mu1_est_list), 1e-2))],
                                             [np.mean(q1_mu2_est_list), np.log(max(np.std(q1_mu2_est_list), 1e-2))]]))
    model_params_est_q1 = np.array(model_params_est_q1)
   
    # 随机效应模型参数
    mu_P_est           = 0.0
    log_sig_P_est      = np.mean(var_params_q3_est[:, 1])
    model_params_est_P = np.array([[[mu_P_est, log_sig_P_est], [-1, -1]]])
    model_params_est   = np.concatenate([model_params_est_q1, model_params_est_P], axis=0)
    
    return (var_params_q2_est, var_params_q3_est), model_params_est


def evaluate_on_simulation_data(data, model_params_real, var_params_init, model_params_init):
    # var_params_init, model_params_init = var_params_bbvi, model_params_bbvi
    
    def evaluate_real2pred(Y, X):
        # 评价指标:  Y是真实值, X是预测值
        # 相关性评价: Pearsonr
        # 距离评价: MAE, MSE
        if len(Y) > 1:
            cor, pvalue = pearsonr(Y, X)
        else:
            cor = -1
        rmse = np.sqrt(mean_squared_error(Y, X))
        mae = mean_absolute_error(Y, X)
        r2  = r2_score(Y, X)
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
    
    
    var_params_q2_est, var_params_q3_est = var_params_init
    # 国家模型参数q1  (errorbar -> 表示估计的精度 + 方差的准确性)
    q1_real, q1_real_sig = model_params_real[:-1, 0][:, 0], model_params_real[:-1, 0][:, 1]
    q1_est,  q1_est_sig  = model_params_init[:-1, 0][:, 0], model_params_init[:-1, 0][:, 1]
    # plot_q(q1_real, np.exp(q1_real_sig), q1_est, np.exp(q1_est_sig), "Country", r"$\hat{Q}$", 'Real', 'Estimated')
    plot_q(q1_real, q1_est, np.exp(q1_est_sig), "国家", "国家科研能力", '真实值', '估计值', "融合合作信息的科研能力量化模型 (模拟配置7)")
    cor_q1, rmse_q1, mae_q1, r2_q1 = evaluate_real2pred(q1_real, q1_est)

    # 机构q2值   (errorbar > 表示估计的精度 + 方差的准确性)
    q2_real, q2_real_sig = data['q2_obs'][:, 0],     data['q2_obs'][:, 1]
    q2_est,  q2_est_err  = var_params_q2_est[:, 0, 0], var_params_q2_est[:, 1, 0]
    # plot_q(q2_real, np.exp(q2_real_sig), q2_est, np.exp(q2_est_err), "Institution", r"$\hat{Q}$", 'Real', 'Estimated')
    plot_q(q2_real, q2_est, np.exp(q2_est_err), "机构", "机构科研能力", '真实值', '估计值', "融合合作信息的科研能力量化模型 (模拟配置7)")
    cor_q2, rmse_q2, mae_q2, r2_q2 = evaluate_real2pred(q2_real, q2_est)

    # 人员q3值        (errorbar -> 表示估计的精度)
    q3_real = data['q3_obs'].flatten()
    q3_est, q3_est_err = var_params_q3_est[:, 0], var_params_q3_est[:, 1]
    # plot_q(q3_real, 0, q3_est, np.exp(q3_est_err), "Scientists", r"$\hat{Q}$", 'Real', 'Estimated')
    plot_q(q3_real, q3_est, np.exp(q3_est_err), "学者", "学者科研能力", '真实值', '估计值', "融合合作信息的科研能力量化模型 (模拟配置7)")
    cor_q3, rmse_q3, mae_q3, r2_q3 = evaluate_real2pred(q3_real, q3_est)

    
    tb = pt.PrettyTable()
    tb.field_names = ["", "cor", "r2", "rmse", 'mae']
    tb.add_row(["国家", "{:.4f}".format(cor_q1), "{:.4f}".format(r2_q1), "{:.4f}".format(rmse_q1), "{:.4f}".format(mae_q1)])
    tb.add_row(["机构", "{:.4f}".format(cor_q2), "{:.4f}".format(r2_q2), "{:.4f}".format(rmse_q2), "{:.4f}".format(mae_q2)])
    tb.add_row(["学者", "{:.4f}".format(cor_q3), "{:.4f}".format(r2_q3), "{:.4f}".format(rmse_q3), "{:.4f}".format(mae_q3)])
    print(tb)
    
    return (cor_q1, r2_q1, rmse_q1, mae_q1), (cor_q2, r2_q2, rmse_q2, mae_q2), (cor_q3, r2_q3, rmse_q3, mae_q3)
    

def data_exchange(data):
    
    i_obs  = data['i_obs']  
    j_obs  = data['j_obs']
    x_obs  = data['x_obs']
    M_coop = data['M_coop']
    q2_obs = data['q2_obs']
    q3_obs = data['q3_obs']    
    
    cou_num = len(set(i_obs))
    org_num = int(len(set(j_obs)) / len(set(i_obs)))
    M_coop  = M_coop > 0
    total_nop_num, total_noa_num = M_coop.shape
    
    data2 = dict()
    for k in range(total_noa_num):
        # 第k个学者的国家i
        cou_i = int(i_obs[k])
        if cou_i not in data2:
            data2[cou_i] = dict()
        
        # 第k个学者的机构j
        org_j    = int(j_obs[k])
        q2_obs_j = q2_obs[org_j]
        org_j_   = org_j % org_num
        if org_j_ not in data2[cou_i]:
            data2[cou_i][org_j_]       = dict()
            data2[cou_i][org_j_]['q2'] = q2_obs_j
            
        aid_k    = len(data2[cou_i][org_j_]) - 1         # 已经加入 q2_obs_j
        x        = x_obs[M_coop[:, k]].squeeze()         # 第k个人的引用观测数据
        q3_obs_k = q3_obs[k].squeeze()
        
        data2[cou_i][org_j_][aid_k]       = dict()
        data2[cou_i][org_j_][aid_k]['x']  = x
        data2[cou_i][org_j_][aid_k]['q3'] = q3_obs_k     # 第k个人的真实q3  
        
    return data2
    

def compared_with_bbvi_em_org(data):

    # 将data格式转为bbvi_em_org_country需要的格式data2
    data2 = data_exchange(data)
    # 调用compared_with_bbvi_em_org函数完成比较
    q2_ORG, q3_ORG = bbvi_em_org_country.compared_with_bbvi_em_org(data2)
    
    return q2_ORG, q3_ORG 
    

def compared_with_bbvi_em_org_country(data, model_params_real):

    # model_params_real的字典形式
    i_obs   = data['i_obs'] 
    cou_num = len(set(i_obs)) 
    model_params_real2 = dict()
    for i in range(cou_num): 
        model_params_real2[i]       = dict()
        model_params_real2[i]['q1'] =  model_params_real[i]
    model_params_real2["P"] = model_params_real[-1][0]
    
    # 将data格式转为bbvi_em_org_country需要的格式data2
    data2 = data_exchange(data)
    var_params_init2, model_params_init2 = bbvi_em_org_country.max_likelihoood(data2)

    mp_num         = 8
    Epochs         = 10
    step_size      = 1e-1
    num_iters      = 100
    batch_size_org = 512
    num_samples    = 1
    var_params     = var_params_init2      
    model_params   = model_params_init2
    for e in range(Epochs):
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))           
        E_start_time    = time.perf_counter()
        var_params_next = bbvi_em_org_country.EStep_MP(data2, var_params, model_params, batch_size_org, step_size, num_iters, num_samples, mp_num)
        E_end_time      = time.perf_counter()                    
        var_params      = var_params_next
        print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
        
        # M-Step
        print("({}) Optimizing model parameters...".format(e))
        M_start_time      = time.perf_counter()
        model_params_next = bbvi_em_org_country.MStep_MP(data2, var_params, model_params, batch_size_org, step_size, num_iters, num_samples, mp_num)
        M_end_time        = time.perf_counter()
        model_params      = model_params_next
        print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time))
    var_params_bbvi2, model_params_bbvi2 = var_params, model_params
    # 评价
    q1_COU, q2_COU, q3_COU = bbvi_em_org_country.evaluate_on_simulation_data(data2, model_params_real2, var_params_bbvi2,  model_params_bbvi2)
    
    return q1_COU, q2_COU, q3_COU
    

#%%
np.set_printoptions(precision=6, suppress=True)
def BBVI_Algorithm():
    # 生成模拟数据
    mu_0, log_sig_0 = 0., -1.
    mu_1, log_sig_1 = -1., -1.
    mu_2, log_sig_2 = 0, -1.      
    mu_3, log_sig_3 = -1, -1.
    
    # 采样参数 
    cou_num = 10    # 国家数目
    org_num = 10    # 机构数目
    aid_num = 10    # 人员数目
    nop_num = 10    # 服从泊松分布 poisson(nop_num) 每位作者的平均发文量数目
    coa_num = 3     # 服从泊松分布 poisson(coa_num) 每篇文章的平均合著作者数目
    group_num = 8
    sampling_params = [cou_num, org_num, aid_num, nop_num, coa_num, group_num]
    
    # 国家模型参数: 
    mu1_cou_list      = sampling_normal(mu_0, log_sig_0, cou_num)       # 国家研究能力的均值: 描述国家下机构研究能力的平均大小情况 —— 其值越大, 该国家下机构内人员能力越大
    log_sig1_cou_list = sampling_normal(mu_1, log_sig_1, cou_num)       # 国家研究能力的均值的方差: 描述国家下机构研究能力的平均大小的浮动情况
    mu2_cou_list      = sampling_normal(mu_2, log_sig_2, cou_num)       # 国家研究能力的方差: 描述国家下机构研究能力的平均浮动情况 —— 其值越大, 该国家下机构内人员能力差异大
    log_sig2_cou_list = sampling_normal(mu_3, log_sig_3, cou_num)       # 国家研究能力的方差的方差: 描述国家下机构研究能力的平均浮动的浮动情况
   
    model_params_real_q1 = np.concatenate([np.concatenate([mu1_cou_list, log_sig1_cou_list], axis=-1),
                                           np.concatenate([mu2_cou_list, log_sig2_cou_list], axis=-1)], axis=-1)
    model_params_real_q1 = model_params_real_q1.reshape((cou_num, 2, 2))
   
    # 随机波动模型参数 & 补两个无意义的参数, 方便拼接
    mu_P_real           = 0
    log_sig_P_real      = 0.5
    model_params_real_P = np.array([[mu_P_real, log_sig_P_real], [-1, -1]])
    model_params_real   = np.concatenate([model_params_real_q1, [model_params_real_P]], axis=0)
    
    # 采样模拟数据
    data = create_simulation_data(model_params_real, sampling_params)
    # 极大似然估计 -> 不考虑合作时, 个能能力被低估
    var_params_init, model_params_init = max_likelihoood(data)
    
    # 贝叶斯后验估计
    mp_num      = 8
    Epochs      = 10
    step_size   = 1e-1
    num_iters   = 100
    num_samples = 1
    var_params, model_params = var_params_init, model_params_init
    for e in range(Epochs):
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))           
        E_start_time    = time.perf_counter()
        var_params_next = Estep(data, model_params, var_params, num_samples, step_size, num_iters)
        E_end_time      = time.perf_counter()                      
        var_params      = var_params_next
        print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
    
        # M-Step
        print("({}) Optimizing model parameters...".format(e))
        M_start_time      = time.perf_counter()
        model_params_next = Mstep(data, model_params, var_params, num_samples, step_size, num_iters)
        M_end_time        = time.perf_counter()
        model_params      = model_params_next
        print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time))
    # 变分参数估计 - BBVI-EM
    var_params_bbvi, model_params_bbvi = var_params, model_params
    var_params_init, model_params_init = max_likelihoood(data)
    
    # 绘图检查
    q1_WSB, q2_WSB, q3_WSB = evaluate_on_simulation_data(data, model_params_real, var_params_init, model_params_init)
    q1_COO, q2_COO, q3_COO = evaluate_on_simulation_data(data, model_params_real, var_params_bbvi, model_params_bbvi)
    q2_ORG, q3_ORG         = compared_with_bbvi_em_org(data)
    q1_COU, q2_COU, q3_COU = compared_with_bbvi_em_org_country(data, model_params_real)
    
    return (q1_WSB, q2_WSB, q3_WSB), (q1_WSB, q2_ORG, q3_ORG), (q1_COU, q2_COU, q3_COU), (q1_COO, q2_COO, q3_COO)


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
    
    times   = 10
    number  = 3  # 记录模拟实验数据编号  
    results = dict()
    for t in range(times):
        (q1_WSB, q2_WSB, q3_WSB), (q1_ORG, q2_ORG, q3_ORG), (q1_COU, q2_COU, q3_COU), (q1_COO, q2_COO, q3_COO) = BBVI_Algorithm()
        results[t] = dict()
        results[t]["wsb"] = (q1_WSB, q2_WSB, q3_WSB)
        results[t]["org"] = (q1_ORG, q2_ORG, q3_ORG)
        results[t]["cou"] = (q1_COU, q2_COU, q3_COU)
        results[t]["coo"] = (q1_COO, q2_COO, q3_COO)
    # 储存
    with open("./Results_coop_org_country/simulation_{}.pkl".format(number), 'wb') as f:
        pickle.dump(results, f)
    # 读取
    with open("./Results_coop_org_country/simulation_{}.pkl".format(number), 'rb') as f:
        results = pickle.load(f)
        
        
    q1_WSB_list, q2_WSB_list, q3_WSB_list = get_list(results, "wsb")
    _,           q2_ORG_list, q3_ORG_list = get_list(results, "org")
    q1_COU_list, q2_COU_list, q3_COU_list = get_list(results, "cou")
    q1_COO_list, q2_COO_list, q3_COO_list = get_list(results, "coo")
    
    # 国家科研能力评价
    q1_COU_mean = np.mean(q1_COU_list, axis=0)
    q1_COO_mean = np.mean(q1_COO_list, axis=0)
    print("COU(q1): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q1_COU_mean))
    print("COO(q1): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q1_COO_mean))
    pvalue_list = list()
    for i in range(4):
        _, pvalue = ttest_rel(q1_COU_list[:, i], q1_COO_list[:, i])
        pvalue_list.append(pvalue)
    print("***(q1): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*pvalue_list))
    
    # 机构科研能力评价
    q2_ORG_mean = np.mean(q2_ORG_list, axis=0)
    q2_COU_mean = np.mean(q2_COU_list, axis=0)
    q2_COO_mean = np.mean(q2_COO_list, axis=0)
    print("ORG(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q2_ORG_mean))
    print("COU(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q2_COU_mean))
    print("COO(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q2_COO_mean))
    pvalue_list = list()
    for i in range(4):
        _, pvalue = ttest_rel(q2_COU_list[:, i], q2_COO_list[:, i])
        pvalue_list.append(pvalue)
    print("***(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*pvalue_list))
    
    # 学者科研能力评价
    q3_WSB_mean = np.mean(q3_WSB_list, axis=0)
    q3_ORG_mean = np.mean(q3_ORG_list, axis=0)
    q3_COU_mean = np.mean(q3_COU_list, axis=0)
    q3_COO_mean = np.mean(q3_COO_list, axis=0)
    print("WSB(q3): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q3_WSB_mean))
    print("ORG(q3): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q3_ORG_mean))
    print("COU(q3): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q3_COU_mean))
    print("COO(q3): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*q3_COO_mean))
    pvalue_list = list()
    for i in range(4):
        _, pvalue = ttest_rel(q3_COU_list[:, i], q3_COO_list[:, i])
        pvalue_list.append(pvalue)
    print("***(q2): Pearsonr: {:.4f}, R2: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}".format(*pvalue_list))
    
    
    
