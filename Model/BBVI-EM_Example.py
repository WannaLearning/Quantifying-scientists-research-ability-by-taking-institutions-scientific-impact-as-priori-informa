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

import multiprocessing
import time
import math
from tqdm import tqdm
from MyQPModel import ml


# q(z)的抽样过程嵌入多进程 multi process

#%%
def log_p_zx_density(z, x, model_params):
    # x: 观测, z: 隐变量, model_params: 模型参数
    mu_Q, mu_P, log_sig_Q, log_sig_P = model_params
    log_Q_density = norm.logpdf(z, mu_Q, np.exp(log_sig_Q)) 
    log_P_density = norm.logpdf(x - z, mu_P, np.exp(log_sig_P))
    # log(q_density * p_density) = log(q_density) + log(p_density) 
    return log_Q_density + log_P_density, log_Q_density, log_P_density


def sampling(mean, log_std, num_samples):
    rs = npr.RandomState()
    samples = rs.randn(num_samples, len(mean)) * np.exp(log_std) + mean
    return samples


def black_box_variational_inference_Estep(logprob, 
                                          num_samples_i,
                                          x_i, model_params):
    def unpack_params(var_params):
        # Variational dist is a diagonal Gaussian.
        D = int(len(var_params) / 2) 
        mu_lamb, log_sig_lamb = var_params[:D], var_params[D:]
        return mu_lamb, log_sig_lamb
    
    def variational_objective(var_params_i, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mu_lamb_i, log_sig_lamb_i = unpack_params(var_params_i)
        
        # 从q(z; mu_lamb, sig_lamb)中抽取样本, 每个作者自身的变分参数
        z_i = sampling(mu_lamb_i, log_sig_lamb_i, num_samples_i)
        
        # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
        log_p_zx, _, _ = logprob(z_i, x_i, model_params)
        part1 = np.mean(log_p_zx)
        part2 = np.mean(norm.logpdf(z_i, mu_lamb_i, np.exp(log_sig_lamb_i)))
        lower_bound = part1 - part2 
        
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound
    
    # 求ELBO的梯度
    gradient = grad(variational_objective)
    return variational_objective, gradient, unpack_params


def black_box_variational_inference_Mstep(logprob, 
                                          num_samples,
                                          x_obs, var_params, update_Q):
    def unpack_params(var_params):
        # Variational dist is a diagonal Gaussian.
        D = int(len(var_params) / 2) 
        mu_lamb, log_sig_lamb = var_params[:D], var_params[D:]
        return mu_lamb, log_sig_lamb
    
    def variational_objective(model_params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        z = list()
        for i, _ in enumerate(var_params):
            # 
            num = sum(num_samples[: i]) 
            num_samples_i = num_samples[i]
            x_obs_i = x_obs[num: num + num_samples_i]
            
            var_params_i  = var_params[i]
            mu_lamb_i, log_sig_lamb_i = unpack_params(var_params_i)
            # 从q(z; mu_lamb, sig_lamb)中抽取样本, 每个作者自身的变分参数
            z_i = sampling(mu_lamb_i, log_sig_lamb_i, num_samples_i)
            z.append(z_i)
        z = np.concatenate(z, axis=0)
        
        # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
        log_p_zx, log_Q_density, log_P_density = logprob(z, x_obs, model_params)
        if update_Q:
            part1 = np.mean(log_Q_density)
        else:
            part1 = np.mean(log_P_density)
        # part1 = np.mean(log_p_zx)
        lower_bound = part1
        
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound
    
    # 求ELBO的梯度
    gradient = grad(variational_objective)
    return variational_objective, gradient, unpack_params


#%%
# 多进程函数
def Estep_MP_Func(num_samples_mp_i, x_obs_mp_i, var_params_mp_i, model_params,
                  step_size, num_iters, 
                  mp_i):
    '''每个进程更新变分参数'''
    # num_samples: 每位作者的文章数目, 对应于需要采样的数目. list()
    # x_obs: 所有作者的观测变量, 对应于每篇文章的应用数目. list()
    # var_params: 变分参数, 通过极大化ELBO估计
    # model_params: 模型参数, 此时固定
    
    num_authors_mp_i = len(num_samples_mp_i)
    for j in range(num_authors_mp_i):
        # 每位作者逐个更新, 第i位作者的数据信息
        num_j = num_samples_mp_i[j]
        num_until_j = sum(num_samples_mp_i[: j]) 
        x_obs_j = x_obs_mp_i[num_until_j: num_until_j + num_j]
        # 第i位作者的变分参数信息
        var_params_j = var_params_mp_i[j]
        
        # Build variational objective.
        objective, gradient, _ = \
            black_box_variational_inference_Estep(log_p_zx_density, num_j, x_obs_j, model_params)
            
        # 变分优化
        init_var_params = var_params_j
        variational_params = adam(gradient, init_var_params, step_size=step_size, num_iters=num_iters)
        var_params_mp_i[j] = variational_params
        
    return (var_params_mp_i, mp_i)
    

def EStep_MP(num_samples, x_obs, var_params, model_params,
             step_size, num_iters,
             mp_num):
    '''多进程更新作者变分参数'''
    num_authors = len(num_samples)
    batchsize = math.ceil(num_authors / mp_num)
    
    pool = multiprocessing.Pool(processes=mp_num)
    results = list()
    for mp_i in range(mp_num):
        begin = batchsize * mp_i 
        end   = min(batchsize * (mp_i + 1), num_authors)

        num_samples_mp_i = num_samples[begin: end]
        x_obs_mp_i = x_obs[sum(num_samples[: begin]): sum(num_samples[: end])]
        var_params_mp_i = var_params[begin: end]
        
        # 储存每个进程更新后的结果
        results.append(pool.apply_async(Estep_MP_Func, (num_samples_mp_i, x_obs_mp_i,
                                                        var_params_mp_i, model_params,
                                                        step_size, num_iters, mp_i, )))
    pool.close()
    pool.join()
    
    var_params_next = list()
    for res in results:
        var_params_mp_i, mp_i = res.get()
        var_params_next.append(var_params_mp_i)
        # print(mp_i)
    var_params_next = np.concatenate(var_params_next, axis=0)
    return var_params_next
    

def MStep_MP_func(num_samples_mp_i, var_params_mp_i, 
                  mp_i):
    '''每个进程采样'''
    def unpack_params(var_params):
        # Variational dist is a diagonal Gaussian.
        D = int(len(var_params) / 2) 
        mu_lamb, log_sig_lamb = var_params[:D], var_params[D:]
        return mu_lamb, log_sig_lamb
    
    z_mp_i = list()
    for j, _ in enumerate(var_params_mp_i):
        # 
        num_j = num_samples_mp_i[j]
        var_params_j  = var_params_mp_i[j]
        mu_lamb_j, log_sig_lamb_j = unpack_params(var_params_j)
        # 从q(z; mu_lamb, sig_lamb)中抽取样本, 每个作者自身的变分参数
        z_j = sampling(mu_lamb_j, log_sig_lamb_j, num_j)
        z_mp_i.append(z_j)
    z_mp_i = np.concatenate(z_mp_i, axis=0)
    return (z_mp_i, mp_i)


def MStep_MP(num_samples, x_obs, var_params, model_params,
             step_size, num_iters,
             mp_num):
    '''多进程采样后更新模型参数'''
    def variational_objective(model_params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        
        # 多进程采样
        num_authors = len(num_samples)
        batchsize = math.ceil(num_authors / mp_num)
        
        pool = multiprocessing.Pool(processes=mp_num)
        results = list()
        for mp_i in range(mp_num):
            begin = batchsize * mp_i 
            end   = min(batchsize * (mp_i + 1), num_authors)
    
            num_samples_mp_i = num_samples[begin: end]
            x_obs_mp_i = x_obs[sum(num_samples[: begin]): sum(num_samples[: end])]
            var_params_mp_i = var_params[begin: end]
            
            # 储存每个进程更新后的结果
            results.append(pool.apply_async(MStep_MP_func, (num_samples_mp_i, var_params_mp_i, mp_i, )))
        pool.close()
        pool.join()
        
        # 合并多进程的抽样结果
        z = list()
        for res in results:
            z_mp_i, mp_i = res.get()
            z.append(z_mp_i)
            # print(mp_i)
        z = np.concatenate(z, axis=0)
        
        log_p_zx, log_Q_density, log_P_density = log_p_zx_density(z, x_obs, model_params)
        if True:
            part1 = np.mean(log_Q_density)
        else:
            part1 = np.mean(log_P_density)
        # part1 = np.mean(log_p_zx)
        lower_bound = part1
        
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound
    
    # 求ELBO的梯度
    gradient = grad(variational_objective)
    
    init_model_params = model_params
    model_params_next = adam(gradient, init_model_params, step_size=step_size, num_iters=num_iters)
    
    return model_params_next


def MStep_non_MP(num_samples, x_obs, var_params, model_params, 
                 step_size, num_iters,
                 update_Q):
    
    # 当作者数目较小时, 多进程采样会更慢
    variational_objective, gradient, _ = black_box_variational_inference_Mstep(log_p_zx_density, 
                                                                                num_samples,
                                                                                x_obs, var_params, 
                                                                                update_Q)
    init_model_params = model_params
    model_params_next = adam(gradient, init_model_params, step_size=step_size, num_iters=num_iters)
    return model_params_next

#%%
def create_simulation_data(model_params, 
                           num_authors, 
                           avg_nop):
    # 生成模拟数据
    # model_params: 模型参数
    # num_authors : 模拟作者数目
    # avg_nop_num : 人均发文量
    
    # 平均每个人发表avg_nop_num篇文章
    mu_Q, mu_P, log_sig_Q, log_sig_P = model_params
    # 抽取每个作者的Q值, 抽取每位作者每篇文章的P值
    q_obs = sampling([mu_Q], [log_sig_Q], num_authors)
    p_obs = sampling([mu_P], [log_sig_P], num_authors * avg_nop)
    # 相加即是观测变量 (引用)
    q_obs_cpy = np.concatenate([q_obs for i in range(avg_nop)], axis=1)
    q_obs_cpy = q_obs_cpy.flatten().reshape((-1, 1))
    x_obs = q_obs_cpy + p_obs
    return x_obs, q_obs, p_obs, q_obs_cpy


def evaluate_on_simulation_data(q_obs,
                                var_params,
                                model_params, model_params_real,
                                normalized):
    # q_obs: 模拟生成的Q能力观测
    # var_params: 变分参数估计值, 理应与q_obs类似
    # model_params: 模型参数估计值
    # model_params_real: 模型真实参数
    
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    def nomralized_func(x):
        return (x - np.mean(x)) / np.std(x)
        
    
    if normalized:
        y1 = q_obs.flatten()
        y2 =  var_params[:, 0].flatten()
        y2_err = var_params[:, 1].flatten()
        
        y1 = nomralized_func(y1)
        y2_err = y2_err / np.std(y2) * 0
        y2 = nomralized_func(y2)
    else:
        y1 = q_obs.flatten()
        y2 =  var_params[:, 0].flatten()
        y2_err = var_params[:, 1].flatten()
        
    plt.plot(np.arange(len(q_obs)), y1, 
             label='Q',
             c='gray', marker='s', alpha=0.75)
    plt.errorbar(np.arange(len(q_obs)), y2, yerr=y2_err, 
                 label=r'$\mu_\lambda$',
                 fmt="o:", color='blue', ecolor='black', capsize=3, markersize=5)
    
    mu_Q, mu_P, log_sig_Q, log_sig_P = model_params
    mu_Q_real, mu_P_real, log_sig_Q_real, log_sig_P_real = model_params_real

    plt.text(len(q_obs) * 0.5, (max(y1) - min(y1)) * 0.15 + min(y1), fontsize=25, color='blue',
             s=r"($\hat\mu_Q$={:.2f}, $\hat\sigma_Q$={:.2f})".format(mu_Q, np.exp(log_sig_Q)))
    plt.text(len(q_obs) * 0.5, (max(y1) - min(y1)) * 0.05 + min(y1), fontsize=25, color='gray',
             s=r"($\mu_Q$={:.2f}, $\sigma_Q$={:.2f})".format(mu_Q_real, np.exp(log_sig_Q_real)))
    plt.ylabel("Q value")
    plt.xlabel("Scientists")
    plt.legend(frameon=False, loc='upper right', fontsize=25)


def BBVI_Algorithm():
    # 真实模型参数: mu_Q, sig_Q, mu_P, sig_P
    mu_Q_real, log_sig_Q_real = 3.0, 1.0
    mu_P_real, log_sig_P_real = 0.0, 1.0     # *** 这个参数已知, 不更新, 区别于王大顺. (因为两个正态分布相加, 导致估计不稳定)
    model_params_real = np.array([mu_Q_real, mu_P_real, log_sig_Q_real, log_sig_P_real])
    
    # 生成过程随机生成数据
    num_authors = 100
    avg_nop = 50
    num_samples = [avg_nop for i in range(num_authors)]
    x_obs, q_obs, p_obs, q_obs_cpy = create_simulation_data(model_params_real, num_authors, avg_nop)

    
    # 待估计模型参数初始化
    mu_Q, log_sig_Q = 0.0, 0.0
    mu_P, log_sig_P = 0.0, 1.0
    model_params = np.array([mu_Q, mu_P, log_sig_Q, log_sig_P])
    
    # 待估计变分参数初始化: mu_lamb, sig_lamb (变分参数个数 = 作者数目 * 2个)
    mu_lamb, log_sig_lamb = 0.0, 1.0
    var_params = np.array([[mu_lamb, log_sig_lamb] for i in range(num_authors)])
    
    np.set_printoptions(precision=6, suppress=True)
    update_Q = True
    Epochs = 50
    step_size = 5e-2
    num_iters = 100
    mp_num = 7
    for e in range(Epochs):    
        start_time = time.perf_counter()
        
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))    
        var_params = EStep_MP(num_samples, x_obs, var_params, model_params, step_size, num_iters, mp_num)
        
        # M-Step
        print("({}) Optimizing model parameters...".format(e))          
        # model_params_next = MStep_MP(num_samples, x_obs, var_params, model_params, step_size, num_iters, mp_num)
        model_params_next = MStep_non_MP(num_samples, x_obs, var_params, model_params, step_size, num_iters, True)
        
        end_time = time.perf_counter()
        epsilon = np.linalg.norm(model_params_next - model_params)      # 判断收敛
        model_params = model_params_next
        print("耗时: {}".format( round(end_time - start_time)))
        print("模型参数: ", model_params_next)
        print("eps = {:.4f} \n".format(epsilon))
        if epsilon < 1e-2:
            break
        
    # 评价结果
    evaluate_on_simulation_data(q_obs, var_params, model_params, model_params_real, True)
    evaluate_on_simulation_data(q_obs, var_params, model_params, model_params_real, False)
    
    model_params_ml, var_params_ml = ml.max_likelihood(num_samples, x_obs)
    evaluate_on_simulation_data(q_obs, var_params_ml, model_params_ml, model_params_real, True)
    evaluate_on_simulation_data(q_obs, var_params_ml, model_params_ml, model_params_real, False) 
