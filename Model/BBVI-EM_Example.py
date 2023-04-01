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

from tqdm import tqdm
import time


#%%
def log_p_zx_density(z, x, model_params):
    # x: 观测, z: 隐变量, model_params: 模型参数
    mu_Q, mu_P, log_sig_Q, log_sig_P = model_params
    log_Q_density = norm.logpdf(z, mu_Q, np.exp(log_sig_Q)) 
    log_P_density = norm.logpdf(x - z, mu_P, np.exp(log_sig_P))   # mu_p = 0
    # log(pq) = logp + logq
    return log_Q_density + log_P_density, log_Q_density, log_P_density


def sampling(mean, log_std, num_samples):
    rs = npr.RandomState()
    samples = rs.randn(num_samples, 1) * np.exp(log_std) + mean
    return samples


def black_box_variational_inference_Estep(logprob, x_i, model_params):
                                          
    def unpack_params(var_params):
        mu_lamb, log_sig_lamb = var_params
        return mu_lamb, log_sig_lamb
    
    def variational_objective(var_params_i, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mu_lamb_i, log_sig_lamb_i = unpack_params(var_params_i)
        
        # 从q(z; mu_lamb, sig_lamb)中抽取样本, 每个作者自身的变分参数
        z_i = sampling(mu_lamb_i, log_sig_lamb_i, 1)
        
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


def black_box_variational_inference_Mstep(logprob, x_obs, var_params):
                                         
                                          
    def unpack_params(var_params):
        mu_lamb, log_sig_lamb = var_params
        return mu_lamb, log_sig_lamb
    
    def variational_objective(model_params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        z = list()
        x = list()
        for i in range(len(x_obs)):
            x_obs_i = x_obs[i]['x']
            var_params_i  = var_params[i]
            mu_lamb_i, log_sig_lamb_i = unpack_params(var_params_i)
            # 从q(z)中抽取样本
            z_i = sampling(mu_lamb_i, log_sig_lamb_i, 1)[0]
            z.append(np.ones((len(x_obs_i), 1)) * z_i)
            #
            x.append(x_obs_i)
        z = np.concatenate(z, axis=0)
        x = np.concatenate(x, axis=0)
         
        # Black box variational inference 内的 ELBO表达式:Eq[logp - logq]
        log_p_zx, log_Q_density, log_P_density = logprob(z, x, model_params)
        part1 = np.mean(log_Q_density)
        lower_bound = part1
        
        # 求ELBO最大, 所以这里加个负号即minimize
        return -lower_bound
    
    # 求ELBO的梯度
    gradient = grad(variational_objective)
    return variational_objective, gradient, unpack_params


#%%
def create_simulation_data(model_params, sampling_params):
    # 抽样参数
    aid_num, nop_num = sampling_params
    
    # 平均每个人发表avg_nop_num篇文章
    mu_Q, mu_P, log_sig_Q, log_sig_P = model_params
    # 抽取每个作者的Q值, 抽取每位作者每篇文章的P值
    q_obs = sampling(mu_Q, log_sig_Q, aid_num)
    x_obs = dict()
    for i in range(aid_num):
        x_obs[i] = dict()
        p_obs_i = sampling(mu_P, log_sig_P, nop_num)
        x_obs[i]['q'] = q_obs[i]
        x_obs[i]['p'] = p_obs_i
        x_obs[i]['x'] = q_obs[i] + p_obs_i
    return x_obs


def evaluate_on_simulation_data(x_obs, var_params, model_params, model_params_real):
    # q_obs: 模拟生成的Q能力观测
    # var_params: 变分参数估计值, 理应与q_obs类似
    # model_params: 模型参数估计值
    # model_params_real: 模型真实参数
    
    q_obs = list()
    var_params_q = list()
    var_params_q_err = list()
    for i in range(len(x_obs)):
        q_obs.append(x_obs[i]['q'][0])
        var_params_q.append(var_params[i][0])
        var_params_q_err.append(var_params[i][1])
    
    
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    plt.plot(np.arange(len(q_obs)), q_obs, 
             label='Q', c='gray', marker='s', alpha=0.75)
             
    plt.errorbar(np.arange(len(q_obs)), var_params_q, yerr=var_params_q_err, 
                 label=r'$\mu_\lambda$', fmt="o:", color='blue', ecolor='black', capsize=6, markersize=3)
                 
    mu_Q, mu_P, log_sig_Q, log_sig_P = model_params
    mu_Q_real, mu_P_real, log_sig_Q_real, log_sig_P_real = model_params_real
    plt.text(len(q_obs) * 0.5, -2, fontsize=25, color='blue',
             s=r"($\hat\mu_Q$={:.2f}, $\hat\sigma_Q$={:.2f})".format(mu_Q, np.exp(log_sig_Q)))
    plt.text(len(q_obs) * 0.5, -3, fontsize=25, color='gray',
             s=r"($\mu_Q$={:.2f}, $\sigma_Q$={:.2f})".format(mu_Q_real, np.exp(log_sig_Q_real)))
    plt.ylabel("Q value")
    plt.xlabel("Scientists")
    plt.legend(frameon=False, loc='upper right', fontsize=25)


np.set_printoptions(precision=6, suppress=True)
def BBVI_Algorithm(): 
    
    
    '''
    模型结构:
        # (1) 抽取个人能力Q值
        Q  ~ Normal(mu_Q_real,       np.exp(log_sig_Q_real))
        
        # (2) 抽取机会随机性P值
        P ~ Normal(mu_P_real, log_sig_P_real)
        
        # 观测引用X = Q + P
        X = Q + P
        
        因此隐变量是 Q
    '''
    
    # 真实模型参数: mu_Q, sig_Q, mu_P, sig_P
    mu_Q_real, log_sig_Q_real = 3.0, 1.0
    mu_P_real, log_sig_P_real = 0.0, 0.0     
    model_params_real = np.array([mu_Q_real, mu_P_real, log_sig_Q_real, log_sig_P_real])
    
    # 生成过程随机生成数据
    aid_num = 100
    nop_num = 20
    sampling_params = [aid_num, nop_num]
    x_obs = create_simulation_data(model_params_real, sampling_params)
    
    
    # 待估计模型参数初始化
    mu_Q, log_sig_Q = 0.0, 1.0
    mu_P, log_sig_P = 0.0, 0.0
    model_params = np.array([mu_Q, mu_P, log_sig_Q, log_sig_P])
    
    # 待估计变分参数初始化: mu_lamb, sig_lamb (变分参数个数 = 作者数目 * 2个)
    mu_lamb, log_sig_lamb = 0.0, 1.0
    var_params = dict()
    for i in range(len(x_obs)):
        var_params[i] = np.array([mu_lamb, log_sig_lamb])
    
    Epochs = 10
    step_size = 5e-2
    num_iters = 100
    for e in range(Epochs):
        
        # E-Step
        print("({}) Optimizing variational parameters...".format(e))    
        time.sleep(.25)
        # 每位作者逐个更新
        for i in tqdm(range(len(x_obs))):
            # 第i位作者的观测数据信息
            x_obs_i = x_obs[i]['x']
            # 第i位作者的变分参数信息
            var_params_i = var_params[i]
            
            # Build variational objective.
            objective, gradient, _ = \
                black_box_variational_inference_Estep(log_p_zx_density, x_obs_i, model_params)
                
            # 变分优化
            init_var_params = var_params_i
            var_params_i_next = adam(gradient, init_var_params, step_size=step_size, num_iters=num_iters)
            var_params[i] = var_params_i_next
        time.sleep(.25)
        
        # M-Step
        print("({}) Optimizing model parameters...".format(e))          
        objective2, gradient2, _ = \
                black_box_variational_inference_Mstep(log_p_zx_density, x_obs, var_params)   
        init_model_params = model_params
        model_params_next = adam(gradient2, init_model_params, step_size=5e-2, num_iters=100)
    
        print(model_params_next)   
        epsilon = np.linalg.norm(model_params_next - model_params)
        model_params = model_params_next
        if epsilon < 1e-2:
            break
        print("eps = {:.4f} \n".format(epsilon))
        
    # 评价结果
    evaluate_on_simulation_data(x_obs, var_params, model_params, model_params_real)
                                    
                                    
