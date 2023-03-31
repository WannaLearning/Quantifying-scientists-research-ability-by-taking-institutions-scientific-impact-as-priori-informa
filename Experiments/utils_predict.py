#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 11:34:43 2022

@author: aixuexi
"""
import os
import pickle
import numpy as np
import autograd.numpy.random as npr
import prettytable as pt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square



def save_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def sort_aid_cc(yearlynop2cc, beforeyear):
    '''获取 beforeyear 年的cc_list'''
    yearlist  = sorted(yearlynop2cc.keys())  # 职业生涯跨度 
    startyear = min(yearlist)                # 发表第一篇文章的时间
    endyear   = max(yearlist)                # 发表最后一篇时间
    
    pid2cc    = dict()
    for year in yearlist:
        if year <= beforeyear:
            
            # 存在标准化引用 czs
            if len(yearlynop2cc[year][0]) == 3:
                for pid, c10, czs in yearlynop2cc[year]:
                    if pid not in pid2cc:
                        pid2cc[pid] = c10  
            # 不存在标准化引用 czs
            if len(yearlynop2cc[year][0]) == 2:  
                for pid, c10 in yearlynop2cc[year]:
                    if pid not in pid2cc:
                        pid2cc[pid] = c10  
                        
    pidlist   = list(pid2cc.keys())
    cclist    = np.array([pid2cc[pid] for pid in pidlist])
    
    return cclist, pidlist


def calculate_h_index(aid_cc_list):
    '''计算h指数'''
    aid_cc_list = sorted(aid_cc_list, reverse = True)
    
    for h in range(len(aid_cc_list), 0, -1):
        if aid_cc_list[h-1] >= h:
            return h 
    return 0


def evaluate_real2pred(Y, X):
    # Y: 真实值; X: 预测值
    cor, pvalue = pearsonr(Y, X)
    r2          = r2_score(Y, X)
    rmse        = np.sqrt(mean_squared_error(Y, X))
    mae         = mean_absolute_error(Y, X)
    return cor, r2, rmse, mae


def print_real2pred_tb(real_pred_WSB, real_pred_OUR, real_pred_AVG, title):
    # 计算评价指标
    cor_WSB, r2_WSB, rmse_WSB, mae_WSB = evaluate_real2pred(real_pred_WSB[:, 0], real_pred_WSB[:, 1])
    cor_OUR, r2_OUR, rmse_OUR, mae_OUR = evaluate_real2pred(real_pred_OUR[:, 0], real_pred_OUR[:, 1])
    cor_AVG, r2_AVG, rmse_AVG, mae_AVG = evaluate_real2pred(real_pred_AVG[:, 0], real_pred_AVG[:, 1])
    # 通过表格展示指标
    tb             = pt.PrettyTable()
    tb.title       = title
    tb.field_names = ["Model", "Pearsonr", "R2", "RMSE", "MAE"]
    tb.add_row(["AVG", "{:.4f}".format(cor_AVG), "{:.4f}".format(r2_AVG), "{:.4f}".format(rmse_AVG), "{:.4f}".format(mae_AVG)])
    tb.add_row(["WSB", "{:.4f}".format(cor_WSB), "{:.4f}".format(r2_WSB), "{:.4f}".format(rmse_WSB), "{:.4f}".format(mae_WSB)])
    tb.add_row(["Our", "{:.4f}".format(cor_OUR), "{:.4f}".format(r2_OUR), "{:.4f}".format(rmse_OUR), "{:.4f}".format(mae_OUR)])
    print(title)
    print(tb)
    
    result_WSB = (cor_WSB, r2_WSB, rmse_WSB, mae_WSB)
    result_OUR = (cor_OUR, r2_OUR, rmse_OUR, mae_OUR)
    result_AVG = (cor_AVG, r2_AVG, rmse_AVG, mae_AVG)
    
    return result_AVG, result_WSB, result_OUR


def plot_real2pred(h_real_pred_WSB, h_real_pred, alpha,
                   logscale, color, Y2,
                   xticks, xlabel, ylabel, label, title):

    def calculate_errorbar(X, Y, xticks):
        # 根据xticks划分bin箱
        bin_value    = dict()
        for i in range(len(xticks)):
            if i < len(xticks) - 1:
                down = xticks[i]
                up   = xticks[i+1]
                bin_value[(down, up)] = list()
        # 分类bin箱内x值
        for x, y in zip(X, Y):
            for down, up in bin_value:
                if down <= x and x < up:
                    bin_value[(down, up)].append([x, y])
        # 计算每个bin箱内的均值和标准差
        bin_avg_x   = list()
        bin_avg_y   = list()
        bin_err_y   = list()
        for key in bin_value:
            xy = np.array(bin_value[key])
            if len(xy) > 1:
                bin_avg_x.append(np.mean(xy[:, 0]))
                bin_avg_y.append(np.mean(xy[:, 1]))
                bin_err_y.append(np.std(xy[:, 1]))
         
        print(len(bin_avg_x), len(bin_avg_y), len(bin_err_y))
        return bin_avg_x, bin_avg_y, bin_err_y
    
    # Y轴是真实值; X轴是预测值
    Y_OUR, X_OUR = h_real_pred[:, 0],     h_real_pred[:, 1]
    Y_WSB, X_WSB = h_real_pred_WSB[:, 0], h_real_pred_WSB[:, 1]
    
    bin_avg_x_OUR, bin_avg_y_OUR, bin_err_y_OUR = calculate_errorbar(X_OUR, Y_OUR, xticks)
    # bin_avg_x_WSB, bin_avg_y_OUR, bin_err_y_WSB = calculate_errorbar(X_WSB, Y_WSB, xticks)

    # print(len(bin_avg_x_OUR), len(bin_avg_y_OUR), len(bin_err_y_OUR))
    
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman", # SimHei
              "font.size" : 22
              }
    rcParams.update(config)
    
    s = 2
    plt.scatter(X_OUR, Y_OUR, c=color,  s=s, alpha=alpha)
    plt.errorbar(bin_avg_x_OUR,  bin_avg_y_OUR, yerr=bin_err_y_OUR, label=label,
                  fmt='o:', ecolor='black', elinewidth=1, ms=10, mfc=color,  mec='black', capsize=5, linewidth=0)
    
    # plt.scatter(X_WSB, Y_WSB, c="gray", s=s, alpha=alpha)
    # plt.errorbar(bin_avg_x_WSB,  bin_avg_y_OUR, yerr=bin_err_y_WSB, label="Q model",
    #               fmt='o:', ecolor='gray', elinewidth=3,  ms=12, mfc='gray', mec='brown', capsize=10, linewidth=0)
    # # 对角线
    plt.plot([xticks[0], xticks[-1]], [xticks[0], xticks[-1]], color='gray', linestyle='--')
    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.xticks(xticks)
    plt.yticks(xticks)
    plt.xlim(xticks[0], xticks[-1])
    plt.ylim(xticks[0], xticks[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left', frameon=False)
    plt.grid(linestyle='--')
    # plt.title(title + "({}年)".format(Y2))
    plt.title(title + " ({})".format(Y2))

def plot_Q(aid2Q, aid2Q_WSB):
    # 观察两个模型预测的Q是否一致
    Q_list = list()
    Q_WSB_list = list()
    for aid in aid2Q:
        Q = aid2Q[aid][0]
        Q_WSB = aid2Q_WSB[aid][0]
        Q_list.append(Q)
        Q_WSB_list.append(Q_WSB)
        
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    plt.scatter(np.arange(len(Q_list)), Q_list, c='red', s=1)
    plt.scatter(np.arange(len(Q_list)), Q_WSB_list, c='gray', s=1)


# 使用beforen年数据预测Q值得, 再根据Q值预测h指数, 累计引用数目, c* 
# 划分时间段进行预测 (using only early-career information to estimate Q parameter)
def predict_func(mu_P, log_sig_P, aid2Q, 
                 targeted_aid,
                 beforeyear, afteryear,
                 sampling_times=100):
    ''' 融合学术环境信息的科研能力量化模型评价预测性
        Sintera基准模型
        融合机构信息的科研能力量化模型
        融合国家信息的科研能力量化模型
        融合合作信息的科研能力量化模型
    '''
    
    aids = list(aid2Q.keys())
    aids = sorted(aids)
    rs   = npr.RandomState()
    
    hx_results = list()  # h-index 结果
    tc_results = list()  # total citatoins 结果
    cs_results = list()  # c* 结果
    for aid in tqdm(aids):
           
        real_cc_list_before, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)
        real_cc_list_after,  _ = sort_aid_cc(targeted_aid[aid]['x_obs'], afteryear)
        
        # 真实结果 (在第after年)
        after_nop    = len(real_cc_list_after)
        h_index_real = calculate_h_index(real_cc_list_after)
        tcc_real     = sum(real_cc_list_after)
        ccstar_real  = max(real_cc_list_after)
        
        # 预测结果 (利用截止Before年的训练Q, 然后预测在第after年)
        Q = aid2Q[aid][0]
        samples_avg =  list()
        for _ in range(sampling_times):
            samples = rs.randn(after_nop, 1) * np.exp(log_sig_P) + mu_P
            samples = Q + samples      # log, 加法效应
            samples = np.exp(samples)
            samples = samples.squeeze()
            samples_avg.append(samples)
        samples_avg = np.array(samples_avg)
        samples_avg = np.mean(samples_avg, axis=0)
        if after_nop == 1:
            samples_avg = np.array([samples_avg])
        
        # 2022-10-1 部分预测: before year之前的是训练集, 不预测
        before_nop = len(real_cc_list_before)
        pred_nop   = after_nop - before_nop
        if pred_nop > 0:
             pred_cc_list = np.concatenate([real_cc_list_before, samples_avg[-pred_nop:]])
        else:
             pred_cc_list = real_cc_list_before
        # pred_cc_list =  samples_avg
        
        # 预测结果 (在第after年)
        h_index_pred = calculate_h_index(pred_cc_list)    # 王大顺论文图C (h-index)
        tcc_pred     = sum(pred_cc_list)                  # 王大顺论文图D (total citations)
        ccstar_pred  = max(pred_cc_list)                  # 王大顺论文图B (C*)
        
        # 添加真实结果和预测结果 (在第after年)
        hx_results.append([h_index_real, h_index_pred])
        tc_results.append([tcc_real,    tcc_pred])
        cs_results.append([ccstar_real, ccstar_pred])
    
    return np.array(hx_results), np.array(tc_results), np.array(cs_results)
   

def avg_func(mu_P, log_sig_P, aid2Q, 
            targeted_aid,
            beforeyear, afteryear,
            sampling_times=100):
    
    # 均值模型评价预测性
    aids = list(aid2Q.keys())
    aids = sorted(aids)
    
    hx_results = list()  # h-index 结果
    tc_results = list()  # total citatoins 结果
    cs_results = list()  # c* 结果
    
    for aid in tqdm(aids):
        real_cc_list_before, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)
        real_cc_list_after,  _ = sort_aid_cc(targeted_aid[aid]['x_obs'], afteryear)
    
        # 真实结果 (在第after年)
        after_nop    = len(real_cc_list_after)
        h_index_real = calculate_h_index(real_cc_list_after)
        tcc_real     = sum(real_cc_list_after)
        ccstar_real  = max(real_cc_list_after)
 
        # 均值预测
        avg_cc       = np.mean(real_cc_list_before)
        before_nop   = len(real_cc_list_before)
        pred_nop     = after_nop - before_nop
        if pred_nop > 0:
            pred_cc_list = np.concatenate([real_cc_list_before, np.ones(pred_nop) * avg_cc])
        else:
            pred_cc_list = real_cc_list_before
        # pred_cc_list =  np.ones(after_nop) * avg_cc
        
        # 预测结果 (在第after年)
        h_index_pred = calculate_h_index(pred_cc_list)    # 王大顺论文图C (h-index)
        tcc_pred     = sum(pred_cc_list)                  # 王大顺论文图D (total citations)
        ccstar_pred  = max(pred_cc_list)                  # 王大顺论文图B (C*)
        
        # 添加真实结果和预测结果
        hx_results.append([h_index_real, h_index_pred])
        tc_results.append([tcc_real,    tcc_pred])
        cs_results.append([ccstar_real, ccstar_pred])
    
    return np.array(hx_results), np.array(tc_results), np.array(cs_results)