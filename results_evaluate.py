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


# 数据挂载盘
abs_path = "/mnt/disk2/"

# 预处理数据存放路径
file_name = "cs"
field_name = "Computer science"
process_data_path = abs_path + "EmpiricalData/StatisticalData_cs"

# file_name = "physics"
# field_name = "Physics"
# process_data_path = abs_path + "EmpiricalData/StatisticalData_physics"


def calculate_h_index(aid_cc_list):
    # 计算h指数
    aid_cc_list = sorted(aid_cc_list, reverse = True)
    
    for h in range(len(aid_cc_list), 0, -1):
        if aid_cc_list[h-1] >= h:
            return h 
    return 0


def evaluate_real2pred(Y, X):
    # Y: 真实值; X: 预测值
    # 评价指标: 
    # 相关性评价: Pearsonr
    # 距离评价: MAE, MSE
    cor, pvalue = pearsonr(Y, X)
    rmse = np.sqrt(mean_squared_error(Y, X))
    mae = mean_absolute_error(Y, X)
    r2  = r2_score(Y, X)
    
    return cor, rmse, mae, r2


def print_real2pred_tb(h_real_pred_WSB, h_real_pred):
    cor_WSB, rmse_WSB, mae_WSB, r2_WSB = evaluate_real2pred(h_real_pred_WSB[:, 0], h_real_pred_WSB[:, 1])
    cor, rmse, mae, r2 = evaluate_real2pred(h_real_pred[:, 0], h_real_pred[:, 1])
    tb = pt.PrettyTable()
    tb.field_names=["Model", "Pearsonr", "R2", "RMSE", "MAE"]
    tb.add_row(["WSB", "{:.4f}".format(cor_WSB), "{:.4f}".format(r2_WSB), "{:.4f}".format(rmse_WSB), "{:.4f}".format(mae_WSB)])
    tb.add_row(["Our", "{:.4f}".format(cor), "{:.4f}".format(r2), "{:.4f}".format(rmse), "{:.4f}".format(mae)])
    print(tb)
    return (cor_WSB, rmse_WSB, mae_WSB, r2_WSB), (cor, rmse, mae, r2)


def plot_real2pred(h_real_pred_WSB, h_real_pred,
                   logscale, color, xticks, label, title, text_xy):
    # Y轴是真实值; X轴是预测值
    Y, X = h_real_pred[:, 0], h_real_pred[:, 1]
    Y_WSB, X_WSB = h_real_pred_WSB[:, 0], h_real_pred_WSB[:, 1]

    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    plt.scatter(X, Y, c=color, s=5, label="Proposed model")
    plt.scatter(X_WSB, Y_WSB, c="gray", s=5, label="Q-model")
    plt.plot([xticks[0], xticks[-1]], [xticks[0], xticks[-1]], color='blue', linestyle='--')
    plt.xlabel("Predicted " + label)
    plt.ylabel("Real " + label)
    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.xticks(xticks)
    plt.yticks(xticks)
    plt.xlim(xticks[0], xticks[-1])
    plt.ylim(xticks[0], xticks[-1])
    plt.legend(loc='upper right', frameon=False)
    plt.title(title)


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


#%%
# 使用全部数据预测Q值得, 再根据Q值复现预测h指数, 累计引用数目, c* # 这是一种Fake prediction
# 划分时间段进行预测 (using only early-career information to estimate Q parameter)
def sort_aid_cc(yearlynop2cc, afteryear):
    yearlist = sorted(yearlynop2cc.keys()) # 职业生涯跨度 
    startyear = min(yearlist)              # 发表第一篇文章的时间
    endyear = max(yearlist)                # 发表最后一篇时间
    
    # startyear -> afteryear
    cclist_after = list()
    for year in yearlist:
        if year <= afteryear:
            for pid, cc in yearlynop2cc[year]:
                cclist_after.append(cc)
                         
    return np.array(cclist_after)


def predict_func(mu_P, log_sig_P, aid2Q, targeted_aid,
                 beforeyear, afteryear,
                 sampling_times=100):
       # 评价预测性
       rs = npr.RandomState()
       h_real_pred = list()
       tcc_real_pred = list()
       ccstar_real_pred = list()
       for aid in tqdm(aid2Q):
           real_cc_list_before = sort_aid_cc(targeted_aid[aid]['x'], beforeyear)
           real_cc_list = sort_aid_cc(targeted_aid[aid]['x'], afteryear)
     
           nop = len(real_cc_list)
           # 真实结果
           h_index_real = calculate_h_index(real_cc_list)
           tcc_real = sum(real_cc_list)
           ccstar_real = max(real_cc_list)
           
           # 预测结果
           Q = aid2Q[aid][0]
           samples_avg =  list()
           for i in range(sampling_times):
               samples = rs.randn(nop, 1) * np.exp(log_sig_P) + mu_P
               samples = Q + samples      # log未原, 加法效应
               samples = np.exp(samples)
               samples = samples.squeeze()
               samples_avg.append(samples)
           samples_avg = np.array(samples_avg)
           samples_avg = np.mean(samples_avg, axis=0)
           
           if nop == 1:
               samples_avg = [samples_avg]
           
           # 2022-10-1 部分预测: before year之前的是训练集, 不预测
           real_num = len(real_cc_list_before)
           pred_num = len(real_cc_list) - real_num
           if pred_num > 0:
               samples_avg_pred = np.concatenate([real_cc_list_before, samples_avg[-pred_num:]])
           else:
               samples_avg_pred = real_cc_list_before
  
           h_index_pred = calculate_h_index(samples_avg_pred)    # 王大顺论文图C
           tcc_pred = sum(samples_avg_pred)                      # 王大顺论文图D
           ccstar_pred = max(samples_avg_pred)                   # 王大顺论文图B
           
           # 添加真实结果和预测结果
           h_real_pred.append([h_index_real, h_index_pred])
           tcc_real_pred.append([tcc_real, tcc_pred])
           ccstar_real_pred.append([ccstar_real, ccstar_pred])
       
       return np.array(h_real_pred), np.array(tcc_real_pred), np.array(ccstar_real_pred)


np.set_printoptions(precision=6, suppress=True)
def predict_h_tcc_cstar_fakepred(targeted_aid):
    # 预测h指数, 累计引用计数tcc, c*
    # 极大似然估计(均值)Q值
    #
    with open(os.path.join(process_data_path, "aid_empirical.pkl"), 'rb') as f:
        targeted_aid = pickle.load(f)
    
    beforeyear = 1997 # 被利用于获取Q的年份
    afteryear  = 2008 # 被利用于预测Q的年份
    
    h_eval_results = dict()   # h指数预测的评价指标结果
    tcc_eval_results = dict() # tcc预测的评价指标结果
    
    h_value_results = dict()   # h指数预测的具体数值
    tcc_value_results = dict() # tcc预测的具体数值
    
    for afteryear in np.arange(1998, 2009):
        
        # 读取BASELINE
        with open("./tmp/results_org_WSB_{}.pkl".format(beforeyear), 'rb') as f:
            results_org_WSB = pickle.load(f)
        [mu_P, log_sig_P], aid2Q_WSB, orgid2Q_WSB = results_org_WSB
        h_real_pred_WSB, tcc_real_pred_WSB, ccstar_real_pred_WSB = predict_func(mu_P, log_sig_P, aid2Q_WSB, targeted_aid, beforeyear, afteryear)
        # 读取我们的结果
        with open("./tmp/results_org_{}.pkl".format(beforeyear), 'rb') as f:
            results_org = pickle.load(f)
        [mu_P, log_sig_P], aid2Q, orgid2Q = results_org
        h_real_pred, tcc_real_pred, ccstar_real_pred = predict_func(mu_P, log_sig_P, aid2Q, targeted_aid, beforeyear, afteryear)
    
        h_value_results[afteryear] = dict()    
        h_value_results[afteryear]['wsb'] = h_real_pred_WSB
        h_value_results[afteryear]['our'] = h_real_pred
        
        tcc_value_results[afteryear] = dict()
        tcc_value_results[afteryear]['wsb'] = tcc_real_pred_WSB
        tcc_value_results[afteryear]['our'] = tcc_real_pred
    
        # 观察aid2Q, aid2Q_WSB中Q的差别 (观察两者的Q是否有差别)
        # plot_Q(aid2Q, aid2Q_WSB)
    
        # 评价结果: 通过表格评价指标; 通过图评价
        h_result_WSB,   h_result_our   = print_real2pred_tb(h_real_pred_WSB, h_real_pred)
        tcc_result_WSB, tcc_result_our = print_real2pred_tb(tcc_real_pred_WSB, tcc_real_pred)
        _, _ = print_real2pred_tb(ccstar_real_pred_WSB, ccstar_real_pred)
        
        # 整体的评价结果, RMSE MAE, Pearsonr, R2 / cor, rmse, mae, r2
        h_eval_results[afteryear] = dict()
        h_eval_results[afteryear]['wsb'] = h_result_WSB
        h_eval_results[afteryear]['our'] = h_result_our
        
        tcc_eval_results[afteryear] = dict()
        tcc_eval_results[afteryear]['wsb'] = tcc_result_WSB
        tcc_eval_results[afteryear]['our'] = tcc_result_our
        
        
    # 论文图: X轴是Predicted Value, Y轴是 Real Value When Y2 = 2008年
    # h指数
    plot_real2pred(h_real_pred_WSB, h_real_pred, logscale=False, color='red', 
                   xticks=np.arange(0, 100, 25), label='h-index (2008)', title=field_name, text_xy=[0.75, 1.00, 1.25, 1.50])
    # 总引用数目
    plot_real2pred(tcc_real_pred_WSB+1, tcc_real_pred+1, logscale=True, color='red', 
                   xticks=10 ** np.arange(1, 6), label=r'$C_{tot}$ (2008)', title=field_name, text_xy=[0.55, 1.00, 1.75, 3.15])
    # c star, 最大引用数目
    plot_real2pred(ccstar_real_pred_WSB+1, ccstar_real_pred+1, logscale=True, color='blue',
                   xticks=10 ** np.arange(0, 5), label=r'$C^{*}(N)$', title=field_name, text_xy=[0.55, 1.00, 1.75, 3.15])
    
    
    # 论文图: X轴是Y2, Y轴是评价指标 Pearsonr, RMSE, MAE, R2
    #         灰色是Q模型的结果, 红色是当前模型结果
    # 1. h-index 指标评价 (Y2 = [1998, 2009])
    h_results_wsb_list = list()
    h_results_our_list = list()
    for afteryear in np.arange(1998, 2009):
        h_results_wsb_list.append(list(h_eval_results[afteryear]['wsb']))
        h_results_our_list.append(list(h_eval_results[afteryear]['our']))
    h_results_wsb_list = np.array(h_results_wsb_list)   
    h_results_our_list = np.array(h_results_our_list)   
    h_results_wsb_list = np.maximum(h_results_wsb_list, 0)
    h_results_our_list = np.maximum(h_results_our_list, 0)
    # 表格Table 3, 4
    for afteryear in [2000, 2005, 2008]:
        print("Year == {}".format(afteryear)) 
        print(h_eval_results[afteryear]['wsb'])
        print(h_eval_results[afteryear]['our'])
        print("\n")
        
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    plt.plot(np.arange(1998, 2009), h_results_wsb_list[:, 2], markersize=10, linewidth=2, c='gray', marker='o', linestyle='--', label='Q-model')
    plt.plot(np.arange(1998, 2009), h_results_our_list[:, 2], markersize=10, linewidth=2, c='red', marker='s', linestyle='--', label='Proposed model')
    plt.yticks(np.arange(0.0, 8.01, 2))
    plt.xticks(np.arange(1998, 2009), rotation=45)
    plt.xlabel(r"$Y_2$")
    plt.ylabel(r"$MAE$") # r"$Pearsonr$" # r"$R^2$" # r"$RMSE$" # r"$MAE$"
    plt.title("h-index")
    plt.legend(frameon=False)
    
    
    # 2. total citation count 标评价 (Y2 = [1998, 2009])
    tcc_results_wsb_list = list()
    tcc_results_our_list = list()
    for afteryear in np.arange(1998, 2009):
        tcc_results_wsb_list.append(list(tcc_eval_results[afteryear]['wsb']))
        tcc_results_our_list.append(list(tcc_eval_results[afteryear]['our']))
    tcc_results_wsb_list = np.array(tcc_results_wsb_list)   
    tcc_results_our_list = np.array(tcc_results_our_list)   
    tcc_results_wsb_list = np.maximum(tcc_results_wsb_list, 0)
    tcc_results_our_list = np.maximum(tcc_results_our_list, 0)
    # 表格Table 3, 4
    for afteryear in [2000, 2005, 2008]:
        print("Year == {}".format(afteryear)) 
        print(tcc_eval_results[afteryear]['wsb'])
        print(tcc_eval_results[afteryear]['our'])
        print("\n")
        
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    plt.plot(np.arange(1998, 2009), tcc_results_wsb_list[:, 2], markersize=10, linewidth=2, c='gray', marker='o', linestyle='--', label='Q-model')
    plt.plot(np.arange(1998, 2009), tcc_results_our_list[:, 2], markersize=10, linewidth=2, c='red', marker='s', linestyle='--', label='Proposed model')
    plt.yticks(np.arange(0, 610, 150))
    # plt.yticks(np.arange(0.2, 1.11, 0.2))
    plt.xticks(np.arange(1998, 2009), rotation=45)
    plt.xlabel(r"$Y_2$")
    plt.ylabel(r"$MAE$") # r"$Pearsonr$" # r"$R^2$" # r"$RMSE$" # r"$MAE$"
    plt.title(r"$C_{tot}$")
    plt.legend(frameon=False)
        
    
    # 案例分析
    real_h_list = h_value_results[2008]['wsb'][:, 0]
    np.argmax(real_h_list)
    # j == 234, 241, 279 300, 301, 213, 219, 237
    j = 21
    real_h = list()
    pred_h_wsb = list()
    pred_h = list()
    for afteryear in h_value_results:
        real_h_j = h_value_results[afteryear]['wsb'][j][0]
        pred_h_wsb_j = h_value_results[afteryear]['wsb'][j][-1]
        pred_h_j = h_value_results[afteryear]['our'][j][-1]
        real_h.append(real_h_j)
        pred_h_wsb.append(pred_h_wsb_j)
        pred_h.append(pred_h_j)
   
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)  
    
    plt.plot(np.arange(1998, 2009), real_h, markersize=10, linewidth=2, c='black', marker='*', linestyle='--', label='Real')
    plt.plot(np.arange(1998, 2009), pred_h_wsb, markersize=10, linewidth=2, c='gray', marker='o', linestyle='--', label='Q-model')
    plt.plot(np.arange(1998, 2009), pred_h, markersize=10, linewidth=2, c='red', marker='s', linestyle='--', label='Proposed model')
    plt.xticks(np.arange(1998, 2009), rotation=45)
    # plt.yticks(np.arange(0, 25, 5))
    plt.legend(frameon=False, loc='upper left')
    plt.ylabel("h-index")
    plt.xlabel(r"$Y_2$")
    plt.title(r"$\alpha_2$")
    
    real_tcc = list()
    pred_tcc_wsb = list()
    pred_tcc = list()
    for afteryear in tcc_value_results:
        real_tcc_j = tcc_value_results[afteryear]['wsb'][j][0]
        pred_tcc_wsb_j = tcc_value_results[afteryear]['wsb'][j][-1]
        pred_tcc_j = tcc_value_results[afteryear]['our'][j][-1]
        real_tcc.append(real_tcc_j)
        pred_tcc_wsb.append(pred_tcc_wsb_j)
        pred_tcc.append(pred_tcc_j)
    
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)  
    
    plt.plot(np.arange(1998, 2009), real_tcc, markersize=10, linewidth=2, c='black', marker='*', linestyle='--', label='Real')
    plt.plot(np.arange(1998, 2009), pred_tcc_wsb, markersize=10, linewidth=2, c='gray', marker='o', linestyle='--', label='Q-model')
    plt.plot(np.arange(1998, 2009), pred_tcc, markersize=10, linewidth=2, c='red', marker='s', linestyle='--', label='Proposed model')
    plt.xticks(np.arange(1998, 2009), rotation=45)
    # plt.yticks(np.arange(0, 850, 150))
    plt.legend(frameon=False)
    plt.ylabel(r"$C_{tot}$")
    plt.xlabel(r"$Y_2$")
    plt.title(r"$\alpha_2$")