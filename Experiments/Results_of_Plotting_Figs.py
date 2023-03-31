#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:32:25 2023

@author: aixuexi
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams

import pickle
import os
import time
import math
import multiprocessing
import prettytable as pt
import pandas as pd
import seaborn as sns
import numpy as np

from MyQPModel_Results.utils_predict import *
from MyQPModel_Results.extract_nobels import *


# 融合学术环境信息的科研能力量化模型 - 模型的科研表现预测效果评价绘图


def read_q(ResultsPath, file_name, beforeyear):
    """读取模型估计的Q值"""
    results_OUR = read_file(os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
    results_WSB = read_file(os.path.join(ResultsPath, "WSB_{}_{}.pkl".format(file_name, beforeyear)))
    return results_OUR, results_WSB


def read_eval(ResultsPath, file_name, beforeyear):
    """读取对h指数和tcc的预测结果"""
    (hx_value_results, tc_value_results, cs_value_results) = read_file(os.path.join(ResultsPath, "value_results_{}_{}.pkl".format(file_name, beforeyear)))
    (hx_eval_results,  tc_eval_results,  cs_eval_results)  = read_file(os.path.join(ResultsPath, "eval_results_{}_{}.pkl".format(file_name, beforeyear)))
    return (hx_value_results, tc_value_results, cs_value_results), (hx_eval_results, tc_eval_results,  cs_eval_results)


def Correlation_Analysis():
    ''' 
    现有指标的相关性分析 
    '''
    titles = {"physics": "物理学", "chemistry": "化学", "computer science": "计算机科学"}
    
    file_name       = "computer science"
    title           = titles[file_name]
    ResultsPath_ORG = "./Results/Results_org"
    ResultsPath_COU = "./Results/Results_org_country"
    ResultsPath_COO = "./Results/Results_coop_org_country"
    
    # 读取实证数据
    targeted_aid    = read_file(os.path.join(ResultsPath_ORG, "empirical_data({}).pkl".format(file_name)))
    
    # 读取估计的q值
    results_ORG, results_WSB = read_q(ResultsPath_ORG, file_name, beforeyear=2000)
    results_COU, _           = read_q(ResultsPath_COU, file_name, beforeyear=2000)
    results_COO, _           = read_q(ResultsPath_COO, file_name, beforeyear=2000)
    
    [mu_P_ORG, log_sig_P_ORG], aid2Q_ORG, orgid2Q_ORG            = results_ORG
    [mu_P_COU, log_sig_P_COU], aid2Q_COU, orgid2Q_COU, cou2Q_COU = results_COU
    [mu_P_COO, log_sig_P_COO], aid2Q_COO, orgid2Q_COO, cou2Q_COO = results_COO
    [mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB            = results_WSB
    
    q3_ORG_list = list()
    q3_COU_list = list()
    q3_COO_list = list()
    q3_WSB_list = list()
    max_c_list  = list()  # 最大引用数目
    h_list      = list()  # h指数
    tcc_list    = list()  # 累计引用数目
    N_list      = list()  # 发文量
    for aid in aid2Q_ORG: 
        # 每位作者截止至beforeyear的发文质量(tcc)列表
        cclist, _  = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear=2000)  
        if len(cclist) == 0:
            continue
        q3_ORG, _ = aid2Q_ORG[aid]            # q 值 (ORG)
        q3_COU, _ = aid2Q_COU[aid]            # q 值 (COU)
        q3_COO, _ = aid2Q_COO[aid]            # q 值 (COO)
        q3_WSB, _ = aid2Q_WSB[aid]            # q 值 (SIN)
        h_index   = calculate_h_index(cclist) # h 指数
        max_c     = max(cclist)               # 最大引用数目
        tcc       = sum(cclist)               # 总引用数目
        N         = len(cclist)               # 产量 Productivity
        
        q3_ORG_list.append(q3_ORG)
        q3_COU_list.append(q3_COU)
        q3_COO_list.append(q3_COO)
        q3_WSB_list.append(q3_WSB)
        max_c_list.append(max_c)
        h_list.append(h_index)
        tcc_list.append(tcc)
        N_list.append(N)
        
    # 计算相关系数
    q3_ORG_list  = np.array(q3_ORG_list).reshape((-1, 1))
    q3_COU_list  = np.array(q3_COU_list).reshape((-1, 1))
    q3_COO_list  = np.array(q3_COO_list).reshape((-1, 1))
    q3_WSB_list  = np.array(q3_WSB_list).reshape((-1, 1))
    max_c_list   = np.array(max_c_list).reshape((-1, 1))
    h_list       = np.array(h_list).reshape((-1, 1))
    tcc_list     = np.array(tcc_list).reshape((-1, 1))
    N_list       = np.array(N_list).reshape((-1, 1))

    all_metrics  = np.concatenate([q3_WSB_list, q3_ORG_list, q3_COU_list, q3_COO_list, 
                                   max_c_list, h_list, tcc_list, N_list], axis=-1)
    all_metrics  = pd.DataFrame(all_metrics)
    
    
    # 计算 pearsonr
    matrix = all_metrics.corr()
    mask   = np.triu(np.ones_like(matrix, dtype=np.bool))
    corr   = matrix.copy()
    # 绘制热力图
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "SimHei", # Times New Roman
              "font.size" : 18
              }
    rcParams.update(config)
    # 列名称
    columns_name = [r"$Q_{\alpha}^{sin}$", r"$Q_{\alpha}^{org}$", r"$Q_{\alpha}^{cou}$", r"$Q_{\alpha}^{coo}$", 
                    r"$C^{*}_{\alpha}$",   r"$H_{\alpha}$",       r"$C_{\alpha}$",     r"$N_{\alpha}$"]
    # SimHei 字体符号不正常显示
    plt.rcParams['axes.unicode_minus'] = False 
    
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    sns.heatmap(corr, 
                mask=mask, annot=True, fmt='.2f', cmap=cmap,
                vmin=-1, vmax=1, 
                cbar_kws={'shrink': 1}, linewidths=5, square=True,
                xticklabels=columns_name, yticklabels=columns_name)
    plt.yticks(rotation=0) 
    plt.title(title)    


def create_eval_table(hx_eval_results_ORG, hx_eval_results_COU, hx_eval_results_COO):
    # 1. h-index 指标评价
    eval_avg_h = list()
    eval_wsb_h = list()
    eval_org_h = list()
    eval_cou_h = list()
    eval_coo_h = list()
    afteryearRange = hx_eval_results_ORG.keys()
    for afteryear in afteryearRange:
        print("(Year == {})".format(afteryear)) 
        
        # Baseline: 平均值估计, Sinatra的Q模型
        cor_AVG, r2_AVG, rmse_AVG, mae_AVG = hx_eval_results_ORG[afteryear]['avg']
        cor_WSB, r2_WSB, rmse_WSB, mae_WSB = hx_eval_results_ORG[afteryear]['wsb']
        # 融合学术环境的科研能力量化模型
        cor_ORG, r2_ORG, rmse_ORG, mae_ORG = hx_eval_results_ORG[afteryear]['our']
        cor_COU, r2_COU, rmse_COU, mae_COU = hx_eval_results_COU[afteryear]['our']
        cor_COO, r2_COO, rmse_COO, mae_COO = hx_eval_results_COO[afteryear]['our']
        
        eval_avg_h.append([cor_AVG, r2_AVG, rmse_AVG, mae_AVG])
        eval_wsb_h.append([cor_WSB, r2_WSB, rmse_WSB, mae_WSB])
        eval_org_h.append([cor_ORG, r2_ORG, rmse_ORG, mae_ORG])
        eval_cou_h.append([cor_COU, r2_COU, rmse_COU, mae_COU])
        eval_coo_h.append([cor_COO, r2_COO, rmse_COO, mae_COO])
        
        tb = pt.PrettyTable()
        tb.field_names = ["Model", "Pearsonr", "R2", "RMSE", "MAE"]
        tb.add_row(["AVG", "{:.4f}".format(cor_AVG), "{:.4f}".format(r2_AVG), "{:.4f}".format(rmse_AVG), "{:.4f}".format(mae_AVG)])
        tb.add_row(["SIN", "{:.4f}".format(cor_WSB), "{:.4f}".format(r2_WSB), "{:.4f}".format(rmse_WSB), "{:.4f}".format(mae_WSB)])
        tb.add_row(["ORG", "{:.4f}".format(cor_ORG), "{:.4f}".format(r2_ORG), "{:.4f}".format(rmse_ORG), "{:.4f}".format(mae_ORG)])
        tb.add_row(["COU", "{:.4f}".format(cor_COU), "{:.4f}".format(r2_COU), "{:.4f}".format(rmse_COU), "{:.4f}".format(mae_COU)])
        tb.add_row(["COO", "{:.4f}".format(cor_COO), "{:.4f}".format(r2_COO), "{:.4f}".format(rmse_COO), "{:.4f}".format(mae_COO)])
        print(tb)
        
    eval_avg_h = np.maximum(np.array(eval_avg_h), 0)  # R2 < 0 无意义
    eval_wsb_h = np.maximum(np.array(eval_wsb_h), 0) 
    eval_org_h = np.maximum(np.array(eval_org_h), 0) 
    eval_cou_h = np.maximum(np.array(eval_cou_h), 0) 
    eval_coo_h = np.maximum(np.array(eval_coo_h), 0) 
    
    return (eval_avg_h, eval_wsb_h, eval_org_h, eval_cou_h, eval_coo_h)


def plot_eval_table(hx_eval_results_ORG, hx_eval_results_COU, hx_eval_results_COO,
                    title1, index, title2):
    """ {0: r"$Pearsonr$", 1: r"$R^2$", 2: r"$RMSE$", 3: r"$MAE$"}"""
    
    eval_avg_h, eval_wsb_h, eval_org_h, eval_cou_h, eval_coo_h = create_eval_table(hx_eval_results_ORG, hx_eval_results_COU, hx_eval_results_COO)
    
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman", # SimHei
              "font.size" : 22
              }
    rcParams.update(config)
    
    afteryearRange = list(hx_eval_results_ORG.keys())
    index_ylabel   = {0: r"$Pearsonr$", 1: r"$R^2$", 2: r"$RMSE$", 3: r"$MAE$"}
    ylabel         = index_ylabel[index]
    
    # plt.plot(afteryearRange, eval_avg_h[:, index], markersize=10, linewidth=2, c='gray', marker='o', linestyle='--', label='均值模型')
    plt.plot(afteryearRange, eval_wsb_h[:, index], markersize=10, linewidth=2, c='brown', marker='s', linestyle='--', label="Q model") # '基准模型'
    plt.plot(afteryearRange, eval_org_h[:, index], markersize=10, linewidth=2, c='red',   marker='^', linestyle='--', label="IQ model") # '融合机构信息的科研能力量化模型'
    plt.plot(afteryearRange, eval_cou_h[:, index], markersize=10, linewidth=2, c='blue',  marker='P', linestyle='--', label="IQ-2 model") # '融合国家信息的科研能力量化模型'
    plt.plot(afteryearRange, eval_coo_h[:, index], markersize=10, linewidth=2, c='black', marker='x', linestyle='--', label="IQ-3 model") # '融合合作信息的科研能力量化模型'
    
    plt.xticks(afteryearRange, rotation=45)
    plt.xlabel("Time") # "年份"
    # plt.yticks(np.arange(0.0, 1.2, 0.2))
    # plt.ylim(0, 1)
    # plt.yticks(np.arange(0.5, 1.01, 0.1))
    # plt.yticks(np.arange(0, 700, 100))

    plt.ylabel(ylabel)
    plt.title("{} ({})".format(title1, title2))
    if index in [0, 1]:
        plt.legend(frameon=False, fontsize=22, loc='upper right')
    else:
        plt.legend(frameon=False, fontsize=22, loc='upper left')


def Prediction_Results_Analysis():
    '''
    预测h指数, 累计引用计数tcc, c* 
    '''
    titles = {"physics": "物理学", "chemistry": "化学", "computer science": "计算机科学"}
    
    file_name       = "computer science"
    title           = titles[file_name]
    ResultsPath_ORG = "./Results/Results_org"
    ResultsPath_COU = "./Results/Results_org_country"
    ResultsPath_COO = "./Results/Results_coop_org_country"
    
    # 读取结果
    (hx_value_results_ORG, tc_value_results_ORG, cs_value_results_ORG), (hx_eval_results_ORG, tc_eval_results_ORG, cs_eval_results_ORG) = read_eval(ResultsPath_ORG, file_name, beforeyear=2000)
    (hx_value_results_COU, tc_value_results_COU, cs_value_results_COU), (hx_eval_results_COU, tc_eval_results_COU, cs_eval_results_COU) = read_eval(ResultsPath_COU, file_name, beforeyear=2000)
    (hx_value_results_COO, tc_value_results_COO, cs_value_results_COO), (hx_eval_results_COO, tc_eval_results_COO, cs_eval_results_COO) = read_eval(ResultsPath_COO, file_name, beforeyear=2000)
    
    '''
        1. 整体指标结果绘图+画表 (Pearsonr, RMSE, MAE, R2)
    ''' 
    # 绘表
    _ = create_eval_table(hx_eval_results_ORG, hx_eval_results_COU, hx_eval_results_COO)
    # 画图
    title_fig = "Physics" # "Computer Science"
    title_suf = "h-index"
    yaxis_idx = 3
    plot_eval_table(hx_eval_results_ORG, hx_eval_results_COU, hx_eval_results_COO, title_fig, yaxis_idx, title_suf)

    # 绘表
    _ = create_eval_table(tc_eval_results_ORG, tc_eval_results_COU, tc_eval_results_COO)
    # 画图
    title_fig = "Physics" # "Computer Science"
    title_suf = r"$C_{tot}$"
    yaxis_idx = 3
    plot_eval_table(tc_eval_results_ORG, tc_eval_results_COU, tc_eval_results_COO, title_fig, yaxis_idx, title_suf)

    
    # 绘制论文要求的图
    '''
       2. 预测数值 (Predicted Value) VS (Real Value)
    '''
    # 论文图: X轴是Predicted Value, Y轴是 Real Value When Y2 = 2008年
    Y2 = 2006
    title_fig = "Physics" # Computer Science
    # h指数
    plot_real2pred(hx_value_results_ORG[Y2]['wsb'], hx_value_results_ORG[Y2]['our'], alpha=0.5,
                   logscale=False, color='red', Y2=Y2,
                   xticks=np.arange(0, 55, 10), 
                   xlabel='Predicted value (h index)', ylabel='Acutal value (h index)', label="IQ model", 
                   title=title_fig)
    # 总引用数目
    plot_real2pred(tc_value_results_ORG[Y2]['wsb'] + 1, tc_value_results_ORG[Y2]['our'] + 1, alpha=0.15,
                   logscale=True, color='red', Y2=Y2,
                   xticks=10 ** np.arange(0, 6), 
                   xlabel=r'Predicted value ($C_{tot}$)', ylabel='Actual value ($C_{tot}$)', label="IQ model", 
                   title=title_fig)
    
     # h指数
    plot_real2pred(hx_value_results_COU[Y2]['wsb'], hx_value_results_COU[Y2]['our'], alpha=0.5,
                   logscale=False, color='blue', Y2=Y2,
                   xticks=np.arange(0, 55, 10), 
                   xlabel='Predicted value (h index)', ylabel='Acutal value (h index)', label="IQ-2 model", 
                   title=title_fig)
    # 总引用数目
    plot_real2pred(tc_value_results_COU[Y2]['wsb'] + 1, tc_value_results_COU[Y2]['our'] + 1, alpha=0.15,
                   logscale=True, color='blue', Y2=Y2,
                   xticks=10 ** np.arange(0, 6), 
                   xlabel='Predicted value ($C_{tot}$)', ylabel='Actual value ($C_{tot}$)', label="IQ-2 model", 
                   title=title_fig)
    
     # h指数
    plot_real2pred(hx_value_results_COO[Y2]['wsb'], hx_value_results_COO[Y2]['our'], alpha=0.5,
                   logscale=False, color='black', Y2=Y2,
                   xticks=np.arange(0, 55, 10),
                   xlabel='Predicted value (h index)', ylabel='Acutal value (h index)', label="IQ-3 model", 
                   title=title_fig)
    # 总引用数目
    plot_real2pred(tc_value_results_COO[Y2]['wsb'] + 1, tc_value_results_COO[Y2]['our'] + 1, alpha=0.15,
                   logscale=True, color='black', Y2=Y2,
                   xticks=10 ** np.arange(0, 6), 
                   xlabel='Predicted value ($C_{tot}$)', ylabel='Actual value ($C_{tot}$)', label="IQ-3 model", 
                   title=title_fig)
    

    '''
        3. 案例分析; 物理学: j == 28, 48
                     化学:   j == 16, 30
                     计算机: j == 9, 12
    '''
    def plot_h_pred(j, hx_value_results_ORG, hx_value_results_COU, hx_value_results_COO, ylabel):
        # h指数和tcc预测 (j号学者)
        real_h     = list()
        pred_h_wsb = list()  # 基准模型
        pred_h_org = list()  # 融合机构信息的科研能力量化模型
        pred_h_cou = list()  # 融合国家信息的科研能力量化模型
        pred_h_coo = list()  # 融合合作信息的科研能力量化模型
        afteryearRange = list(hx_eval_results_ORG.keys())
        for afteryear in afteryearRange:
            real_h_j     = hx_value_results_ORG[afteryear]['wsb'][j][0]
            pred_h_wsb_j = hx_value_results_ORG[afteryear]['wsb'][j][-1]
            pred_h_org_j = hx_value_results_ORG[afteryear]['our'][j][-1]
            pred_h_cou_j = hx_value_results_COU[afteryear]['our'][j][-1]
            pred_h_coo_j = hx_value_results_COO[afteryear]['our'][j][-1]
            real_h.append(real_h_j)
            pred_h_wsb.append(pred_h_wsb_j)
            pred_h_org.append(pred_h_org_j)
            pred_h_cou.append(pred_h_cou_j)
            pred_h_coo.append(pred_h_coo_j)
            
        fig = plt.figure(figsize=(10, 8))
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['figure.dpi'] = 300
        config = {
                  "font.family" : "Times New Roman",# SimHei
                  "font.size" : 22
                  }
        rcParams.update(config)  
        
        plt.plot(afteryearRange, real_h,     markersize=10, linewidth=2, c='gray',  marker='o', linestyle='--', label='Actual value')
        plt.plot(afteryearRange, pred_h_wsb, markersize=10, linewidth=2, c='brown', marker='s', linestyle='--', label='Q model') # 基准模型
        plt.plot(afteryearRange, pred_h_org, markersize=10, linewidth=2, c='red',   marker='^', linestyle='--', label='IQ model') # 融合机构信息的科研能力量化模型
        plt.plot(afteryearRange, pred_h_cou, markersize=10, linewidth=2, c='blue',  marker='P', linestyle='--', label='IQ-2 model')
        plt.plot(afteryearRange, pred_h_coo, markersize=10, linewidth=2, c='black', marker='x', linestyle='--', label='IQ-3 model')
        plt.xticks(afteryearRange, rotation=45)
        # plt.yticks(np.arange(0, 35, 10))
        plt.yticks(np.arange(0, 2500, 500))
        plt.legend(frameon=False, loc='upper left', fontsize=20)
        plt.ylabel(ylabel, fontsize=25)
        plt.xlabel("Time", fontsize=25)
        plt.title(r"Physics ($\alpha_4$)")
    
    j = 48
    plot_h_pred(j, hx_value_results_ORG, hx_value_results_COU, hx_value_results_COO, r"h index")
    plot_h_pred(j, tc_value_results_ORG, tc_value_results_COU, tc_value_results_COO, r"$C_{tot}$")
    

#%%
def calculate_roc(aids_list, q3_OUR_list, nobel_laureates, take_exp):
    # ROC Curve
    nobel_laureates_num = len(nobel_laureates)     # 诺奖得主数目
    total_aid_num       = len(aids_list)           # 总人数
    rank_threshold      = np.arange(1, 102, 2) / 100
    
    aid2metric = list()
    for i in range(total_aid_num):
        if take_exp:
            metric = np.exp(q3_OUR_list[i])
        else:
            metric = q3_OUR_list[i]
        aid2metric.append((aids_list[i], metric))
    aid2metric = sorted(aid2metric, key=lambda x: x[-1], reverse=True)
    
    Y = list()
    X = list()
    for rank_i in rank_threshold:
        part_num        = int(rank_i * total_aid_num)
        aid2metric_part = aid2metric[: part_num]
        
        Ture_num  = 0
        False_num = 0
        for aid, metric in aid2metric_part:
            if aid in nobel_laureates:
                Ture_num  += 1
            else:
                False_num += 1
        roc_y = Ture_num  / nobel_laureates_num
        roc_x = False_num / (total_aid_num - nobel_laureates_num)
        Y.append(roc_y)
        X.append(roc_x)
    
    # Roc面积
    X = np.array(X)
    Y = np.array(Y)
    S = np.sum(np.multiply(X[1:] - X[:-1], Y[:-1]))
    return X, Y, S


def Prediction_Nobel_Analysis():
    '''识别Nobel获奖者'''
    titles = {"physics": "物理学", "chemistry": "化学", "computer science": "计算机科学"}
    
    file_name       = "chemistry"
    title           = titles[file_name]
    ResultsPath_ORG = "./Results/Results_org"
    ResultsPath_COU = "./Results/Results_org_country"
    ResultsPath_COO = "./Results/Results_coop_org_country"
    
    # 读取实证数据
    targeted_aid    = read_file(os.path.join(ResultsPath_ORG, "empirical_data({}).pkl".format(file_name)))
    
    # 读取估计的q值
    results_ORG, results_WSB = read_q(ResultsPath_ORG, file_name, beforeyear=2000)
    results_COU, _           = read_q(ResultsPath_COU, file_name, beforeyear=2000)
    results_COO, _           = read_q(ResultsPath_COO, file_name, beforeyear=2000)
    
    [mu_P_ORG, log_sig_P_ORG], aid2Q_ORG, orgid2Q_ORG            = results_ORG
    [mu_P_COU, log_sig_P_COU], aid2Q_COU, orgid2Q_COU, cou2Q_COU = results_COU
    [mu_P_COO, log_sig_P_COO], aid2Q_COO, orgid2Q_COO, cou2Q_COO = results_COO
    [mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB            = results_WSB
    
    # 读取年诺贝尔奖数据 --- 由extract_nobel.py生成
    nobel_data       = read_nobel_data(nobel_file_path)
    nobel_data_field = nobel_data[file_name]
    
    # 统计数据
    pid2aid = dict()
    for aid in targeted_aid:
        x_obs = targeted_aid[aid]["x_obs"]
        for year in x_obs:
            for pid, cc, czs in x_obs[year]:
                pid = str(pid)
                if pid not in pid2aid:
                    pid2aid[pid] = dict()
                if aid not in pid2aid[pid]:
                    pid2aid[pid][aid] = ''
    print("当前实验数据涉及{}学者的{}篇文章".format(len(targeted_aid), len(pid2aid)))
    
    # 检查诺奖有多少篇保留在抽取的样本集中
    nobel_laureates = dict()
    find_nobel_num = 0
    for pid in nobel_data_field:
        
        # 获奖年分过滤(Prize_Year) 过滤 : 如果在训练集时段已经获奖, 则不能预测
        Prize_Year = nobel_data_field[pid]['Prize_Year']
        Pub_Year   = nobel_data_field[pid]['Pub_Year']
        beforeyear = 2000
        if Prize_Year <= beforeyear:
            continue
        
        # 实证数据集(pid2aid) 过滤 : 如果不再我们筛选的target_aid内, 则不能预测
        if pid in pid2aid:
            find_nobel_num += 1
            for aid in pid2aid[pid]:
                if aid not in nobel_laureates:
                    nobel_laureates[aid] = 1
                else:
                    nobel_laureates[aid] += 1
    print("当前实验数据涉及{}学者的{}诺奖文章".format(len(nobel_laureates), find_nobel_num))
    
    # 所有指标
    aids_list   = list()
    q3_ORG_list = list()
    q3_COU_list = list()
    q3_COO_list = list()
    q3_WSB_list = list()
    max_c_list  = list()
    h_index_list= list()
    tcc_list    = list()
    N_list      = list()
    for aid in aid2Q_ORG: 
        cclist, _  = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear=2000)  # 每位作者截止至beforeyear的发文质量(tcc)列表
        if len(cclist) == 0:
            continue
        
        q3_ORG, _ = aid2Q_ORG[aid]            # q 值 (OUR)
        q3_COU, _ = aid2Q_COU[aid]
        q3_COO, _ = aid2Q_COO[aid]
        q3_WSB, _ = aid2Q_WSB[aid]            # q 值 (WSB)
        h_index   = calculate_h_index(cclist) # h 指数
        max_c     = max(cclist)               # 最大引用数目
        tcc       = sum(cclist)               # 总引用数目
        N         = len(cclist)               # 产量 Productivity
        
        aids_list.append(aid)
        q3_ORG_list.append(q3_ORG)
        q3_COU_list.append(q3_COU)
        q3_COO_list.append(q3_COO)
        q3_WSB_list.append(q3_WSB)
        max_c_list.append(max_c)
        h_index_list.append(h_index)
        tcc_list.append(tcc)
        N_list.append(N)

    # ROC curve
    X_ORG, Y_ORG, S_ORG = calculate_roc(aids_list, q3_ORG_list, nobel_laureates, False)
    X_COU, Y_COU, S_COU = calculate_roc(aids_list, q3_COU_list, nobel_laureates, False)
    X_COO, Y_COO, S_COO = calculate_roc(aids_list, q3_COO_list, nobel_laureates, False)
    X_WSB, Y_WSB, S_WSB = calculate_roc(aids_list, q3_WSB_list, nobel_laureates, False)
    X_CS,  Y_CS,  S_CS  = calculate_roc(aids_list, max_c_list,  nobel_laureates, False)
    X_HX,  Y_HX,  S_HX  = calculate_roc(aids_list, h_index_list,nobel_laureates, False)
    X_TC,  Y_TC,  S_TC  = calculate_roc(aids_list, tcc_list,    nobel_laureates, False)
    X_N,   Y_N,   S_N   = calculate_roc(aids_list, N_list,      nobel_laureates, False)
    
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "SimHei",
              "font.size" : 18
              }
    rcParams.update(config)  
    
    ls = 2
    s  = 5
    plt.plot(X_ORG, Y_ORG, c='red',    marker='^', markersize=s,  linewidth=ls,  label=r"$Q^{org}_{\alpha}$" + "({:.2f})".format(S_ORG))
    plt.plot(X_COU, Y_COU, c='blue',    marker='P', markersize=s, linewidth=ls, label=r"$Q^{cou}_{\alpha}$" + "({:.2f})".format(S_COU))
    plt.plot(X_COO, Y_COO, c='black',    marker='x', markersize=s, linewidth=ls, label=r"$Q^{coo}_{\alpha}$" +"({:.2f})".format(S_COO))
    plt.plot(X_WSB, Y_WSB, c='brown',  marker='s', markersize=s,  linewidth=ls, label=r"$Q^{sin}_{\alpha}$" + "({:.2f})".format(S_WSB))
    plt.plot(X_CS,  Y_CS,  c='green',  marker='*', markersize=s,  linewidth=ls, label=r"$C^*_{\alpha}$" + "({:.2f})".format(S_CS))
    plt.plot(X_HX,  Y_HX,  c='purple',   marker='d', markersize=s, linewidth=ls, label=r"$H_{\alpha}$" +"({:.2f})".format(S_HX))
    plt.plot(X_TC,  Y_TC,  c='gray',  marker='o', markersize=s,   linewidth=ls, label=r"$C_{\alpha}$" +"({:.2f})".format(S_TC))
    plt.plot(X_N,   Y_N,   c='orange', marker='1', markersize=s,  linewidth=ls, label=r"$N_{\alpha}$" +"({:.2f})".format(S_N))
    plt.plot(np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2), linestyle='--', c='gray', linewidth=ls)
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(linestyle='--') 
    plt.legend(frameon=False, loc=4)
    plt.xlabel("假阳性率 (False Positive Rate)", fontsize=22)
    plt.ylabel("真阳性率 (True Positive Rate)", fontsize=22)
    plt.title(title)