#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:14:10 2022

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

from MyQPModel.bbvi_em_org import *
from MyQPModel_Results.utils_predict import *
from MyQPModel_Results.extract_nobels import *


ResultsPath = "./Results/Results_org"


#%%
def get_emprical_data(save_path, file_name, beforeyear):
    ''' 实证数据读取 '''

    # 读取实证数据 --- 由mag_aid.py生成
    targeted_aid = read_file(os.path.join(save_path,  "empirical_data.pkl"))
    save_file(targeted_aid, os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))

    # 训练集确定: 机构信息, 作者每篇论文累计引用数目列表
    # 确定机构数; 由此确定机构模型参数数目
    org_id2aid = dict()
    for aid in targeted_aid:
        org_id    = targeted_aid[aid]['org_id']
        cclist, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
        if len(cclist) == 0:
            continue
        if org_id not in org_id2aid:
            org_id2aid[org_id] = list()
        org_id2aid[org_id].append(aid)      # 机构下包含那些作者
    
    print("结果存放路径: {}".format(ResultsPath))
    print("待分析学者数目: {}".format(len(targeted_aid)))
    print("待分析机构数目: {}".format(len(org_id2aid)))
    
    # 转换成所需要的data格式 (见上述模拟数据生成的data)
    col_id1   = dict()    
    col_id2   = dict() 
    org_dict1 = dict()
    org_dict2 = dict()
    data      = dict()
    for i, org_id in enumerate(org_id2aid):
        # 第i个机构
        org_dict1[org_id] = i
        org_dict2[i]      = org_id
        
        data[i] = dict()         
        data[i]['q2'] = ["", ""]
        for j, aid in enumerate(org_id2aid[org_id]):
            cclist, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)
            if len(cclist) == 0:
                continue
            # 第j个人
            col_id1[aid]    = (i, j)
            col_id2[(i, j)] = aid
            
            data[i][j] = dict()  
            data[i][j]['q3'] = ""
            data[i][j]['x'] = np.log(cclist + 1)
    
    return data, org_id2aid
    
    
def BBVI_Algorithm_For_EmpiricalAnalysis(save_path, file_name, beforeyear):
    ''' 融合机构先验信息的科研能力量化模型实证研究 '''

    # (1) 读取实证数据
    data, org_id2aid = get_emprical_data(save_path, file_name, beforeyear)
    
    # 极大似然估计 (WSB)
    var_params_init, model_params_init = max_likelihoood(data)

    # (2)变分估计
    mp_num         = 8
    Epochs         = 10
    step_size      = 1e-1
    batch_size_org = 128
    num_iters      = 100
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
    
    # (3) 存放结果
    mu_P_OUR      = 0        # 假定为0 不影响结果  
    log_sig_P_OUR = 0        # 引用范式待估参数
    aid2Q_OUR     = dict()   # 提出的基于机构能力先验 
    orgid2Q_OUR   = dict()   # 提出的基于机构能力先验 
    
    mu_P_WSB      = 0        # 假定为0 不影响结果
    log_sig_P_WSB = 0        # 引用范式待估参数
    aid2Q_WSB     = dict()   # WSB的平均估计(极大似然)
    orgid2Q_WSB   = dict()   # WSB的平均估计(极大似然) -> 我们简单将其扩展到机构
    
    for i, org_id in enumerate(org_id2aid):
        orgid2Q_OUR[org_id] = model_params_bbvi[i]['q2']
        orgid2Q_WSB[org_id] = model_params_init[i]['q2']
        
        for j, aid in enumerate(org_id2aid[org_id]):
            aid2Q_OUR[aid]  = var_params_bbvi[i][j]['q3']
            aid2Q_WSB[aid]  = var_params_init[i][j]['q3']
            
    log_sig_P_OUR = model_params_bbvi["P"][1]
    log_sig_P_WSB = model_params_init["P"][1]
    
    # 模型估计结果: 论文质量随机参数, 作者q2, 机构q1
    results_OUR = ([mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR)
    results_WSB = ([mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB)
    save_file(results_OUR, os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
    save_file(results_WSB, os.path.join(ResultsPath, "WSB_{}_{}.pkl".format(file_name, beforeyear)))


def Prediction_For_EmpiricalAnalysis(save_path, file_name, beforeyear, afteryearRange):
    # 读取模型估计的Q值
    results_OUR  = read_file(os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
    results_WSB  = read_file(os.path.join(ResultsPath, "WSB_{}_{}.pkl".format(file_name, beforeyear)))    # 
    targeted_aid = read_file(os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))
    
    [mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR = results_OUR
    [mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB = results_WSB
    
    hx_value_results = dict() # h指数预测的具体数值
    tc_value_results = dict() # tcc预测的具体数值
    cs_value_results = dict() # c*预测的具体数值

    hx_eval_results  = dict() # h指数预测的评价指标结果
    tc_eval_results  = dict() # tcc预测的评价指标结果
    cs_eval_results  = dict() # c*预测的具体数值

    for afteryear in afteryearRange:
        # 预测结果 (抽样预测)
        hx_value_WSB, tc_value_WSB, cs_value_WSB = predict_func(mu_P_WSB, log_sig_P_WSB, aid2Q_WSB, targeted_aid, beforeyear, afteryear)
        hx_value_OUR, tc_value_OUR, cs_value_OUR = predict_func(mu_P_OUR, log_sig_P_OUR, aid2Q_OUR, targeted_aid, beforeyear, afteryear)
        hx_value_AVG, tc_value_AVG, cs_value_AVG = avg_func(mu_P_OUR, log_sig_P_OUR, aid2Q_OUR, targeted_aid, beforeyear, afteryear)
    
        hx_value_results[afteryear] = dict()    
        hx_value_results[afteryear]['wsb'] = hx_value_WSB
        hx_value_results[afteryear]['our'] = hx_value_OUR
        hx_value_results[afteryear]['avg'] = hx_value_AVG
        
        tc_value_results[afteryear] = dict()
        tc_value_results[afteryear]['wsb'] = tc_value_WSB
        tc_value_results[afteryear]['our'] = tc_value_OUR
        tc_value_results[afteryear]['avg'] = tc_value_AVG
        
        cs_value_results[afteryear] = dict()
        cs_value_results[afteryear]['wsb'] = cs_value_WSB
        cs_value_results[afteryear]['our'] = cs_value_OUR
        cs_value_results[afteryear]['avg'] = cs_value_AVG
        
        # 评价结果: 通过表格评价指标 + 通过图评价
        hx_eval_AVG, hx_eval_WSB, hx_eval_OUR = print_real2pred_tb(hx_value_WSB, hx_value_OUR, hx_value_AVG, "H index")
        tc_eval_AVG, tc_eval_WSB, tc_eval_OUR = print_real2pred_tb(tc_value_WSB, tc_value_OUR, tc_value_AVG, "Total citations")
        cs_eval_AVG, cs_eval_WSB, cs_eval_OUR = print_real2pred_tb(cs_value_WSB, cs_value_OUR, cs_value_AVG, "C*")
        
        # 评价结果: Pearsonr, RMSE, MAE, R2
        hx_eval_results[afteryear] = dict()
        hx_eval_results[afteryear]['wsb'] = hx_eval_WSB
        hx_eval_results[afteryear]['our'] = hx_eval_OUR
        hx_eval_results[afteryear]['avg'] = hx_eval_AVG
        
        tc_eval_results[afteryear] = dict()
        tc_eval_results[afteryear]['wsb'] = tc_eval_WSB
        tc_eval_results[afteryear]['our'] = tc_eval_OUR
        tc_eval_results[afteryear]['avg'] = tc_eval_AVG
        
        cs_eval_results[afteryear] = dict()
        cs_eval_results[afteryear]['wsb'] = cs_eval_WSB
        cs_eval_results[afteryear]['our'] = cs_eval_OUR
        cs_eval_results[afteryear]['avg'] = cs_eval_AVG
        
    value_results = (hx_value_results, tc_value_results, cs_value_results)
    eval_results  = (hx_eval_results,  tc_eval_results,  cs_eval_results)
    # 储存结果
    save_file(value_results, os.path.join(ResultsPath, "value_results_{}_{}.pkl".format(file_name, beforeyear)))
    save_file(eval_results,  os.path.join(ResultsPath, "eval_results_{}_{}.pkl".format(file_name, beforeyear)))
    

def main():
    beforeyear = 2000  # beforeyear 之前的被用作训练数据
    for file_name in ["physics", "chemistry", "computer science"]:
        save_path = "/mnt/disk2/EmpiricalData/StatisticalData_{}".format(file_name)
        BBVI_Algorithm_For_EmpiricalAnalysis(save_path, file_name,  beforeyear)
    
    # 预测afteryearRang的学者表现
    afteryearRange = np.arange(2001, 2011) 
    for file_name in ["physics", "chemistry", "computer science"]:
        save_path = "/mnt/disk2/EmpiricalData/StatisticalData_{}".format(file_name)
        Prediction_For_EmpiricalAnalysis(save_path, file_name, beforeyear, afteryearRange)
  