#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 15:15:14 2022

@author: aixuexi
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pickle
import os
import time
import math
import copy
import multiprocessing
import prettytable as pt
import pandas as pd
import seaborn as sns
import numpy as np

from MyQPModel.bbvi_em_org_country import *
from MyQPModel_Results.utils_predict import *


ResultsPath = "./Results/Results_org_country"


def get_emprical_data(save_path, file_name, beforeyear):
    ''' 实证数据读取 '''
   
    # 读取实证数据 --- 由mag_aid.py生成
    targeted_aid = read_file(os.path.join(save_path, "empirical_data.pkl"))
    save_file(targeted_aid, os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))
    
    # 给每个国家编号 (在编号时, 提前根据国家内人员数等分, 方便多进程中各进程处理数据大小近似相同)
    cou_num = dict()   
    for aid in targeted_aid:
        cou = targeted_aid[aid]['cou']
        if cou not in cou_num:
            cou_num[cou] = 1
        else:
            cou_num[cou] += 1
    cou_num  = [(cou, cou_num[cou]) for cou in cou_num] # 统计每个国家的学者数目
    cou_num  = sorted(cou_num, key=lambda x: x[-1], reverse=True)
    
    mp_num      = 8                                     # 假定8个进程
    mp_cou_size = int(np.ceil(len(cou_num) / mp_num))   # 每个进程近似需要处理的国家数
    cou_num_1   = copy.copy(cou_num)
    cou_num_2   = list()
    for i in range(mp_num):
        # 选 1个 国家内人数多的国家
        cou_num_2 += cou_num_1[:1]                      
        del cou_num_1[:1]
        # 选 mp_cou_size-1 个 国家内人数少的国家
        cou_num_2 += cou_num_1[-min((mp_cou_size-1), len(cou_num_1)):] 
        del cou_num_1[-(mp_cou_size-1):]  
    assert len(cou_num) == len(set(cou_num_2))
    
    cou_dict1 = dict()
    cou_dict2 = dict()
    for i, (cou, _) in enumerate(cou_num_2):
        cou_dict1[cou] = i
        cou_dict2[i]   = cou
    
    # 检查是否存在一个机构属于多个国家 --- 不存在这种非法情况
    check = dict()
    for aid in targeted_aid:
        org_id = targeted_aid[aid]['org_id']
        cou    = targeted_aid[aid]['cou']
        cou_id = cou_dict1[cou]
        if org_id not in check:
            check[org_id] = dict()
        if cou_id not in check[org_id]:
            check[org_id][cou_id] = ''
    
    # 训练集确定: 国家信息, 机构信息, 作者每篇论文累计引用数目列表
    cou_id2org_id = dict()
    for aid in targeted_aid:
        org_id = targeted_aid[aid]['org_id']
        cou    = targeted_aid[aid]['cou']
        cou_id = cou_dict1[cou]
        
        cclist = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
        if len(cclist) == 0:
            continue
        
        if cou_id not in cou_id2org_id:
            cou_id2org_id[cou_id] = dict()
        if org_id not in cou_id2org_id[cou_id]:
            cou_id2org_id[cou_id][org_id] = list() # 国家下包含那些机构
        cou_id2org_id[cou_id][org_id].append(aid)  # 机构下包含那些作者
                
    print("结果存放路径: {}".format(ResultsPath))
    print("待分析学者数目: {}".format(len(targeted_aid)))      
    print("待分析国家数目: {}".format(len(cou_id2org_id)))
    
    # 转换成所需要的x_obs格式 (见上述模拟数据生成的x_obs)
    data = dict()
    for _, cou_id in enumerate(cou_id2org_id):
        i = cou_id
        data[i]       = dict()                # 第i个国家
        data[i]['q1'] = [["", ""], ["", ""]]
        
        for j, org_id in enumerate(cou_id2org_id[cou_id]):
            data[i][j]       = dict()         # 第j个机构  
            data[i][j]['q2'] = ["", ""]
            
            for k, aid in enumerate(cou_id2org_id[cou_id][org_id]):
                cclist, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)
                if len(cclist) == 0:
                    continue
                data[i][j][k]       = dict()  # 第k个学者
                data[i][j][k]['q3'] = ""
                data[i][j][k]['x']  = np.log(cclist + 1)
                
    return data, cou_id2org_id, cou_dict2


def BBVI_Algorithm_For_EmpiricalAnalysis(save_path, file_name, beforeyear):
    ''' 融合国家先验信息的科研能力量化模型实证研究 '''
    
    # (1) 读取实证数据
    data, cou_id2org_id, cou_dict2 = get_emprical_data(save_path, file_name, beforeyear)    
                
    # Q-model 均值估计
    var_params_init, model_params_init = max_likelihoood(data)

    # (2) 变分估计
    mp_num         = 8
    Epochs         = 10
    step_size      = 1e-1
    batch_size_org = 512
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
    mu_P_OUR      = 0            # 假定为0 不影响结果  
    log_sig_P_OUR = 0            # 引用范式待估参数
    aid2Q_OUR     = dict()       # 学者科研能力
    orgid2Q_OUR   = dict()       # 机构科研能力
    cou2Q_OUR     = dict()       # 国家科研能力
    
    mu_P_WSB      = 0            # 假定为0 不影响结果
    log_sig_P_WSB = 0            # 引用范式待估参数
    aid2Q_WSB     = dict()       # 王大顺的平均估计(极大似然)
    orgid2Q_WSB   = dict()       # 王大顺的平均估计(极大似然) ---> 我们简单将其扩展到机构
    cou2Q_WSB     = dict()
    
    for _, cou_id in enumerate(cou_id2org_id):
        i   = cou_id
        cou = cou_dict2[i]
        cou2Q_OUR[cou] = model_params_bbvi[i]['q1']
        cou2Q_WSB[cou] = model_params_init[i]['q1']
        
        for j, org_id in enumerate(cou_id2org_id[cou_id]):
            orgid2Q_OUR[org_id] = var_params_bbvi[i][j]['q2']
            orgid2Q_WSB[org_id] = var_params_init[i][j]['q2']
            
            for k, aid in enumerate(cou_id2org_id[cou_id][org_id]):
                aid2Q_OUR[aid]  = var_params_bbvi[i][j][k]['q3']
                aid2Q_WSB[aid]  = var_params_init[i][j][k]['q3']
                
    log_sig_P_OUR = model_params_bbvi["P"][1]
    log_sig_P_WSB = model_params_init["P"][1]
    
    # 模型估计结果: 论文质量随机参数, 作者q2, 机构q1
    results_OUR = ([mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR, cou2Q_OUR)
    results_WSB = ([mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB, cou2Q_WSB)
    save_file(results_OUR, os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
    save_file(results_WSB, os.path.join(ResultsPath, "WSB_{}_{}.pkl".format(file_name, beforeyear)))
    
    
def Prediction_For_EmpiricalAnalysis(save_path, file_name, beforeyear, afteryearRange):
    # 读取模型估计的Q值
    results_OUR  = read_file(os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
    results_WSB  = read_file(os.path.join(ResultsPath, "WSB_{}_{}.pkl".format(file_name, beforeyear)))
    targeted_aid = read_file(os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))
     
    [mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR, cou2Q_OUR = results_OUR
    [mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB, cou2Q_WSB = results_WSB
    
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
    save_file(eval_results,  os.path.join(ResultsPath, "eval_results_{}_{}.pkl".format(file_name,  beforeyear)))
    
    
def main():
    # beforeyear 之前的被用作训练数据
    beforeyear = 2000       
    for file_name in ["physics", "chemistry", "computer science"]:
        save_path = "/mnt/disk2/EmpiricalData/StatisticalData_{}".format(file_name)
        BBVI_Algorithm_For_EmpiricalAnalysis(save_path, file_name, beforeyear)
     
    # 预测afteryearRang的学者表现
    afteryearRange = np.arange(2001, 2011) 
    for file_name in ["physics", "chemistry", "computer science"]:
        save_path = "/mnt/disk2/EmpiricalData/StatisticalData_{}".format(file_name)
        Prediction_For_EmpiricalAnalysis(save_path, file_name, beforeyear, afteryearRange)
     