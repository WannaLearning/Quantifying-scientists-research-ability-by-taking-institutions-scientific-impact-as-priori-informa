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
import pandas as pd
import prettytable as pt
import seaborn as sns
import numpy as np
import random
from queue import Queue

from MyQPModel.bbvi_em_coop_org_country_mp import *
from MyQPModel_Results.utils_predict import *
from MyQPModel.utils_Mcoop_split import *


ResultsPath = "./Results/Results_coop_org_country"


#%%
def get_empirical_data(save_path, file_name, beforeyear, chunks_1, chunks_2, epsilon_1, epsilon_2, coop):
    
    def coop_split(targeted_aid, chunks_1, beforeyear, epsilon1):
        ''' 所有学者根据合作关系切割成chunks块 
            被 Results_coop_org_country.py 调用
        '''
        data       = dict()
        pid2cc     = dict()      # 每篇文章的logcc
        aid2pid    = dict()      # 每位学者的文章
        pid2aid    = dict()      # 每篇文章的作者
        total_nop_num = 0        # 非unique的总发文量 (每篇文章重复独立的归属于每一位学者)
        for aid in targeted_aid:
            cclist, pidlist = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
            if len(cclist) == 0:
                continue
            for pid, cc in zip(pidlist, cclist):
                # 统计每篇文章的logcc
                if pid not in pid2cc:
                    pid2cc[pid] = np.log(cc + 1)
                # 统计每篇文章的作者
                if pid not in pid2aid:
                    pid2aid[pid] = list()
                pid2aid[pid].append(aid)
            total_nop_num += len(pidlist)
            aid2pid[aid]   = pidlist
        total_unique_nop_num = len(pid2cc)  # 总unique文章数
        total_noa_num        = len(aid2pid) # 总人数
        print("总人数: {}, 总发文量: {} / {}".format(total_noa_num, total_unique_nop_num, total_nop_num))
        
        # 统计每位学者的合作情况
        aid2aid = dict()      
        for pid in pid2aid:
            if len(pid2aid[pid]) == 1:
                # 独著
                aid_i = pid2aid[pid][0]
                if aid_i not in aid2aid:
                    aid2aid[aid_i] = dict()
            else:
                # pid由多位学者撰写
                for aid_i in pid2aid[pid]:
                    for aid_j in pid2aid[pid]:
                        if aid_i == aid_j:
                            continue
                        else:
                            if aid_i not in aid2aid:
                                aid2aid[aid_i] = dict()
                            if aid_j not in aid2aid[aid_i]:
                                aid2aid[aid_i][aid_j] = 0
                            # aid_i 和 aid_j 合作一次
                            aid2aid[aid_i][aid_j] += 1
        
        # 划分成多块
        visited_aids_all  = dict()
        visited_aids_list = list()
        while len(visited_aids_all) < total_noa_num:
            # 已经被选过, 则不作为起点备选集
            alternative_source = dict()
            for aid in aid2aid:
                if aid not in visited_aids_all:  
                    alternative_source[aid] = ''
            source = random.sample(alternative_source.keys(), 1)[0]  # 从起点开始遍历 - 起点采样至起点备选集
        
            # 开始一次广度优先遍历
            queue = Queue()     
            visited_aids = dict()      # 存放本次已经遍历的作者
            visited_aids[source] = ''  # 起点
            queue.put(source)
            while not queue.empty():
                # 第vertex位作者
                vertex = queue.get()  
                # 确定第vertex位作者的合作者
                vertex_coa = list()
                for aid in aid2aid[vertex]:
                    freq = aid2aid[vertex][aid]  # 两人合作次数
                    vertex_coa.append((aid, freq))
 
                for aid, freq in vertex_coa:  
                    # 如果该合作者 本次未见过 且 之前为见过 且合作次数超过epsilon, 则添加
                    if freq >= epsilon1:
                        if aid not in visited_aids and aid not in visited_aids_all:
                            visited_aids[aid] = ''
                            queue.put(aid)
                            
            for aid in visited_aids:
                if aid in visited_aids_all:
                    print("重复包括")
                visited_aids_all[aid] = ''
                
            visited_aids_list.append(visited_aids)
            # print("{} / {} = {:.6f}".format(len(visited_aids_all), total_noa_num, len(visited_aids_all)/ total_noa_num))
        
        # 检查是否有作者丢失 (无丢失)
        check_aids_num = dict()
        for aids in visited_aids_list:
            for aid in aids:
                if aid not in check_aids_num:
                    check_aids_num[aid] = ''
        assert(len(check_aids_num) == total_noa_num)   
        # 检查闭集大小
        closedset_size = list()
        for visited_aids in visited_aids_list:
            closedset_size.append(len(visited_aids))
        print("未合并每块近似闭集的大小", sorted(closedset_size, reverse=True)[:10])
        # 合并过小集合
        visited_aids_list_ = Merge_ColosedSet_Size(visited_aids_list, chunks_1, threshold=100)
        print("合并后每块近似闭集的大小:", sorted([len(i) for i in visited_aids_list_], reverse=True))
        
        aids_chunks = dict()
        for c in range(chunks_1):
            aids_chunks[c] = visited_aids_list_[c]

        # # 方法1: 利用合作矩阵划分
        # '''生成合作矩阵'''
        # row_id1 = dict()
        # row_id2 = dict()
        # col_id1 = dict()    
        # col_id2 = dict()    
        # for row_i, pid in enumerate(pid2cc):
        #     row_id1[pid]   = row_i
        #     row_id2[row_i] = pid
        # for col_j, aid in enumerate(aid2pid):
        #     col_id1[aid]   = col_j
        #     col_id2[col_j] = aid
        
        # # M_coop (i,j)元编号:第i行是文章编号; 第j列是人员编号
        # x_obs   = np.array([pid2cc[row_id2[i]] for i in range(total_unique_nop_num)])   # logcc观测数据
        # M_coop  = np.zeros((total_unique_nop_num, total_noa_num), dtype=np.int8)        # 合作矩阵
        # for aid in aid2pid:
        #     col_j = col_id1[aid]
        #     for pid in aid2pid[aid]:
        #         row_i = row_id1[pid]
        #         M_coop[row_i, col_j] = 1
        # # M_coop = M_coop / np.sum(M_coop, axis=-1, keepdims=True)
        # data["M_coop"]  = M_coop
        # data["x_obs"]   = x_obs
        # epsilon1        = 20   # 近似合作闭集
        # row_col_list    = utils_mp_split2(data, row_id1, row_id2, col_id1, col_id2, aid2pid, mp_num=chunks_1, epsilon=epsilon1, threshold=100)
        
        # aids_chunks = dict()
        # for c in range(chunks_1):
        #     aids_chunk             = list()
        #     row_c_list, col_c_list = row_col_list[c]
        #     for col_j, col_j_b in enumerate(col_c_list):
        #         if col_j_b:
        #             aids_chunk.append(col_id2[col_j])
        #     aids_chunks[c] = aids_chunk
    
        return aids_chunks
    
    def get_data(targeted_aid_c, chunks_2, epsilon2, coop, c):
    
        # 训练集确定: 国家信息, 机构信息, 作者每篇论文累计引用数目列表
        # 转换成所需要的x_obs格式 (见上述模拟数据生成的x_obs)
        data_c     = dict()
        pid2cc     = dict()      # 每篇文章的 logcc
        aid2pid    = dict()      # 每位学者的 文章
        total_nop_num = 0        # 非unique的总发文量 (每篇文章重复独立的归属于每一位学者)
        for aid in targeted_aid_c:
            cclist, pidlist = sort_aid_cc(targeted_aid_c[aid]['x_obs'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
            if len(cclist) == 0:
                continue
            # 每篇文章的logc(10)
            for pid, cc in zip(pidlist, cclist):
                if pid not in pid2cc:
                    pid2cc[pid] = np.log(cc + 1)
            total_nop_num += len(pidlist)
            # 学者的文章
            aid2pid[aid] = pidlist
        total_unique_nop_num = len(pid2cc)  # 总unique文章数
        total_noa_num        = len(aid2pid) # 总人数
        print("总人数: {}, 总发文量: {} / {}".format(total_noa_num, total_unique_nop_num, total_nop_num))
        
        
        '''生成合作矩阵 和 x_obs'''
        row_id1 = dict()
        row_id2 = dict()
        col_id1 = dict()    
        col_id2 = dict()    
        for row_i, pid in enumerate(pid2cc):
            row_id1[pid]   = row_i
            row_id2[row_i] = pid
        for col_j, aid in enumerate(aid2pid):
            col_id1[aid]   = col_j
            col_id2[col_j] = aid
        
        if coop == "not":
            # (1) 每篇论文独立属于每位作者 - 即无视合作关系
            x_obs  = list()
            M_coop =  np.zeros((total_nop_num, total_noa_num), dtype=np.int8)
            row_st = 0
            row_en = 0
            for col_j in range(len(col_id2)):
                # 合作矩阵 (i,j) 元赋值
                aid     = col_id2[col_j]
                pidlist = aid2pid[aid]
                nop     = len(pidlist)
                row_en  = row_st + nop
                M_coop[row_st:row_en, col_j] = 1
                row_st  = row_en
                # 引用观测
                for pid in pidlist:
                    logcc = pid2cc[pid]
                    x_obs.append(logcc)
            x_obs             = np.array(x_obs)
            data_c["M_coop"]  = M_coop
            data_c["x_obs"]   = x_obs
            row_col_list      = utils_mp_split(data_c, chunks_2)  # 合作闭集
            data_c['rc']      = row_col_list
        
        if coop == "frac":
            # (2) 每篇论文均分给每位作者 - fractional counting    
            # M_coop (i,j)元编号:第i行是文章编号; 第j列是人员编号
            x_obs   = np.array([pid2cc[row_id2[i]] for i in range(total_unique_nop_num)])   # logcc观测数据
            M_coop  = np.zeros((total_unique_nop_num, total_noa_num), dtype=np.float16)     # 合作矩阵
            for aid in aid2pid:
                col_j = col_id1[aid]
                for pid in aid2pid[aid]:
                    row_i = row_id1[pid]
                    M_coop[row_i, col_j] = 1
            M_coop            = M_coop / np.sum(M_coop, axis=-1, keepdims=True)
            data_c["M_coop"]  = M_coop
            data_c["x_obs"]   = x_obs
            row_col_list      = utils_mp_split2(data_c, row_id1, row_id2, col_id1, col_id2, aid2pid, mp_num=chunks_2, epsilon=epsilon2)
            data_c['rc']      = row_col_list
        
        '''机构所属国家(i_obs) 和 学者隶属的机构(j_obs)'''
        i_obs = np.zeros((total_noa_num, ), dtype=np.float16)
        j_obs = np.zeros((total_noa_num, ), dtype=np.float16)
    
        # 给每个国家编号
        cou_num = dict()  
        for aid in targeted_aid_c:
            cou = targeted_aid_c[aid]['cou']
            cclist, pidlist = sort_aid_cc(targeted_aid_c[aid]['x_obs'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
            if len(cclist) == 0:
                continue
            if cou not in cou_num:
                cou_num[cou] = 1
            else:
                cou_num[cou] += 1
        cou_num   = [(cou, cou_num[cou]) for cou in cou_num]  # 统计每个国家的学者数目, 并排序
        cou_num   = sorted(cou_num, key=lambda x: x[-1], reverse=True)
        cou_dict1 = dict()
        cou_dict2 = dict()
        for i, (cou, _) in enumerate(cou_num):
            cou_dict1[cou] = i
            cou_dict2[i]   = cou
            
        # 给每个机构编号
        org_num = dict()        # 每个机构的学者数目
        for aid in targeted_aid_c:
            org_id = targeted_aid_c[aid]['org_id']
            cclist, pidlist = sort_aid_cc(targeted_aid_c[aid]['x_obs'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
            if len(cclist) == 0:
                continue
            if org_id not in org_num:
                org_num[org_id] = 1
            else:
                org_num[org_id] += 1
        org_num   = [(org, org_num[org]) for org in org_num]  # 统计每个机构的学者数目, 并排序
        org_num   = sorted(org_num, key=lambda x: x[-1], reverse=True)
        org_dict1 = dict()
        org_dict2 = dict()
        for i, (org, _) in enumerate(org_num):
            org_dict1[org] = i
            org_dict2[i]   = org
        
        for aid in targeted_aid_c:
            cclist, pidlist = sort_aid_cc(targeted_aid_c[aid]['x_obs'], beforeyear)  # 每位作者截止至beforeyear的发文质量(tcc)列表
            if len(cclist) == 0:
                continue
            org_id       = targeted_aid_c[aid]['org_id']
            cou          = targeted_aid_c[aid]['cou']
            col_j        = col_id1[aid]               # 学者aid 对应 M_coop矩阵中col_j 列
            i_obs[col_j] = cou_dict1[cou]             # 学者aid 的国家编号
            j_obs[col_j] = org_dict1[org_id]          # 学者aid 的机构编号
            
        data_c["i_obs"] = i_obs
        data_c["j_obs"] = j_obs
    
        return data_c, col_id2, org_dict2, cou_dict2, org_num, cou_num
    
    
    # 读取实证数据 --- 由mag_aid.py生成
    targeted_aid = read_file(os.path.join(save_path,  "empirical_data.pkl"))
    save_file(targeted_aid, os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))
    
    if coop == "not":
        print("无视合作")
        aids_chunks = random_split(targeted_aid, chunks_1)
    if coop == "frac":
        print("均分合作")
        epsilon_1 = 20
        aids_chunks = coop_split(targeted_aid, chunks_1, beforeyear, epsilon_1)
    
    # 根据 aids_chunks 划分数据集
    data       = dict()
    col_id2    = dict()
    org_dict2  = dict()
    cou_dict2  = dict()
    org_num    = dict()
    cou_num    = dict()
    for c in range(chunks_1):
        # 第c块包含的学者
        aids_chunk     = aids_chunks[c]
        targeted_aid_c = dict()
        for aid in aids_chunk:
            targeted_aid_c[aid] = targeted_aid[aid]
        # 训练数据, 学者编号, 机构编号, 国家编号, 该块机构下学者人数, 该块国家下学者人数
        data_c, col_id2_c, org_dict2_c, cou_dict2_c, org_num_c, cou_num_c = get_data(targeted_aid_c, chunks_2, epsilon_2, coop, c)
        data[c]      = data_c
        col_id2[c]   = col_id2_c
        org_dict2[c] = org_dict2_c
        cou_dict2[c] = cou_dict2_c
        org_num[c]   = org_num_c
        cou_num[c]   = cou_num_c

    return data, col_id2, org_dict2, cou_dict2, org_num, cou_num


def BBVI_Algorithm_For_EmpiricalAnalysis(save_path, file_name, beforeyear, chunks_1, chunks_2, epsilon_1, epsilon_2, coop):
    # 融合机构先验信息的科研能力量化模型实证研究
    
    data, col_id2, org_dict2, cou_dict2, org_num, cou_num = get_empirical_data(save_path, file_name, beforeyear, 
                                                                               chunks_1, chunks_2, epsilon_1, epsilon_2, coop)
    
    results_OUR_ = dict()
    results_WSB_ = dict()
    for c in data:
        # 第 c / chunks_1 块数据
        data_c      = data[c]
        col_id2_c   = col_id2[c]
        org_dict2_c = org_dict2[c]
        cou_dict2_c = cou_dict2[c]
        
        # 极大似然估计
        var_params_init, model_params_init = max_likelihoood(data_c)
        
        # 贝叶斯后验估计
        mp_num       = 8
        Epochs       = 10
        step_size    = 1e-1
        num_iters    = 100
        num_samples  = 1
        var_params   = var_params_init   # 初始化变分参数
        model_params = model_params_init # 初始化模型参数
        row_col_list = data_c['rc']
        datas_c      = split_data(data_c, row_col_list, mp_num)  # 当mp_num == chunks_2时, 多进程内无循环
        for e in range(Epochs):
            # E-Step
            print("({}) Optimizing variational parameters...".format(e))           
            E_start_time    = time.perf_counter()
            var_params_next = Estep_MP(datas_c, row_col_list, model_params, var_params, num_samples, step_size, num_iters)
            E_end_time      = time.perf_counter()                      
            var_params      = var_params_next
            print("Estep 耗时: {:.4f}".format(E_end_time-E_start_time))
            
            # M-Step
            print("({}) Optimizing model parameters...".format(e))
            M_start_time      = time.perf_counter()
            model_params_next = Mstep_MP(datas_c, row_col_list, model_params, var_params, num_samples, step_size, num_iters)
            M_end_time        = time.perf_counter()
            model_params      = model_params_next
            print("Estep 耗时: {:.4f}".format(M_end_time-M_start_time))
        
        # 变分参数估计
        var_params_bbvi,   model_params_bbvi = var_params, model_params
        var_params_q2_OUR, var_params_q3_OUR = var_params_bbvi
        var_params_init,   model_params_init = max_likelihoood(data_c)
        var_params_q2_WSB, var_params_q3_WSB = var_params_init
        
        aid2Q_OUR   = dict()   # 学者能力
        orgid2Q_OUR = dict()   # 机构能力
        cou2Q_OUR   = dict()   # 国家能力
        aid2Q_WSB   = dict()   # 学者能力
        orgid2Q_WSB = dict()   # 机构能力
        cou2Q_WSB   = dict()   # 国家能力
        # 学者科研能力
        for col_j in col_id2_c:
            aid            = col_id2_c[col_j]
            q3_OUR         = var_params_q3_OUR[col_j]
            q3_WSB         = var_params_q3_WSB[col_j]
            aid2Q_OUR[aid] = q3_OUR
            aid2Q_WSB[aid] = q3_WSB
        # 机构科研能力
        for org_j in org_dict2_c:
            org_id              = org_dict2_c[org_j]
            q2_OUR              = var_params_q2_OUR[org_j]
            q2_WSB              = var_params_q2_WSB[org_j]
            orgid2Q_OUR[org_id] = q2_OUR
            orgid2Q_WSB[org_id] = q2_WSB
        # 国家科研能力
        for cou_j in cou_dict2_c:
            cou            = cou_dict2_c[cou_j]
            q1_OUR         = model_params_bbvi[cou_j]
            q1_WSB         = model_params_init[cou_j]
            cou2Q_OUR[cou] = q1_OUR
            cou2Q_WSB[cou] = q1_WSB
        # 
        mu_P_OUR, log_sig_P_OUR = model_params_bbvi[-1][0] # 引用随机性参数
        mu_P_WSB, log_sig_P_WSB = model_params_init[-1][0]  # 引用随机性参数
        
        # 模型估计结果: 论文质量随机参数, 作者q3, 机构q2, 国家q1
        results_OUR_[c] = ([mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR, cou2Q_OUR)
        results_WSB_[c] = ([mu_P_WSB, log_sig_P_WSB], aid2Q_WSB, orgid2Q_WSB, cou2Q_WSB)
    
    save_data = (data, col_id2, org_dict2, cou_dict2, org_num, cou_num)
    # 储存结果
    save_file(results_OUR_, os.path.join(ResultsPath, "OUR_{}_{}_.pkl".format(file_name, beforeyear)))
    save_file(results_WSB_, os.path.join(ResultsPath, "WSB_{}_{}_.pkl".format(file_name, beforeyear)))   
    save_file(save_data,    os.path.join(ResultsPath, "DATA_{}_{}_.pkl".format(file_name, beforeyear)))   
    
    
    # 读取结果
    results_OUR_ = read_file(os.path.join(ResultsPath, "OUR_{}_{}_.pkl".format(file_name, beforeyear)))
    results_WSB_ = read_file(os.path.join(ResultsPath, "WSB_{}_{}_.pkl".format(file_name, beforeyear)))
    save_data    = read_file(os.path.join(ResultsPath, "DATA_{}_{}_.pkl".format(file_name, beforeyear)))
    data, col_id2, org_dict2, cou_dict2, org_num, cou_num = save_data
    
    def take_average_func(results_OUR_):
        ''' 加权更新 机构变分参数 和 国家模型参数
            以人数比例作为加权权重
        '''
        noa_total_num = 0         # 总学者数
        org_total_num = dict()    # 每个机构下总学者数
        cou_total_num = dict()    # 每个国家下总学者数
        for c in data:
            org_num_c  = org_num[c]
            cou_num_c  = cou_num[c]
            org_num_c_ = dict()
            cou_num_c_ = dict()
            for org, num1 in org_num_c:
                org_num_c_[org] = num1
            for cou, num2 in cou_num_c:
                cou_num_c_[cou] = num2 
            org_num_c  = org_num_c_
            cou_num_c  = cou_num_c_
    
            for org in org_num_c:
                if org not in org_total_num:
                    org_total_num[org]  = org_num_c[org]
                else:
                    org_total_num[org] += org_num_c[org]
            for cou in cou_num_c:
                if cou not in cou_total_num:
                    cou_total_num[cou]  = cou_num_c[cou]
                else:
                    cou_total_num[cou] += cou_num_c[cou]
        # 总学者数目
        for cou in cou_total_num:
            noa_total_num += cou_total_num[cou]
            
        mu_P_OUR      = 0
        log_sig_P_OUR = 0
        aid2Q_OUR     = dict()
        orgid2Q_OUR   = dict()
        cou2Q_OUR     = dict()
        for c in data:
            org_num_c  = org_num[c]
            cou_num_c  = cou_num[c]
            org_num_c_ = dict()
            cou_num_c_ = dict()
            for org, num1 in org_num_c:
                org_num_c_[org] = num1
            for cou, num2 in cou_num_c:
                cou_num_c_[cou] = num2 
            org_num_c  = org_num_c_
            cou_num_c  = cou_num_c_
            
            [mu_P_OUR_c, log_sig_P_OUR_c], aid2Q_OUR_c, orgid2Q_OUR_c, cou2Q_OUR_c = results_OUR_[c]
            
            noa_num_c = 0  # 第c块数据内学者数目 
            for cou in cou_num_c:
                noa_num_c += cou_num_c[cou]
            ratio_c        = noa_num_c / noa_total_num
            mu_P_OUR      += ratio_c * mu_P_OUR_c
            log_sig_P_OUR += ratio_c * log_sig_P_OUR_c
            
            for aid in aid2Q_OUR_c:
                if aid not in aid2Q_OUR:
                    aid2Q_OUR[aid] = aid2Q_OUR_c[aid]
            
            for org in orgid2Q_OUR_c:
                if org in org_num_c: # 每次切割有随机性
                    ratio_c = org_num_c[org] / org_total_num[org]
                    if org not in orgid2Q_OUR:
                        orgid2Q_OUR[org]  = ratio_c * orgid2Q_OUR_c[org]
                    else:
                        orgid2Q_OUR[org] += ratio_c * orgid2Q_OUR_c[org]
            
            for cou in cou2Q_OUR_c:
                if cou in cou_num_c: # 每次切割有随机性
                    ratio_c = cou_num_c[cou] / cou_total_num[cou]
                    if org not in cou2Q_OUR:
                        cou2Q_OUR[cou]  = ratio_c * cou2Q_OUR_c[cou]
                    else:
                        cou2Q_OUR[cou] += ratio_c * cou2Q_OUR_c[cou]
        
        results_OUR = ([mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR, cou2Q_OUR)
        return results_OUR
    
    results_OUR = take_average_func(results_OUR_)
    results_WSB = take_average_func(results_WSB_)
    # 储存结果
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
        hx_value_AVG, tc_value_AVG, cs_value_AVG = avg_func(mu_P_OUR,     log_sig_P_OUR, aid2Q_OUR, targeted_aid, beforeyear, afteryear)
    
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
    chunks_1   = 10      # 切割成chunks_1块 (循环)
    chunks_2   = 64      # 上述每块再利用utils_Mcoop_split.utils_mp_split切割成chunks_2块 (多进程)
    epsilon_1  = 10
    epsilon_2  = 20
    coop       = 'frac'  # 合作加权方式
    beforeyear = 2000    # beforeyear 之前的被用作训练数据
    for file_name in ["computer science"]: # "physics", ,"computer science"
        save_path = "/mnt/disk2/EmpiricalData/StatisticalData_{}".format(file_name)
        BBVI_Algorithm_For_EmpiricalAnalysis(save_path, file_name, beforeyear, chunks_1, chunks_2, epsilon_1, epsilon_2, coop)
    
    # 开始预测
    afteryearRange = np.arange(2001, 2011)
    for file_name in ["physics", "chemistry", "computer science"]: # , "computer science",
        save_path = "/mnt/disk2/EmpiricalData/StatisticalData_{}".format(file_name)
        Prediction_For_EmpiricalAnalysis(save_path, file_name, beforeyear, afteryearRange)  
    
