#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 10:01:35 2022

@author: aixuexi
"""
import os
import pickle
import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm


# 根据特定条件筛选满足aid进行实证分析


# 数据挂载盘
abs_path = "/mnt/disk2/"

# 预处理数据存放路径
# file_name = "cs"
# process_data_path = abs_path + "EmpiricalData/StatisticalData_cs"

file_name = "physics"
process_data_path = abs_path + "EmpiricalData/StatisticalData_physics"


def extract_aids_for_empirical_analysis():
    # 筛选准则 - 真实数据检验Q模型预测能力
    '''(1)-(4) 筛选部分作者进行实证分析 (避免作者太多, 算法太慢)
        # (1) 生涯累计发文量超过20篇
        # (2) 1990-1995年间首次发表论文的人员
        # (3) 职业生涯跨度超过span_year年
        # (4) 最大发文量限定在max_nop以内 —— 避免是数据集内消歧造成
        # (5) 只关注人数超过阈值的机构的作者
    '''
    # 读取机构信息
    # 含有 机构名称, 机构国家, 机构坐标
    with open(os.path.join(process_data_path, "org_id_dict.pkl"), 'rb') as f:
        org_id_dict = pickle.load(f)
    # Key: 作者id, Value: 机构id
    with open(os.path.join(process_data_path, "aid2orgid.pkl"), 'rb') as f:
        aid2orgid = pickle.load(f)
    
    
    i_list = [6, 7, 8, 9, 10]
    born_year = [1990, 1995]      # 第一篇文章发表的时间
    span_year = 10                # 职业生涯长度
    min_nop = 30                  # 所使用的时间段, 至少20篇文章
    max_nop = 300                 # 避免是数据集内作者消歧错误
    truncated_begin_year = 2009   # 保证存在c10
    truncated_end_year   = 2018   # 保证存在c10  
    
    targeted_aid1 = dict()
    for i in i_list:
        # 作者发文量信息 ---> 由Extract Aid Fos.py确定
        with open(os.path.join(process_data_path, "aid2fos_{}.pkl".format(i)), 'rb') as f:
            aid2fos = pickle.load(f)
        # 作者引用量信息 ---> 确定C(10)
        with open(os.path.join(process_data_path, "aid2cc_{}.pkl".format(i)), 'rb') as f:
            aid2cc = pickle.load(f)

        # 统计发文量区间 
        aid2nop  = dict()
        nop_list = list()
        for aid in aid2fos:
            nop = 0
            for pubyear in aid2fos[aid]:
                nop += len(aid2fos[aid][pubyear])
            nop_list.append(nop)
            aid2nop[aid] = nop
        print("发文量: {} - {}".format(min(nop_list), max(nop_list))) 
        
        count = 0
        for aid in aid2fos:
            year_list = list(sorted(aid2fos[aid].keys()))
            aid_born = min(year_list)
            
            # 筛选准则 - (1)(2)(4)
            if aid_born < born_year[0] or aid_born > born_year[1]:
                continue
            if max(year_list) - aid_born + 1 < span_year:
                continue
            if aid2nop[aid] > max_nop:
                continue
            # 筛选准则 - (3)
            nop = 0
            for year in aid2fos[aid]:
                if year > truncated_begin_year:
                    continue 
                else:
                    for pid, _ in aid2fos[aid][year]:
                        nop += 1
            if nop < min_nop:
                continue
            
            # 以下是经历筛选后, 预留的人员
            if aid not in targeted_aid1:
                count += 1
                targeted_aid1[aid] = dict()
                for year in aid2fos[aid]:
                    # 只取 <= truncated_begin_year的发文记录
                    if year > truncated_begin_year:
                        continue
                    # 收集发文信息和引用信息
                    targeted_aid1[aid][year] = list()
                    for pid, _ in aid2fos[aid][year]:
                        #
                        cc_10 = 0
                        pubyear = year
                        for t in range(pubyear, pubyear + 10):
                            if t in aid2cc[pid]:
                                cc_10 += aid2cc[pid][t]
                        targeted_aid1[aid][year].append((pid, cc_10))
        print("新增入{}人员".format(count))
    print("(1)-(4) 准则过滤操作后, 剩余作者数目为: {}".format(len(targeted_aid1)))
        
    '''
    (5) 通过机构筛选部分作者进行实证分析 (避免机构太多, 算法太慢)
        第一步: 读取由mag_affiliation抽取的机构信息
        第二步: 统计发现大部分过滤得到的作者属于多个机构
        第三步: 机构缺失的作者剔除, 多个机构的通过文章数目确定主机构
        第四步: 机构人数很少的机构剔除评估
    '''
    # 简单统计一个人属于多个机构的情况
    count1 = 0  # 超过一个机构
    count2 = 0  # 缺少机构信息
    count3 = 0  # 只隶属于一个机构
    for aid in targeted_aid1:
        org_ids = aid2orgid[aid]
        if len(org_ids) > 1:
            count1 += 1
        if len(org_ids) == 0:
            count2 += 1
        if len(org_ids) == 1:
            count3 += 1
    # 大部分人具备多个机构 (可能是由于筛选过后, 作者的职业生涯较长)
    tb = pt.PrettyTable()    
    tb.field_names = ["机构缺失", "唯一机构", "多个机构"]
    tb.add_row([count2, count3, count1])
    print(tb)
    
    # 处理方式: 机构缺失的作者剔除, 多个机构的通过文章数目确定主机构
    print("(5) 机构缺失的作者剔除, 多个机构的通过文章数目确定主机构")
    targeted_aid2 = dict()
    for aid in targeted_aid1:
        # 发文量信息
        x = targeted_aid1[aid]
        # 机构信息
        org_ids = aid2orgid[aid]
        if len(org_ids) == 0:  # 机构缺失的作者剔除分析过陈
            continue
        org_id_nop = list()    # 多个机构的通过文章数目确定主机构
        for org_id in org_ids:
            org_id_nop.append((org_id, org_ids[org_id]))
        org_id_nop = sorted(org_id_nop, key=lambda x: x[-1], reverse=True)
        org_id = org_id_nop[0][0]
        # 国家信息
        coun = org_id_dict[org_id]['loc'][0]
        
        targeted_aid2[aid] = dict()
        targeted_aid2[aid]['x'] = x            # 观测数据
        targeted_aid2[aid]['org_id'] = org_id  # 主机构id
        targeted_aid2[aid]['coun'] = coun      # 国家缩写
    
    # 随后只挑选机构人数多的机构
    noa = 30 
    org_id2noa = dict()
    for aid in targeted_aid2:
        org_id = targeted_aid2[aid]['org_id']
        if org_id not in org_id2noa:
            org_id2noa[org_id] = 1
        else:
            org_id2noa[org_id] += 1
        # org_ids = aid2orgid[aid] 
        # for org_id in org_ids:
        #     if org_id not in org_id2noa:
        #         org_id2noa[org_id] = 1
        #     else:
        #         org_id2noa[org_id] += 1
    org_id_filter_dict = dict()  # 人数超过noa的机构id
    for org_id in org_id2noa:
        if org_id2noa[org_id] > noa:
            org_id_filter_dict[org_id] = org_id2noa[org_id]
    print("(5) 人数超过{}的机构有{}个".format(noa, len(org_id_filter_dict)))
    # 
    targeted_aid3 = dict()
    for aid in targeted_aid2:
        org_id = targeted_aid2[aid]['org_id'] 
        if org_id in org_id_filter_dict:
            targeted_aid3[aid] = targeted_aid2[aid]
    print("(5) 准则过滤操作后, 剩余作者数目为: {}".format(len(targeted_aid3)))

    # 储存aid (这是实证研究所需要用的aid)
    with open(os.path.join(process_data_path, "aid_empirical.pkl"), 'wb') as f:
        pickle.dump(targeted_aid3, f)           
    

def check_targeted_aid():
    
    def get_xy(total_nop_list):
        # X轴是累计发文量大小(累计引用量大小)
        # Y轴是作者数目
        X_dict =  dict()
        for x in total_nop_list:
            if x not in X_dict:
                X_dict[x] = 0
            X_dict[x] += 1
        X = list(set(X_dict.keys()))
        Y = [X_dict[x] for x in X]
        return X, Y
    
    # 在targeted_aid基础上, 将没有机构信息的作者剔除了
    with open("./tmp/targeted_aid.pkl", 'rb') as f:
        targeted_aid = pickle.load(f)
    print("待分析作者数目: {}".format(len(targeted_aid)))
    
    # 检查抽取结果的统计特性
    total_noc_list = list()
    total_nop_list = list()
    for aid in targeted_aid:
        aid_total_noc = 0
        aid_total_nop = 0
        for year in targeted_aid[aid]['x']:
            for pid, cc in targeted_aid[aid]['x'][year]:
                aid_total_noc += cc
                aid_total_nop += 1
        total_noc_list.append(aid_total_noc)
        total_nop_list.append(aid_total_nop)
    
    # (1) 发文量分布
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    X, Y = get_xy(total_nop_list)
    plt.scatter(X, Y, c='black', s=3)
    plt.xscale('log')
    plt.yscale('log')    
    plt.xlabel(r"Accumulative number of publications ($N_{\alpha}$)")
    plt.ylabel("Number of authors")
    plt.xticks(10 ** np.arange(1, 4))
    
    # (2) 引用量分布
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    config = {
              "font.family" : "Times New Roman",
              "font.size" : 30
              }
    rcParams.update(config)
    
    X, Y = get_xy(total_noc_list)
    plt.scatter(X, Y, c='black', s=3)
    plt.xscale('log')
    plt.yscale('log')    
    plt.xlabel(r"Accumulative number of citations ($C_{\alpha}$)")
    plt.ylabel("Number of authors")
    plt.xticks(10 ** np.arange(0, 6))
    
    # (3) 个人机构数目分布
