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

from MyQPModel_Results import extract_nobels
from MyQPModel_Results.utils_predict import *


# 根据特定条件筛选满足aid进行实证分析

# 数据挂载盘
abs_path = "/mnt/disk2/"


def descriptive_analysis(save_path):
    '''
    # 对抽取得到的targeted_aid3数据进行描述性统计分析
    '''
    
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
    
    with open(os.path.join(save_path, "empirical_data.pkl"), 'rb') as f:
        targeted_aid = pickle.load(f)
    print("待分析作者数目: {}".format(len(targeted_aid)))
    
    # 检查抽取结果的统计特性
    total_noc_list = list()
    total_nop_list = list()
    for aid in targeted_aid:
        aid_total_noc = 0
        aid_total_nop = 0
        for year in targeted_aid[aid]['x_obs']:
            for pid, cc in targeted_aid[aid]['x_obs'][year]:
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


def extract_aids_for_empirical_analysis(configs):
    '''
    筛选学者进行实证分析
    '''
    
    file_name      = configs['file_name']
    i_list         = configs['i_list']
    born_year      = configs['born_year']
    span_year      = configs['span_year']
    min_nop        = configs['min_nop']
    max_nop        = configs['max_nop']
    cc_year        = configs['cc_year']
    begin_year     = configs['begin_year']
    end_year       = configs['end_year']
    org_noa_filter = configs['org_noa_filter']
    cou_noa_filter = configs['cou_noa_filter']
    save_path      = configs['save_path']
            
    # 读取诺贝尔奖数据 (实证研究) - 来自于王大顺收集
    nobel_data            = extract_nobels.read_nobel_data(extract_nobels.nobel_file_path)
    nobel_laureates_field = dict()
    if file_name in nobel_data:
        # 存在诺贝尔奖数据
        nobel_data_field  = nobel_data[file_name]
        i_list            = np.arange(2, 11)
        for i in i_list:
            # 作者发文量信息 ---> 由Extract Aid Fos.py确定
            with open(os.path.join(save_path, "aid2fos_{}.pkl".format(i)), 'rb') as f:
                aid2fos = pickle.load(f)
            # 遍历作者发文量, 判断其是否是诺奖得主
            for aid in aid2fos:
                for year in aid2fos[aid]:
                    for pid, _ in aid2fos[aid][year]:
                        if str(pid) in nobel_data_field:   # 是否属于该领域的诺奖得主
                            nobel_laureates_field[aid] = ''
            print("涉及获得诺奖的学者: {}".format(len(nobel_laureates_field)))
            
    # 读取该领域年均引用数据 (计算标准化引用需要) - 来自于ExtractFilteredFoS.py处理得到
    pubyear2cc_mean_std = read_file(os.path.join(save_path, "pubyear2cc_mean_std.pkl"))
    
    
    ''' 
    (1)-(4) 
    挑选学者进行实证研究
    '''
    targeted_aid1 = dict()
    for i in i_list:
        # 作者发文量信息 (由Extract Aid Fos.py确定)
        aid2fos = read_file(os.path.join(save_path, "aid2fos_{}.pkl".format(i)))
        # 作者引用量信息 (确定cc10)
        aid2cc  = read_file(os.path.join(save_path, "aid2cc_{}.pkl".format(i)))
        # 统计文件i的发文量区间 
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
            aid_born  = min(year_list)
            
            # 筛选准则 - (2)(3)(4)
            if aid_born < born_year[0] or aid_born > born_year[1]:
                continue
            # if max(year_list) - aid_born + 1 < span_year:
            #     continue
            if aid2nop[aid] > max_nop:
                continue
            # 筛选准则 - (1)
            nop = 0
            for year in aid2fos[aid]:
                if year > begin_year:
                    continue 
                else:
                    for pid, _ in aid2fos[aid][year]:
                        nop += 1
            if nop < min_nop and aid not in nobel_laureates_field:
                continue
            # 以下是经历筛选后的学者
            if aid not in targeted_aid1:
                count += 1
                targeted_aid1[aid] = dict()
                for year in aid2fos[aid]:
                    if year > begin_year:              # 只取 <= begin_year的发文记录
                        continue
                    targeted_aid1[aid][year] = list()  # 收集学者的发文量信息和引用信息
                    for pid, _ in aid2fos[aid][year]:
                        # 统计论文pid在cc_year年的累计引用量
                        cc      = 0
                        pubyear = year
                        for t in range(pubyear, pubyear + cc_year):
                            if t in aid2cc[pid]:
                                cc += aid2cc[pid][t]
                        # 非标准化的cc10
                        cc_10 = cc
                        # 引用数目是否进行标准化
                        mean_cc_t, std_cc_t = pubyear2cc_mean_std[year]
                        cc_zs = (cc - mean_cc_t) / max(std_cc_t, 1e-1)
                        cc_zs = np.exp(cc_zs) - 1
                           
                        targeted_aid1[aid][year].append((pid, cc_10, cc_zs))
        print("新增: {}人员 / {}人员".format(count, len(aid2fos)))
        
    count = 0
    for aid in targeted_aid1:
        if aid in nobel_laureates_field:
            count += 1
    print("(1)-(4) 准则过滤操作后, 剩余学者数目: {}".format(len(targeted_aid1)))
    print("(1)-(4) 准则过滤操作后, 剩余诺奖得主: {}".format(count))
    
    '''
    (5) 
    读取由mag_affiliation.py抽取的机构和国家信息
    '''
    # 读取机构位置信息 (机构名称, 机构国家, 机构坐标)
    org_id_dict = read_file(os.path.join(save_path, "org_id_dict.pkl"))
    # 读取学者所属机构信息 (Key: 作者id, Value: 机构id)
    aid2orgid   = read_file(os.path.join(save_path, "aid2orgid.pkl"))
    
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
    tb = pt.PrettyTable()    
    tb.field_names = ["机构缺失", "唯一机构", "多个机构"]
    tb.add_row([count2, count3, count1])
    print(tb)
    
    # 处理方式: 机构缺失的作者剔除, 多个机构的通过文章数目确定主机构
    print("(5) 机构缺失的作者剔除, 多个机构的通过文章数目确定主机构")
    targeted_aid2 = dict()
    for aid in targeted_aid1:
        x_obs   = targeted_aid1[aid]
        org_ids = aid2orgid[aid]
        if len(org_ids) == 0:  # 机构缺失的作者剔除
            continue
        org_id_nop = list()    # 多个机构的通过文章数目确定主机构
        for org_id in org_ids:
            org_id_nop.append((org_id, org_ids[org_id]))
        org_id_nop = sorted(org_id_nop, key=lambda x: x[-1], reverse=True)
        org_id     = org_id_nop[0][0]
        cou        = org_id_dict[org_id]['loc'][0]  # 国家信息
        
        targeted_aid2[aid] = dict()
        targeted_aid2[aid]['x_obs']  = x_obs    # 观测数据 (发文量和引用量)
        targeted_aid2[aid]['org_id'] = org_id   # 观测数据 (作者隶属机构)
        targeted_aid2[aid]['cou']    = cou      # 观测数据 (机构隶属国家) 
    
    cou2noa = dict()   # 统计每个国家内人数
    org2noa = dict()   # 统计每个机构内人数
    for aid in targeted_aid2:
        org_id = targeted_aid2[aid]['org_id']
        cou    = targeted_aid2[aid]['cou']
        if org_id not in org2noa:
            org2noa[org_id] = 1    
        else:
            org2noa[org_id] += 1    
        if cou not in cou2noa:
            cou2noa[cou] = 1
        else:    
            cou2noa[cou] += 1
            
    # 人数超过org_noa_filter的机构
    org2noa_filter = dict()  
    for org_id in org2noa:
        if org2noa[org_id] >= org_noa_filter:
            org2noa_filter[org_id] = org2noa[org_id]
    print("(5) 人数超过{}的机构有{}个".format(org_noa_filter, len(org2noa_filter)))
    # 人数超过cou_noa_filter的国家
    cou2noa_filter = dict()
    for cou in cou2noa:
        if cou2noa[cou] >= cou_noa_filter:
            cou2noa_filter[cou] = cou2noa[cou]
    print("(5) 人数超过{}的国家有{}个".format(cou_noa_filter, len(cou2noa_filter)))
    
    targeted_aid3 = dict()
    for aid in targeted_aid2:
        org_id = targeted_aid2[aid]['org_id'] 
        cou    = targeted_aid2[aid]['cou']
        if org_id in org2noa_filter and cou in cou2noa_filter:
            targeted_aid3[aid] = targeted_aid2[aid]
    print("(5) 准则过滤操作后, 剩余作者数目为: {} / {}".format(len(targeted_aid3), len(targeted_aid2)))
    
    # 统计待分析的文章数
    pid_dict = dict()
    for aid in targeted_aid3:
        for year in targeted_aid3[aid]['x_obs']:
            for pid in targeted_aid3[aid]['x_obs'][year]:
                if pid not in pid_dict:
                    pid_dict[pid] = ''
    print("文章数目 : {}".format(len(pid_dict)))
    # 统计待分析的诺奖得主数
    count = 0
    for aid in targeted_aid3:
        if aid in nobel_laureates_field:
            count += 1
    print("诺奖得主 : {}".format(count))

    # 储存实证研究所需要用的学者
    save_file(targeted_aid3, os.path.join(save_path, "empirical_data.pkl"))
    
    
def extract_aids_for_empirical_analysis2(configs2):
    ''' 
    '''

    file_name      = configs2['file_name']
    i_list         = configs2['i_list']
    span_year      = configs2['span_year']      # 核心 [1980, 1985]
    min_nop        = configs2['min_nop']        # 核心 5
    max_nop        = configs2['max_nop']
    cc_year        = configs2['cc_year']
    org_noa_filter = configs2['org_noa_filter'] # 0
    cou_noa_filter = configs2['cou_noa_filter'] # 0
    save_path      = configs2['save_path']
         
    # 读取该领域年均引用数据 (计算标准化引用需要) - 来自于ExtractFilteredFoS.py处理得到
    pubyear2cc_mean_std = read_file(os.path.join(save_path, "pubyear2cc_mean_std.pkl"))

    targeted_aid1 = dict()
    for i in i_list:
        # 作者发文量信息 (由Extract Aid Fos.py确定)
        aid2fos = read_file(os.path.join(save_path, "aid2fos_{}.pkl".format(i)))
        # 作者引用量信息 (确定cc10)
        aid2cc  = read_file(os.path.join(save_path, "aid2cc_{}.pkl".format(i)))
        # 统计文件i的学者发文量区间 
        aid2nop  = dict()
        nop_list = list()
        for aid in aid2fos:
            nop = 0
            for pubyear in aid2fos[aid]:
                nop += len(aid2fos[aid][pubyear])
            nop_list.append(nop)
            aid2nop[aid] = nop
        print("发文量: {} - {}".format(min(nop_list), max(nop_list))) 
        # 筛选学者
        count = 0
        for aid in aid2fos:
            year_list = list(sorted(aid2fos[aid].keys()))
            span_down = span_year[0]
            span_up   = span_year[1]
            nop       = 0
            for year in year_list:
                if span_down <= year and year < span_up:
                    for pid, _ in aid2fos[aid][year]:
                        nop += 1
            if nop < min_nop or nop > max_nop:
                continue
            
            # 以下是经历筛选后的学者
            if aid not in targeted_aid1:
                count += 1
                targeted_aid1[aid] = dict()
                for year in year_list:
                    if span_down <= year and year < span_up:
                        targeted_aid1[aid][year] = list()  # 收集学者的发文量信息和引用信息
                        for pid, _ in aid2fos[aid][year]:
                            
                            # 统计论文pid在cc_year年的累计引用量
                            cc      = 0
                            pubyear = year
                            for t in range(pubyear, pubyear + cc_year):
                                if t in aid2cc[pid]:
                                    cc += aid2cc[pid][t]
                            # 非标准化的cc10
                            cc_10 = cc
                            # 引用数目是否进行标准化
                            mean_cc_t, std_cc_t = pubyear2cc_mean_std[year]
                            cc_zs = (cc - mean_cc_t) / max(std_cc_t, 1e-1)
                            cc_zs = np.exp(cc_zs) - 1
                            
                            targeted_aid1[aid][year].append((pid, cc_10, cc_zs))
        print("新增: {}人员 / {}人员".format(count, len(aid2fos)))
        
    print("{}-{}年内剩余学者数目: {}".format(span_down, span_up, len(targeted_aid1)))

    # 读取机构位置信息 (机构名称, 机构国家, 机构坐标)
    org_id_dict = read_file(os.path.join(save_path, "org_id_dict.pkl"))
    # 读取学者所属机构信息 (Key: 作者id, Value: 机构id)
    aid2orgid   = read_file(os.path.join(save_path, "aid2orgid.pkl"))
    
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
    tb = pt.PrettyTable()    
    tb.field_names = ["机构缺失", "唯一机构", "多个机构"]
    tb.add_row([count2, count3, count1])
    print(tb)
    
    # 处理方式: 机构缺失的作者剔除, 多个机构的通过文章数目确定主机构
    print("(5) 机构缺失的作者剔除, 多个机构的通过文章数目确定主机构")
    targeted_aid2 = dict()
    for aid in targeted_aid1:
        x_obs   = targeted_aid1[aid]
        org_ids = aid2orgid[aid]
        if len(org_ids) == 0:  # 机构缺失的作者剔除
            continue
        org_id_nop = list()    # 多个机构的通过文章数目确定主机构
        for org_id in org_ids:
            org_id_nop.append((org_id, org_ids[org_id]))
        org_id_nop = sorted(org_id_nop, key=lambda x: x[-1], reverse=True)
        org_id     = org_id_nop[0][0]
        cou        = org_id_dict[org_id]['loc'][0]  # 国家信息
        
        targeted_aid2[aid] = dict()
        targeted_aid2[aid]['x_obs']  = x_obs    # 观测数据 (发文量和引用量)
        targeted_aid2[aid]['org_id'] = org_id   # 观测数据 (作者隶属机构)
        targeted_aid2[aid]['cou']    = cou      # 观测数据 (机构隶属国家) 
    
    cou2noa = dict()   # 统计每个国家内人数
    org2noa = dict()   # 统计每个机构内人数
    for aid in targeted_aid2:
        org_id = targeted_aid2[aid]['org_id']
        cou    = targeted_aid2[aid]['cou']
        if org_id not in org2noa:
            org2noa[org_id] = 1    
        else:
            org2noa[org_id] += 1    
        if cou not in cou2noa:
            cou2noa[cou] = 1
        else:    
            cou2noa[cou] += 1
            
    # 人数超过org_noa_filter的机构
    org2noa_filter = dict()  
    for org_id in org2noa:
        if org2noa[org_id] >= org_noa_filter:
            org2noa_filter[org_id] = org2noa[org_id]
    print("(5) 人数超过{}的机构有{}个".format(org_noa_filter, len(org2noa_filter)))
    # 人数超过cou_noa_filter的国家
    cou2noa_filter = dict()
    for cou in cou2noa:
        if cou2noa[cou] >= cou_noa_filter:
            cou2noa_filter[cou] = cou2noa[cou]
    print("(5) 人数超过{}的国家有{}个".format(cou_noa_filter, len(cou2noa_filter)))
    
    targeted_aid3 = dict()
    for aid in targeted_aid2:
        org_id = targeted_aid2[aid]['org_id'] 
        cou    = targeted_aid2[aid]['cou']
        if org_id in org2noa_filter and cou in cou2noa_filter:
            targeted_aid3[aid] = targeted_aid2[aid]
    print("(5) 准则过滤操作后, 剩余作者数目为: {} / {}".format(len(targeted_aid3), len(targeted_aid2)))
    
    # 统计待分析的文章数
    pid_dict = dict()
    for aid in targeted_aid3:
        for year in targeted_aid3[aid]['x_obs']:
            for pid in targeted_aid3[aid]['x_obs'][year]:
                if pid not in pid_dict:
                    pid_dict[pid] = ''
    print("文章数目 : {}".format(len(pid_dict)))

    # 储存实证研究所需要用的学者
    save_file(targeted_aid3, os.path.join(save_path, "empirical_data_{}-{}.pkl".format(*span_year)))



def calculate_avg_max_min(temp):
    temp2 = np.array([temp[t] for t in temp])
    mean2 = np.mean(temp2)
    max2  = np.max(temp2)
    min2  = np.min(temp2)
    return mean2, max2, min2
    

def main():
    
    # 读取发文量信息
    i_list         = [5, 6, 7, 8, 9, 10]
    born_year      = [1990, 2000]      # 第一篇文章发表的时间
    span_year      = 10                # 职业生涯长度
    min_nop        = 30                # 所使用的时间段, 至少min_nop篇文章 (50)
    max_nop        = 300               # 避免是数据集内作者消歧错误
    cc_year        = 10                # 一篇论文发表cc_year年的引用衡量该篇论文的质量
    begin_year     = 2010              # 保证存在c10
    end_year       = 2018              # 保证存在c10  
    org_noa_filter = 10
    cou_noa_filter = 100
    # file_name = "physics" # "chemistry" # "medicine" # "computer science" # 选择领域

    for file_name in ["physics", "chemistry", "computer science"]:
        # 实证数据配置
        configs = dict()
        configs['file_name']      = file_name       # 领域
        configs['i_list']         = i_list          # 源文件
        configs['born_year']      = born_year       # 介于该年份区间的学者
        configs['span_year']      = span_year       # 
        configs['min_nop']        = min_nop         # 学者最少发文量
        configs['max_nop']        = max_nop         # 学者最大发文量
        configs['cc_year']        = cc_year         # 一篇论文发表后*年的引用量衡量其质量
        configs['begin_year']     = begin_year      # 保证c10
        configs['end_year']       = end_year        #
        configs['org_noa_filter'] = org_noa_filter  # 机构最少人数
        configs['cou_noa_filter'] = cou_noa_filter  # 国家最少人数
        configs['save_path']      = os.path.join(abs_path, "EmpiricalData/StatisticalData_{}".format(file_name))
        # 抽取实证数据
        extract_aids_for_empirical_analysis(configs)
      
    # 统计分析数据中学者和论文数目   
    for file_name in ["physics", "chemistry", "computer science"]:
        save_path    = os.path.join(abs_path, "EmpiricalData/StatisticalData_{}".format(file_name))
        targeted_aid = read_file(os.path.join(save_path, "empirical_data.pkl"))
        
        cou2num = dict()
        org2num = dict()
        aid2nop = dict()   
        aid2c10 = dict()
        aid2czs = dict()
        pid2c10 = dict()  # 10 years citation count
        pid2czs = dict()  # z-score citation count        
        for aid in targeted_aid:
            cou = targeted_aid[aid]['cou']
            org = targeted_aid[aid]['org_id']
            x   = targeted_aid[aid]['x_obs']
            if cou not in cou2num:
                cou2num[cou] = 1
            else:
                cou2num[cou] += 1
            if org not in org2num:
                org2num[org] = 1
            else:
                org2num[org] += 1

            nop   = 0
            noc10 = 0
            noczs = 0
            for year in x:
                for pid, c10, czs in x[year]:
                    if pid not in pid2c10:
                        pid2c10[pid] = c10
                    if pid not in pid2czs:
                        pid2czs[pid] = np.log(czs + 1)
                    nop   += 1
                    noc10 += c10
                    noczs += np.log(czs + 1)
                
            if aid not in aid2nop:
                aid2nop[aid] = nop
                aid2c10[aid] = noc10
                aid2czs[aid] = noczs
        # 统计特征
        print("----------------------------")
        print("国家数目: {}".format(len(cou2num)))
        print("机构数目: {}".format(len(org2num)))
        print("学者数目: {}".format(len(aid2nop)))
        print("论文数目: {}".format(len(pid2c10)))
        cou_avg, cou_max, cou_min = calculate_avg_max_min(cou2num)
        org_avg, org_max, org_min = calculate_avg_max_min(org2num)
        print("国家平均人数: {:.4f}, 国家最大人数: {}, 国家最小人数: {}".format(cou_avg, cou_max, cou_min))
        print("机构平均人数: {:.4f}, 机构最大人数: {}, 机构最小人数: {}".format(org_avg, org_max, org_min))
        
        nop_avg, nop_max, nop_min = calculate_avg_max_min(aid2nop)
        c10_avg, c10_max, c10_min = calculate_avg_max_min(aid2c10)
        czs_avg, czs_max, czs_min = calculate_avg_max_min(aid2czs)
        print("学者平均发文量: {:.4f},  学者最大发文量: {},  学者最小发文量: {}".format(nop_avg, nop_max, nop_min))
        print("学者平均累计C10: {:.4f}, 学者最大累计C10: {}, 学者最小累计C10: {}".format(c10_avg, c10_max, c10_min))
        print("学者平均累计CZS: {:.4f}, 学者最大累计CZS: {:.4f}, 学者最小累计CZS: {:.4f}".format(czs_avg, czs_max, czs_min))
        
        pc10_avg, pc10_max, pc10_min = calculate_avg_max_min(pid2c10)
        pczs_avg, pczs_max, pczs_min = calculate_avg_max_min(pid2czs)
        print("论文平均累计C10: {:.4f}, 论文最大累计C10: {}, 论文最小累计C10: {}".format(pc10_avg, pc10_max, pc10_min))
        print("论文平均累计CZS: {:.4f}, 论文最大累计CZS: {:.4f}, 论文最小累计CZS: {:.4f}".format(pczs_avg, pczs_max, pczs_min))
        print("----------------------------")
    
      
    
    # 融合学术环境与选题策略的科研能力量化模型预测分析的实证数据抽取
    for file_name in ["physics", "chemistry", "computer science"]:
        span_down_list = np.arange(1970, 2015, 5)[:, np.newaxis]
        span_up_list   = span_down_list + 5
        span_year_list = np.concatenate([span_down_list, span_up_list], axis=-1)
        for span_year in span_year_list:
            # 实证数据配置
            configs2 = dict()
            configs2['file_name']      = file_name       # 领域
            configs2['i_list']         = i_list          # 源文件
            configs2['span_year']      = span_year       # 
            configs2['min_nop']        = 5               # 学者最少发文量
            configs2['max_nop']        = 300             # 学者最少发文量
            configs2['cc_year']        = 10              # 一篇论文发表后*年的引用量衡量其质量
            configs2['org_noa_filter'] = 50              # 机构最少人数
            configs2['cou_noa_filter'] = 500             # 国家最少人数
            configs2['save_path']      = os.path.join(abs_path, "EmpiricalData/StatisticalData_{}".format(file_name))
            # 抽取实证数据
            extract_aids_for_empirical_analysis2(configs2)
        break
        
    # 统计分析数据中学者和论文数目
    for file_name in ["physics", "chemistry", "computer science"]:
        span_down_list = np.arange(1970, 2015, 5)[:, np.newaxis]
        span_up_list   = span_down_list + 5
        span_year_list = np.concatenate([span_down_list, span_up_list], axis=-1)
        save_path      = os.path.join(abs_path, "EmpiricalData/StatisticalData_{}".format(file_name))
        
        tb = pt.PrettyTable()
        tb.field_names = ["时间", "国家数目", "机构数目", "学者数目", "论文数目"]
            
        for span_year in span_year_list:
            targeted_aid3 = read_file(os.path.join(save_path, "empirical_data_{}-{}.pkl".format(*span_year)))
            
            cous = dict()
            orgs = dict()
            pids = dict()
            for aid in targeted_aid3:
                org    = targeted_aid3[aid]['org_id']
                cou    = targeted_aid3[aid]['cou']
                x_obs  = targeted_aid3[aid]['x_obs']
                for year in x_obs:
                    for pid, cc, cc_zs in x_obs[year]:
                        pids[pid] = ''
                if cou not in cous:
                    cous[cou] = 0
                if org not in orgs:
                    orgs[org] = 0
                
            nop = len(pids)           # 论文数目
            noo = len(orgs)           # 机构数目
            noc = len(cous)           # 国家数目
            noa = len(targeted_aid3)  # 学者数目
            tb.add_row(["{}-{}".format(*span_year), "{}".format(noc), "{}".format(noo), "{}".format(noa), "{}".format(nop)])
        
        print(file_name)
        print(tb) 
        
