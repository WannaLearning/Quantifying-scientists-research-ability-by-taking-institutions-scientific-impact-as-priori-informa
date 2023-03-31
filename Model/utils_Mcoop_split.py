#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:39:36 2022

@author: aixuexi
"""
import random
import numpy as np
from tqdm import tqdm
from queue import Queue


# 将合作矩阵M_coop划分成近似闭包 - 每个闭集多进程更新
# (1) M_coop 本质是一个连通图 (可将弱链接(即只合作一次)的边剔除)
# (2) 利用广度遍历算法, 找寻闭集


def check_i_j_coop_num(i, j, M_coop):
    # 检查 i 和 j的合作次数; M_coop : 文章数目 x 人员数目 的合作矩阵
    i_  = M_coop[:, i] > 0
    j_  = M_coop[:, j] > 0
    i_j = np.array(i_, np.int32) + np.array(j_, np.int32)
    print("{} 和 {} 合作{}次".format(i, j, np.sum(i_j > 1)))
    

def Identify_ColosedSet_in_Mcoop(M_coop):
    # 利用广度优先遍历寻找合作矩阵中合作闭集
    total_nop_num, total_noa_num = M_coop.shape
    visited_aids_all  = dict()  # 第i-1次看过的结点
    visited_aids_list = list()  # 
    while len(visited_aids_all) < total_noa_num:
        # 已经被选过, 则不作为起点备选集
        alternative_source = dict()
        for i in np.arange(0, total_noa_num, 1, dtype=np.int32):
            if i not in visited_aids_all:  
                alternative_source[i] = ''
        # 从起点开始遍历 - 起点采样至起点备选集
        source = random.sample(alternative_source.keys(), 1)[0]
    
        # 开始一次广度优先遍历
        queue = Queue()     
        visited_aids = dict()      # 存放本次已经遍历的作者
        visited_aids[source] = ''  # 起点
        queue.put(source)
        while not queue.empty():
            # 第vertex位作者
            vertex = queue.get()  
            # 确定第vertex位作者的文章
            vertex_papers_idx = M_coop[:, vertex] > 0
            vertex_papers     = M_coop[vertex_papers_idx]
            # 确定第vertex位作者的合作者
            vertex_coa = np.sum(vertex_papers, axis=0) > 0
            for j, vertex_coa_j in enumerate(vertex_coa):  
                # 如果 是合作者 且 本次未见过, 则添加
                if vertex_coa_j:
                    if j not in visited_aids:  
                        visited_aids[j] = ''
                        queue.put(j)
                        
        for aid in visited_aids:
            if aid in visited_aids_all:
                print("重复包括")
            visited_aids_all[aid] = ''
        
        visited_aids_list.append(visited_aids)
        # print("{} / {} = {:.6f}".format(len(visited_aids_all), total_noa_num, len(visited_aids_all)/ total_noa_num))
        
        
    # 检查是否有作者丢失
    check_aids_num = dict()
    for aids in visited_aids_list:
        for aid in aids:
            if aid not in check_aids_num:
                check_aids_num[aid] = ''
    assert(len(check_aids_num) == total_noa_num)   
    return visited_aids_list


def Merge_ColosedSet_Size(visited_aids_list, mp_num, threshold):
    # 分组数目不能超过进程数目 < mp &  合并太小的集合的阈值 < threshold
    # 合并小于阈值的集合
    visited_aids_list  = sorted(visited_aids_list, key=lambda x: len(x), reverse=True)
    visited_aids_list_ = list()  
    group_i_ = dict()
    for group_i in visited_aids_list:
        if len(group_i) > threshold:
            visited_aids_list_.append(group_i)
        else:
            for j in group_i:
                group_i_[j] = ''
            if len(group_i_) > threshold:
                visited_aids_list_.append(group_i_)
                group_i_ = dict()
    if len(visited_aids_list_) == 0: #  
        visited_aids_list_.append(group_i_)
    else:
        if len(group_i_) != 0:       # 余下的补入倒数第一个组
            for j in group_i_:
                visited_aids_list_[-1][j] = ''
                
    if mp_num >= 2:
        # 检查进程数目要求, 集合过多合并
        while len(visited_aids_list_) > mp_num:
            visited_aids_list_  = sorted(visited_aids_list_, key=lambda x: len(x), reverse=True)
            # 将最小两个合并
            group_i_ = dict()
            for j in visited_aids_list_[-1]:
                group_i_[j] = ''
            for j in visited_aids_list_[-2]:
                group_i_[j] = ''
            del visited_aids_list_[-1]
            del visited_aids_list_[-1]  # 删了倒数第一, 倒数第二是倒数第一
            visited_aids_list_.append(group_i_)
    else:
        # 当进程数为1, 返回一个全集
        group_i_ = dict()
        for group_i in visited_aids_list_:
            for j in group_i:
                group_i_[j] = ''
        visited_aids_list_ = [group_i_]
            
    visited_aids_list_  = sorted(visited_aids_list_, key=lambda x: len(x), reverse=True)
    
    # 检查是否有作者丢失
    check_aids_num = dict()
    for aids in visited_aids_list_:
        for aid in aids:
            if aid not in check_aids_num:
                check_aids_num[aid] = ''
    print("(算法检查)学者数目: {}".format(len(check_aids_num)))
    
    return visited_aids_list_


def utils_mp_split(data, mp_num=8, threshold=100):
    # 准备多进程切分数据
    M_coop = data["M_coop"]
    total_nop_num, total_noa_num = M_coop.shape
    # 寻找合作矩阵连通图中闭集
    visited_aids_list  = Identify_ColosedSet_in_Mcoop(M_coop)
    # 融合较小闭集 & 保证集合数目不超过进程要求数
    visited_aids_list_ = Merge_ColosedSet_Size(visited_aids_list, mp_num, threshold)
    # 划分数据集
    rou_num = 0
    col_num = 0
    row_col_list = list()  # (row, col): col是该进程计算的作者-BOOL, row是该进程计算的作者的文章-BOOL
    for i, group_i_ in enumerate(visited_aids_list_):
        group_i        = list(sorted(group_i_.keys()))
        col_i          = np.zeros(total_noa_num)
        col_i[group_i] = 1
        col_i          = np.array(col_i, dtype=np.bool)
        M_coop_i       = M_coop[:, col_i]
        row_i          = np.array(np.sum(M_coop_i, axis=-1) > 0, dtype=np.bool)
        M_coop_i       = M_coop_i[row_i, :]
        row_col_list.append(( row_i, col_i ))
        
        # M_coop_i = np.array(M_coop_i > 0, dtype=np.int32)
        # print(np.sum(np.sum(M_coop_i, axis=-1) > 1))
        print("({})M_coop被分解为:".format(i), M_coop_i.shape)
        rou_num += M_coop_i.shape[0]
        col_num += M_coop_i.shape[1]
    
    print("M_coop累计: {} x {}".format(rou_num, col_num))
    return row_col_list
    

#%%
def Identify_ColosedSet_in_Mcoop2(M_coop, 
                                  row_id1, row_id2, col_id1, col_id2, aid2pid,
                                  epsilon):
    # 利用广度优先遍历寻找合作矩阵中合作闭集
    # 利用 M_coop 矩阵过大, 切片太慢,
    M_coop = np.array(M_coop > 0, dtype=np.int32)
    
    total_nop_num, total_noa_num = M_coop.shape
    visited_aids_all  = dict()                  # 第i-1次看过的结点
    visited_aids_list = list() 
    while len(visited_aids_all) < total_noa_num:
        
        # 已经被选过, 不作为起点备选集
        alternative_source = dict()
        for i in np.arange(0, total_noa_num, 1, dtype=np.int16):
            if i not in visited_aids_all:  
                alternative_source[i] = ''
       
        # 从起点开始遍历 - 起点采样至起点备选集
        source = random.sample(alternative_source.keys(), 1)[0]
    
        # 开始一次广度优先遍历
        queue        = Queue()     
        visited_aids = dict()                 # 存放本次已经遍历的作者
        visited_aids[source] = ''             # 起点 source 压栈
        queue.put(source)
        
        while not queue.empty():
            # 第vertex位作者
            vertex = queue.get()  
            # 确定第vertex位作者的文章
            aid     = col_id2[vertex]         # 学者的真实aid
            pidlist = list(set(aid2pid[aid])) # 学者的发文情况
    
            # 计算与每位学者的合作次数            
            temp = np.zeros((total_noa_num, ), dtype=np.float16)
            for pid in pidlist:
                temp += M_coop[row_id1[pid], :]
            vertex_coa = temp > epsilon       # 合作次数超过 epsilon 次的密切合作者
            
            for j, vertex_coa_j in enumerate(vertex_coa):  
                # 如果是密切合作者 & 本次未见过 & 之前未见过 ---> 添加进入visited_aids
                if vertex_coa_j:
                    if j not in visited_aids and j not in visited_aids_all:  
                        visited_aids[j] = ''
                        queue.put(j)
        # 将本次近似闭集中点记录至已见过
        for j in visited_aids:
            visited_aids_all[j] = ''
        # 添加新近似闭集
        visited_aids_list.append(visited_aids)
        # print("{} / {} = {:.4f}".format(len(visited_aids_all), total_noa_num, len(visited_aids_all)/ total_noa_num))
    
    # 检查识别的闭集大小
    num_list = list()
    for visited_aids in visited_aids_list:
        num_list.append(len(visited_aids))
    print("最大的10个近似闭集", sorted(num_list, reverse=True)[:10])
         
    # 检查是否有作者丢失
    check_aids_num = dict()
    for visited_aids in visited_aids_list:
        for j in visited_aids:
            if j not in check_aids_num:
                check_aids_num[j] = ''
    assert(len(check_aids_num) == total_noa_num)
    print("(算法检查)学者数目: {}".format(len(check_aids_num)))
    
    return visited_aids_list


def utils_mp_split2(data, 
                    row_id1, row_id2, col_id1, col_id2, aid2pid,
                    mp_num=10, threshold=50, epsilon=5):
    # 准备多进程切分数据
    M_coop = data["M_coop"]
    total_nop_num, total_noa_num = M_coop.shape
    # 寻找合作矩阵连通图中闭集 (改动处)
    visited_aids_list  = Identify_ColosedSet_in_Mcoop2(M_coop, row_id1, row_id2, col_id1, col_id2, aid2pid, epsilon)
    # 融合较小闭集 & 保证集合数目不超过进程要求数
    visited_aids_list_ = Merge_ColosedSet_Size(visited_aids_list, mp_num, threshold)

    # 划分数据集
    row_num = 0
    col_num = 0
    row_col_list = list()  # (row, col): col是该进程计算的作者-BOOL, row是该进程计算的作者的文章-BOOL
    for i, group_i_ in enumerate(visited_aids_list_):
        group_i        = list(sorted(group_i_.keys()))
        col_i          = np.zeros(total_noa_num)
        col_i[group_i] = 1
        col_i          = np.array(col_i, dtype=np.bool)
        M_coop_i       = M_coop[:, col_i]
        row_i          = np.array(np.sum(M_coop_i, axis=-1) > 0, dtype=np.bool)
        M_coop_i       = M_coop_i[row_i, :]
        row_col_list.append(( row_i, col_i ))
        
        print("({})M_coop被分解为:".format(i), M_coop_i.shape)
        row_num += M_coop_i.shape[0]
        col_num += M_coop_i.shape[1]

    print("M_coop累计: {} x {}".format(row_num, col_num))
    return row_col_list



#%%
def random_split(targeted_aid, chunks_1):
    ''' 所有学者随机切割成chunks块 
    '''
    aids        = list(targeted_aid.keys())
    batch_size  = int(np.ceil(len(aids) / chunks_1))
    start       = 0
    end         = 0
    aids_chunks = list()
    for c in range(chunks_1):
        end        = min(start + batch_size, len(aids))
        aids_chunk = aids[start: end]
        aids_chunks.append(aids_chunk)
        start      = end
    return aids_chunks
