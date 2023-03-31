#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:01:24 2022

@author: aixuexi
"""
import os
import csv
import pickle
import json
from tqdm import tqdm


FilePath = "/mnt/disk2/EmpiricalData/Nobels"
nobel_file_path = os.path.join(FilePath, "Nobel_laureates")


#%%
def read_nobel_data(nobel_file_path):
    # 数据来自 "A dataset of publication records for Nobel laureates"
    for file_name in os.listdir(nobel_file_path):
        print(file_name)
    
    # Prize-winning paper record.csv
    file_path = os.path.join(nobel_file_path, "Prize-winning paper record.csv")
            
    # 诺奖文章
    nobel_data = dict()
    with open(file_path, 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        count  = 0
        for row in reader:
            count += 1
            if count == 1:
                print(row)
                continue
            else:
                Field         = row[0].lower()
                Pid           = row[6]
                Aid           = int(row[1])
                Prize_Year    = int(row[3])
                Pub_Year      = int(row[5])
                Laureate_Name = row[2]
                Title         = row[4]
            
            # 写入字典
            if Field not in nobel_data:
                nobel_data[Field] = dict()
            if Pid == "":
                Pid = "xxx" + str(count)
            if Pid not in nobel_data[Field]:
                nobel_data[Field][Pid]                  = dict()
                nobel_data[Field][Pid]["Prize_Year"]    = Prize_Year
                nobel_data[Field][Pid]["Pub_Year"]      = Pub_Year
                nobel_data[Field][Pid]["Laureate_Name"] = Laureate_Name
                nobel_data[Field][Pid]["Title"]         = Title
                nobel_data[Field][Pid]["Aids"]          = [Aid]     
            else:
                nobel_data[Field][Pid]["Aids"].append(Aid)
                     
    return nobel_data


def match_nobel_paper(field="physics"):
    # 将nobel paper 匹配MAG中 pid
    # 两者Pid已经对齐, 无需进一步匹配
    
    # 读取所有获得诺奖文章
    nobel_data  = read_nobel_data(nobel_file_path)
    nobel_field = list(nobel_data.keys())
    nobel_field = [field.lower() for field in nobel_field] # 'physics', 'chemistry', 'medicine'
    
    if field not in nobel_field:
        print(field, "exit -1")
        return -1
     
    nobel_data_field = nobel_data[field]
    Time2Title = dict()
    for Pid in nobel_data_field:
        Pub_Year = nobel_data_field[Pid]['Pub_Year']
        Title    = nobel_data_field[Pid]['Title']
        if Pub_Year not in Time2Title:
            Time2Title[Pub_Year] = list()
        Time2Title[Pub_Year].append(Title)  
    
    # 多进程匹配
    file_out_dir    = os.path.join(FilePath, "MAGv2.1-meta-{}".format(field))
    file_out_path_x = os.path.join(file_out_dir, "mag_papers_{}.txt")
    match_results   = list()
    for file_id in tqdm(range(0, 17)):
        file_out_path = file_out_path_x.format(file_id)
        with open(file_out_path, 'r') as f:
             while True:
                   oneline = f.readline().strip()
                   if oneline:
                      oneline_json = json.loads(oneline)
                      pid          = str(oneline_json['pid'])
                      pubyear      = oneline_json['t']
                      
                      
                      if pid in nobel_data_field:
                          match_results.append(oneline_json)
                          print("One match")
                   else:
                       break
    
    # 检查匹配结果
    for oneline_json in match_results:
        pid   = str(oneline_json['pid'])
        title1 = oneline_json['ti'].lower()
        title2 = nobel_data_field[pid]['Title']
        print(title1)
        print(title2)