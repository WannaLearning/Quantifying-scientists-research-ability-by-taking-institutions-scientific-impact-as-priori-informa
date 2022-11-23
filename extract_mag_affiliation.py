#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 09:26:53 2022

@author: aixuexi
"""
import os
import pickle
import json
import numpy as np
import reverse_geocoder as rg
from tqdm import tqdm


# 读取每个作者(cs/ physics)所属机构

# 数据挂载盘
abs_path = "/mnt/disk2/"

# mag内约2.5w个机构的信息
mag_affiliation_path = abs_path + "MAG_DATA_SET/MAGv2.1-affiliations/mag_affiliations.txt"

# # MAG_cs的数据路径
# mag_papers_meta_cs = abs_path + "MAG_DATA_SET/MAGv2.1-meta-cs/mag_papers_{}.txt"
# mag_papers_meta = mag_papers_meta_cs
# # 预处理数据存放路径
# field_name = "computer science"
# file_name = "cs"
# process_data_path = abs_path + "EmpiricalData/StatisticalData_cs"

# MAG_physics的数据路径
mag_papers_meta_physics = abs_path + "MAG_DATA_SET/MAGv2.1-meta-physics/mag_papers_{}.txt"
mag_papers_meta = mag_papers_meta_physics
# 预处理数据存放路径
field_name = "physics"
file_name = "physics"
process_data_path = abs_path + "EmpiricalData/StatisticalData_physics"


def read_mag_affiliation():
    # 读取mag内所有机构信息
    org_id_dict = dict()
    with open(mag_affiliation_path, 'r') as f:
        while True:
            oneline = f.readline().strip()
            if oneline:
                oneline_json = json.loads(oneline)
                org_id = oneline_json['id']
                org_name = oneline_json['NormalizedName']
                org_xy = (float(oneline_json["Latitude"]), float(oneline_json["Longitude"]))
                
                if org_id not in org_id_dict:
                    org_id_dict[org_id] = dict()
                    org_id_dict[org_id]['name'] = org_name
                    org_id_dict[org_id]['xy'] = org_xy
            else:
                break
    return org_id_dict
        

def extract_loc(org_id_dict):
    # 抽取mag内所有机构所隶属的国家 : org_id_dict
    def getplace(latitude, longtitude):
        # 根据精度和维度获取国家
        xy  = (latitude, longtitude)
        loc = rg.search(xy)[0]
        city    = loc['name']     # 城市
        county  = loc['admin2']   # 郡
        state   = loc['admin1']   # 州
        country = loc['cc']       # 国家
        # print(country + " ," + state + " ," + county + " ," + city)
        return (country, state, county, city)
    
    # 查询缓慢
    for org_id in tqdm(org_id_dict):
        latitude, longtitude = org_id_dict[org_id]['xy']
        loc = getplace(latitude, longtitude)
        org_id_dict[org_id]['loc'] = loc 
    # 储存.pkl文件
    with open(os.path.join(process_data_path, "org_id_dict.pkl"), 'wb') as f:
        pickle.dump(org_id_dict, f)
    return org_id_dict


def extract_aid2orgid(mag_papers_meta_physics, file_name):
    # Key: 是作者id, Value: 是机构id
    # Step 2.1: 抽取这些作者(aid2nop_part)的文章 meta 信息 (fos, pubyear, aids)
    aid2orgid = dict()
    for i in tqdm(range(0, 17)):
        with open(mag_papers_meta_physics.format(i), 'r') as f:
            while True:
                oneline = f.readline().strip()
                if oneline:
                    oneline_json = json.loads(oneline)
                    fos_list = oneline_json['f']
                    year = oneline_json['t']
                    pid = oneline_json['pid']
                    aid_list = oneline_json['aid']
                    for aid, org_id in aid_list:
                        if aid not in aid2orgid:
                            aid2orgid[aid] = dict()
                        if org_id == "":                 # 删除空字串
                            continue
                        if org_id not in aid2orgid[aid]: # 统计特定机构发文次数
                            aid2orgid[aid][org_id] = 1
                        else:
                            aid2orgid[aid][org_id] += 1    
                else:
                    break
    # 储存.pkl
    with open(os.path.join(process_data_path, "aid2orgid.pkl"), 'wb') as f:
        pickle.dump(aid2orgid, f)

if __name__ == "__main__":
    # extract_aid2orgid(mag_papers_meta, file_name)
    pass
