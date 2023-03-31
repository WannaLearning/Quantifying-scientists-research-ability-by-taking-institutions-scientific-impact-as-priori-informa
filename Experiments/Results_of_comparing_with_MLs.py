#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:42:55 2022

@author: aixuexi
"""
import os
import pickle
import random
import numpy as np
import prettytable as pt

import tensorflow as tf
import autograd.numpy.random as npr

from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from MyQPModel_Results.utils_predict import *


# 我们选择1990-2010年学者进行实证分析
beginyear = 1990

# beforeyear前的数据被用来构建训练集
# afteryear当年的数据被用作测试时刻

def create_inputoutput_pair(targeted_aid, train_aids, beforeyear, afteryear):
    ''' 构建机器学习模型和深度学习模型需要的输入输出数据 '''
    X_train_h   = list()
    Y_train_h   = list()
    X_train_tcc = list()
    Y_train_tcc = list()
    for aid in train_aids:
        # 输入是时间序列数据 (beginyear, beforeyear)
        input_h   = list()
        input_tcc = list()
        for before_year_i in range(beginyear, beforeyear + 1):
            real_cclist_before, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], before_year_i)
            if len(real_cclist_before) == 0:
                input_h.append(0)
                input_tcc.append(0)
            else:
                input_h.append(calculate_h_index(real_cclist_before))
                input_tcc.append(sum(real_cclist_before))
        # 输出是时间点数据 (afteryear)
        real_cclist_after, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], afteryear)
        output_h  = calculate_h_index(real_cclist_after)
        output_tcc= sum(real_cclist_after)
        #
        X_train_h.append(input_h)
        X_train_tcc.append(input_tcc)
        Y_train_h.append(output_h)
        Y_train_tcc.append(output_tcc)
        
    X_train_h   = np.array(X_train_h)
    Y_train_h   = np.array(Y_train_h)
    X_train_tcc = np.array(X_train_tcc)
    Y_train_tcc = np.array(Y_train_tcc)
    return (X_train_h, Y_train_h), (X_train_tcc, Y_train_tcc)


def svr_model(X_train_h, Y_train_h,
              X_valid_h, Y_valid_h,
              X_test_h,  Y_test_h,
              takelog=False):
    
    if takelog:
        X_train_h = np.log(np.maximum(X_train_h, 1))
        X_valid_h = np.log(np.maximum(X_valid_h, 1))
        X_test_h  = np.log(np.maximum(X_test_h,  1))
        
        Y_train_h = np.log(np.maximum(Y_train_h, 1))
        Y_valid_h = np.log(np.maximum(Y_valid_h, 1))
        # Y_test_h  = np.log(np.maximum(Y_test_h, 1))
    
    # 训练集上训练
    svr = SVR()
    svr.fit(X_train_h, Y_train_h)
    # 测试集上评价
    Y_test_pred = svr.predict(X_test_h)
    Y_test_real = Y_test_h
    
    if takelog:
        Y_test_pred = np.exp(Y_test_pred)
    # 返回测试集上表现
    cor, r2, rmse, mae = evaluate_real2pred(Y_test_real, Y_test_pred)
    return cor, r2, rmse, mae 
    

def rf_model(X_train_h, Y_train_h,
             X_valid_h, Y_valid_h,
             X_test_h,  Y_test_h,
             takelog=False):
    
    if takelog:
        X_train_h = np.log(np.maximum(X_train_h, 1))
        X_valid_h = np.log(np.maximum(X_valid_h, 1))
        X_test_h  = np.log(np.maximum(X_test_h,  1))
        
        Y_train_h = np.log(np.maximum(Y_train_h, 1))
        Y_valid_h = np.log(np.maximum(Y_valid_h, 1))
        # Y_test_h  = np.log(np.maximum(Y_test_h, 1))
    
    # 训练集上训练
    rf = RandomForestRegressor()
    rf.fit(X_train_h, Y_train_h)
    # 测试集上评价
    Y_test_pred = rf.predict(X_test_h)
    Y_test_real = Y_test_h
    if takelog:
        Y_test_pred = np.exp(Y_test_pred)
    # 返回测试集上表现
    cor, r2, rmse, mae = evaluate_real2pred(Y_test_real, Y_test_pred)
    return cor, r2, rmse, mae
    

def lstm_model(X_train_h, Y_train_h,
               X_valid_h, Y_valid_h,
               X_test_h,  Y_test_h,
               takelog=False):
    
    if takelog:
        X_train_h = np.log(np.maximum(X_train_h, 1))
        X_valid_h = np.log(np.maximum(X_valid_h, 1))
        X_test_h  = np.log(np.maximum(X_test_h, 1))
        
        Y_train_h = np.log(np.maximum(Y_train_h, 1))
        Y_valid_h = np.log(np.maximum(Y_valid_h, 1))
        # Y_test_h  = np.log(np.maximum(Y_test_h, 1))
    
    seq_len = X_train_h.shape[1]
    # LSTM 模型构建
    lstm = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(seq_len, 1)),
                                       tf.keras.layers.LSTM(32),
                                       tf.keras.layers.Dense(1)])
    lstm.summary()
    lstm.compile(tf.keras.optimizers.Adam(1e-3), loss='mse')
    # 模型训练: 训练集训练 + 验证集挑选
    validation_data = (X_valid_h[:, :, np.newaxis], Y_valid_h[:, np.newaxis])
    lstm.fit(X_train_h[:, :, np.newaxis], Y_train_h[:, np.newaxis], epochs=20, validation_data=validation_data)
    
    # 测试集上评价
    Y_test_pred = lstm.predict(X_test_h[:, :, np.newaxis]).ravel()
    Y_test_real = Y_test_h
    
    if takelog:
        Y_test_pred = np.exp(Y_test_pred)
    # 返回测试集上表现
    cor, r2, rmse, mae = evaluate_real2pred(Y_test_real, Y_test_pred)
    return cor, r2, rmse, mae 


def rnn_model(X_train_h, Y_train_h,
              X_valid_h, Y_valid_h,
              X_test_h,  Y_test_h,
              takelog=False):
    
    if takelog:
        X_train_h = np.log(np.maximum(X_train_h, 1))
        X_valid_h = np.log(np.maximum(X_valid_h, 1))
        X_test_h  = np.log(np.maximum(X_test_h, 1))
        
        Y_train_h = np.log(np.maximum(Y_train_h, 1))
        Y_valid_h = np.log(np.maximum(Y_valid_h, 1))
        # Y_test_h  = np.log(np.maximum(Y_test_h, 1))
    
    seq_len = X_train_h.shape[1]
    # LSTM 模型构建
    gru = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(seq_len, 1)),
                                      tf.keras.layers.GRU(32),
                                      tf.keras.layers.Dense(1)])
    gru.summary()
    gru.compile(tf.keras.optimizers.Adam(1e-3), loss='mse')
    # 模型训练: 训练集训练 + 验证集挑选
    validation_data = (X_valid_h[:, :, np.newaxis], Y_valid_h[:, np.newaxis])
    gru.fit(X_train_h[:, :, np.newaxis], Y_train_h[:, np.newaxis], epochs=20, validation_data=validation_data)
    
    # 测试集上评价
    Y_test_pred = gru.predict(X_test_h[:, :, np.newaxis]).ravel()
    Y_test_real = Y_test_h
    
    if takelog:
        Y_test_pred = np.exp(Y_test_pred)
    # 返回测试集上表现
    cor, r2, rmse, mae = evaluate_real2pred(Y_test_real, Y_test_pred)
    return cor, r2, rmse, mae 


def our_model(test_aids, model_name, file_name, beforeyear, afteryear):
    # 读取我们的Q值
    model_name = model_name.lower()
    if model_name == "org":
        print("IQ Model")
        ResultsPath  = "./Results/Results_org"
        results_OUR  = read_file(os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
        targeted_aid = read_file(os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))
        [mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR = results_OUR
    if model_name == "cou":
        print("IQ-2 Model")
        ResultsPath  = "./Results/Results_org_country"
        results_OUR  = read_file(os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
        targeted_aid = read_file(os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))
        [mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR, cou2Q_OUR = results_OUR
    if model_name == "coo":
        print("IQ-3 Model")
        ResultsPath  = "./Results/Results_coop_org_country"
        results_OUR  = read_file(os.path.join(ResultsPath, "OUR_{}_{}.pkl".format(file_name, beforeyear)))
        targeted_aid = read_file(os.path.join(ResultsPath, "empirical_data({}).pkl".format(file_name)))
        [mu_P_OUR, log_sig_P_OUR], aid2Q_OUR, orgid2Q_OUR, cou2Q_OUR = results_OUR
        
    # 评价预测性
    rs = npr.RandomState()
    sampling_times   = 100
    h_real_pred      = list()
    tcc_real_pred    = list()
    ccstar_real_pred = list()
    
    for aid in test_aids:
        real_cclist_before, _ = sort_aid_cc(targeted_aid[aid]['x_obs'], beforeyear)
        real_cclist_after, _  = sort_aid_cc(targeted_aid[aid]['x_obs'], afteryear)
  
        # 真实结果 (afteryear时刻)
        after_nop    = len(real_cclist_after)
        h_index_real = calculate_h_index(real_cclist_after)
        tcc_real     = sum(real_cclist_after)
        ccstar_real  = max(real_cclist_after)
        
        # 预测结果
        Q = aid2Q_OUR[aid][0]
        samples_avg =  list()
        for i in range(sampling_times):
            samples = rs.randn(after_nop, 1) * np.exp(log_sig_P_OUR) + mu_P_OUR
            samples = Q + samples      # log, 加法效应
            samples = np.exp(samples)
            samples = samples.squeeze()
            samples_avg.append(samples)
        samples_avg = np.array(samples_avg)
        samples_avg = np.mean(samples_avg, axis=0)
        if after_nop == 1:
            samples_avg = np.array([samples_avg])
        
        # 2022-10-1 部分预测: before year之前的是训练集, 不预测
        before_nop = len(real_cclist_before)
        pred_nop   = after_nop - before_nop
        if pred_nop > 0:
            samples_avg_pred = np.concatenate([real_cclist_before, samples_avg[-pred_nop:]])
        else:
            samples_avg_pred = real_cclist_before
  
        h_index_pred = calculate_h_index(samples_avg_pred)
        tcc_pred     = sum(samples_avg_pred)                      
        ccstar_pred  = max(samples_avg_pred)                  
        
        # 添加真实结果和预测结果
        h_real_pred.append([h_index_real, h_index_pred])
        tcc_real_pred.append([tcc_real, tcc_pred])
        ccstar_real_pred.append([ccstar_real, ccstar_pred])
    
    h_real_pred   = np.array(h_real_pred)
    tcc_real_pred = np.array(tcc_real_pred)

    cor_h,   r2_h,   rmse_h,   mae_h,  = evaluate_real2pred(h_real_pred[:, 0],   h_real_pred[:, 1])
    cor_tcc, r2_tcc, rmse_tcc, mae_tcc = evaluate_real2pred(tcc_real_pred[:, 0], tcc_real_pred[:, 1])
    return (cor_h, r2_h, rmse_h, mae_h), (cor_tcc, r2_tcc, rmse_tcc, mae_tcc)


#%%
# 与机器习模型进行预测能力对比: SVR, LSTM
def main():
        
    # 读取实证数据
    file_name    = 'physics'# computer science # physics
    save_path    = "/mnt/disk2/EmpiricalData/StatisticalData_{}".format(file_name)
    targeted_aid = read_file(os.path.join(save_path, "empirical_data.pkl"))

    beforeyear = 2000  # 被利用于获取Q的年份 [1990: 2000]
    afteryear  = 2009  # 被利用于预测Q的年份 
    
    results  = dict()
    # 存放h指数预测 - SVR, LSTM, 我们的模型运行*次
    svr_h    = list()
    rf_h     = list()
    rnn_h    = list()
    lstm_h   = list()
    org_h    = list()
    cou_h    = list()
    coo_h    = list()
    # 存放tcc预测   - SVR, LSTM, 我们的模型运行*次
    svr_tcc  = list()
    rf_tcc   = list()
    rnn_tcc  = list()
    lstm_tcc = list()
    org_tcc  = list()
    cou_tcc  = list()
    coo_tcc  = list()
    # 重复*次实验
    times = 10
    for t in range(times):
        # 8:1:1 划分训练集, 验证集, 测试集
        train_size = int(0.8 * len(targeted_aid))
        valid_size = int(0.1 * len(targeted_aid))
        test_size  = len(targeted_aid) - train_size - valid_size
        # 洗牌
        aids       = list(targeted_aid.keys())
        random.shuffle(aids)  
        train_aids = aids[:train_size]
        valid_aids = aids[train_size: train_size + valid_size]
        test_aids  = aids[train_size + valid_size: train_size + valid_size + test_size]
        
        # 生成时间序列数据 
        (X_train_h, Y_train_h), (X_train_tcc, Y_train_tcc) = create_inputoutput_pair(targeted_aid, train_aids, beforeyear, afteryear)
        (X_valid_h, Y_valid_h), (X_valid_tcc, Y_valid_tcc) = create_inputoutput_pair(targeted_aid, valid_aids, beforeyear, afteryear)
        (X_test_h,  Y_test_h),  (X_test_tcc,  Y_test_tcc)  = create_inputoutput_pair(targeted_aid, test_aids,  beforeyear, afteryear)
        
        # 预测任务1: h-index 预测
        cor_svr,  r2_svr,  rmse_svr,  mae_svr   = svr_model(X_train_h,  Y_train_h, X_valid_h, Y_valid_h, X_test_h, Y_test_h)
        cor_rf,   r2_rf,   rmse_rf,   mae_rf    = rf_model(X_train_h,   Y_train_h, X_valid_h, Y_valid_h, X_test_h, Y_test_h)
        cor_rnn,  r2_rnn,  rmse_rnn,  mae_rnn   = rnn_model(X_train_h, Y_train_h, X_valid_h, Y_valid_h, X_test_h, Y_test_h)
        cor_lstm, r2_lstm, rmse_lstm, mae_lstm  = lstm_model(X_train_h, Y_train_h, X_valid_h, Y_valid_h, X_test_h, Y_test_h)
        (cor_org, r2_org, rmse_org, mae_org), _ = our_model(test_aids, 'ORG', file_name, beforeyear, afteryear)
        (cor_cou, r2_cou, rmse_cou, mae_cou), _ = our_model(test_aids, 'COU', file_name, beforeyear, afteryear)
        (cor_coo, r2_coo, rmse_coo, mae_coo), _ = our_model(test_aids, "COO", file_name, beforeyear, afteryear)
        
        svr_h.append([cor_svr, r2_svr, rmse_svr, mae_svr])
        rf_h.append([cor_rf,   r2_rf,  rmse_rf,  mae_rf])
        rnn_h.append([cor_rnn,  r2_rnn, rmse_rnn, mae_rnn])
        lstm_h.append([cor_lstm, r2_lstm, rmse_lstm, mae_lstm])
        org_h.append([cor_org, r2_org, rmse_org, mae_org])
        cou_h.append([cor_cou, r2_cou, rmse_cou, mae_cou])
        coo_h.append([cor_coo, r2_coo, rmse_coo, mae_coo])
        
        # 预测任务2: tcc 预测
        cor_svr,  r2_svr,  rmse_svr,  mae_svr   = svr_model(X_train_tcc,  Y_train_tcc, X_valid_tcc, Y_valid_tcc, X_test_tcc, Y_test_tcc, True)
        cor_rf,   r2_rf,   rmse_rf,   mae_rf    = rf_model(X_train_tcc,  Y_train_tcc, X_valid_tcc, Y_valid_tcc, X_test_tcc, Y_test_tcc, True)
        cor_rnn,  r2_rnn,  rmse_rnn,  mae_rnn   = rnn_model(X_train_tcc, Y_train_tcc, X_valid_tcc, Y_valid_tcc, X_test_tcc, Y_test_tcc, True)
        cor_lstm, r2_lstm, rmse_lstm, mae_lstm  = lstm_model(X_train_tcc, Y_train_tcc, X_valid_tcc, Y_valid_tcc, X_test_tcc, Y_test_tcc, True)
        _, (cor_org, r2_org, rmse_org, mae_org) = our_model(test_aids, 'ORG', file_name, beforeyear, afteryear)
        _, (cor_cou, r2_cou, rmse_cou, mae_cou) = our_model(test_aids, 'COU', file_name, beforeyear, afteryear)
        _, (cor_coo, r2_coo, rmse_coo, mae_coo) = our_model(test_aids, "COO", file_name, beforeyear, afteryear)
        
        svr_tcc.append([cor_svr, r2_svr, rmse_svr, mae_svr])
        rf_tcc.append([cor_rf,   r2_rf,   rmse_rf,   mae_rf])
        rnn_tcc.append([cor_rnn,  r2_rnn,  rmse_rnn,  mae_rnn])
        lstm_tcc.append([cor_lstm, r2_lstm, rmse_lstm, mae_lstm])
        org_tcc.append([cor_org, r2_org, rmse_org, mae_org])
        cou_tcc.append([cor_cou, r2_cou, rmse_cou, mae_cou])
        coo_tcc.append([cor_coo, r2_coo, rmse_coo, mae_coo])
        
    # h指数预测结果
    svr_h    = np.array(svr_h)
    rf_h     = np.array(rf_h)
    rnn_h    = np.array(rnn_h)
    lstm_h   = np.array(lstm_h)
    org_h    = np.array(org_h)
    cou_h    = np.array(cou_h)
    coo_h    = np.array(coo_h)
    # tcc预测结果
    svr_tcc  = np.array(svr_tcc)
    rf_tcc   = np.array(rf_tcc)
    rnn_tcc = np.array(rnn_tcc)
    lstm_tcc = np.array(lstm_tcc)
    org_tcc  = np.array(org_tcc)
    cou_tcc  = np.array(cou_tcc)
    coo_tcc  = np.array(coo_tcc)
    
    # 字典格式存储
    results["h"]   = (svr_h,   rf_h,   rnn_h,   lstm_h,   org_h,   cou_h,   coo_h)
    results["tcc"] = (svr_tcc, rf_tcc, rnn_tcc, lstm_tcc, org_tcc, cou_tcc, coo_tcc)
    save_file(results, "./Results/Results_compared_with_ml/Results_{}_{}.pkl".format(file_name, afteryear))

    # 读取
    results = read_file("./Results/Results_compared_with_ml/Results_{}_{}.pkl".format(file_name, afteryear))
    (svr_h,   rf_h,   rnn_h,   lstm_h,   org_h,   cou_h,   coo_h)   = results["h"]
    (svr_tcc, rf_tcc, rnn_tcc, lstm_tcc, org_tcc, cou_tcc, coo_tcc) = results["tcc"]

    # the h-index prediction
    svr_h       = np.maximum(svr_h,  0)  # 注意取均值不要有负数
    rf_h        = np.maximum(rf_h,  0)   # 注意取均值不要有负数
    rnn_h       = np.maximum(rnn_h,  0)  # 注意取均值不要有负数
    lstm_h      = np.maximum(lstm_h, 0)
    org_h       = np.maximum(org_h,  0)
    cou_h       = np.maximum(cou_h,  0)
    coo_h       = np.maximum(coo_h,  0)
    svr_h_mean  = np.mean(svr_h, axis=0)
    rf_h_mean   = np.mean(rf_h, axis=0)
    rnn_h_mean  = np.mean(rnn_h, axis=0)
    lstm_h_mean = np.mean(lstm_h, axis=0)
    org_h_mean  = np.mean(org_h, axis=0)
    cou_h_mean  = np.mean(cou_h, axis=0)
    coo_h_mean  = np.mean(coo_h, axis=0)
    tb = pt.PrettyTable()
    tb.field_names = ["模型", "Pearsonr", "R2", "RMSE", "MAE"]
    tb.add_row(["RF", "{:.4f}".format(rf_h_mean[0]),     "{:.4f}".format(rf_h_mean[1]),   "{:.4f}".format(rf_h_mean[2]),   "{:.4f}".format(rf_h_mean[3])])
    tb.add_row(["SVR", "{:.4f}".format(svr_h_mean[0]),   "{:.4f}".format(svr_h_mean[1]),  "{:.4f}".format(svr_h_mean[2]),  "{:.4f}".format(svr_h_mean[3])])
    tb.add_row(["GRU", "{:.4f}".format(rnn_h_mean[0]),   "{:.4f}".format(rnn_h_mean[1]),  "{:.4f}".format(rnn_h_mean[2]),  "{:.4f}".format(rnn_h_mean[3])])
    tb.add_row(["LSTM", "{:.4f}".format(lstm_h_mean[0]), "{:.4f}".format(lstm_h_mean[1]), "{:.4f}".format(lstm_h_mean[2]), "{:.4f}".format(lstm_h_mean[3])])
    tb.add_row(["ORG", "{:.4f}".format(org_h_mean[0]),   "{:.4f}".format(org_h_mean[1]),  "{:.4f}".format(org_h_mean[2]),  "{:.4f}".format(org_h_mean[3])])
    tb.add_row(["COU", "{:.4f}".format(cou_h_mean[0]),   "{:.4f}".format(cou_h_mean[1]),  "{:.4f}".format(cou_h_mean[2]),  "{:.4f}".format(cou_h_mean[3])])
    tb.add_row(["COO", "{:.4f}".format(coo_h_mean[0]),   "{:.4f}".format(coo_h_mean[1]),  "{:.4f}".format(coo_h_mean[2]),  "{:.4f}".format(coo_h_mean[3])])
    print(tb)

   
