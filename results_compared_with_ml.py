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
import tensorflow as tf
from sklearn.svm import SVR
from MyQPModel.evaluate_results import *


def create_input_set(targeted_aid, train_aids, afteryear):
    # 构建训练集 ---> 表格数据
    X_train_h = list()
    Y_train_h = list()
    X_train_tcc = list()
    Y_train_tcc = list()
    for aid in train_aids:
        # 输入
        input_h = list()
        input_tcc = list()
        for before_year_i in range(1990, beforeyear + 1):
            real_cc_list_before = sort_aid_cc(targeted_aid[aid]['x'], before_year_i)
            if len(real_cc_list_before) == 0:
                input_h.append(0)
                input_tcc.append(0)
            else:
                input_h.append(calculate_h_index(real_cc_list_before))
                input_tcc.append(sum(real_cc_list_before))
        # 输出
        real_cc_list = sort_aid_cc(targeted_aid[aid]['x'], afteryear)
        #
        X_train_h.append(input_h)
        Y_train_h.append(calculate_h_index(real_cc_list))
        X_train_tcc.append(input_tcc)
        Y_train_tcc.append(sum(real_cc_list))
    X_train_h = np.array(X_train_h)
    Y_train_h = np.array(Y_train_h)
    X_train_tcc = np.array(X_train_tcc)
    Y_train_tcc = np.array(Y_train_tcc)
    return (X_train_h, Y_train_h), (X_train_tcc, Y_train_tcc)


def svr_model(X_train_h, Y_train_h,
              X_valid_h, Y_valid_h,
              X_test_h,  Y_test_h,
              log=False):
    Y_test_real = Y_test_h
    if log:
        X_train_h = np.log(np.maximum(X_train_h, 1))
        Y_train_h = np.log(np.maximum(Y_train_h, 1))
        
        X_valid_h = np.log(np.maximum(X_valid_h, 1))
        Y_valid_h = np.log(np.maximum(Y_valid_h, 1))
        
        X_test_h  = np.log(np.maximum(X_test_h, 1))
        # Y_test_h  = np.log(np.maximum(Y_test_h, 1))
        
    # SVR 模型
    svr = SVR()
    svr.fit(X_train_h, Y_train_h)
    Y_test_pred = svr.predict(X_test_h)
    if log:
        Y_test_pred = np.exp(Y_test_pred)
    cor, rmse, mae, r2 = evaluate_real2pred(Y_test_real, Y_test_pred)
    # 返回测试集上表现
    return cor, rmse, mae, r2
    
    
def lstm_model(X_train_h, Y_train_h,
               X_valid_h, Y_valid_h,
               X_test_h,  Y_test_h,
               log=False):
    Y_test_real = Y_test_h
    if log:
        X_train_h = np.log(np.maximum(X_train_h, 1))
        Y_train_h = np.log(np.maximum(Y_train_h, 1))
        
        X_valid_h = np.log(np.maximum(X_valid_h, 1))
        Y_valid_h = np.log(np.maximum(Y_valid_h, 1))
        
        X_test_h  = np.log(np.maximum(X_test_h, 1))
        # Y_test_h  = np.log(np.maximum(Y_test_h, 1))
    
    # LSTM 模型
    lstm = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(8, 1)),
                                       tf.keras.layers.Dense(8), 
                                       tf.keras.layers.LSTM(32),
                                       tf.keras.layers.Dense(1)])
    lstm.summary()
    lstm.compile(tf.keras.optimizers.Adam(0.001), loss='mse')
    # 训练集上训练 + 验证集上挑选
    validation_data = (X_valid_h[:, :, np.newaxis], Y_valid_h[:, np.newaxis])
    lstm.fit(X_train_h[:, :, np.newaxis], Y_train_h[:, np.newaxis], epochs=20, validation_data=validation_data)
    # 测试集上评价
    Y_test_pred = lstm.predict(X_test_h[:, :, np.newaxis]).ravel()
    if log:
        Y_test_pred = np.exp(Y_test_pred)
    cor, rmse, mae, r2 = evaluate_real2pred(Y_test_real, Y_test_pred)
    # 返回测试集上表现
    return cor, rmse, mae, r2


def IQ_model(test_aids):
    # Institution Q-model 
    # 读取我们的Q值 --- 注意 tmp文件夹
    with open("./tmp/results_org_{}.pkl".format(beforeyear), 'rb') as f:
        results_org = pickle.load(f)
    [mu_P, log_sig_P], aid2Q, orgid2Q = results_org

    # 评价预测性
    sampling_times=100
    rs = npr.RandomState()
    h_real_pred = list()
    tcc_real_pred = list()
    ccstar_real_pred = list()
    for aid in tqdm(test_aids):
        real_cc_list_before = sort_aid_cc(targeted_aid[aid]['x'], beforeyear)
        real_cc_list = sort_aid_cc(targeted_aid[aid]['x'], afteryear)
  
        nop = len(real_cc_list)
        # 真实结果
        h_index_real = calculate_h_index(real_cc_list)
        tcc_real = sum(real_cc_list)
        ccstar_real = max(real_cc_list)
        
        # 预测结果
        Q = aid2Q[aid][0]
        samples_avg =  list()
        for i in range(sampling_times):
            samples = rs.randn(nop, 1) * np.exp(log_sig_P) + mu_P
            samples = Q + samples      # log未原, 加法效应
            samples = np.exp(samples)
            samples = samples.squeeze()
            samples_avg.append(samples)
        samples_avg = np.array(samples_avg)
        samples_avg = np.mean(samples_avg, axis=0)
        
        if nop == 1:
            samples_avg = [samples_avg]
        
        # 2022-10-1 部分预测: before year之前的是训练集, 不预测
        real_num = len(real_cc_list_before)
        pred_num = len(real_cc_list) - real_num
        if pred_num > 0:
            samples_avg_pred = np.concatenate([real_cc_list_before, samples_avg[-pred_num:]])
        else:
            samples_avg_pred = real_cc_list_before
  
        h_index_pred = calculate_h_index(samples_avg_pred)    # 王大顺论文图C
        tcc_pred = sum(samples_avg_pred)                      # 王大顺论文图D
        ccstar_pred = max(samples_avg_pred)                   # 王大顺论文图B
        
        # 添加真实结果和预测结果
        h_real_pred.append([h_index_real, h_index_pred])
        tcc_real_pred.append([tcc_real, tcc_pred])
        ccstar_real_pred.append([ccstar_real, ccstar_pred])
    #
    h_real_pred, tcc_real_pred = np.array(h_real_pred), np.array(tcc_real_pred)

    cor_h, rmse_h, mae_h, r2_h = evaluate_real2pred(h_real_pred[:, 0], h_real_pred[:, 1])
    cor_tcc, rmse_tcc, mae_tcc, r2_tcc = evaluate_real2pred(tcc_real_pred[:, 0], tcc_real_pred[:, 1])
    return (cor_h, rmse_h, mae_h, r2_h), (cor_tcc, rmse_tcc, mae_tcc, r2_tcc)


#%%
# 与机器习模型进行预测能力对比: SVR, LSTM
def main():
    # 读取实证数据
    with open(os.path.join(process_data_path, "aid_empirical.pkl"), 'rb') as f:
        targeted_aid = pickle.load(f)
    # Y1 = bearfore, Y2 = afteryear
    beforeyear = 1997 # 被利用于获取Q的年份
    afteryear  = 2008 # 被利用于预测Q的年份
    
    # 存放h指数预测 - SVR, LSTM, IQ 运行多次
    svr_performance_h = list()
    lstm_performance_h = list()
    q_performace_h = list()
    # 存放tcc预测 - SVR, LSTM, IQ 运行多次
    svr_performance_tcc = list()
    lstm_performance_tcc = list()
    q_performace_tcc = list()
    
    # 重复10次实验
    for t in range(10):
        # 8:1:1 划分训练集和测试集
        train_size = int(0.8 * len(targeted_aid))
        valid_size = int(0.1 * len(targeted_aid))
        test_size  = len(targeted_aid) - train_size - valid_size
        # 洗牌
        aids = list(targeted_aid.keys())
        random.shuffle(aids)  
        train_aids = aids[:train_size]
        valid_aids = aids[train_size: train_size+valid_size]
        test_aids  = aids[train_size+valid_size: train_size+valid_size+test_size]
        # 生成时间序列数据 
        (X_train_h, Y_train_h), (X_train_tcc, Y_train_tcc) = create_input_set(targeted_aid, train_aids, afteryear)
        (X_valid_h, Y_valid_h), (X_valid_tcc, Y_valid_tcc) = create_input_set(targeted_aid, valid_aids, afteryear)
        (X_test_h,  Y_test_h),  (X_test_tcc,  Y_test_tcc)  = create_input_set(targeted_aid, test_aids,  afteryear)
        
        # h-index 预测
        cor_svr, rmse_svr, mae_svr, r2_svr = svr_model(X_train_h, Y_train_h, X_valid_h, Y_valid_h, X_test_h, Y_test_h)
        cor_lstm, rmse_lstm, mae_lstm, r2_lstm = lstm_model(X_train_h, Y_train_h, X_valid_h, Y_valid_h, X_test_h, Y_test_h)
        (cor_Q, rmse_Q, mae_Q, r2_Q), _ = IQ_model(test_aids)
        
        svr_performance_h.append([cor_svr, r2_svr, rmse_svr, mae_svr])
        lstm_performance_h.append([cor_lstm, r2_lstm, rmse_lstm, mae_lstm])
        q_performace_h.append([cor_Q, r2_Q, rmse_Q, mae_Q])
        
        # tcc 预测
        cor_svr, rmse_svr, mae_svr, r2_svr = svr_model(X_train_tcc, Y_train_tcc, X_valid_tcc, Y_valid_tcc, X_test_tcc, Y_test_tcc, True)
        cor_lstm, rmse_lstm, mae_lstm, r2_lstm = lstm_model(X_train_tcc, Y_train_tcc, X_valid_tcc, Y_valid_tcc, X_test_tcc, Y_test_tcc, True)
        _, (cor_Q, rmse_Q, mae_Q, r2_Q) = IQ_model(test_aids)
        
        svr_performance_tcc.append([cor_svr, r2_svr, rmse_svr, mae_svr])
        lstm_performance_tcc.append([cor_lstm, r2_lstm, rmse_lstm, mae_lstm])
        q_performace_tcc.append([cor_Q, r2_Q, rmse_Q, mae_Q])
    
    # 存放h指数预测
    svr_performance_h = np.array(svr_performance_h)
    lstm_performance_h = np.array(lstm_performance_h)
    q_performace_h = np.array(q_performace_h)
    # 存放tcc预测
    svr_performance_tcc = np.array(svr_performance_tcc)
    lstm_performance_tcc = np.array(lstm_performance_tcc)
    q_performace_tcc = np.array(q_performace_tcc)
    # 字典格式存储
    results = dict()
    results["h"] = dict()
    results["tcc"] = dict()
    results["h"]["svr"] = svr_performance_h
    results["h"]["lstm"] = lstm_performance_h
    results["h"]["q"] = q_performace_h
    results["tcc"]["svr"] = svr_performance_tcc
    results["tcc"]["lstm"] = lstm_performance_tcc
    results["tcc"]["q"] = q_performace_tcc
    with open("./tmp/compared_with_mls_{}.pkl".format(afteryear), 'wb') as f:
        pickle.dump(results, f)
    
    # 注意取均值不要有负数
    # the h-index prediction
    print("svr(h)", np.mean(svr_performance_h, axis=0))
    print("lstm(h)", np.mean(lstm_performance_h, axis=0))
    print("IQ(h)", np.mean(q_performace_h, axis=0))
    # total citation count prediction
    print("svr(tcc)", np.mean(svr_performance_tcc, axis=0))
    print("lstm(tcc)", np.mean(lstm_performance_tcc, axis=0))
    print("IQ(tcc)", np.mean(q_performace_tcc, axis=0))
