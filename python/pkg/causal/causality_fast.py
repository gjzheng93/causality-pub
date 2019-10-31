# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 16:13:18 2016

@author: gjz5038
"""

# causality.py

import statsmodels
import numpy as np

import time
import datetime
import timeit

from statsmodels.tsa.stattools import adfuller as adfuller

from . import granger_std
from . import parts
from . import output
from . import prune


import sys


class cnts_prune:
    
    def __init__(self, cnt_promising, cnt_promising_not, cnt_not_sure, cnt_initial):
        self.cnt_promising=  cnt_promising
        self.cnt_promising_not = cnt_promising_not
        self.cnt_not_sure = cnt_not_sure
        self.cnt_initial = cnt_initial
        
class time_prune:
    
    def __init__(self, time_promising, time_promising_not, time_not_sure, time_initial):
        self.time_promising=  time_promising
        self.time_promising_not = time_promising_not
        self.time_not_sure = time_not_sure
        self.time_initial = time_initial
    
                  
        
def granger(array_X, array_Y, X_name, Y_name, para_set, path_to_output):
        
    step = para_set.step
    lag = para_set.lag
    test_mode = para_set.test_mode
    significant_thres = para_set.significant_thres
    min_segment_len = para_set.min_segment_len
    max_segment_len = para_set.max_segment_len     
    
    n_sample = len(array_X)    

    print('sample size: ' + str(n_sample))   
    
    # ===================================================  initialization =================================================

    
    cnt_prune_YX = cnts_prune(0, 0, 0, 0)
    cnt_prune_XY = cnts_prune(0, 0, 0, 0)
    
    time_prune_XY = time_prune(0, 0, 0, 0)
    time_prune_YX = time_prune(0, 0, 0, 0)
    
    print(X_name)

    time1 = timeit.default_timer()
    
    time_granger = 0
    time_adf = 0
    
    array_YX = np.concatenate((array_Y, array_X), axis = 1)
    array_XY = np.concatenate((array_X, array_Y), axis = 1)
    
    n_step = int(n_sample/step-1)
    list_segment_split = [step * i for i in range(n_step)]
    list_segment_split.append(n_sample-1)
        
    
    start = 0
    end = 0
    
    total_cnt_segment_YX = 0
    total_cnt_segment_XY = 0
    total_cnt_segment_adf = 0
    total_cnt_segment_cal_adf = 0
    total_cnt_segment_examine_adf_Y = 0
    
    array_results_YX = np.full((n_step+1, n_step+1), -2, dtype = float)
    array_results_XY = np.full((n_step+1, n_step+1), -2, dtype = float)
    
    array_adf_results_X = np.full((n_step+1, n_step+1), -2, dtype = float)
    array_adf_results_Y = np.full((n_step+1, n_step+1), -2, dtype = float)
    
    # get lagged data
    
    dta_YX, dtaown_YX, dtajoint_YX = parts.get_lagged_data(array_YX, lag, addconst = True, verbose = False)
    dta_XY, dtaown_XY, dtajoint_XY = parts.get_lagged_data(array_XY, lag, addconst = True, verbose = False)
    
    # make the data to the original length
    
#     dta_YX = np.concatenate((np.zeros((lag, np.shape(dta_YX)[1])), dta_YX), axis = 0)
#     dtaown_YX = np.concatenate((np.zeros((lag, np.shape(dtaown_YX)[1])), dtaown_YX), axis = 0)
#     dtajoint_YX = np.concatenate((np.zeros((lag, np.shape(dtajoint_YX)[1])), dtajoint_YX), axis = 0)
#     dta_XY = np.concatenate((np.zeros((lag, np.shape(dta_XY)[1])), dta_XY), axis = 0)
#     dtaown_XY = np.concatenate((np.zeros((lag, np.shape(dtaown_XY)[1])), dtaown_XY), axis = 0)
#     dtajoint_XY = np.concatenate((np.zeros((lag, np.shape(dtajoint_XY)[1])), dtajoint_XY), axis = 0)
#     
#     dtaown_YX[:lag,-1] = 1
#     dtajoint_YX[:lag,-1] = 1
#     dtaown_XY[:lag,-1] = 1
#     dtajoint_XY[:lag,-1] = 1
    
    # maintain a non_zero flag to update degree of freedom
    
    if_non_zero_columns_YX = np.zeros(np.shape(dtajoint_YX)[1])
    if_non_zero_columns_XY = np.zeros(np.shape(dtajoint_XY)[1])
    
    
    # begin loop
    

    for i in range(n_step):
        start = list_segment_split[i]
        
        print(str(start)+'/'+str(len(array_YX)))
        
        reset_cnt_YX = -1
        res2down_YX = None
        res2djoint_YX =None
        res2down_ssr_upper_YX = 0
        res2down_ssr_lower_YX = 0
        res2djoint_ssr_upper_YX = 0
        res2djoint_ssr_lower_YX = 0 
        res2djoint_df_resid_YX = 0
        
        reset_cnt_XY = -1 
        res2down_XY = None
        res2djoint_XY = None
        res2down_ssr_upper_XY = 0
        res2down_ssr_lower_XY = 0
        res2djoint_ssr_upper_XY = 0
        res2djoint_ssr_lower_XY = 0 
        res2djoint_df_resid_XY = 0
        
        
        for j in range(i+1, n_step+1):
            
            end = list_segment_split[j]
            
            dta_start = start
            dta_end = end - lag
                
            
            if (len(array_YX[start:end, :]) < min_segment_len or len(array_YX[start:end, :]) > max_segment_len):
                
                if_non_zero_columns_YX = np.logical_or(np.sum(dtajoint_YX[dta_end-step:dta_end,:], axis = 0) != 0, if_non_zero_columns_YX)
                if_non_zero_columns_XY = np.logical_or(np.sum(dtajoint_XY[dta_end-step:dta_end,:], axis = 0) != 0, if_non_zero_columns_XY)
                
                continue
            
            
            # =======================================================  F test =======================================================
            
            time3 = timeit.default_timer()
            

            if test_mode == 'standard':                  
                
                p_value_YX, res2down_YX, res2djoint_YX = granger_std.grangercausalitytests(dta_YX[dta_start:dta_end], dtaown_YX[dta_start:dta_end], dtajoint_YX[dta_start:dta_end], lag, addconst = True, verbose = False)  
                if p_value_YX < significant_thres:
                    p_value_XY, res2down_XY, res2djoint_XY = granger_std.grangercausalitytests(dta_XY[dta_start:dta_end], dtaown_XY[dta_start:dta_end], dtajoint_XY[dta_start:dta_end], lag, addconst = True, verbose = False)
                else:
                    p_value_XY = -1 
                
                
            elif test_mode == 'fast_version_1': #only check F_upper

                p_value_YX, res2down_YX, res2djoint_YX, res2down_ssr_upper_YX, res2djoint_ssr_lower_YX, res2djoint_df_resid_YX, reset_cnt_YX, if_non_zero_columns_YX = prune.grangercausalitytests_check_F_upper(dta_YX[dta_start:dta_end], dtaown_YX[dta_start:dta_end], dtajoint_YX[dta_start:dta_end], lag, res2down_YX, res2djoint_YX, res2down_ssr_upper_YX, res2djoint_ssr_lower_YX, res2djoint_df_resid_YX, if_non_zero_columns_YX, significant_thres, step, reset_cnt_YX, addconst=True, verbose= False)
                if p_value_YX < significant_thres and p_value_YX >= 0:
                    p_value_XY, res2down_XY, res2djoint_XY = granger_std.grangercausalitytests(dta_XY[dta_start:dta_end], dtaown_XY[dta_start:dta_end], dtajoint_XY[dta_start:dta_end], lag, addconst = True, verbose = False)
                else:
                    p_value_XY = -1                 
            
            
            elif test_mode == 'fast_version_2': # check F_upper then check F_lower
                
                total_cnt_segment_YX += 1
                
                p_value_YX, res2down_YX, res2djoint_YX, res2down_ssr_upper_YX, res2down_ssr_lower_YX, res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX, res2djoint_df_resid_YX, reset_cnt_YX, cnt_prune_YX, time_prune_YX, if_non_zero_columns_YX \
                = prune.grangercausalitytests_check_F_upper_lower(dta_YX[dta_start:dta_end], dtaown_YX[dta_start:dta_end], dtajoint_YX[dta_start:dta_end], lag, res2down_YX, res2djoint_YX, res2down_ssr_upper_YX, res2down_ssr_lower_YX, res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX, res2djoint_df_resid_YX, if_non_zero_columns_YX, significant_thres, step, reset_cnt_YX, cnt_prune_YX, time_prune_YX, addconst=True, verbose=False)
                    
                if p_value_YX < significant_thres and p_value_YX >= 0:
                    total_cnt_segment_XY += 1
                    p_value_XY, res2down_XY, res2djoint_XY = granger_std.grangercausalitytests(dta_XY[dta_start:dta_end], dtaown_XY[dta_start:dta_end], dtajoint_XY[dta_start:dta_end], lag, addconst = True, verbose = False)
                else:
                    p_value_XY = -1 
            
                
            elif test_mode == 'fast_version_3': # check YX then check XY
                
                total_cnt_segment_YX += 1
            
                p_value_YX, res2down_YX, res2djoint_YX, res2down_ssr_upper_YX, res2down_ssr_lower_YX, res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX, res2djoint_df_resid_YX, reset_cnt_YX, cnt_prune_YX, time_prune_YX, if_non_zero_columns_YX \
                = prune.grangercausalitytests_check_F_upper_lower(dta_YX[dta_start:dta_end], dtaown_YX[dta_start:dta_end], dtajoint_YX[dta_start:dta_end], lag, res2down_YX, res2djoint_YX, res2down_ssr_upper_YX, res2down_ssr_lower_YX, res2djoint_ssr_upper_YX, res2djoint_ssr_lower_YX, res2djoint_df_resid_YX, if_non_zero_columns_YX, significant_thres, step, reset_cnt_YX, cnt_prune_YX, time_prune_YX, addconst=True, verbose=False)
                    
                if p_value_YX < significant_thres and p_value_YX >= 0:
                    total_cnt_segment_XY += 1
                    p_value_XY, res2down_XY, res2djoint_XY, res2down_ssr_upper_XY, res2down_ssr_lower_XY, res2djoint_ssr_upper_XY, res2djoint_ssr_lower_XY, res2djoint_df_resid_XY, reset_cnt_XY, cnt_prune_XY, time_prune_XY, if_non_zero_columns_XY \
                = prune.grangercausalitytests_check_F_upper_lower(dta_XY[dta_start:dta_end], dtaown_XY[dta_start:dta_end], dtajoint_XY[dta_start:dta_end], lag, res2down_XY, res2djoint_XY, res2down_ssr_upper_XY, res2down_ssr_lower_XY, res2djoint_ssr_upper_XY, res2djoint_ssr_lower_XY, res2djoint_df_resid_XY, if_non_zero_columns_XY, significant_thres, step, reset_cnt_XY, cnt_prune_XY, time_prune_XY, addconst=True, verbose=False)
                else:
                    p_value_XY = -1
                    
                    if res2down_XY != None and res2djoint_XY != None:
                        res2down_ssr_upper_XY, res2down_ssr_lower_XY, res2djoint_ssr_upper_XY, res2djoint_ssr_lower_XY, res2djoint_df_resid_XY, if_non_zero_columns_XY = prune.update_bound(dta_XY[dta_start:dta_end], dtaown_XY[dta_start:dta_end], dtajoint_XY[dta_start:dta_end], res2down_XY, res2djoint_XY, res2down_ssr_upper_XY, res2down_ssr_lower_XY, res2djoint_ssr_upper_XY, res2djoint_ssr_lower_XY, res2djoint_df_resid_XY, if_non_zero_columns_XY, lag, step, addconst = True, verbose = False)
                    
                        if res2down_XY.ssr > res2down_ssr_upper_XY or res2djoint_XY.ssr > res2djoint_ssr_upper_XY:
                            print ('error')
                            
                
            array_results_YX[i,j] = p_value_YX  
            array_results_XY[i,j] = p_value_XY   
                
                       
            time4 = timeit.default_timer()
            
            time_granger += (time4 - time3)
                
            # ====================================== stationary test ====================================================
                
            time5 = timeit.default_timer()
            
            if para_set.cal_stationary_separately == 0:
                
                total_cnt_segment_adf += 1
                
                if p_value_YX < significant_thres and p_value_YX >= 0 and p_value_XY > significant_thres:
                    
                    total_cnt_segment_examine_adf_Y += 1
                
                    adfstat_Y, pvalue_Y, usedlag_Y, nobs_Y, critvalues_Y, icbest_Y = adfuller(array_XY[start:end, 1], lag)
                    
                    if pvalue_Y < significant_thres and pvalue_Y >= 0:
                    
                        adfstat_X, pvalue_X, usedlag_X, nobs_X, critvalues_X, icbest_X = adfuller(array_XY[start:end, 0], lag)
                        total_cnt_segment_cal_adf += 1
                        
                    else:
                        
                        pvalue_X = -1
                        
                else:
                    pvalue_Y = -1
                    pvalue_X = -1
                    
            else:
                
                
                total_cnt_segment_examine_adf_Y += 1
                
                adfstat_Y, pvalue_Y, usedlag_Y, nobs_Y, critvalues_Y, icbest_Y = adfuller(array_XY[start:end, 1], lag)
                    
                if pvalue_Y < significant_thres and pvalue_Y >= 0:
                
                    adfstat_X, pvalue_X, usedlag_X, nobs_X, critvalues_X, icbest_X = adfuller(array_XY[start:end, 0], lag)
                    total_cnt_segment_cal_adf += 1
                        
                else:
                    
                    pvalue_X = -1
                    
                    
            array_adf_results_Y[i, j] = pvalue_Y
            array_adf_results_X[i, j] = pvalue_X
                
            time6 = timeit.default_timer()
            
            time_adf += (time6 - time5)
                

    
    time2 = timeit.default_timer()
    
    total_time = time2-time1

    print('total time: ' + str(time2 - time1))

    time_set = [time1, time2, time_granger, time_adf]  
    cnt_set = [total_cnt_segment_YX, cnt_prune_YX, time_prune_YX, total_cnt_segment_XY, cnt_prune_XY, time_prune_XY, total_cnt_segment_adf, total_cnt_segment_cal_adf, total_cnt_segment_examine_adf_Y]
    
    output.output_causal(path_to_output, X_name, Y_name, time_set, cnt_set, array_results_YX, array_results_XY, array_adf_results_X, array_adf_results_Y, list_segment_split, para_set)
        
    return total_time, time_granger, time_adf, total_cnt_segment_YX, cnt_prune_YX, time_prune_YX, total_cnt_segment_XY, cnt_prune_XY, time_prune_XY   
    
   
