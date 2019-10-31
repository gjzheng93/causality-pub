

import numpy as np
from . import parts

import scipy
import timeit



def grangercausalitytests_check_F_upper(dta, dtaown, dtajoint, lag, pre_res2down, pre_res2djoint, pre_res2down_ssr_upper, pre_res2djoint_ssr_lower, pre_res2djoint_df_resid, if_non_zero_columns, significant_thres, step, cnt, addconst=True, verbose=False):
    
    if cnt == -1:
        #initialization
    
        res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
        
        if res2djoint.ssr == 0:
            p_value = 1
        else:
            result = parts.f_test(res2down, res2djoint, lag)
            p_value = result['ssr_ftest'][1] 
        
        res2down_ssr_upper = res2down.ssr
        res2djoint_ssr_lower = res2djoint.ssr
        
        if_non_zero_columns = np.logical_or(np.sum(dtajoint[-step:,:], axis = 0) != 0, if_non_zero_columns)
        
        return p_value, res2down, res2djoint, res2down_ssr_upper, res2djoint_ssr_lower, res2djoint.df_resid, 0, if_non_zero_columns
    
    else:
        
        res2down_fit_new_point_error = np.dot(dtaown[-step:,:], pre_res2down.params)-dta[-step:,0]
        res2down_ssr_upper = np.dot(res2down_fit_new_point_error, res2down_fit_new_point_error)+pre_res2down_ssr_upper
        res2djoint_ssr_lower = pre_res2djoint_ssr_lower
    
        if res2djoint_ssr_lower == 0:
            
            #not sure
            
            res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
                
            if res2djoint.ssr == 0:
                p_value = 1
            else:
                result = parts.f_test(res2down, res2djoint, lag)
                p_value = result['ssr_ftest'][1] 
            
            res2down_ssr_upper = res2down.ssr
            res2djoint_ssr_lower = res2djoint.ssr 
            
            if_non_zero_columns = np.logical_or(np.sum(dtajoint[-step:,:], axis = 0) != 0, if_non_zero_columns)           
            
            return p_value, res2down, res2djoint, res2down_ssr_upper, res2djoint_ssr_lower, res2djoint.df_resid, 0, if_non_zero_columns
            
        else:
            
            if_non_zero_columns = np.logical_or(np.sum(dtajoint[-step:,:], axis = 0) != 0, if_non_zero_columns)
            non_zero_column = np.sum(if_non_zero_columns)
            res2djoint_df_resid = len(dtajoint) - non_zero_column  
            
            F_upper = (res2down_ssr_upper/res2djoint_ssr_lower - 1)*(res2djoint_df_resid)/lag
            p_value_lower = 1-scipy.stats.f.cdf(F_upper, lag, (res2djoint_df_resid))
            
            if p_value_lower < significant_thres:
        
                res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
                
                if res2djoint.ssr == 0:
                    p_value = 1
                    print ('lower bound estimation error')
                else:
                    result = parts.f_test(res2down, res2djoint, lag)
                    p_value = result['ssr_ftest'][1] 
                
                res2down_ssr_upper = res2down.ssr
                res2djoint_ssr_lower = res2djoint.ssr            
                
                return p_value, res2down, res2djoint, res2down_ssr_upper, res2djoint_ssr_lower, res2djoint.df_resid, 0, if_non_zero_columns
                
            else:
                p_value = 1 
                
                return p_value, pre_res2down, pre_res2djoint, res2down_ssr_upper, res2djoint_ssr_lower, res2djoint_df_resid, cnt+1, if_non_zero_columns
            
            
            
def update_bound(dta, dtaown, dtajoint, pre_res2down, pre_res2djoint, pre_res2down_ssr_upper, pre_res2down_ssr_lower, pre_res2djoint_ssr_upper, pre_res2djoint_ssr_lower, pre_res2djoint_df_resid, if_non_zero_columns, lag, step, addconst, verbose):

    res2down_fit_new_point_error = np.dot(dtaown[-step:,:], pre_res2down.params)-dta[-step:,0]
    res2djoint_fit_new_point_error = np.dot(dtajoint[-step:,:], pre_res2djoint.params)-dta[-step:,0]
    
    
    res2down_ssr_upper = np.dot(res2down_fit_new_point_error, res2down_fit_new_point_error)+pre_res2down_ssr_upper
    res2djoint_ssr_lower = pre_res2djoint_ssr_lower
    res2down_ssr_lower = pre_res2down_ssr_lower
    res2djoint_ssr_upper = np.dot(res2djoint_fit_new_point_error, res2djoint_fit_new_point_error)+pre_res2djoint_ssr_upper
    
    if_non_zero_columns = np.logical_or(np.sum(dtajoint[-step:,:], axis = 0) != 0, if_non_zero_columns)
    non_zero_column = np.sum(if_non_zero_columns)
    res2djoint_df_resid = len(dtajoint) - non_zero_column 
 
    return res2down_ssr_upper, res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower, res2djoint_df_resid, if_non_zero_columns



def grangercausalitytests_check_F_upper_lower(dta, dtaown, dtajoint, lag, pre_res2down, pre_res2djoint, pre_res2down_ssr_upper, pre_res2down_ssr_lower, pre_res2djoint_ssr_upper, pre_res2djoint_ssr_lower, pre_res2djoint_df_resid, if_non_zero_columns, significant_thres, step, cnt, cnt_prune, time_prune, addconst=True, verbose=False):

    
    if cnt == -1:
        #initialization
    
        block_time1 = timeit.default_timer()

        res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
        
        if res2djoint.ssr == 0:
            p_value = 1
        else:
            result = parts.f_test(res2down, res2djoint, lag)
            p_value = result['ssr_ftest'][1] 
        
        res2down_ssr_upper = res2down.ssr
        res2down_ssr_lower = res2down.ssr
        res2djoint_ssr_upper = res2djoint.ssr        
        res2djoint_ssr_lower = res2djoint.ssr

        
        if_non_zero_columns = np.logical_or(np.sum(dtajoint[-step:,:], axis = 0) != 0, if_non_zero_columns)
        
        block_time2 = timeit.default_timer()
        cnt_prune.cnt_initial += 1
        time_prune.time_initial += (block_time2-block_time1)
        
        return p_value, res2down, res2djoint, res2down_ssr_upper, res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower, res2djoint.df_resid, 0, cnt_prune, time_prune, if_non_zero_columns
    
    else:
        
        #fit the new data   
        
        #prune promising not
        
        block_time1 = timeit.default_timer()
        
        res2down_fit_new_point_error = np.dot(dtaown[-step:,:], pre_res2down.params)-dta[-step:,0]
        res2djoint_fit_new_point_error = np.dot(dtajoint[-step:,:], pre_res2djoint.params)-dta[-step:,0]
        
        
        res2down_ssr_upper = np.dot(res2down_fit_new_point_error, res2down_fit_new_point_error)+pre_res2down_ssr_upper
        res2djoint_ssr_lower = pre_res2djoint_ssr_lower
        res2down_ssr_lower = pre_res2down_ssr_lower
        res2djoint_ssr_upper = np.dot(res2djoint_fit_new_point_error, res2djoint_fit_new_point_error)+pre_res2djoint_ssr_upper
              
        
#        res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
#        if (res2down_ssr_upper - res2down.ssr) < -0.00001 or (res2djoint_ssr_upper < res2djoint.ssr) < -0.00001:
#            print 'error'
#            
#        if res2djoint.df_resid != res2djoint_df_resid:
#            print 'error'
#        
        # check F_upper
        
        if res2djoint_ssr_lower == 0:
            
            # not sure
            res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
                    
            if res2djoint.ssr == 0:
                p_value = 1
            else:                
                result = parts.f_test(res2down, res2djoint, lag)
                p_value = result['ssr_ftest'][1] 
            
            res2down_ssr_upper = res2down.ssr
            res2down_ssr_lower = res2down.ssr
            res2djoint_ssr_upper = res2djoint.ssr        
            res2djoint_ssr_lower = res2djoint.ssr   
            
            if_non_zero_columns = np.logical_or(np.sum(dtajoint[-step:,:], axis = 0) != 0, if_non_zero_columns) 
            
            cnt_prune.cnt_not_sure += 1
            block_time2 = timeit.default_timer()
            time_prune.time_not_sure += (block_time2-block_time1)
                
            return p_value, res2down, res2djoint, res2down_ssr_upper, res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower, res2djoint.df_resid, 0, cnt_prune, time_prune, if_non_zero_columns
        
            
        else:
            
            if_non_zero_columns = np.logical_or(np.sum(dtajoint[-step:,:], axis = 0) != 0, if_non_zero_columns)
            non_zero_column = np.sum(if_non_zero_columns)
            res2djoint_df_resid = len(dtajoint) - non_zero_column              
            
            F_upper = (res2down_ssr_upper/res2djoint_ssr_lower - 1)*(res2djoint_df_resid)/lag
            p_value_lower = 1-scipy.stats.f.cdf(F_upper, lag, (res2djoint_df_resid))
        
        
            if p_value_lower >= significant_thres: # promising not
                p_value = 1
                cnt_prune.cnt_promising_not += 1
                block_time2 = timeit.default_timer()
                time_prune.time_promising_not += (block_time2-block_time1) 
                
                return p_value, pre_res2down, pre_res2djoint, res2down_ssr_upper, res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower, res2djoint_df_resid, cnt + 1, cnt_prune, time_prune, if_non_zero_columns
            
            else:
            
                # check F_lower
            
                if res2djoint_ssr_upper == 0:
                    p_value = 1
                    print ('bound estimation error')
                    return p_value, pre_res2down, pre_res2djoint, res2down_ssr_upper, res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower, res2djoint_df_resid, cnt + 1, cnt_prune, time_prune, if_non_zero_columns
                else:
                    F_lower = (res2down_ssr_lower/res2djoint_ssr_upper - 1)*(res2djoint_df_resid)/lag
                    p_value_upper = 1-scipy.stats.f.cdf(F_lower, lag, (res2djoint_df_resid))   
                    
                    if p_value_upper < significant_thres:
                        # promising
                        p_value = 0
                        cnt_prune.cnt_promising += 1
                        block_time2 = timeit.default_timer()
                        time_prune.time_promising += (block_time2-block_time1)                       
                        
                        return p_value, pre_res2down, pre_res2djoint, res2down_ssr_upper, res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower, res2djoint_df_resid, cnt + 1, cnt_prune, time_prune, if_non_zero_columns
         
                    else:
                        # not sure
                        res2down, res2djoint = parts.fit_regression(dta, dtaown, dtajoint)
                        
                        if res2djoint.ssr == 0:
                            p_value = 1
                            print ('bound estimation error')
                        else:                
                            result = parts.f_test(res2down, res2djoint, lag)
                            p_value = result['ssr_ftest'][1] 
                        
                        res2down_ssr_upper = res2down.ssr
                        res2down_ssr_lower = res2down.ssr
                        res2djoint_ssr_upper = res2djoint.ssr        
                        res2djoint_ssr_lower = res2djoint.ssr    
                        
                        cnt_prune.cnt_not_sure += 1
                        block_time2 = timeit.default_timer()
                        time_prune.time_not_sure += (block_time2-block_time1)
                            
                        return p_value, res2down, res2djoint, res2down_ssr_upper, res2down_ssr_lower, res2djoint_ssr_upper, res2djoint_ssr_lower, res2djoint.df_resid, 0, cnt_prune, time_prune, if_non_zero_columns
        
  
            
            
            