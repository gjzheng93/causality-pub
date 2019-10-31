# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:22:24 2016

@author: gjz5038
"""

#system modules
import os
import sys
import numpy as np
from multiprocessing import Pool
import shutil
import pickle


#self modules
from .input_funcs import load_data
from .input_funcs import load_para
from .output_funcs import output_tools

from .causal import causality_fast as causality
from . import parameters


if __debug__:
    print('in debug mode')
else:
    print('in run mode')
    
    
def main():
    # =========================================== load and set parameters ====================================

    path_to_preprocessed_data_file = '../../data/preprocessed/'

    path_to_para = 'pkg/'
    para_file = 'taxi.conf'
    
    [data_source, X_name, Y_name, analysis_type, this_run_name] = load_para.load_para(['data_source', 'X_name', 'Y_name', 'analysis_type', 'this_run_name'], path_to_para, para_file)

    path_to_sample = path_to_preprocessed_data_file+data_source+'/'
    X_file_name = X_name+'.csv'
    Y_file_name = Y_name+'.csv'


    path_to_output = '../../data/'+data_source+'/'+this_run_name+'/'
           
    [step, trip, lag, test_mode, significant_thres, min_segment_len, max_segment_len, cal_stationary_separately] = load_para.load_para(['step', 'trip', 'lag', 'test_mode', 'significant_thres', 'min_segment_len', 'max_segment_len', 'cal_stationary_separately'], path_to_para, para_file)
    
    para_set = parameters.paraSet(step, trip, lag, test_mode, significant_thres, min_segment_len, max_segment_len, cal_stationary_separately)

    # =========================================== load data ====================================

    # load X
    array_X = load_data.load_data_without_head(path_to_sample, X_file_name)
    array_X = array_X.astype(float)        
    # load Y
    array_Y = load_data.load_data_without_head(path_to_sample, Y_file_name)
    array_Y = array_Y.astype(float) 
    # load time
    array_time = pickle.load(open(path_to_sample+'time.pickle', 'rb'))
    
    array_Y = array_Y[:744]
    array_X = array_X[:744]
    array_time = array_time[:744]
        
    if len(array_X) < 100:
        print ('too few points')
        return
        
    if os.path.exists(path_to_output):
        print('already exists')
        sys.exit(path_to_output+'  exists')
        
    else:
        os.mkdir(path_to_output)
        
    causality.granger(array_X, array_Y, X_name, Y_name, para_set, path_to_output)
    shutil.copyfile(path_to_para+para_file, path_to_output+para_file)
                
        
if __name__ == '__main__':
    main()

