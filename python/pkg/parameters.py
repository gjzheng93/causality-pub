# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:11:26 2016

@author: gjz5038
"""

#para_set

class paraSet:
    def __init__(self, step, trip, lag, test_mode, significant_thres, min_segment_len, max_segment_len, cal_stationary_separately):
        self.step = step
        self.trip = trip
        self.lag = lag
        self.test_mode = test_mode
        self.significant_thres = significant_thres
        self.min_segment_len = min_segment_len
        self.max_segment_len = max_segment_len
        self.cal_stationary_separately = cal_stationary_separately