

from ..output_funcs import output_tools


def output_causal(path_to_output, X_name, Y_name, time_set, cnt_set, array_results_YX, array_results_XY, array_adf_results_X, array_adf_results_Y, list_segment_split, para_set):
    
    [start_time, end_time, time_granger, time_adf] = time_set  
    [total_cnt_segment_YX, cnt_prune_YX, time_prune_YX, total_cnt_segment_XY, cnt_prune_XY, time_prune_XY, total_cnt_segment_adf, total_cnt_segment_cal_adf, total_cnt_segment_examine_adf_Y] = cnt_set
    
    file_time = open(path_to_output+'time', 'a+')
    file_time.write(X_name+',total time: ' + str(end_time - start_time)+'\n')
    file_time.write(X_name+',granger time: '+ str(time_granger)+'\n')
    file_time.write(X_name+',adf time: '+ str(time_adf)+'\n')
    file_time.write('\n')
    file_time.close()
    
    
    if para_set.test_mode == 'fast_version_2' or para_set.test_mode == 'fast_version_3':
        file_cnts_prune = open(path_to_output+'prune_cnts', 'a+')
        file_cnts_prune.write(X_name+',for YX:\n')
        file_cnts_prune.write(X_name+',total cnt: ' + str(total_cnt_segment_YX)+'\n')
        file_cnts_prune.write(X_name+',promising cnt: '+ str(cnt_prune_YX.cnt_promising)+'\n')
        file_cnts_prune.write(X_name+',promising not cnt: '+ str(cnt_prune_YX.cnt_promising_not)+'\n')            
        file_cnts_prune.write(X_name+',not sure cnt: '+ str(cnt_prune_YX.cnt_not_sure)+'\n')
        file_cnts_prune.write(X_name+',initial cnt: '+ str(cnt_prune_YX.cnt_initial)+'\n')
        file_cnts_prune.write(X_name+',for XY:\n')
        file_cnts_prune.write(X_name+',total cnt: ' + str(total_cnt_segment_XY)+'\n')
        file_cnts_prune.write(X_name+',promising cnt: '+ str(cnt_prune_XY.cnt_promising)+'\n')
        file_cnts_prune.write(X_name+',promising not cnt: '+ str(cnt_prune_XY.cnt_promising_not)+'\n')            
        file_cnts_prune.write(X_name+',not sure cnt: '+ str(cnt_prune_XY.cnt_not_sure)+'\n')
        file_cnts_prune.write(X_name+',initial cnt: '+ str(cnt_prune_XY.cnt_initial)+'\n')
        file_cnts_prune.write(X_name+',total adf cnt: '+ str(total_cnt_segment_adf)+'\n')
        file_cnts_prune.write(X_name+',total cal adf cnt: '+ str(total_cnt_segment_cal_adf)+'\n')
        file_cnts_prune.write(X_name+', total_cnt_segment_examine_adf_Y: '+str(total_cnt_segment_examine_adf_Y)+'\n')
        file_cnts_prune.write('\n')
        file_cnts_prune.close() 
    
    
    output_tools.output_2d_data(array_results_YX, [], path_to_output, str(para_set.trip) +'_'+Y_name+'_caused_by_'+X_name+'.csv')
    output_tools.output_2d_data(array_results_XY, [], path_to_output, str(para_set.trip) +'_'+X_name+'_caused_by_'+Y_name+'.csv')
    
    
    output_tools.output_2d_data(array_adf_results_X, [], path_to_output, str(para_set.trip) +'_'+X_name+'_adf'+'.csv')
    output_tools.output_2d_data(array_adf_results_Y, [], path_to_output, str(para_set.trip) +'_'+Y_name+'_with_'+X_name+'_adf'+'.csv')

    
    output_tools.output_list(list_segment_split, path_to_output, 'segment_split_id.csv')