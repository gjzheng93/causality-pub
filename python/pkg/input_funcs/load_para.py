# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 14:16:12 2016

@author: gjz5038
"""

# filename: load_para.py


def get_para(para_name, f):
    for line in f:
        if line[0] == '#':
            continue
        if line[:len(para_name)] == para_name:
            line = line.replace('\n', '')
            [para_head, para] = line.split('=')
            break
        
    para = para.replace(' ', '')
    
    if para[0] == "'" and para[-1] == "'":
        para = para[1:-1]
    elif para[0] == '[' and para[-1] == ']':
        para = para[1:-1].split(',')
        for i in range(len(para)):
            if para[i][0] == "'" and para[i][-1] == "'":
                para[i] = para[i][1:-1]
            else:
                try:
                    para[i] = float(para[i])
                    if para[i] == int(para[i]):
                        para[i] = int(para[i])
                except:
                    raise TypeError('wrong input type for para')
    else:
        try:
            para = float(para)
            if para == int(para):
                para = int(para)
        except:
            raise TypeError('wrong input type for para')
    
    return para
    

def load_para(list_para_name, path_to_para, para_file):
    list_para = []
    for para_name in list_para_name:
        f = open(path_to_para+para_file, 'r')
        list_para.append(get_para(para_name, f))
        f.close()
    
    return list_para    
