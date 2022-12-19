# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:30:03 2022

@author: zhong
"""
import pandas as pd
import numpy as np
import datetime


#%%  make one-day matrix


def getfullmatrix(data):
    stations=data['StopAreaNumber'].unique()
    dates=data['CalendarDateKey'].unique()
    #default matrix
    stack_matrix=pd.DataFrame(np.nan, index=data['interval_start'].unique(),
                              columns=stations).stack(dropna=False).to_frame()
    stack_matrix.index.names=['interval_start', 'StopAreaNumber']
    
    #process the matrix
    matrix=data.pivot(index='interval_start',
                      columns=['CalendarDateKey','StopAreaNumber',],
                      values='n_validations'
                      )
    matrix=matrix.stack(1,dropna=False)
    stack_matrix=stack_matrix.merge(matrix,
                                    how='left',
                                    left_index=True,
                                    right_index=True)
    
    stack_matrix = stack_matrix.iloc[: , 1:]
    dates=data['CalendarDateKey'].dt.date.unique()     # remove hour...
    stack_matrix.columns=dates
    
    # how many non values
    print('there are %d unrecorded Nans' % stack_matrix.isna().sum().sum())
    return stack_matrix

def getfullmatrix_grid(data):
    stations=data['index'].unique()
    dates=data['CalendarDateKey'].unique()
    #default matrix
    stack_matrix=pd.DataFrame(np.nan, index=data['interval_start'].unique(),
                              columns=stations).stack(dropna=False).to_frame()
    stack_matrix.index.names=['interval_start', 'index']
    
    #process the matrix
    matrix=data.pivot(index='interval_start',
                      columns=['CalendarDateKey','index',],
                      values='n_validations'
                      )
    matrix=matrix.stack(1,dropna=False)
    
    #merge the matrix
    stack_matrix=stack_matrix.merge(matrix,
                                    how='left',
                                    left_index=True,
                                    right_index=True)
    
    stack_matrix = stack_matrix.iloc[: , 1:]
    dates=data['CalendarDateKey'].dt.date.unique()     # remove hour...
    stack_matrix.columns=dates
    # how many non values
    print('there are %d unrecorded Nans' % stack_matrix.isna().sum().sum())
    return stack_matrix



def getfullmatrix_grid_wide(data): # columns just stations
    
    stations=data['index'].unique()
    #default matrix
    stack_matrix=pd.DataFrame(np.nan,index=stations,columns=['Default'])
    stack_matrix.index.names=['index']
    stack_matrix.columns=pd.MultiIndex.from_product([stack_matrix.columns, ['C']])
    
    #process the matrix
    matrix=data.pivot(index='index',
                      columns=['CalendarDateKey','interval_start',],
                      values='n_validations'
                      )
    # matrix=matrix.stack(1,dropna=False)
    
    #merge the matrix
    stack_matrix=stack_matrix.merge(matrix,
                                    how='left',
                                    left_index=True,
                                    right_index=True)
    
    stack_matrix = stack_matrix.iloc[: , 1:]
    dates=data['CalendarDateKey'].dt.date.unique()     # remove hour...
    stack_matrix.columns=stack_matrix.columns.set_levels(dates,level=0)
    # how many non values
    print('there are %d unrecorded Nans' % stack_matrix.isna().sum().sum())
    return stack_matrix

#%% get one matrix

def getonematrix(full_matrix,date=datetime.date(2018, 9, 1)):
    matrix=full_matrix[date].unstack(level=-1)
    return matrix



