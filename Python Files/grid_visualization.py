# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:51:09 2022

@author: zhong
"""

import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import jenkspy
import matplotlib.pyplot as plt
import data_obtain as do
# from pyproj import Proj, transform
from pyproj import Transformer
import making_matrix as mm
import PCA_method as Pca
import pandas as pd
import datetime



def make_grid(size):
    file=r'.\new study area.shp'
    gdf=gpd.read_file(file)
    gdf.to_crs(3006,inplace=True)
    # full grids
    
    xmin,ymin,xmax,ymax =  gdf.total_bounds
    width = size #meters
    height = size
    rows = int(np.ceil((ymax-ymin) /  height))
    cols = int(np.ceil((xmax-xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin
        for j in range(rows):
            polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width
    
    grid = gpd.GeoDataFrame({'geometry':polygons})
    grid.crs = "EPSG:3006"
    # grid figure
    grid_gdf=gpd.overlay(grid, gdf, how='intersection')
    grid_gdf.reset_index(inplace=True)
    
    return grid_gdf


def draw_colormap(data,geo_df,column_name,bounds,figure_name,color='RdYlBu'):

    cmap = mpl.cm.get_cmap(color)   #OrRd
    norm_b=BoundaryNorm(boundaries=bounds,ncolors=cmap.N)
    fig, ax = plt.subplots(figsize = (18,14))
    result=geo_df.merge(data, on='index',how='left')
    result=result.fillna({column_name:0})
    result.to_crs(epsg=4326).plot(
        column=column_name,
        cmap=cmap,
        ax=ax,
        norm=norm_b,
        legend= False
        )
    ax.set_xlabel('Longitude',fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.set_ylabel('Latitude',fontsize=35)
    ax.set_title(figure_name,fontsize=45)
    fig, ax = plt.subplots(figsize=(1, 12))
    fig.subplots_adjust(bottom=0.5)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm_b, cmap=cmap),
                  cax=ax, orientation='vertical', label='colorbar')

#%%
def matrix_to_df(faces,df,n): #faces list  and the Nth face
    data=pd.DataFrame(faces[n],index=df.index,columns=['values'])
    return data

def eigen_image(file,face_threshhold=0.95):
    data=pd.read_csv(file)
    data=data.sort_values(by=['CalendarDateKey','interval_start'])
    data['CalendarDateKey'] = pd.to_datetime(data['CalendarDateKey'], format='%Y%m%d').dt.normalize()
    # data['interval_start'] = pd.to_datetime(data['interval_start'], format='%H:%M:%S').dt.time
    stack_matrix=mm.getfullmatrix_grid_wide(data)
    stack_matrix=stack_matrix.fillna(0)
    faces,weights,day_average,var_list,eigenvalues=Pca.obtainfaces(stack_matrix,face_threshhold)
    print('under threshold value of %.3f, there are %d eigenfaces'%(face_threshhold,len(faces)))
    return faces,weights,day_average,stack_matrix,eigenvalues

def draw_eigenface(faces,matrix,bounds,name,eigenvalues,color='RdYlBu'):
    column_name='values'
    for n in range(len(faces)):
        # figure_name=name+' %d'%n + ' (Eigenvalue: %.3e'%eigenvalues[0][n] +')'
        figure_name=name+' %d'%n
        # figure_name=name  # for average face
        faces_df=matrix_to_df(faces,matrix,n)
        # bounds = jenkspy.jenks_breaks(faces.flatten(),nb_class=nb_class)
        draw_colormap(faces_df,grid_gdf,column_name,bounds,figure_name=figure_name,color=color)

def bound_range_max(*arg):
    max_list=[]
    for i in arg:
        max_value=max(abs(i.flatten()))
        max_list.append(max_value)
    return max(max_list)



#%% draw demand figures

def getDemandBounds(data,nb_class):
    bounds = jenkspy.jenks_breaks(data[data != 0],nb_class=nb_class) # remove all the 0s
    if min(bounds)>0:
        bounds.insert(0,0)
    else:
        bounds.append(0)
        bounds.sort()
    return bounds

def getEigenBounds(data,nb_class):
    bounds=np.linspace(min(data),max(data),nb_class)
    bounds=list(bounds)
    bounds.append(0)
    bounds.sort()
    return bounds

def drawDemandDistribution(matrix,realmatrix,grid_gdf,name,bounds):
    column_name=realmatrix.columns[0]
    real_first_day=realmatrix[[column_name]]
    real_first_day.reset_index(inplace=True)
    real_first_day[column_name]=matrix
    draw_colormap(real_first_day,grid_gdf,column_name,bounds,name,color='OrRd')
    
    
    
    
    
def draw_weight_figure(k_means,matrix,weights,figure_name):
    labels=k_means.labels_
    dict_time=dict(zip(matrix.columns.get_level_values(1).unique(),range(0,35)))
    dict_date=dict(zip(matrix.columns.get_level_values(0).unique(),range(0,len(matrix.columns.get_level_values(0).unique()))))
    for i in set(labels):
        df_=pd.DataFrame(matrix.columns[np.where(labels == i)].get_level_values(0),columns=['date'])

        df_['time']=matrix.columns[np.where(labels == i)].get_level_values(1)

        df_['time']=df_['time'].map(lambda t :dict_time[t])

        df_['X']=df_['date'].map(lambda t :dict_date[t])
        fig,ax=plt.subplots(figsize=(32,15))
        ax.set_title(figure_name+'with label %d and centroid %.2f'%(i,np.mean(weights[0][np.where(labels == i)])),fontsize=25)

        ax.plot(df_['X'], df_['time'], 'o', color='black')


        ax.set_xticks(list(dict_date.values()))
        x_ticks=[x.isoformat().split('-',1)[1] for x in list(dict_date.keys())]            # remove the year
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(list(dict_time.values()))

        ax.set_yticklabels(list(dict_time.keys()))
        ax.tick_params(axis='x', labelrotation = 65)
        plt.show()
        
        

#%% 
if __name__=="__main__":

    grid_gdf=make_grid(1500)
    #%%
    file1=r'./2019_1500_merge_data'
    file2=r'./2021_1500_merge_data'
    #%% New
    
    face_19,weights_19,day_average_19,matrix_19,eigenvalues_19=eigen_image(file1,face_threshhold=0.97)
    
    
    
    face_21,weights_21,day_average_21,matrix_21,eigenvalues_21=eigen_image(file2,face_threshhold=0.97)
    
    

    
    
    #%%draw average face
    
    
    nb_class=9
    bounds=getDemandBounds(day_average_19.flatten(),nb_class)
    name='2019 average demand' 
    draw_eigenface(day_average_19.T,matrix_19,bounds,name,eigenvalues_19,color='OrRd')
    
    
    #%% draw faces
    
    nb_class=9
    bounds=getEigenBounds(np.append(face_19.flatten(),face_21.flatten()),nb_class)
    name='Eigen Image'
    draw_eigenface(face_19,matrix_19,bounds,name,eigenvalues_19,color='RdYlBu')
    
    
    
    
    
    name='Eigen Image'
    draw_eigenface(face_21,matrix_21,bounds,name,eigenvalues_21,color='RdYlBu')
    
    

    

    

