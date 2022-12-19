# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:07:03 2022
@author: czhong
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import making_matrix as mm
import PCA_method as Pca
import datetime
import random

class eigen_prediction:
    
    def __init__(self):
        self.real_demand=None
        self.faces=None
        self.day_average=None
        self.weights=None
        self.demand_noise=[]
        self.Loc_list=[]
        # self.Last_days=None  # the 30% day tag
        
    def get_data_and_training(self,file,face_threshhold):
        data=pd.read_csv(file)
        data=data.sort_values(by=['CalendarDateKey','interval_start'])
        data['CalendarDateKey'] = pd.to_datetime(data['CalendarDateKey'], format='%Y%m%d').dt.normalize()
        self.real_demand=mm.getfullmatrix_grid(data)
        self.real_demand=self.real_demand.fillna(0)
        self.faces,self.weights,self.day_average,var_list,eigenvalues=Pca.obtainfaces(self.real_demand,face_threshhold)
        
    
    
    def linear_predictor(self,date,num_period):   # predict a random day
        # date=self.x_test.columns[0]   # get the information of next day
        real_demand=self.real_demand[date].values.reshape(-1,1)
        num=int(len(real_demand)/35)*num_period # get the know half an hour
        known_demand=real_demand[:num,0].reshape(-1,1)
        weight=np.linalg.inv(self.faces[:,:num].dot(self.faces[:,:num].T)).dot(self.faces[:,:num]).dot(known_demand-self.day_average[:num])
        result=self.faces.T.dot(weight)+self.day_average
        result[:num]=known_demand    # replace predicted with known data
        return result, real_demand
        

    # clustering predictor in the paper disadvantage
    def clustering_training(self,kmax, clusters):
        
        self.clusters=Pca.calculate_WSS(self.weights.T, kmax, clusters) # 2 clusters
        
        return self.clusters.labels_ , self.clusters.cluster_centers_
        
    def clustering_predictor(self,cluster_number):
        re_demand=(self.faces.T.dot(self.clusters.cluster_centers_[cluster_number])).reshape(-1,1)+self.day_average
        return re_demand

    def linear_predictor_with_noise(self,date,num_period,missing,noise_level):
        real_demand=self.real_demand[date].values.reshape(-1,1)
        period_num=int(len(real_demand)/35)
        num=period_num*num_period # get the know half an hour
        known_demand=np.array(list(real_demand[num-period_num:num,0])).reshape(-1,1)
        #add missing 
        Loc=random.sample(range(0,period_num),int(period_num*(1-missing))) #take only these cells
        Loc+=np.array(num-period_num)
        self.Loc_list.extend(Loc.reshape(-1).tolist())
        #add noise
        val=np.array(random.choices(noise_level,k=int(period_num))).reshape(-1,1)
        # random locations to add noise
        known_demand=known_demand*val  
        self.demand_noise.extend(known_demand.reshape(-1).tolist())
        
        #calculate
        faces,demand_noises,average_faces=self.faces[:,:num][:,Loc],np.array(self.demand_noise).reshape(-1,1)[Loc],self.day_average[:num][Loc]
        weight=np.linalg.inv(faces.dot(faces.T)).dot(faces).dot(np.array(demand_noises).reshape(-1,1)-average_faces)
        result=self.faces.T.dot(weight)+self.day_average
        # result[:num]=known_demand    # replace predicted with known data
        return result, real_demand


    def data_for_21(self,file):
        data=pd.read_csv(file)
        data=data.sort_values(by=['CalendarDateKey','interval_start'])
        data['CalendarDateKey'] = pd.to_datetime(data['CalendarDateKey'], format='%Y%m%d').dt.normalize()
        self.demand_in_21=mm.getfullmatrix_grid(data)
        self.demand_in_21=self.demand_in_21.fillna(0)

    def linear_predictor_transportability(self,date,num_period):
        face_19=self.convert_matrix(self.faces.T,self.demand_in_21)
        face_19=face_19.T
        day_average=self.convert_matrix(self.day_average,self.demand_in_21)
        real_demand=self.demand_in_21[date].values.reshape(-1,1)
        num=int(len(real_demand)/35)*num_period # get the know half an hour
        known_demand=real_demand[:num,0].reshape(-1,1)
        weight=np.linalg.inv(face_19[:,:num].dot(face_19[:,:num].T)).dot(face_19[:,:num]).dot(known_demand-day_average[:num])
        result=face_19.T.dot(weight)+day_average
        result[:num]=known_demand    # replace predicted with known data
        return result, real_demand   
    
    
    def convert_matrix(self,array_19,des_matrix): #the grids in 2019 are not the same as in 2021
        index_21=pd.DataFrame(None,des_matrix.index)
        converted_matrix=pd.DataFrame(array_19,index=self.real_demand.index)
        converted_matrix=index_21.join(converted_matrix, how='left')  
        converted_matrix=converted_matrix.fillna(0)
        
        return converted_matrix.values
    
    
    
    
    
# class Time_series_predictor:
#     # LSTM predictor  time series model
    
#     # Linear predictor   time series model
    
#     pass


def measure_the_error(actual,pred,period=0,interval=1):
    rmsl=[]
    mae=[]
    num=int(len(pred)/35)

    if period!=0:
        length=len(pred)
        limit=period*num+interval*num
        lower=num*period
        upper=lower+num
        while upper<=length and upper<=limit:
            ac=actual[lower:upper,:]
            pr=pred[lower:upper,:]
            rmsl.append((np.square(np.subtract(ac,pr)).mean())**0.5)
            mae.append(abs(np.subtract(ac,pr)).mean())
            lower+=num
            upper+=num
    else:
        lower=num*period
        upper=lower+num
        while upper<=len(pred):
            ac=actual[lower:upper,:]
            pr=pred[lower:upper,:]
            rmsl.append((np.square(np.subtract(ac,pr)).mean())**0.5)
            mae.append(abs(np.subtract(ac,pr)).mean())
            lower+=num
            upper+=num
#    root_mean_square=np.square(np.subtract(actual,pred)).mean()
    root_mean_square=sum(rmsl)/len(rmsl)
    mean_abosolute=sum(mae)/len(mae)
    # mean_absolute_percentage=np.mean(np.abs((actual - pred)/actual))*100
    return root_mean_square,mean_abosolute
    
    #%%
    

if __name__=="__main__":
    Predictor_19=eigen_prediction()
    Predictor_19.get_data_and_training('./2019_1500_merge_data',0.95)
    
    
    
    #%% linear prediction -1 accuracy
    #every 30 min every 2 hours, every 4 hours and every 8 hours  in October
    def linear_predictor_frequency(interval):
        
        date_=Predictor_19.real_demand.columns
        date_range=list(date_)
        day_mse=[]
        day_mae=[]
        for d in date_range:
            period_mse=[]
            period_mae=[]
            
            for i in range(interval,35,interval):
                pred,actual=Predictor_19.linear_predictor(d,i)
                mse,mae=measure_the_error(actual,pred,period=i,interval=interval)
                period_mse.append(mse)
                period_mae.append(mae)
            day_mse.append(sum(period_mse)/len(period_mse))
            day_mae.append(sum(period_mae)/len(period_mae))
        return day_mse,day_mae
    
    
    def linear_predictor_drawing():
        interval=[1,4,8,16]
        data_mse=pd.DataFrame()
        for i in interval:
            name='frequency: %.1f hour'%(i/2)
            data_mse[name]=linear_predictor_frequency(i)
        x_tick=['Oct'+'-'+str(j) for j in  range(1,32,3)]+['Nov'+'-'+str(j) for j in  range(1,31,3)]
        plt.figure(figsize=(13,8))
        plt.plot(data_mse)
        plt.legend(labels=data_mse.columns)
        plt.xticks(range(0,61,3), x_tick,rotation=45)
        plt.show()

    
    list1_mse,list1_mae=linear_predictor_frequency(8)
    list2_mse,list2_mae=linear_predictor_frequency(16)
    
    #%% linear prediction -2 robustness

    def linear_predictor_noise(Predictor_19,interval,missing,noise_level):
        
        date_=Predictor_19.real_demand.columns
        date_range=list(date_)[-30:]
        day_mse=[]
        for d in date_range:
            Predictor_19.demand_noise=[]
            period_mse=[]
            for i in range(interval,35,interval):
                pred,actual=Predictor_19.linear_predictor_with_noise(d,i,missing,noise_level)
                mse,mae=measure_the_error(actual,pred,period=i,interval=interval)
                period_mse.append(mse)
                
            day_mse.append(sum(period_mse)/len(period_mse))
            

        Predictor_19=eigen_prediction()

        return day_mse

    mse_list_missing=pd.DataFrame()
    mse_list_magnitude=pd.DataFrame()
    for i in range(1,6):
        day_mse=linear_predictor_noise(Predictor_19,1,i*0.1,[1.3,0.7])
        mse_list_missing['mis'+str(i)]=day_mse
    del day_mse
    for i in range(1,6):
        day_mse=linear_predictor_noise(Predictor_19,1,0.3,[1-i*0.1,1+i*0.1])
        mse_list_magnitude['mag'+str(i)]=day_mse
    # day_mse2=linear_predictor_noise(1,0.5,[0.7,1.3])
    



    #%% linear prediction -3 Transferability
    Predictor_19=eigen_prediction()
    Predictor_19.get_data_and_training('./2019_1500_merge_data.csv',0.95) # update for unknown bug
    Predictor_19.data_for_21('./2021_1500_merge_data.csv.csv')
    
    
    def linear_predictor_transferability(interval):
        
        
        date_=Predictor_19.demand_in_21.columns
        date_range=list(date_)
        day_mse=[]
        day_mae=[]
        for d in date_range:
            period_mse=[]
            period_mae=[]
            for i in range(interval,35,interval):
                pred,actual=Predictor_19.linear_predictor_transportability(d,i)
                mse,mae=measure_the_error(actual,pred,period=i,interval=interval)
                period_mse.append(mse)
                period_mae.append(mae)
            day_mse.append(sum(period_mse)/len(period_mse))
            day_mae.append(sum(period_mae)/len(period_mae))
        return day_mse,day_mae
    

    
    
    list1_mse,list1_mae=linear_predictor_transferability(16)
    
    
    
    
    
    
    
    
    
    
    
    
    
    #%% clustering prediction -1 accuracy
    label_19,cent_weight=Predictor_19.clustering_training(15,5) 
    def clustering_mse(date_list):
        weekday_info=[i.weekday()for i in date_list]
        mse_list=[]
        mae_list=[]
        demand_clusters=[]
        for i in range(0,5):
            demand_clusters.append(Predictor_19.clustering_predictor(i))
        
        for i,d in enumerate(date_list):
            if d < datetime.date(2019, 7, 8):
                if weekday_info[i]<5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[2])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[3])
                    mse_list.append(mse) 
                    mae_list.append(mae)
            elif d<datetime.date(2019, 8, 2):
                if weekday_info[i]<5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[0])
                    mse_list.append(mse) 
                    mae_list.append(mae)

                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[3])
                    mse_list.append(mse) 
                    mae_list.append(mae)

            elif d<datetime.date(2019, 8, 16):
                if weekday_info[i]<5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[2])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[3])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                
            else:
                if weekday_info[i]<4:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[1])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                elif weekday_info[i]==4:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[4])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[0])
                    mse_list.append(mse) 
                    mae_list.append(mae)
        return mse_list,mae_list
                    

    clustering_mse_list,clustering_mae_list=clustering_mse(list(Predictor_19.real_demand.columns))
    
    #%% clustering prediction -3 Tranferability
    Predictor_19=eigen_prediction()
    Predictor_19.get_data_and_training('./2019_1500_merge_data.csv',0.95) # update for unknown bug
    Predictor_19.data_for_21('./2019_1500_merge_data.csv')
    label_19,cent_weight=Predictor_19.clustering_training(15,5) 
    def clustering_mse(date_list):
        weekday_info=[i.weekday()for i in date_list]
        mse_list=[]
        mae_list=[]
        demand_clusters=[]
        for i in range(0,5):
            cluster_demand=Predictor_19.clustering_predictor(i)
            cluster_demand=Predictor_19.convert_matrix(cluster_demand,Predictor_19.demand_in_21)
            demand_clusters.append(cluster_demand)
            
            
        for i,d in enumerate(date_list):
            if weekday_info[i]<4:
                mse,mae=measure_the_error(Predictor_19.demand_in_21[d].values.reshape(-1,1),demand_clusters[1])
                mse_list.append(mse) 
                mae_list.append(mae)
            elif weekday_info[i]==4:
                mse,mae=measure_the_error(Predictor_19.demand_in_21[d].values.reshape(-1,1),demand_clusters[4])
                mse_list.append(mse) 
                mae_list.append(mae)
            else:
                mse,mae=measure_the_error(Predictor_19.demand_in_21[d].values.reshape(-1,1),demand_clusters[0])
                mse_list.append(mse) 
                mae_list.append(mae)
        return mse_list,mae_list
            
            


    clusterting_transfer_mse,clusterting_transfer_mae=clustering_mse(Predictor_19.demand_in_21.columns)
    
    #%% clustering prediction -1 accuracy（1000）
    label_19,cent_weight=Predictor_19.clustering_training(15,5) 
    def clustering_mse(date_list):
        weekday_info=[i.weekday()for i in date_list]
        mse_list=[]
        mae_list=[]
        demand_clusters=[]
        for i in range(0,5):
            demand_clusters.append(Predictor_19.clustering_predictor(i))
        
        for i,d in enumerate(date_list):
            if d < datetime.date(2019, 7, 8):
                if weekday_info[i]<5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[0])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[3])
                    mse_list.append(mse) 
                    mae_list.append(mae)
            elif d<datetime.date(2019, 8, 2):
                if weekday_info[i]<5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[2])
                    mse_list.append(mse) 
                    mae_list.append(mae)

                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[3])
                    mse_list.append(mse) 
                    mae_list.append(mae)

            elif d<datetime.date(2019, 8, 16):
                if weekday_info[i]<5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[0])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[3])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                
            else:
                if weekday_info[i]<5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[1])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                elif weekday_info[i]==5:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[2])
                    mse_list.append(mse) 
                    mae_list.append(mae)
                else:
                    mse,mae=measure_the_error(Predictor_19.real_demand[d].values.reshape(-1,1),demand_clusters[3])
                    mse_list.append(mse) 
                    mae_list.append(mae)
        return mse_list,mae_list
                    

    clustering_mse_list,clustering_mae_list=clustering_mse(list(Predictor_19.real_demand.columns))
    
    #%% clustering prediction -3 Tranferability （1000）
    Predictor_19=eigen_prediction()
    Predictor_19.get_data_and_training('../Data_pol2/2019_1000_merge_data.csv',0.95) # update for unknown bug
    Predictor_19.data_for_21('../Data_pol2/2021_1000_merge_data.csv')
    label_19,cent_weight=Predictor_19.clustering_training(15,5) 
    def clustering_mse(date_list):
        weekday_info=[i.weekday()for i in date_list]
        mse_list=[]
        mae_list=[]
        demand_clusters=[]
        for i in range(0,5):
            cluster_demand=Predictor_19.clustering_predictor(i)
            cluster_demand=Predictor_19.convert_matrix(cluster_demand,Predictor_19.demand_in_21)
            demand_clusters.append(cluster_demand)
            
            
        for i,d in enumerate(date_list):
            if weekday_info[i]<5:
                mse,mae=measure_the_error(Predictor_19.demand_in_21[d].values.reshape(-1,1),demand_clusters[1])
                mse_list.append(mse) 
                mae_list.append(mae)
            elif weekday_info[i]==5:
                mse,mae=measure_the_error(Predictor_19.demand_in_21[d].values.reshape(-1,1),demand_clusters[2])
                mse_list.append(mse) 
                mae_list.append(mae)
            else:
                mse,mae=measure_the_error(Predictor_19.demand_in_21[d].values.reshape(-1,1),demand_clusters[3])
                mse_list.append(mse) 
                mae_list.append(mae)
        return mse_list,mae_list
            
            


    clusterting_transfer_mse,clusterting_transfer_mae=clustering_mse(Predictor_19.demand_in_21.columns)
    

    #%% history average prediction -1 Accuracy
    def AverageDemandPrediction():
        date=Predictor_19.real_demand.columns
        date_tag=[i.weekday()for i in date]
        traning_end=np.where(date==datetime.date(2019, 10, 31))[0][0]+1
        x_train_tag=date_tag[0:traning_end]
        y_tag=date_tag[traning_end:]
        ha_mse=[]
        ha_mae=[]
        for i,v in enumerate(y_tag):
            y_predict=Predictor_19.real_demand.iloc[:,[t for t,x in enumerate(x_train_tag) if x==v]].mean(axis=1)
            mse,mae=measure_the_error(Predictor_19.real_demand.iloc[:,traning_end+i].values.reshape(-1,1), y_predict.values.reshape(-1,1))
            ha_mse.append(mse)         
            ha_mae.append(mae)   
            print(Predictor_19.real_demand.columns[traning_end+i])
        return ha_mse,ha_mae
    
    ha_mse,ha_mae=AverageDemandPrediction()
    
    

    #%% history average prediction -2 Tranferability
    def AverageDemandPrediction():
        # 2019 training
        date=Predictor_19.real_demand.columns
        date_tag=[i.weekday()for i in date]
        traning_end=np.where(date==datetime.date(2019, 10, 31))[0][0]+1
        x_train_tag=date_tag[0:traning_end]
        # test in 2021
        date_2021=Predictor_19.demand_in_21.columns
        testing_start=np.where(date_2021==datetime.date(2021, 10, 31))[0][0]+1
        date_tag_21=[i.weekday()for i in date_2021]
        y_tag=date_tag_21[testing_start:]
        ha_mse_transfer=[]
        ha_mae_transfer=[]
        for i,v in enumerate(y_tag):
            y_predict=Predictor_19.real_demand.iloc[:,[t for t,x in enumerate(x_train_tag) if x==v]].mean(axis=1)
            y_predict=Predictor_19.convert_matrix(y_predict.values,Predictor_19.demand_in_21)
            y_real=Predictor_19.demand_in_21.iloc[:,testing_start+i].values.reshape(-1,1)
            mse,mae=measure_the_error(y_real, y_predict)
            ha_mse_transfer.append(mse)   
            ha_mae_transfer.append(mae) 
            print(Predictor_19.real_demand.columns[testing_start+i])
        return ha_mse_transfer,ha_mae_transfer
    
    ha_mse_transfer,ha_mae_transfer=AverageDemandPrediction()
    
    
    
    #%% random forest prediction
    
    
  