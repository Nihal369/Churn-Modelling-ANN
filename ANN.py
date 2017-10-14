# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:17:12 2017

@author: nihal369
"""
#Importing libraries
import numpy as np
import pandas as pd

#!!!-Data Preprocessing-!!!

#Importing the dataset
dataset=pd.read_csv("Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

#Encoding the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder=LabelEncoder()
x[:,1]=labelEncoder.fit_transform(x[:,1])
x[:,2]=labelEncoder.fit_transform(x[:,2])
oneHotEncoder=OneHotEncoder(categorical_features=[1],dtype=np.int)
x=oneHotEncoder.fit_transform(x).toarray()
x=x[:,1:]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)

#Training and test set splitting
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,train_size=0.9)




