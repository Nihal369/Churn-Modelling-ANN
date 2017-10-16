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
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)

#!!!-Building the ANN-!!!

#Importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Dropout

#Initalizing the ANN
classifier=Sequential()

#Creating the layers
#First Hidden layer and Input Layer
classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform",input_dim=11))
classifier.add(Dropout(rate=0.1))
#Second Hidden Layer
classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))
#Output layer
classifier.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))

#Compiling the ANN
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

#Fitting the ANN
classifier.fit(xtrain,ytrain,batch_size=10,epochs=100)

#Testing the ANN
ypred=classifier.predict(xtest)
ypred=(ypred>0.5)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)

#Evaluvating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform",input_dim=11))
    classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))
    classifier.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=10)
accuracies=cross_val_score(classifier,xtrain,ytrain,cv=10,n_jobs=-1)

#Parameter tuning
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform",input_dim=11))
    classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))
    classifier.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
    classifier.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[10,25,32,16],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']
            }
gridSearchCV=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=2)
gridSearchCV.fit(xtrain,ytrain)
bestParams=gridSearchCV.best_params_
bestAccuracy=gridSearchCV.best_score_
