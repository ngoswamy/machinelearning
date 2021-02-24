# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:35:16 2021

@author: Neeraj Goswamy
"""

# Load CSV from Pima Indian Dataset
import pandas as pd
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset = pd.read_csv("salary_train.csv") # split into input (X) and output (Y) variables
dataset['prediction'] = dataset['prediction'].map({'>50K': 1, '<=50K': 0})
dataset['workclass'] = dataset['workclass'].map({'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3, 'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked': 7})
dataset['education'] = dataset['education'].map({'Bachelors':0, 'Some-college':1, '11th':2, 'HS-grad':3, 'Prof-school':4, 'Assoc-acdm':5, 'Assoc-voc':6, '9th':7, '7th-8th':8, '12th':9, 'Masters':10, '1st-4th':11, '10th':12, 'Doctorate':13, '5th-6th':14, 'Preschool':15})
dataset['marital-status'] = dataset['marital-status'].map({'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6})
dataset['occupation'] = dataset['occupation'].map({'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13})
dataset['sex'] = dataset['sex'].map({'Male': 1, 'Female': 0})
dataset['relationship'] = dataset['relationship'].map({'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5})
dataset['race'] = dataset['race'].map({'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4})
dataset['native-country'] = dataset['native-country'].map({'United-States':0,'Cuba':1,'Jamaica':2,'India':3,'?':4,'Mexico':5,'South':6,'Honduras':7,'Mexico':8,'Puerto-Rico':9,'England':10,'Germany':11,'Iran':12,'Philippines':13,'Italy':14,'Poland':15,'Columbia':16,'Cambodia':17,'Thailand':18,'Canada':19,'Ecuador':20,'Laos':21,'Taiwan':22,'Haiti':23,'Portugal':24,'Dominican-Republic':25,'Philippines':26,'El-Salvador':27,'Poland':28,'France':29,'Honduras':30,'Haiti':31,'Guatemala':32,'China':33,'Japan':34,'Yugoslavia':35,'Peru':36,'Scotland':37,'Haiti':38,'Trinadad&Tobago':39,'Greece':40,'Nicaragua':41,'Vietnam':42,'Nicaragua':43,'Hong':44,'Outlying-US(Guam-USVI-etc)':45,'Ireland':46,'Hungary':47,'Cambodia':48})
X_train = dataset.iloc[1:, 0:13].values
y_train = dataset.iloc[1:, 14].values
y_train = keras.utils.to_categorical(y_train, num_classes = 2)
datasett = pd.read_csv("salary_test.csv") # split into input (X) and output (Y) variables
datasett['prediction'] = datasett['prediction'].map({'>50K': 1, '<=50K': 0})
datasett['workclass'] = datasett['workclass'].map({'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3, 'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked': 7})
datasett['education'] = datasett['education'].map({'Bachelors':0, 'Some-college':1, '11th':2, 'HS-grad':3, 'Prof-school':4, 'Assoc-acdm':5, 'Assoc-voc':6, '9th':7, '7th-8th':8, '12th':9, 'Masters':10, '1st-4th':11, '10th':12, 'Doctorate':13, '5th-6th':14, 'Preschool':15})
datasett['marital-status'] = datasett['marital-status'].map({'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3, 'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6})
datasett['occupation'] = datasett['occupation'].map({'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 'Armed-Forces':13})
datasett['sex'] = datasett['sex'].map({'Male': 1, 'Female': 0})
datasett['relationship'] = datasett['relationship'].map({'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5})
datasett['race'] = datasett['race'].map({'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4})
datasett['native-country'] = datasett['native-country'].map({'United-States':0,'Cuba':1,'Jamaica':2,'India':3,'?':4,'Mexico':5,'South':6,'Honduras':7,'Mexico':8,'Puerto-Rico':9,'England':10,'Germany':11,'Iran':12,'Philippines':13,'Italy':14,'Poland':15,'Columbia':16,'Cambodia':17,'Thailand':18,'Canada':19,'Ecuador':20,'Laos':21,'Taiwan':22,'Haiti':23,'Portugal':24,'Dominican-Republic':25,'Philippines':26,'El-Salvador':27,'Poland':28,'France':29,'Honduras':30,'Haiti':31,'Guatemala':32,'China':33,'Japan':34,'Yugoslavia':35,'Peru':36,'Scotland':37,'Haiti':38,'Trinadad&Tobago':39,'Greece':40,'Nicaragua':41,'Vietnam':42,'Nicaragua':43,'Hong':44,'Outlying-US(Guam-USVI-etc)':45,'Ireland':46,'Hungary':47,'Cambodia':48})
X_test = datasett.iloc[1:, 0:13].values
y_test = datasett.iloc[1:, 14].values
y_test = keras.utils.to_categorical(y_test, num_classes = 2)

model = keras.Sequential()
dim = X_train.shape[1]#Layer 1
model.add(layers.Dense(32, input_dim = dim))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.25))#Layer 2
model.add(layers.Dense(32))
model.add(layers.LeakyReLU())
model.add(layers.Dropout(0.25))#output layer
model.add(layers.Dense(2))
model.add(layers.Activation('sigmoid'))
opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer = opt,loss = 'mse', metrics = ['accuracy'])
#Fit/Train the model
bsize = 200  
model.fit(X_train, y_train, batch_size = bsize, epochs = 20, verbose = 0)
lss,acc = model.evaluate(X_test, y_test,verbose=0)
print('Test Accuracy: %.3f' % acc)



