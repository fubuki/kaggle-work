#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Dropout
import matplotlib.pyplot as plt


numpy.random.seed(10)

all_df = pd.read_csv("./train.csv")


cols=['PassengerId', 'Survived','Pclass', 'Name', 'Sex', 'Age', 'SibSp',
              'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

all_df=all_df[cols]
msk = numpy.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

print('all:', len(all_df),
    'train:',len(train_df),
    'test:', len(test_df))


def PreprocessData(raw_df):
    df=raw_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    fare_mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(fare_mean)
    df['Sex']= df['Sex'].map({'female':0, 'male': 1}).astype(int)
    x_OneHot_df = pd.get_dummies(data=df,columns=["Embarked" ])

    ndarray = x_OneHot_df.values
    Features = ndarray[:,1:]
    Label = ndarray[:,0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

train_Features,train_Label=PreprocessData(train_df)
test_Features,test_Label=PreprocessData(test_df)

model = Sequential()


model.add(Dense(units=40, input_dim=9, 
                kernel_initializer='uniform', 
                activation='relu'))


model.add(Dense(units=30, 
                kernel_initializer='uniform', 
                activation='relu'))

model.add(Dense(units=1, 
                kernel_initializer='uniform',
                activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


train_history =model.fit(x=train_Features, 
                         y=train_Label, 
                         validation_split=0.1, 
                         epochs=30, 
                         batch_size=30,verbose=2)


def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history,'acc','val_acc')

show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x=test_Features, 
                                y=test_Label)

print(scores[1])

model.save("my_model.h5")
