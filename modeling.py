import streamlit as st

# for ML model
import numpy as np 
import pandas as pd
from math import sqrt
import seaborn as sns
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

uploaded_file = st.file_uploader("heart.csv")
if uploaded_file is not None:
    heart = pd.read_csv("heart.csv")

    # === encoding ===
    # encode categorical variables
    le = preprocessing.LabelEncoder()
    for name in heart.columns:
        if heart[name].dtypes == 'O':
         heart[name] = heart[name].astype(str)
         le.fit(heart[name])
         heart[name] = le.transform(heart[name])
    
    # fill missing values based on probability of occurrence
    for column in heart.columns:
        null_vals = heart.isnull().values
        a, b = np.unique(heart.values[~null_vals], return_counts = 1)
        heart.loc[heart[column].isna(), column] = np.random.choice(a, heart[column].isnull().sum(), p = b / b.sum())

    # apply log transformation to reduce skewness over .75 by taking log(feature + 1)
    skewed_train = heart.apply(lambda x: skew(x.dropna()))
    skewed_train = skewed_train[skewed_train > .75]
    heart[skewed_train.index] = np.log1p(heart[skewed_train.index])

    # === Modeling ===
    #Defisinisikan X dan Y
    X = heart.loc[:, heart.columns !='target']
    y = heart['target']

    #Train test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)
    
    # model
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(X_train, y_train)
    
    # make predictions based on model
    y_pred2 = model.predict(X_test)
    
    # plot
    # alpha helps to show overlapping data
    plt.scatter(y_pred2, y_test, alpha = 0.7, color = 'b')
    plt.xlabel('Predicted')
    plt.ylabel('HeartDesease')
    plt.title('Linear Regression Model')
    
    st.write(heart.head())
    st.pyplot()