import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\Aniruddha Sultane\Downloads\abc.csv")

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='mean')

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()

#labelencoder_X.fit_transform(X[:,0])
X[:,0]=labelencoder_X.fit_transform(X[:,0])

labelencoder_y = LabelEncoder()

Y=labelencoder_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
