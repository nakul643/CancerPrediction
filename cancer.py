import numpy as np
import pandas as pd 
import seaborn as sns
from scipy import stats
import pickle
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
data1= pd.read_csv("data1.csv")
data2=pd.read_csv("data2.csv")
data2.isna().sum()
numeric = data2.select_dtypes(include=np.number)
numeric_columns=numeric.columns
data2[numeric_columns]=data2[numeric_columns].fillna(data2.mean)
xtest1=data2.iloc[:,range(0,30)].values
xtest1=ss.fit_transform(xtest1)

data1.isna().sum()
numeric = data1.select_dtypes(include=np.number)
numeric_columns=numeric.columns
data1[numeric_columns]=data1[numeric_columns].fillna(data1.mean)
#sns=boxplot(x=data1[''])
x=data1.iloc[:,range(3,33)].values
y=data1.iloc[:,2].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30)
# print(xtest)



xtrain=ss.fit_transform(xtrain)
#print(x.shape)
print(xtest)
xtest=ss.transform(xtest)
# from sklearn.decomposition import PCA 
# pca=PCA(n_components=10)
# xpca=pca.fit_transform(x)

# print(xpca.shape)
# x_train_pca,x_test_pca,ytrain,ytest=train_test_split(xpca,y,test_size=0.30,random_state=30)
# x_train_pca=ss.fit_transform(x_train_pca)
# x_test_pca=ss.transform(x_test_pca)
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(random_state=0,max_iter=1000)
input=[[13.03, 18.42, 82.61, 523.8, 0.08983, 0.03766, 0.02562, 0.02923, 0.1467, 0.05863, 0.1839, 2.342, 1.17, 14.16,
        0.004352, 0.004899, 0.01343, 0.01164, 0.02671, 0.001777, 13.3, 22.81, 84.46, 545.9, 0.09701, 0.04619, 0.04833, 0.05013, 0.1987, 0.06169]]
input = np.asarray(input)
#print(input)
model1.fit(xtrain,ytrain)
print(model1.score(xtest,ytest))
pickle.dump(model1, open("model.pkl", "wb"))
yprediction =model1.predict(xtest1)
print(yprediction)
#if(yprediction=="M"):
    #print("person is mignate")

#from sklearn.metrics import confusion_matrix
#c=confusion_matrix(ytest,yprediction)
#print(c)
