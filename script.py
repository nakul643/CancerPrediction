import csv
import os
import numpy as np
import pandas as pd 

import pickle
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

model=pickle.load(open("model.pkl","rb"))   
def process_csv(filename):
    
    data1=pd.read_csv(filename)
    # data1.isna().sum()
    # numeric = data1.select_dtypes(include=np.number)
    # numeric_columns=numeric.columns
    # data1[numeric_columns]=data1[numeric_columns].fillna(data1.mean);
#sns=boxplot(x=data1[''])
    
   
    sn=data1.iloc[:,0].values
    id=data1.iloc[:,1].values
    data1.isna().sum()
    numeric = data1.select_dtypes(include=np.number)
    numeric_columns=numeric.columns
    data1[numeric_columns]=data1[numeric_columns].fillna(data1.mean)
    x=data1.iloc[:,range(2,32)].values
    x=ss.fit_transform(x)

    x=np.array(x)
    prediction=model.predict(x)
    output_file = "result.csv"
    with open(os.path.join('output', output_file), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['sn','id', 'diagnosis'])
        for i in range(0,sn.size):
            writer.writerow([sn[i],id[i],prediction[i]])

        return output_file
        # print(id[1])
        # print(type(sn))
        # print(id)

        # print(type(id))
        # print(prediction)
        # print(type(prediction))
        # print(sn.size)

       

    

    

