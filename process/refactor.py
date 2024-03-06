import pandas as pd
import numpy as np
df=pd.read_csv('data.csv')

def fun(x):
    list=[]

    for ch in x.lower():
        list.append(ord(ch))
    return list
df.replace(np.nan, "", inplace=True)

df['label_digit']=df.label.apply(fun)


df.to_pickle('data.pickle')
