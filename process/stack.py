import pandas as pd
df=pd.read_pickle('data.pickle')

def fun(x):
    if(len(x)<23):
        x.extend([0] * (23 - len(x)))
    return x

df.label_digit.apply(fun)
df.to_pickle('data.pickle')






