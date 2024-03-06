import cv2
import os
import pandas as pd
import numpy as np
# 指定目标文件夹路径
mypath = '../data/train_image/'
df=pd.read_csv('data.csv')
df['image'] = np.nan
# 获取目标文件夹下所有文件和目录的名称
all_items = os.listdir(mypath)

# 筛选出文件的路径
file_paths = [os.path.join(mypath, item) for item in all_items if os.path.isfile(os.path.join(mypath, item))]
head=0
width=0
for i,file_path in enumerate(file_paths):


    img = cv2.imread(file_path)
    width+=img.shape[0]
    head+=img.shape[1]
width=width//6000
head=head//6000
print('{}  and {}'.format(width,head))
##实际宽高（196，383）最好refactor为（128，384）





