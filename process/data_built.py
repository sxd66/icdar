import cv2
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
import torch

def get_data():
    mypath = '../data/train_image/'
    df=pd.read_pickle('data.pickle')

    # 获取目标文件夹下所有文件和目录的名称
    all_items = os.listdir(mypath)

    # 筛选出文件的路径
    file_paths = [os.path.join(mypath, item) for item in all_items if os.path.isfile(os.path.join(mypath, item))]
    return df.label_digit.values,file_paths

class Patchdataset(Dataset):
    def __init__(self,label,file_paths,transform=None):
        self.label=label
        self.filepath=file_paths
        self.transform=transform
    def __len__(self):
        return self.label.shape[0]
    def __getitem__(self, item):
        img=torch.from_numpy(cv2.imread(self.filepath[item]).astype(float)).permute(2,0,1)
        if(self.transform is not None):
            img=self.transform(img)

        target=torch.tensor(self.label[item])
        return (img,target)

def data_load(train_batch,val_batch):
    label,file_paths=get_data()

    #mean([129.5605, 135.2879, 148.4153])  and   std([58.6972, 62.4920, 63.5567])
    dataset=Patchdataset(label,file_paths,
    transform=transforms.Compose([
        transforms.Resize((128,384)),
        transforms.Normalize(([129.5605, 135.2879, 148.4153]),([58.6972, 62.4920, 63.5567]))
    ]))
    """
    trainload=DataLoader(dataset,1,shuffle=True)
    mean=torch.zeros(3)
    std=torch.zeros(3)
    for i,(x,y) in enumerate(trainload):
        for channel in range(3):
            mean[channel]+=x[:,channel,:,:].mean()
            std[channel]+=x[:,channel,:,:].std()
    mean=mean/6000
    std=std/6000
    print('{}  and   {}'.format(mean,std))
    """
    train_size=int(len(dataset)*0.8)
    text_size=len(dataset)-train_size
    train_set,text_set=random_split(dataset,[train_size,text_size])
    train_load=DataLoader(train_set,train_batch,shuffle=True)
    val_load=DataLoader(text_set,val_batch,shuffle=True)
    return train_load,val_load
if __name__=="__main__":
    train_load,val_load=data_load(64,64)
    for i,(x,y) in enumerate(train_load):
        if(i<3):
            print(x.shape)
            print(y.shape)
        else :
            break
