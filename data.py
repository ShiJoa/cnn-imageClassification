import torch

batch_size=8

def load_image(path):
    import cv2
    import numpy as np
    import os

    data = []
    flag = []
    for fileName in os.listdir(path):
        if not fileName.endswith(".jpg"):
            continue

        #用opencv读取图片
        x=cv2.imread(path+"/"+fileName)

        #数值压缩到0-1之间
        x=torch.FloatTensor(np.array(x))/255

        #变形，使通道数在前面
        x=x.permute(2,0,1)

        #标签
        y=int(fileName[0])

        data.append(x)
        flag.append(y)

    return data,flag

#读取数据
data,flag=load_image("data\cifar10")


#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(data)
    def __getitem__(self, item):
        return data[item],flag[item]

#建立数据集
dataset=Dataset()


#定义加载器
loader=torch.utils.data.DataLoader(dataset=dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)