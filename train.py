import torch
from model import CNNModule
from data import loader

#系统变量
device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
epochs=10

model=CNNModule()
model.to(device)

def train():
    optimizer= torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fu=torch.nn.CrossEntropyLoss()
    loss_fu.to(device)
    model.train()


    for epoch in range(epochs):
        for i,(x,y) in enumerate(loader):

            #数据移到GPU
            x=x.to(device)
            y=y.to(device)

            #forward
            out=model(x)
            loss=loss_fu(out,y)

            #backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i% 2000==0:
                acc= (out.argmax(dim=1)==y).sum().item()/len(y)
                print(epoch,i,loss.item(),acc)

    torch.save(model,'cnn.model')

train()

