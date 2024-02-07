import torch
from data import loader

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

#测试
@torch.no_grad()
def test():
    model=torch.load("cnn.model")
    
    model.to(device)

    model.eval()

    correct=0
    total=0
    for i in range(100):
        x,y=next(iter(loader))


        #GPU
        x=x.to(device)
        y=y.to(device)


        out=model(x).argmax(dim=1)

        correct+=(out==y).sum().item()
        total+=len(y)

    print(correct/total)

test()
