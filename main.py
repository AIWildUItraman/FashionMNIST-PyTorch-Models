import resnet4
import torch

from torch import nn, optim
import time
import torchvision
from torch.utils.data import DataLoader



def load_data_fashion_mnist(batch_size, resize=None, root='/home/mxs/Deepleaning/FashionMNIST-PyTorch-Models/FashionMNIST-PyTorch-Models/datasets'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    # 图像增强
    transform = torchvision.transforms.Compose(trans)
    
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # If device is the GPU, copy the data to the GPU.
        X, y = X.to(device), y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            # [[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item() / n

def train(net, train_iter, test_iter, criterion, epochs, device, lr=0.001):
    print('Training on ', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    best_test_acc = 0.92
    for epoch in range(epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            net.train()
            optimizer.zero_grad()  # 清空梯度
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, device)  
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model_pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), '/home/mxs/Deepleaning/FashionMNIST-PyTorch-Models/FashionMNIST-PyTorch-Models/model_pth/best_{}.pth'.format(test_acc))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
net = resnet4.Resnet_4()
lr, num_epochs = 0.002, 20
criterion = nn.CrossEntropyLoss()   #交叉熵描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近
train(net, train_iter, test_iter, criterion, num_epochs, device, lr)