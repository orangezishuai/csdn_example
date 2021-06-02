import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary 
from getdata import GetData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




# 冻结网络层的参数
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.required_grad = False

# VGG网络结构定义
class VGG16net(nn.Module):
    def __init__(self, feature_extract = True, num_class = 2):
        super(VGG16net, self).__init__()
        model = models.vgg16(pretrained = True)
        self.features = model.features
        set_parameter_requires_grad(self.features, feature_extract)
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 512),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512, 128),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(128, num_class),

            # 
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # 改变tensor形状，拉伸成一维
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

# 画图函数
def plt_image(x_input, y_input, title, xlabel, ylabel):
    plt.plot(x_input, y_input, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def main():
    # GPU选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG16net().to(device) 

    # 关键参数设置
    learning_rate=0.001
    num_epochs = 1        
    train_batch_size = 16
    test_batch_size = 16

    # 优化器设置            
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # 加载数据集
    train_dataset = GetData('./data', 224, 'train')
    test_dataset = GetData('./data', 224, 'test')

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    # 画图需要的参数
    epochs = []
    evaloss = []
    acc = []

    # 打印模型结构
    backbone = summary(model, (3, 224, 224))

    for epoch in range(num_epochs):
        epochs.append(epoch+1)
        # train过程
        
        total_step = len(train_loader)
        train_epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()

            # 加载标签与图片
            images = images.to(device)
            labels = labels.to(device)

            # 前向计算
            output = model(images)
            loss = criterion(output, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 累加每代中所有步数的loss    
            train_epoch_loss += loss.item()

            # 打印部分结果
            if (i + 1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            if (i + 1) == total_step:
                epoch_eva_loss = train_epoch_loss / total_step
                evaloss.append(epoch_eva_loss)
                print('Epoch_eva loss is : {:.5f}'.format(epoch_eva_loss))

        # test过程
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                print(predicted)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc.append(100*(correct/total))
            print('Test Accuracy  {} %'.format(100*(correct/total)))

    # print(model.state_dict())
    torch.save(obj = model.state_dict(), f='model/model.pth')
    # 训练结束后绘图
    plt_image(epochs, evaloss, 'loss', 'Epochs', 'EvaLoss')
    plt_image(epochs, acc, 'ACC', 'Epochs', 'EvaAcc' )
if __name__ == "__main__" :
    main()

