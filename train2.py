import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR

# 定义数据增强和预处理
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_data = datasets.ImageFolder('New Masks Dataset/Train', transform=train_transforms)
test_data = datasets.ImageFolder('New Masks Dataset/Test', transform=test_transforms)
# val_data = datasets.ImageFolder('New Masks Dataset/Validation', transform=test_transforms)

# 定义数据集加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
# val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# 加载预训练的ResNet18模型
resnet18 = models.resnet18(pretrained=True)

# 冻结所有卷积层的参数
for param in resnet18.parameters():
    param.requires_grad = False

# 替换最后一层全连接层
resnet18.fc = nn.Linear(in_features=resnet18.fc.in_features, out_features=2)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)


# 将模型移动到GPU设备（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)

# 调整学习率
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

print(resnet18)

# 训练模型
num_epochs = 10
# best_val_acc = 0.0
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 输出训练损失和准确率
        running_loss += loss.item()
        if i % 10 == 9:
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = resnet18(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('[Epoch %d, Batch %5d] loss: %.3f accuracy: %.2f %%' %
                  (epoch + 1, i + 1, running_loss / 10, 100 * correct / total))
            running_loss = 0.0
            # update the learning rate
            scheduler.step()

# 保存模型
torch.save(resnet18.state_dict(), 'resnet18_mask_detection.pth')
print("模型已保存")

'''
            # 输出训练损失
running_loss += loss.item()
    if i % 100 == 99:  # 每 100 个小批次输出一次损失
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 100))
        running_loss = 0.0

# 在验证集上计算模型的准确率
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = 100 * correct / total
print('Validation accuracy: %d %%' % val_acc)

# 如果在验证集上获得了更好的准确率，就保存模型
if val_acc > best_val_acc:
    torch.save(resnet18.state_dict(), 'resnet18_mask_detection.pth')
    best_val_acc = val_acc
'''
