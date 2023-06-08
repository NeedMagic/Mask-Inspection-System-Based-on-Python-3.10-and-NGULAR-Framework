import multiprocessing as mp

if __name__ == '__main__':
    try:
        mp.get_context('spawn')
    except RuntimeError:
        pass

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

# Define the YOLOv5 model
class YOLOv5(nn.Module):
    def __init__(self):
        super(YOLOv5, self).__init__()

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def forward(self, x):
        output = self.model(x)
        return output

# Define the custom dataset for the mask detection task
class MaskDetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = []
        self.labels = []

        for image_name in os.listdir(root):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                self.images.append(os.path.join(root, image_name))

                # Extract the label from the image filename
                if 'mask' in image_name.lower():
                    self.labels.append([1, 0])
                else:
                    self.labels.append([0, 1])

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.images)

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the transformation for the training and validation set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = MaskDetectionDataset('./images/train', transform)
val_dataset = MaskDetectionDataset('./images/val', transform)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    def convert_target_to_dict(target):
        return {
            'boxes': torch.as_tensor(target['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(target['labels'], dtype=torch.int64)
        }
    transforms.append(T.Lambda(lambda image, target: (image, convert_target_to_dict(target))))
    return T.Compose(transforms)

# add the following code to fix the shape of the target tensor
def fix_target_shape(batch):
    images, targets = zip(*batch)
    targets = [{k: v for k, v in t.items()} for t in targets]
    for t in targets:
        t['boxes'] = torch.as_tensor(t['boxes'], dtype=torch.float32)
        t['labels'] = torch.as_tensor(t['labels'], dtype=torch.int64)
    return images, targets

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=fix_target_shape)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=fix_target_shape)

# Define the model, loss function, and optimizer
model = YOLOv5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        print(inputs.shape, labels.shape)  # add this line
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch [%d/50], Loss: %.4f' % (epoch + 1, running_loss / len(train_loader)))

# Evaluate the model on the validation set
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the validation set: %.2f %%' % (100 * correct / total))
