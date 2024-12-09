import pandas as pd
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as pt
import os 
import torch
import torch.nn as nn
import torch.optim as optimizer
from torchvision.models import googlenet

model = googlenet(pretrained=True, aux_logits=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels_path = 'data/cifar-10/trainLabels.csv'
train = 'data/cifar-10/train'
imgsz = 224  # Image size (height and width)
image_no = 50000
0000  # Maximum number of images to process
train_path = 'data/cifar-10/train/'  # Path to training images
count = 0
images = []

labels_data = pd.read_csv(labels_path)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('\n\nairplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
encoded_labels = pd.get_dummies(labels_data['label'], columns=labels).astype(np.int8)
encoded_labels = np.array(encoded_labels)
print(labels_data.head())
labels = np.array(encoded_labels)


for i in range(1,image_no+1):
    if count >= image_no:  # Stop if the desired number of images is reached
        break
    img_path = os.path.join(train_path,  f"{i}.png")
    # Read the image in color
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    rs_img = cv.resize(img, (imgsz, imgsz))
    # Store the resized image in the array
    images.append(rs_img)
    count += 1
# Trim the `images` array to the actual number of processed images

print(f"Images loaded and resized successfully! Total images processed: {count}")
images = np.array(images)

batch_sz = 60
num_batches = len(images) // batch_sz
batch_img = []
batch_lab = []

for h in range(num_batches):
    start = h * batch_sz
    end = (h + 1) * batch_sz
    batch = images[start:end]
    batch_l = labels[start:end]
    batch_img.append(batch)
    batch_lab.append(batch_l)
batch_img = np.array(batch_img)
#batch_img = torch.tensor(batch_img).to('cuda')
batch_lab = np.array(batch_lab)
#batch_lab = torch.tensor(batch_lab).to('cuda')
num_batches

model.fc = nn.Linear(model.fc.in_features , 10)
model.to(device)
optimizers = optimizer.Adam(model.parameters() ,lr=0.001)
loss_function = nn.CrossEntropyLoss()


def train():
    for e in range (epochs):
        total = 0
        optimizers.zero_grad()
        print('Starting epoch no ',e+1)
        for i in range(num_batches):
            optimizers.zero_grad()
            bat_img = torch.tensor(batch_img[i] , dtype = torch.float32 ).permute(0 ,3 ,1 ,2).to(device)
            bat_lab = torch.tensor(batch_lab[i] ,dtype = torch.long).argmax(dim=1).to(device)
            out = model(bat_img)
            
            
            if isinstance(out, tuple):
                main_output, aux1_output, aux2_output = out
                loss_main = loss_function(main_output, bat_lab)
                loss_aux1 = loss_function(aux1_output, bat_lab)
                loss_aux2 = loss_function(aux2_output, bat_lab)

                loss = (1 - 0.3)*loss_main + 0.3 *( loss_aux1  +  loss_aux2) # Weighted sum of losses
            else:
                loss = loss_function(out, batch_lab)
          
            loss.backward()
            optimizers.step()
            total += loss
        print('Loss per epoch ',total/ num_batches)

epochs = 20
results = train()

torch.save(model, "inceptionet_10.pth")


