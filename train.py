import torch
import numpy as np
import data_process as dp
import os
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.cuda.FloatTensor')

lfw_dir = '../../../../Courses_data'
lfw_dataset_dir = '../../../../Courses_data/lfw'
train_set_path = os.path.join(lfw_dir, 'LFW_annotation_train.txt')
max_epochs = 80
learning_rate = 0.0001
pretrained = True
str_pre = 'pre'
file_name = 'lfw_alexnet_'+str(learning_rate)+'_'+str(max_epochs)+'_'+str_pre

train_list = dp.get_list(train_set_path)
valid_list = train_list[-2000: ]
train_list = train_list[: -2000]
transform = ['flip', 'rcrop']
train_dataset = dp.LFWDataSet(train_list, transform=transform)
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=4,
                                                shuffle=False,
                                                num_workers=0)
print('train items:', len(train_dataset))

valid_dataset = dp.LFWDataSet(valid_list, transform=transform)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=0)
print('validation items:', len(valid_dataset))

## visualize some data
idx, (img, lm) = next(enumerate(train_data_loader))
nd_img = img.cpu().numpy()
nd_lm = lm.cpu().numpy()
dp.show_landmarks(nd_img, nd_lm)

## load alexnet
net = models.alexnet(pretrained=pretrained)
net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 14),
        )
net.cuda()

criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

train_losses = []
valid_losses = []
itr = 0
for epoch_idx in range(0, max_epochs):
    for train_batch_idx, (train_input, train_lm) in enumerate(train_data_loader):
       
        itr += 1
        net.train()
    
        optimizer.zero_grad()

        train_input = torch.transpose(train_input, 1, 3)
        train_input = Variable(train_input.cuda())
        train_out = net.forward(train_input)

        train_lm = Variable(train_lm.cuda())
        loss = criterion(train_out.view((-1, 2, 7)), train_lm)
        
        loss.backward()
        optimizer.step()
        train_losses.append((itr, loss.item()))
        
        if train_batch_idx % 200 == 0:
            print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))
            net.eval() 
            valid_loss_set = []
            valid_itr = 0

            for valid_batch_idx, (valid_input, valid_lm) in enumerate(valid_data_loader):
                net.eval()
                valid_input = torch.transpose(valid_input, 1, 3)
                valid_input = Variable(valid_input.cuda())
                valid_out = net.forward(valid_input)

                valid_lm = Variable(valid_lm.cuda())
                valid_loss = criterion(valid_out.view((-1, 2, 7)), valid_lm)
                valid_loss_set.append(valid_loss.item())

                valid_itr += 1
                if valid_itr > 5:
                    break
            
            avg_valid_loss = np.mean(np.asarray(valid_loss_set))
            print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, avg_valid_loss))
            valid_losses.append((itr, avg_valid_loss))

train_losses = np.asarray(train_losses)
valid_losses = np.asarray(valid_losses)
plt.plot(train_losses[:, 0],
         train_losses[:, 1])
plt.plot(valid_losses[:, 0],
         valid_losses[:, 1])
# plt.show()
plt.savefig(file_name+'.jpg')

net_state = net.state_dict()
torch.save(net_state, os.path.join(lfw_dir, file_name+'.pth'))
