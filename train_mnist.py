# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:45:42 2024

@author: fbtek
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR, ReduceLROnPlateau
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchinfo import summary

from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np


from AdaptiveLocal2DLayerv2 import AdaptiveLocal2DLayer  # your implementation

def get_parameter_groups(model, lr_mu, lr_sigma, lr_weights, lr_other):
    mu_params     = []
    sigma_params  = []
    weight_params = []
    other_params  = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'mu_' in name:
            mu_params.append(p)
        elif 'sigma_' in name:
            sigma_params.append(p)
        elif 'weights' in name:
            weight_params.append(p)
        else:
            other_params.append(p)
    return [
        {'params': mu_params,     'lr': lr_mu, 'weight_decay': 0.0, 'name':'mu'},
        {'params': sigma_params,  'lr': lr_sigma,   'weight_decay': 1e-10,'name':'sigma'},
        {'params': weight_params, 'lr': lr_weights, 'weight_decay': 0.0,'name':'weight'},
        {'params': other_params,  'lr': lr_other,   'weight_decay': 0.0,'name':'other'},
    ]


import torch.nn.functional as F


class SimpleFNN(nn.Module):
    def __init__(self):
        super(SimpleFNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 28*28)
        self.fc2 = nn.Linear(28*28, 28*28)
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.5)
        #self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(64*7*7, 10)  # Should be 128 -> 10 if using fc1
        #self.fc2 = nn.Linear(64, 10)  # Should be 128 -> 10 if using fc1

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        #x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.globpool(x) globpool does not train well!
        x = x.view(-1, 64 * 7 * 7)
        #x = x.view(-1, 64)
        # x = F.relu(self.fc1(x))  # Uncomment to use fc1 (requires adjusting fc2)
        x = self.drop(x)
        x = self.fc2(x)
        return x
    

class PreAdaptiveCNN(nn.Module): # does not work greate
    def __init__(self,input_size, hidden_sizes, output_size):
        super(PreAdaptiveCNN, self).__init__()
        init_si = (0.03,0.03) #0.1 was also ok1
        self.layer1     = AdaptiveLocal2DLayer(input_size, hidden_sizes[0],
                                               channel_separate=False,
                                               normed=True, layer_norm=False,
                                               n_embedding=None,
                                               si_init=init_si,
                                               shared_weights=False, activ='relu')
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(0.5)
        #self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(64*7*7, 10)  # Should be 128 -> 10 if using fc1
        #self.fc2 = nn.Linear(64, 10)  # Should be 128 -> 10 if using fc1

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(F.relu(self.conv1(x)))
        #x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.globpool(x) globpool does not train well!
        x = x.view(-1, 64 * 7 * 7)
        #x = x.view(-1, 64)
        # x = F.relu(self.fc1(x))  # Uncomment to use fc1 (requires adjusting fc2)
        x = self.drop(x)
        x = self.fc2(x)
        return x
class TwoLayerNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        N_embedding= None
        #N_embedding = 1 # I could not train embedding version in torhc
        layer1_out_channels = hidden_sizes[0][0] + 4 * int(N_embedding!=None)
        layer2_out_channels = hidden_sizes[1][0] + 4 * int(N_embedding!=None)
        init_si = (0.03,0.03) #0.1 was also ok1
        if N_embedding is not None and N_embedding>1: 
            layer1_out_channels = N_embedding
            layer2_out_channels = N_embedding
        self.layer1     = AdaptiveLocal2DLayer(input_size, hidden_sizes[0],
                                               channel_separate=False,
                                               normed=True, layer_norm=False,
                                               n_embedding=N_embedding,
                                               si_init=init_si,
                                               shared_weights=False, activ='relu')
        
        #print(" SEPARATE EMBEDDNG FROM THE CHANNEL OUTPUT, FIRST OUTPUT EVERTHING THENP PUT IT TO EMBEDDING I YOU LIKE. 1")
        
        self.activation1 = nn.ReLU()
        self.maxpool    = nn.MaxPool2d(2, stride=2)
        self.layer2     = AdaptiveLocal2DLayer((layer1_out_channels, 
                                                *np.array(hidden_sizes[0][1:])),
                                               hidden_sizes[1],
                                               channel_separate=False,
                                               normed=True, layer_norm=False,
                                               n_embedding=N_embedding,
                                               si_init=init_si,
                                               shared_weights=False, activ='relu')
        
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        
        
        
        self.activation2 = nn.ReLU()
        self.dropout0   = nn.Dropout(0.2)
        self.dropout1   = nn.Dropout(0.25)
        self.dropout2   = nn.Dropout(0.25)
        if N_embedding is None: N_embedding=1
        #print(layer2_out_channels,np.prod(hidden_sizes[1][2:]))
        #self.linear     = nn.Linear(layer2_out_channels*np.prod(hidden_sizes[1][1:]),output_size)
        self.linear     = nn.Linear(layer2_out_channels*np.prod(hidden_sizes[1][1:]),output_size)
        #self.linear     = nn.Linear(layer2_out_channels,output_size)

    def forward(self, x):
         # added the dropout to input. 
        #
        x = self.dropout1(x)
        x = self.layer1(x)
        x = self.dropout1(x)
        #print(x.size())
        # removed the nonlinearity 
        #x = self.activation1(x)
        #x = self.maxpool(x)
        x = self.layer2(x)
        
        
        #print(x.size())
        #x = self.activation2(x)
        #x = self.dropout2(x)
        #x = self.maxpool(x)
        #print(x.size())
        x = self.dropout2(x)
        #x = self.globpool(x) # global average almost untrainable?
        x = x.view(x.size(0), -1)
        return self.linear(x)

def train_model(model,
                train_loader,
                val_loader,
                lr_max,
                lr_mu,
                lr_sigma,
                lr_weights,
                lr_other,
                l1_lambda,
                epochs=20,
                device='cpu'
                ):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # build parameter groups with distinct LRs and L2 decays
    base_lrs = [lr_mu, lr_sigma, lr_weights, lr_other]
    param_groups = get_parameter_groups(model,
                                        *base_lrs)
    params = [p for group in param_groups for p in group['params']]
    #print(params)
    #optimizer = optim.SGD(param_groups, momentum=0.9)
    optimizer = optim.SGD(param_groups, momentum=0.9)
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = optim.Adam(param_groups) # sgd performs better
    #scheduler = ReduceLROnPlateau(optimizer,
    #                              mode='min',
    #                              factor=0.9,
    #                              patience=25,
    #                              verbose=True)

    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [],   'val_acc': []}
    batch_counter = 1
    viz_interval  = 500
    # for optimizer scheduler. 
    # midpoint = epochs / 2
    # sigma = (epochs / 2) * 0.4
    #lr_scales = np.concat(np.linspace(1.0, lr_max, int(midpoint)), 
    #                   np.maximum(1.0, np.linspace(lr_max, 1.0, int(midpoint/2))) )
    lr_scales = np.concatenate([np.ones(int(epochs/8))*lr_weights, 
                                np.linspace(lr_weights, lr_max*lr_weights, int(1*epochs/8)),
          np.linspace(lr_max*lr_weights, lr_weights, int(2*epochs/8)),
          np.linspace(lr_weights, 0.9*lr_weights, int(4*epochs/8)+2),
          ])
    
    #print("scales:", len(lr_scales), lr_scales)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        # apply learning rate scale to parameters. 
        # scale = lr_max * np.exp(-((epoch - midpoint) ** 2) / (sigma ** 2))  
        scale = lr_scales[epoch]
        print("scale:", scale)
        for group in optimizer.param_groups:
           if not any(sub in group.get('name', '').lower() for sub in ['mu', 'sigma']):
               #group['lr'] = compute_new_lr(epoch)  # Define your LR schedule
               group['lr'] = scale
               print(group.get('name', ''), group['lr'])
               #s = input("continue")
            #print("Group lr: ", group['lr'])
        
        for images, labels in pbar:
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # L1 penalty on Gaussian weights
            #weight_params = param_groups[2]['params']
            #l1_norm = sum(p.abs().sum() for p in weight_params)
            #loss = loss + l1_lambda * l1_norm
            loss.backward()
            
            # Clip gradients by value (e.g., between -1 and 1)  
            torch.nn.utils.clip_grad_value_(params, clip_value=1.0)

            optimizer.step()
           
            if 'layer1' in model.__dict__.get('_modules').keys():
                model.layer1.apply_constraints()
            
            if 'layer2' in model.__dict__.get('_modules').keys():
                
                model.layer2.apply_constraints()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total   += labels.size(0)
            correct += preds.eq(labels).sum().item()

            if batch_counter % viz_interval == 0:
                if 'layer1' in model.__dict__.get('_modules').keys():
                    grads = [p.grad.abs().mean().item()
                             for p in model.layer1.parameters()
                             if p.grad is not None]
                    #print(f"Mean grad magnitude: {np.mean(grads):.2e}")
                    with torch.no_grad():
                        if 'layer1' in model.__dict__.get('_modules').keys():
                            model.layer1.visualize_gaussian_masks()
                        else:
                            pass
                                
                plt.show()
            batch_counter += 1

            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
            #break

        val_loss, val_acc = evaluate(model, val_loader, device)
        #scheduler.step(val_loss)
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(100. * correct / total)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        current_lr = optimizer.param_groups[0]['lr']
        #print(f"Epoch {epoch}: LR = {current_lr:.2e}")
        print(f"Epoch {epoch}: LR scale = {scale:.6f}")

    return history

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    criterion  = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total   += labels.size(0)
            correct += preds.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    class ScaleAndShift01:
        def __init__(self, K=4.0):
            self.K = K
    
        def __call__(self, tensor):
            return (tensor * self.K)-self.K/2
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ScaleAndShift01()
    ])
    train_set = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
    test_set  = datasets.MNIST(root='./data', train=False,
                               transform=transform)
    train_loader = DataLoader(train_set, batch_size=128,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=32,
                              shuffle=False)

    batch = next(iter(train_loader))
    channels, height, width = batch[0].size()[1:]
    
    # Stack all images into a single tensor
    all_data = torch.stack([img for img, _ in train_set])  # Shape: [N, 1, 28, 28]
    
    mean = all_data.mean()
    std = all_data.std()
    min_val = all_data.min()
    max_val = all_data.max()
    
    print(f"Mean: {mean.item():.4f}")
    print(f"Std: {std.item():.4f}")
    print(f"Min: {min_val.item():.4f}")
    print(f"Max: {max_val.item():.4f}")
    num_outputs = 1 # more than 1 did not perform better in mnist
    model = TwoLayerNetwork((channels, height, width),
                              [(num_outputs, height, width), 
                              (num_outputs, height, width)],
                              output_size=10)
    
    # model = PreAdaptiveCNN((channels, height, width),
    #                           [(num_outputs, height, width), 
    #                           (num_outputs, height, width)],
    #                           output_size=10)
    #model = SimpleFNN()
    
    #print('layer1' in model.__dict__.get('_modules').keys())

    #model = SimpleTwoLayerConvNet(1,10)
    #model = SimpleCNN()
    
    s = summary(model, (1, channels, height,width))
    print(s)

    # pass custom learning rates here
    history = train_model(model,
                          train_loader,
                          test_loader,
                          lr_max=1.0, 
                          lr_mu=1e-2, #lr multipliers
                          lr_sigma=1e-2,
                          lr_weights=1e-2, 
                          lr_other=1e-2,
                          l1_lambda=0,
                          epochs=200,
                          device=device)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'],   label='val_loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
