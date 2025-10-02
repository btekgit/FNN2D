# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:01:41 2024

@author: fbtek
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import math
from torchvision import datasets, transforms


class AdaptiveLocal2DLayer(nn.Module):
    ''' This is the pytorch implementation of ALC2D Layer.'''
    def __init__(self, input_size, output_size, device=None, dtype=None,
                 mu_init='spread', si_init=(0.1, 0.1), bias=True, channel_separate=False,
                 normed=True, layer_norm=False, n_embedding=None, 
                 shared_weights=False, activ = None):
        super(AdaptiveLocal2DLayer, self).__init__()

       
        self.in_channels = input_size[0]  # assuming PyTorch order.
        self.output_size = output_size  # this is also the 1D or 2D layout.
        self.bias = bias
        
        if len(output_size)==2:
            self.neurons = np.prod(output_size) # output is single channel
            self.output_channels = 1
            self.output_height = output_size[0]
            self.output_width = output_size[1]
        else:
            self.neurons = np.prod(output_size[1:]) # output is multi channel
            self.output_channels = output_size[0]
            self.output_height = output_size[1]
            self.output_width = output_size[2]
      
        #print("Hererere:",self.neurons)
        # Store device properly as an instance variable
        self.device = device if device is not None else torch.device('cpu')
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        train_mus = True
        train_sis = True
        if mu_init == 'spread':
            # Initialize mu values in a regular 2D grid with some margin from borders
            margin = 0.2  # Define your margin here
            grid_y, grid_x = np.meshgrid(np.linspace(margin, 1 - margin, self.output_height),
                                          np.linspace(margin, 1 - margin, self.output_width))
            self.mu_x = nn.Parameter(torch.tensor(grid_x.flatten(), **self.factory_kwargs), requires_grad=train_mus)
            self.mu_y = nn.Parameter(torch.tensor(grid_y.flatten(), **self.factory_kwargs), requires_grad=train_mus)
        elif mu_init == 'middle':
            # Initialize all mu's in the center of the 2D input map
            center_x = 0.5 * np.ones(self.neurons)
            center_y = 0.5 * np.ones(self.neurons)
            self.mu_x = nn.Parameter(torch.tensor(center_x, **self.factory_kwargs),requires_grad=train_sis)
            self.mu_y = nn.Parameter(torch.tensor(center_y, **self.factory_kwargs),requires_grad=train_sis)
        else:
            raise ValueError("Invalid value for mu_init. Choose 'spread' or 'middle'.")

        # Initialize sigma with the given value
        if type(si_init)=='float':
            si_init = list([si_init,si_init])
            
       
 
        self.sigma_x = nn.Parameter(torch.full((self.neurons,), si_init[0], **self.factory_kwargs),requires_grad=train_sis)
        self.sigma_y = nn.Parameter(torch.full((self.neurons,), si_init[1], **self.factory_kwargs),requires_grad=train_sis)

        self.device = device    
        
           
        #prepare grid for forward function
        self.normed = normed
        self.layer_norm = layer_norm
        self.input_height = input_size[1]
        self.input_width = input_size[2]
        
        
        # Grid coordinates should be buffers (not parameters)
        x = torch.linspace(0., 1., self.input_width, **self.factory_kwargs)
        y = torch.linspace(0., 1., self.input_height, **self.factory_kwargs)
        self.register_buffer('x', x.view(1, -1, 1))  # [1, W, 1]
        self.register_buffer('y', y.view(-1, 1, 1))  # [H, 1, 1]
        
        
        # Initialize weights with normal distribution and adjust standard deviation
        self.channel_separate = channel_separate
        masks = self.calc_U()
        
        #print(masks.shape)
        
        # mask sum(U^2) must be fan_in
        if self.channel_separate:
            # sum over spatial dims for each channel
            sq_sum = masks.square().sum(dim=(1,2))    # shape: (in_channels,)
            expected = torch.full_like(sq_sum, self.input_height * self.input_width*1.0)
        else:
            # sum over all dims
            sq_sum = masks[:,:,:,0].square().sum().to(torch.float)             # scalar
            expected = torch.tensor(self.in_channels * self.input_height * self.input_width * 1.0,dtype=torch.float)
        
        # compare
        #assert torch.allclose(sq_sum, expected, atol=1e-5), \
        #       f"mask norm² {sq_sum} != fan_in {expected}"
        
        
        if shared_weights:
            num_weights = 1
        else:
            num_weights = self.neurons
            
        # determine weight‐tensor shape
        if self.channel_separate:
            w_shape = (self.input_height, self.input_width, num_weights)
            b_shape = (self.in_channels,)
        else:
            w_shape = (self.in_channels, self.input_height, self.input_width, num_weights)
            b_shape = (1,)
        
        if self.output_channels>1:
            w_shape = w_shape + (self.output_channels,)
            b_shape = (self.output_channels,)

        # 1) allocate an uninitialized tensor of the right shape & dtype
        self.weights = nn.Parameter(
            torch.empty(w_shape, **self.factory_kwargs),
            requires_grad=True
        )
        
        self.bias= nn.Parameter(torch.zeros(b_shape,**self.factory_kwargs),
        requires_grad=True
        )

        # 2) choose a gain (for ReLU, gain = √2)
        gain = init.calculate_gain('relu')
        

        # 3a) Xavier‐uniform (balances forward & backward)
        eff_fan_in = torch.sum(self.calc_U()**2)  # after your mask‐normalization
        fan_out    = self.neurons
        bound = gain * math.sqrt(6.0 / (eff_fan_in + fan_out))
        init.uniform_(self.weights, -bound, bound)
        
        self.n_embedding = n_embedding
        if n_embedding is not None:
            # Normalize the (C + 4)-dimensional feature vector
            self.pos_norm = nn.LayerNorm(self.output_channels + 4, elementwise_affine=True).to(device)

            
            # Linear map from (C + 4) → n_embedding
            self.pos_emb = nn.Linear(self.output_channels + 4, n_embedding, bias=False).to(device)
            
            # Xavier init (since no activation is applied after this linear layer)
            nn.init.xavier_uniform_(self.pos_emb.weight, gain=1.0)
        
        # Layer norm for the main outputs (unchanged)
        self.out_norm = nn.LayerNorm(
            (self.output_channels, self.output_height, self.output_width),
            elementwise_affine=True
        ).to(device)
        self.out_bnorm2d = nn.BatchNorm2d(self.output_channels).to(device)
        # Layer norm for the main outputs (unchanged)
        self.out_w_pos_norm = nn.LayerNorm(
            (self.output_channels+4, self.output_height, self.output_width),
            elementwise_affine=True
        ).to(device)
        
        self.activation = None
        if activ:
            self.activation = activ
            print("layer applied its own activation because it is siimpleer to do it here. ") 
        
        #-------------------------------------------------------------------

    def calc_U(self):
        # Compute the Gaussian mask for each neuron
    
        gauss_y = torch.exp(-0.5 * ((self.y - self.mu_y.view(1, 1, -1)) / self.sigma_y.view(1, 1, -1))**2)
        gauss_x = torch.exp(-0.5 * ((self.x - self.mu_x.view(1, 1, -1)) / self.sigma_x.view(1, 1, -1))**2)
        gauss_mask = gauss_x * gauss_y
        if self.in_channels>1:
            gauss_mask = gauss_mask.unsqueeze(0).repeat(self.in_channels, 1, 1, 1)
            #gauss_mask = gauss_mask[None,:,:,:]
        else:
            gauss_mask = gauss_mask.unsqueeze(0)
            
        #print("Size",gauss_mask.size())

        if self.normed:
            if self.channel_separate:
                mask_normed = gauss_mask / torch.sqrt(torch.sum(torch.square(gauss_mask), dim=(1, 2), keepdim=True))
                #
                mask_normed *= torch.sqrt(torch.tensor(self.input_height * self.input_width, device=self.device))
                mask = mask_normed
            else:
                mask_normed = gauss_mask / torch.sqrt(torch.sum(torch.square(gauss_mask), dim=(0, 1, 2), keepdim=True))
                #print("Check this")
                mask_normed *= torch.sqrt(torch.tensor(self.input_height * self.input_width * self.in_channels, device=self.device))
                mask = mask_normed
        else:
            mask = gauss_mask
        return mask
    
    
    
    def forward(self, inputs, training=False):
        if not self.device:
            device = inputs.device
        else:
            device = self.device
    
        batch_size, channels, height, width = inputs.size()
        
        #calculate gaussian masks. 
        mask = self.calc_U()
      
        # remove this after debugging. 
        self.gaussian_masks = mask  # save for plotting
        
        #print("Mask and weights:", mask.shape,self.weights.shape)
        #if self.channel_separate:
        #    mask = mask.squeeze()
        if self.output_channels>1:
            mask = mask.unsqueeze(4)
        
        # calculate effective weights            
        weighted_mask = mask * self.weights
        #print("inputs dim", inputs.shape)
        inputs_dtype = inputs.dtype
        weighted_mask = weighted_mask.to(device=inputs.device, dtype=inputs_dtype)
        #print("weighted mask", weighted_mask.shape, "inputs size", inputs.shape)
        if self.channel_separate:
            outputs = torch.tensordot(inputs, weighted_mask, dims=([2,3], [1,2]))
            #print("outputs mask", outputs.shape)
            print("separate channel Does NOT WORK")
            outputs = outputs.reshape((inputs.shape[0],inputs.shape[1]*self.output_channels,
                                       self.output_height, self.output_width)) # add channel dim.
        else:
            outputs = torch.tensordot(inputs, weighted_mask, dims=([1,2,3], [0,1,2]))
            outputs = outputs.reshape((inputs.shape[0], self.output_channels,self.output_height,
                                       self.output_width)) # add channel dim.
        
        
        B, C, H, W = outputs.shape
        
        if self.layer_norm:
            #outputs = self.out_norm(outputs) + self.bias.reshape(1,C,1,1)
            outputs = self.out_bnorm2d(outputs +self.bias.reshape(1,C,1,1))
            #print(outputs.shape, self.output_size)
            #outputs = (outputs.reshape((inputs.shape[0],*self.output_size)).unsqueeze(1)) # add channel dim.
        if self.activation:
            outputs = nn.functional.relu(outputs)
        # if embedding is disabled, return exactly the original output
        if self.n_embedding is None:
            return outputs
        
        
        # Prepare position features (shape [B, C + 4, H, W])
        # After concatenation: pos_features shape = [B, C+4, H, W]
        output_w_pos = torch.cat([
            outputs,  # [B, C, H, W]
            self.mu_x.view(1, 1, H, W).expand(B, 1, H, W),
            self.mu_y.view(1, 1, H, W).expand(B, 1, H, W) ,
            self.sigma_x.view(1, 1, H, W).expand(B, 1, H, W),
            self.sigma_y.view(1, 1, H, W).expand(B, 1, H, W)
        ], dim=1).to(device=inputs.device, dtype=inputs_dtype)  # [B, C+4, H, W]
        
        #out_w_norm = self.out_w_pos_norm(pos_features)
        return output_w_pos



    def stats(self, functions):
        stats_dict = {}

        # Apply the functions to the trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats_dict[name] = {}
                for func in functions:
                    func_name = func.__name__
                    stats_dict[name][func_name] = func(param.detach().cpu().numpy())

        return stats_dict
    
    def apply_constraints(self):
        """Apply constraints to the parameters to keep them within the required range."""
        with torch.no_grad():
            # Ensure mu_x and mu_y stay in range [0, 1]
            MIN_MU = 0.0
            MAX_MU = 1.0
            MIN_SI = 1.0/float(max(self.input_height, self.input_width))
            MAX_SI = 2.0
            self.mu_x.data.clamp_(MIN_MU, MAX_MU)
            self.mu_y.data.clamp_(MIN_MU, MAX_MU)

        
            self.sigma_x.data.clamp_(MIN_SI, MAX_SI)
            self.sigma_y.data.clamp_(MIN_SI, MAX_SI)

    def visualize_gaussian_masks(self):
        #print(self.gaussian_masks.size())
        masks = self.gaussian_masks.detach().cpu()

        num_neurons = masks.size(2)  # Get the number of neurons
        height, width = masks.size(0), masks.size(1)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 5))
        #print("Num neurons:", num_neurons)
        for i in range(0,min(num_neurons, 12),1):
            plt.subplot(1, min(num_neurons, 12), i + 1)
            gmask =masks[0,:, :, i].numpy() # plots single channel only
            #print(gmask.shape)
            plt.imshow(gmask, cmap='viridis')
            plt.title(f'Neuron {i + 1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(15, 5))
        #print("Num neurons:", num_neurons)
        for i in range(0,min(num_neurons, 12),1):
            plt.subplot(1, min(num_neurons, 12), i + 1)
            gmask =masks[0,:, :, i].numpy() #plots single channel only
            #print(gmask.shape)
            plt.imshow(gmask, cmap='viridis')
            plt.title(f'Neuron {i + 1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # edited until here F.B.T. April 7 2024
   



if __name__ == '__main__':
    #main()
    test_imagenet= False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if test_imagenet:
        
        output_size = (10,7,7)
        # Generate a dummy input tensor for testing without MNIST
        batch_size, channels, height, width = 4, 3, 224, 224
        adaptive_layer = AdaptiveLocal2DLayer((channels, height, width), output_size, 
                                                      channel_separate=True, 
                                                      n_embedding=None, 
                                                      shared_weights=True,
                                                      device=device)
        
    
    else:
            
        output_size = (3,40,50)
        # Generate a dummy input tensor for testing without MNIST
        batch_size, channels, height, width = 4, 3, 80, 60
        adaptive_layer = AdaptiveLocal2DLayer((channels, height, width), output_size, 
                                                      channel_separate=False, 
                                                      n_embedding=5, 
                                                      shared_weights=True,
                                                      device=device)
        
    # Create input with known statistics
    input_mean = 0.0
    input_std = 1.0
    dummy_input = torch.randn(batch_size, channels, height, width, device=device) * input_std + input_mean
    
    # Calculate input statistics
    input_variance = torch.var(dummy_input)
    input_mean = torch.mean(dummy_input)
    
    print(f"\nInput statistics:")
    print(f"Mean: {input_mean.item():.4f}")
    print(f"Variance: {input_variance.item():.4f}")
    print(f"Std: {input_std:.4f} (theoretical)")
    
    # Forward pass
    output = adaptive_layer(dummy_input, training=True)
    
    # Calculate output statistics
    output_variance = torch.var(output)
    output_mean = torch.mean(output)
    
    print(f"\nOutput statistics:")
    print(f"Mean: {output_mean.item():.4f}")
    print(f"Variance: {output_variance.item():.4f}")
    
    # Variance ratio (how much variance is preserved)
    variance_ratio = output_variance / input_variance
    print(f"\nVariance ratio (output/input): {variance_ratio.item():.4f}")
    
    # Optional: Print per-channel statistics
    if output.dim() == 4:  # [batch, channels, height, width]
        print("\nPer-channel output variance:")
        for c in range(output.size(1)):
            chan_var = torch.var(output[:, c])
            print(f"Channel {c}: {chan_var.item():.4f}")
    
    adaptive_layer.visualize_gaussian_masks()
    
    # Print the input and output shapes
    print("\nShape information:")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    
    gausspool_stats = adaptive_layer.stats([np.max, np.mean, np.min])
    print("\nLayer parameter statistics:", gausspool_stats)