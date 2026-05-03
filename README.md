# FNN2D

This is the code repository of 2D Adaptive Local Connected Neuron Paper.
The newest code is written in pytorch. 
AdaptiveLocal2DLayerv2.py
- If you run this file, you just test a simple adaptive locally connected 2D layer. 
- If you run train_mnist you will train and test adaptive locally connected 2D layer in a simple 2 layer network configuration. 
- Code is in progress will be updated soon.

Some earlier code has tensorflow base.
Code is based on Tensorflow 2+ and Keras. 
utility code is also supplied. 

To try our code simply run the main file. 
Focus2D_tf2_v2.py
It will create a simple 2 layer network and train on MNIST. 


1 May 2026

- I have included new pytorch implementation of AdaptiveLocal2DLayerv2.py
This should run and test layer with a random input. 

- I have included imagenet_trainer code inet_trained_alc2d.py, sorry this code is chaotic tried lots of stuff. But it is running to create resnet_w_alc.py and vit_w_alc.py 

- run_spatial_benchmarks.py is the newest code which compares alc2d with deformconv and coordconv in a simple convolutional network. 



We will provide more information in the future. 

