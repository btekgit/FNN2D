# -*- coding: utf-8 -*-
import os
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if not os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH'):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Lambda, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, GaussianNoise
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device Details: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install Tensorflow that supports GPU")
#physical_devices = tf.config.list_physical_devices('GPU') 
#for gpu_instance in physical_devices: 
#    tf.config.experimental.set_memory_growth(gpu_instance, True)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow_addons.image import random_cutout

from Focus2D_tf2_v2 import FocusedLayer2D
from Kfocusingtf2 import FocusedLayer1D

from keras_utils_tf2 import ClipCallback

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
#from RandomCutout import RandomCutout

import numpy as np

def make_grid(t, num_images, num_rows=2):
    '''takes stack of images as (batch, w, h, num_images) and tiles them into a grid'''
    #t = tf.squeeze(t) # remove single batch, TODO make more flexible to work with higher batch size
    t = tf.unstack(t, num=num_images, axis=-1) # split last axis (num_images) into list of (h, w)
    t = tf.concat(t, axis=2) # tile all images horizontally into single row
    t = tf.split(t, num_rows, axis=2) # split into desired number of rows
    t = tf.concat(t, axis=1) # tile rows vertically
    if len(t.shape)==3:
        t = t[:,:,:,tf.newaxis]
    return t

def build_model(settings={},verbose=True):
    data_augmentation = tf.keras.Sequential([
            ##RandomCutout((8,8)),  this is slow, slightly reduces overfit. 
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1,width_factor=0.1, fill_mode='constant', interpolation='nearest'),
            ])
    
    if settings.dset=="cifar10":
        input_shape = (32,32,3)
        dset =  cifar10
        img_rows, img_cols,n_channels = input_shape  # 32,32
        (x_train, y_train), (x_test, y_test) = dset.load_data()
        #(x_test, y_test), (x_train, y_train) = dset.load_data()
        num_classes = 10 
    elif settings.dset=='tiny':
        input_shape = (64,64,3)
        dset = 'tiny'
        num_classes = 100 #200#00
        #path='/home/btek/datasets/tiny/tiny-imagenet-200'
        path='/home/fbtek/datasets/tiny-imagenet-200'
        #path='/home/btek/datasets/tiny/tiny-imagenet-200'
        img_rows, img_cols, n_channels = input_shape  # 32,32
        from load_images import load_images
        x_train, y_train, x_test, y_test = load_images(path, num_classes)
        
        
        
    inputs = tf.keras.Input(shape=input_shape)
    #MAXCUT=10
    #mask_size = tf.random.uniform([],1,5,dtype=tf.dtypes.int32)*2
    #print(mask_size)
    
    #
    
    if settings.augment:
        
        x = data_augmentation(inputs)
    
        x = preprocess_input(x)
    else:
        x = preprocess_input(inputs)
    use_imagenet_weights =  settings.use_imagenet_weights #False

    if use_imagenet_weights:
        base_model = ResNet50V2(input_shape=input_shape, weights='imagenet', include_top=False, input_tensor=x)
        print("....Using ImageNet Weights....")
    else:
        base_model = ResNet50V2(input_shape=input_shape, include_top=False, input_tensor=x) #try without imagenet weights
        print("....Not Using ImageNet Weights....")
    
    if verbose:
        print(base_model.summary())
        #input('wait')

    '''
    from keras.utils.vis_utils import plot_model
    plot_model(base_model, to_file='base_model_summary.png', show_shapes=True, show_layer_names=True)
    '''

    print("conv2_block2_out's output: ",base_model.get_layer('conv2_block2_out').output)

    print("conv2_block3_out's output: ", base_model.get_layer('conv2_block3_out').output)

    print("base_model's output: ", base_model.output)
    
    input("wait")

    # add a global spatial average pooling layer
    global second_layer
    second_layer = settings.second_layer # False #@param {type:"boolean"}
    #8,12,16
    units = settings.units #14 #@param [8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 30, 32, 48, 64, 196, 256] {type:"raw"}
    kernel_initializer = settings.kernel_init #@param ["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"] {type: "string"}
    resnetv2_output  = settings.resnetv2_output# "conv2_block2_out" #@param ["conv2_block2_out", "conv2_block3_out"] {type: "string"}

    normed =  settings.normed #2 #@param [0, 1, 2] {type:"raw"}
    neuron_type = settings.neuron_type #"Focus2D" # Focus2D, Focus1D, Dense
    activation = settings.activation
    init_sigma = settings.focus_init_sigma
    
    if neuron_type=="Focus1D":
        MIN_SIG = 1./64
        MAX_SIG = 2.0
    else:
        MIN_SIG = 1./64
        MAX_SIG = 1./2
    
    #init_sigma =  MIN_SIG*3 #@param {type:"slider", min:0.05, max:0.5, step:0.01}
    
    ccp1 = ClipCallback('Mu',[MIN_SIG, 1-MIN_SIG])
    ccp2 = ClipCallback('Sigma',[MIN_SIG, MAX_SIG])

    #inputs = tf.keras.Input(shape=(32, 32, 3))
    #x = data_augmentation(inputs)
    #x = preprocess_input(x)
    #print(type(x))
    #x = base_model(x,training=False)
    x = base_model.get_layer(resnetv2_output).output

    x = BatchNormalization()(x)  
    x = Activation('relu')(x)

    #x = base_model.get_layer('conv2_block3_out').output
    x = Dropout(0.25)(x) # Experimental
    #x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    #x = Dense(1024, activation='relu')(x)
    #x = Reshape((32,32,2))(x)
    #28,28,512
    reg2 = None
    if settings.kernel_regularizer:
        reg2 = tf.keras.regularizers.l2(settings.kernel_regularizer)
    regsi = None
    if settings.sigma_regularizer:
        regsi = tf.keras.regularizers.l2(settings.sigma_regularizer)
        
    if neuron_type=="Focus2D":
        # to transform channels into a 2D map. 
        #x = make_grid(x, x.shape[-1], np.int(np.sqrt(x.shape[-1])))
        x = FocusedLayer2D(units=units, name='focus-1',initer='glorot',
                                 distribution='uniform',
                                 kernel_initializer=kernel_initializer,
                                 train_sigma=settings.focus_train_si, 
                                 train_mu = settings.focus_train_mu,
                                 train_weights=settings.focus_train_weights,
                                 si_regularizer=regsi,
                                 normed=normed,
                                 init_mu = 'spread', init_sigma=init_sigma,#30
                                 activation=activation, kernel_regularizer=reg2, 
                                 verbose=False)(x)
        split1, split2 = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': [1,4]})(x)
        split1= Activation('relu')(split1)
        x = BatchNormalization()(split1)
        #x = Activation('relu')(x)    
        #x = Concatenate()([split1, split2])
        x = Dropout(0.25)(x)
        if second_layer:
            x = FocusedLayer2D(units=units, name='focus-2',initer='glorot',
                                 distribution='uniform',
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=reg2,
                                train_sigma=settings.focus_train_si, 
                                 train_mu = settings.focus_train_mu,
                                 train_weights=settings.focus_train_weights,
                                 si_regularizer=regsi,
                                 normed=normed,
                                 init_mu = 'spread', init_sigma=init_sigma,#30
                                 activation=activation, verbose=False)(x)
            split1, split2 = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': [1,4]})(x)
            split1= Activation('relu')(split1)
            split1 = BatchNormalization()(split1)    
            x = Concatenate()([split1, split2])
            x = Dropout(0.25)(x)
    elif neuron_type=="Focus1D":
        x = Flatten(data_format='channels_last')(x)
        #x = tf.keras.layers.Reshape(x.shape+(1,))(x)
        x = FocusedLayer1D(units=units,
                                   name='focus-1',
                                   activation=activation,
                                   init_sigma=init_sigma, 
                                   kernel_initializer=kernel_initializer,
                                   init_mu='spread',
                                   init_w= None,
                                   train_sigma=settings.focus_train_si, 
                                   train_mu = settings.focus_train_mu,
                                   train_weights=settings.focus_train_weights,
                                   si_regularizer=regsi,
                                   normed=normed,
                                   gain=1.0)(x)
        x = BatchNormalization()(x)
        #node_ = LeakyReLU()(node_)
        
        x = Activation('relu')(x)
        
        x = Dropout(0.25)(x)
    elif neuron_type=="Dense":
        x = Flatten(data_format='channels_last')(x)
        x = Dense(units,name='dense-1',activation=activation,
                          kernel_initializer=initializers.glorot_uniform())(x)
        x = BatchNormalization()(x)
        #node_ = LeakyReLU()(node_)
        
        x = Activation('relu')(x)
        
        x = Dropout(0.25)(x)
    
                                 # and a logistic layer -- let's say we have 200 classes
    if neuron_type=="Focus2D" or neuron_type=="Focus1D":
        x = Flatten(data_format='channels_last')(x)
    
    predictions = Dense(num_classes, activation='linear',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    print(model.summary())

    '''
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    '''

    """#**Dataset Settings - Preprocessing and Data Augmentation**"""

    epochs = settings.epochs
    batch_size = settings.batch_size
    
    
            
    
    
    num_classes = np.unique(y_train).shape[0]
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)


    
    '''
    x_train = data_augmentation(x_train)
    x_test = data_augmentation(x_test)

    x_train = preprocess_input(x_train) 
    '''
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')  
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    #x_train= x_train[0:1000] crop dataset for fast training
    #y_train = y_train[0:1000]
    #import matplotlib.pyplot as plt
    #plt.imshow(x_train[0])
    #plt.show()
    #input('wait')
    train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(x_train.shape[0],
                                reshuffle_each_iteration=True).batch(batch_size, 
                                        drop_remainder=True).repeat(epochs)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Callbacks. 
    callbacks = [ccp1,ccp2]
    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
    silent_mode = False
    if not silent_mode:
        from keras_utils_tf2 import PrintLayerVariableStats

        pr_1 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
        #pr_3 = PrintLayerVariableStats("focus-2","Weights:0",stat_func_list,stat_func_name)
        #pr_4 = PrintLayerVariableStats("focus-2","Sigma:0",stat_func_list,stat_func_name)            
        callbacks+=[pr_1, pr_2, pr_3] #,rv_weights_1,rv_sigma_1]
    else:
        pass

    """All layers' trainable attribute is true
    

    #**Without ImageNet Weights Data Augmentation - Adding BatchNorm and Activation after Conv2_block2_out**
    """
    #max_lr = settings['max_lr'] #1e-2 # focus1d 1e-1
    def gauss_lr(epoch, lr, min_lr=settings.min_lr, max_lr=settings.max_lr, 
                center=epochs/2, lr_sigma=0.25):
        max_lr = settings.max_lr
        min_lr = settings.min_lr
        print("testing", epoch, lr, min_lr, max_lr, center, lr_sigma)
        lr = (min_lr + max_lr * np.exp(-(epoch-center)**2 / (center*lr_sigma)**2))
        return lr
   
    
    
    
    
    if settings.optimizer=='gauss':
        lr_callback = tf.keras.callbacks.LearningRateScheduler(gauss_lr, verbose=1) 
        opt = tf.keras.optimizers.SGD(lr=settings.min_lr, momentum=0.9, clipvalue=1.0)
    elif settings.optimizer=='sgd':
        '''lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=epochs//3, 
                                                           factor=0.9,
                                                           verbose=1,
                                                           min_lr=settings.min_lr)
        '''
        ## lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=epochs//2,  result for Focus1d, Dense
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=epochs//5, 
                                                           factor=0.5,
                                                           verbose=1,
                                                           min_lr=settings.min_lr)
        opt = tf.keras.optimizers.SGD(lr=settings.max_lr, momentum=0.9, clipvalue=1.0)
    elif settings.optimizer=='adam':
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=epochs//3,
                                                           verbose=1,
                                                           factor=0.9,
                                                           min_lr=settings.min_lr)
        opt = tf.keras.optimizers.Adam(lr=settings.max_lr, clipvalue=1.0)
    
    callbacks +=[lr_callback]

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #opt = tf.keras.optimizers.Adam(lr=1e-3, clipvalue=1.0)
    

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=opt, loss=loss_object, metrics=['accuracy'])

    # train the model on the new data for a few epochs
    if neuron_type=="Focus2D" or neuron_type=="Focus1D":
        model.fit(
                train_ds,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=test_ds, callbacks=callbacks)
    else:
        model.fit(
                train_ds,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=test_ds)
        # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    print(neuron_type)
    return model

class Settings:
    def __init__(self,dset="cifar10", neuron_type='Focus2D', units=14,
                 activation='linear', focus_init_sigma=0.05, focus_init_mu='spread',
                 focus_train_mu=True, focus_train_si=True, focus_train_weights=True,
                 normed=2, augment=True, epochs=15,batch_size=32,
                 repeats=5, use_imagenet_weights=False,max_lr=0.01,min_lr=0.0001,
                 kernel_init='glorot_uniform', resnetv2_output='conv2_block2_out',
                 second_layer=False,kernel_regularizer=None, sigma_regularizer=None,
                 optimizer='adam',notes='None'):
        self.dset = dset
        self.neuron_type = neuron_type
        self.units = units
        self.activation = activation
        self.focus_init_sigma = focus_init_sigma
        self.focus_init_mu = focus_init_mu
        self.focus_train_mu = focus_train_mu
        self.focus_train_si = focus_train_si
        self.focus_train_weights = focus_train_weights
        self.normed = normed
        self.augment = augment
        self.epochs = epochs
        self.batch_size = batch_size
        self.repeats = repeats
        self.use_imagenet_weights = use_imagenet_weights
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.kernel_init = kernel_init
        self.resnetv2_output = resnetv2_output
        self.second_layer = second_layer
        self.kernel_regularizer = kernel_regularizer
        self.sigma_regularizer = sigma_regularizer
        self.optimizer = optimizer
        self.notes = notes
        
    
    def __str__(self):
        return str(self.__dict__)
        

if __name__ == "__main__":
    #print("Result saving part is not finished, we will finish within the day (30.07.2021)")
    import sys
    from datetime import datetime
    delayed_start = 0*3600
    import time 
    from shutil import copyfile
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    now = datetime.now()
    timestr = now.strftime("%Y%m%d_%H%M%S")
    dset = 'tiny'
    settings = Settings(dset=dset, neuron_type='Focus2D', units=16, activation='linear',
         focus_init_sigma=4/16, focus_init_mu='spread', focus_train_mu=True,
         focus_train_si=True, focus_train_weights=True, normed=2, augment=True,
         epochs=35, batch_size=128, repeats=5, use_imagenet_weights=False,
         max_lr=2e-4, min_lr= 1e-5, kernel_init='glorot_uniform', 
         resnetv2_output='conv2_block2_out',
         second_layer=False, kernel_regularizer=None,
         sigma_regularizer=None, optimizer='gauss',
         notes='Focus2D with old spread 0.2-0.8')

    '''
    settings={'dset':dset, 'neuron_type':'Focus2D', 'units':14, 'activation':'linear',
         'focus_init_sigma':0.1,'focus_init_mu':'spread','focus_train_mu':True, 
         'focus_train_si':True,'focus_train_weights':True,'normed':2, 'augment':True, 
         'epochs':15, 'batch_size':32,'repeats':1, 'use_imagenet_weights':False,
         'max_lr':0.01, 'min_lr': 0.0001,'kernel_init':'glorot_uniform', 'resnetv2_output': 'conv2_block2_out'}
    '''
    if len(sys.argv) > 1:
        settings.dset = sys.argv[1]
        settings.dset = settings.dset.replace('\r', '')
    if len(sys.argv) > 2:
        settings.neuron_type = sys.argv[2]
        settings.neuron_type = settings.neuron_type.replace('\r', '')
    if len(sys.argv) > 3:
        settings.units = int(sys.argv[3])
        print("units ", settings.units)
        print(type(settings['units']))
    if len(sys.argv) > 4:
        settings.activation = sys.argv[4]
        settings.activation = settings.activation.replace('\r', '')
        print("activation", sys.argv[4])
    if len(sys.argv) > 5:
        settings.focus_init_sigma = float(sys.argv[5])
        print("focus_init_sigma ", settings.focus_init_sigma)
        print(type(settings.focus_init_sigma))
    if len(sys.argv) > 6:
        settings.focus_init_mu = sys.argv[6]
        settings.focus_init_mu = settings.focus_init_mu.replace('\r', '')
    if len(sys.argv) > 7:
        if sys.argv[7]=='False':
            settings.focus_train_mu = False
        else:
            settings.focus_train_mu = True
        print("focus_train_mu", sys.argv[7])
    if len(sys.argv) > 8:
        if sys.argv[8]=='False':
            settings.focus_train_si = False
        else:
            settings.focus_train_si = True
        print("focus_train_si", sys.argv[8])
    if len(sys.argv) > 9:
        if sys.argv[9]=='False':
            settings.focus_train_weights = False
        else:
            settings.focus_train_weights = True
        print("focus_train_weights", sys.argv[9])
    if len(sys.argv) > 10:
        settings.normed = int(sys.argv[10])
        print("units ", settings.normed, "type: ", type(settings.normed))
    if len(sys.argv) > 11:
        if sys.argv[11]=='False':
            settings.augment = False
        else:
            settings.augment = True
        print("augment", sys.argv[11])
    if len(sys.argv) > 12:
        settings.epochs = int(sys.argv[12])
        print("Epochs", int(sys.argv[12]))
    if len(sys.argv) > 13:
        settings.batch_size = int(sys.argv[13])
        print("batch_size", int(sys.argv[13]))
    if len(sys.argv) > 14:
        settings.repeats = int(sys.argv[14])
        print("repeats", int(sys.argv[14]))
    if len(sys.argv) > 15:
        if sys.argv[15]=='False':
            settings.use_imagenet_weights = False
        else:
            settings.use_imagenet_weights = True
        print("use_imagenet_weights", sys.argv[15])
    if len(sys.argv) > 16:
        settings.max_lr = float(sys.argv[16])
        print("max_lr", float(sys.argv[16]))
    if len(sys.argv) > 17:
        settings.min_lr = float(sys.argv[17])
        print("max_lr", float(sys.argv[17]))
    if len(sys.argv) > 18:
        settings.kernel_init = sys.argv[18]
        settings.kernel_init = settings.kernel_init.replace('\r', '')
        print("kernel_init", sys.argv[18])
    if len(sys.argv)>19:
        settings.resnetv2_output = sys.argv[19]
        settings.resnetv2_output = settings.resnetv2_output.replace('\r', '')
    if len(sys.argv) > 20:
        if sys.argv[20]=='False':
            settings.second_layer = False
        else:
            settings.second_layer = True
            
    if len(sys.argv) > 21:
        if sys.argv[21]=='None':
            settings.kernel_regularizer = None
        else:
            settings.kernel_regularizer = float(sys.argv[21])
    
    if len(sys.argv) > 22:
            settings.optimizer = sys.argv[22].replace('r','')

    
    
        
    K.clear_session()
    print(settings)
    val_acc = []
    val_loss = []
    acc = []
    loss = []
    import os
    if os.name=='nt':
        folder = 'results\\'
    else:
        folder = 'results/'
    
    if settings.use_imagenet_weights:
        filename = folder + timestr + '_ResNetV2_' + settings.dset + '_' + settings.neuron_type + '_' + str(settings.units) + '_with_ImageNet' + '_model_results.npz'
    else:
        filename = folder + timestr + '_ResNetV2_' + settings.dset + '_'+ settings.neuron_type + '_' + str(settings.units) + '_without_ImageNet' +'_model_results.npz'

    for i in range(settings.repeats):
        model = build_model(settings=settings)
        val_acc.append(model.history.history['val_accuracy'])
        val_loss.append(model.history.history['val_loss'])
        acc.append(model.history.history['accuracy'])
        loss.append(model.history.history['loss'])
        
    histories = [{'val_acc':val_acc,'val_loss':val_loss,
                  'acc':acc,'loss':loss}]
        
    mx_scores = [np.max(val_acc[i]) for i in range(len(val_acc))]
    settings_a= [settings.__dict__]
    np.savez_compressed(filename, history=histories, mx_scores=mx_scores, settings=settings_a)
    #model.layers[4]
    
    
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    '''
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)
        '''
    # 87.96 weight reg 1e-4 ,sgd, max lr 0.1, minlr 1e-4
    # 87.95 no reg sgd, max lr 0.1, minlr 1e-4
    # 88.20 focus-1d
    # 0.8702 dense
    
    plt = True
    if plt:
        import matplotlib.pyplot as plt
        plt.plot(np.array(val_acc).T)
        plt.grid()
        plt.ylim([0.85,0.885])