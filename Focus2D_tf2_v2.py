# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:41:42 2020

@author: talha
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, regularizers, constraints
from tensorflow.keras import initializers
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.initializers import constant
from Kfocusingtf2 import FocusedLayer1D
import matplotlib.pyplot as plt
from shutil import copyfile

class SIInitializer(tf.keras.initializers.Initializer):

    def __init__(self, init_sigma, input_shape, units):
        self.init_sigma = init_sigma
        self.units = units
        self.incoming_height = input_shape[1]
        self.incoming_width = input_shape[2]

    def __call__(self, shape, dtype=None):
        if isinstance(self.init_sigma, list):
            si = np.ones(shape=shape,dtype='float32') * self.init_sigma
        if isinstance(self.init_sigma, float):
            si = np.ones(shape=shape,dtype='float32') * self.init_sigma
        elif isinstance(self.init_sigma, str) and self.init_sigma=='random':
            si = np.random.uniform(low=1./self.incoming_width,high=1.0,size=shape)
        elif isinstance(self.init_sigma, str) and self.init_sigma=='hv':
            aa = np.float32([[1./self.incoming_height, 2./self.incoming_width],
                             [2./self.incoming_width, 1./self.incoming_height]])
            si = np.tile(aa,(int(self.units[0]),int(self.units[0]/2),1))
            print(shape, si.shape)
        elif isinstance(self.init_sigma, str) and self.init_sigma=='gauss':
            mx = self.units[0]//2
            my = self.units[1] //2
            x_range = np.arange(0.0,self.units[0]) 
            y_range = np.arange(0.0, self.units[1])
            xx,yy = np.meshgrid(x_range, y_range)
            aa =  0.15*np.exp(-((xx-mx)**2+(yy-my)**2)/(4*mx**2))
            print(aa)
            #input('wait')
            si = np.zeros(shape=shape,dtype='float32') 
            si[:,:,0]  = aa
            si[:,:,1] = aa
            print(shape, si.shape)
        return tf.convert_to_tensor(si, dtype=dtype)

    def get_config(self):  # To support serialization
      return {'init_sigma': self.init_sigma,'units': self.units, 'incoming_height': self.incoming_height,
              'incoming_width': self.incoming_width}
  
   
class MUInitializer(tf.keras.initializers.Initializer):

    def __init__(self, init_mu, input_shape, units, sigma_init):
        self.init_mu = init_mu
        self.units = units
        self.incoming_height = input_shape[1]
        self.incoming_width = input_shape[2]
        if isinstance(sigma_init, float):
            self.marginx = 1.0*sigma_init
            self.marginy = 1.0*sigma_init
        elif isinstance(sigma_init, list):
            self.marginx = 1.0*sigma_init[0]
            self.marginy = 1.0*sigma_init[1]
        elif isinstance(sigma_init, str): # assuming random 
            self.marginx = 1.0*(1./self.incoming_width+1.)/2
            self.marginy = 1.0*(1./self.incoming_height+1.)/2

    def __call__(self, shape, dtype=None):
        total_units = np.prod(self.units)    
        
        x_range = np.linspace(0.0,1.0,self.incoming_width)
        x_mid = x_range[x_range.shape[0]//2]
        y_range = np.linspace(0.0,1.0,self.incoming_height)
        y_mid =  y_range[y_range.shape[0]//2]
        mu = None
        init_mu = self.init_mu
        if dtype == tf.float32:
            dtype = 'float32'
        else:
            print("Warning! datatype not implemented")
        if isinstance(init_mu, str):
            if init_mu == 'middle':
        
                mux = x_mid * np.ones(self.units[1],dtype=dtype)
                muy = y_mid * np.ones(self.units[0],dtype=dtype)
        
            elif init_mu =='middle_random':
                mux = x_mid * np.ones(self.units[1],dtype=dtype)
                muy = y_mid * np.ones(self.units[0],dtype=dtype)
                rx = (np.random.rand(self.units[1])-0.5)*(1.0/(float(self.units[1])))
                ry = (np.random.rand(self.units[0])-0.5)*(1.0/(float(self.units[0])))
                mux +=rx.T
                muy +=ry.T
            
            elif init_mu == 'spreadnew':
            #print(num_units[0])
                #paper results were taken with this. IT EFFECTS RESULTS!!!
                mux = np.linspace(self.marginx,1-self.marginx,self.units[0])
                muy = np.linspace(self.marginy, 1-self.marginy,self.units[1])
                # new method 
                
                # bu yeni iyi diye düşünmüştüm. MNIST, CIFAR-10 aynı
                # maalesef küçük spacelerde mesela 8x8 kötü sonuçlanıyor
                #mux = np.linspace(x_range[3],x_range[-4],self.units[0])
                #muy = np.linspace(y_range[3],y_range[-4],self.units[1])
                
                # new spread divide the space into n_units+4 space
                # use the 2:-2
                #marginx = np.int(np.ceil(np.sqrt(self.units[0])))
                #x_broad = np.linspace(0,1,self.units[0]+marginx)
                #mux = x_broad[marginx//2:-marginx//2]
                #marginy = np.int(np.ceil(np.sqrt(self.units[1])))
                #y_broad = np.linspace(0,1,self.units[1]+marginy)
                #muy = y_broad[marginy//2:-marginy//2]
                #print(marginx)
                print(mux,muy)
                #input('wait')
                
            elif init_mu == 'spread':
            #print(num_units[0])
                #spaper results were taken with this. IT EFFECTS RESULTS!!!
                mux = np.linspace(0.2,0.8,self.units[0])
                muy = np.linspace(0.2,0.8,self.units[1])
                
                
                
            
            elif init_mu == 'random':
            # select random points from idx
            # tested with si=0.1, did not work well for mnist. 
            
            
                mux = np.random.choice(x_range, size= self.units[0])
                muy = np.random.choice(y_range, size= self.units[1])
            
                      
        elif isinstance(init_mu, float):  #initialize it with the given scalar
            assert init_mu > 0. and init_mu < 1.0, "The argument 'init_mu' should be in the interval (0, 1)."
            mux = np.repeat(init_mu, self.units[0])  # 
            muy = np.repeat(init_mu, self.units[1])  # 

        elif isinstance(init_mu,np.ndarray):  #initialize it with the given array , must be same length of num_units
            if init_mu.max() > 1.0:
                print("Mu must be [0,1.0] Normalizing initial Mu value")
                init_mu =np.clip(init_mu, 0,1)
                mu = init_mu       
            else:
                mu = init_mu
        
        else:
            print("Not implemented.")
        
        if mu is None:
            print(shape)
            mu = np.zeros(shape, dtype=dtype)
            
            
            for ix,x in enumerate(mux):
                for iy,y in enumerate(muy):
                    mu[iy,ix,0] = y
                    mu[iy,ix,1] = x
            
        
        return tf.convert_to_tensor(mu, dtype=dtype)

    def get_config(self):  # To support serialization
      return {'init_mu': self.init_mu,'units': self.units, 'incoming_height': self.incoming_height,
              'incoming_width': self.incoming_width}


class FocusedLayer2D(Layer):
    def __init__(self, units, #in github units =40 ???
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 si_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 gain=1.0,
                 u_gain=1.0,
                 init_mu = 'spread',
                 init_sigma=0.1,
                 initer='Glorot',
                 distribution='normal',
                 init_bias = initializers.Constant(0.0),
                 train_mu=True,
                 train_sigma=True, 
                 train_weights=True,
                 reg_bias=None,
                 normed=2,
                 verbose=False,
                 add_mu_si=True,
                 input_output_scale_norm=False,
                 **kwargs):
        super(FocusedLayer2D, self).__init__(**kwargs)
        
        if isinstance(units,int):
            self.units=(units,units)
            units=self.units
            
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.si_regularizer = regularizers.get(si_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)
        self.supports_masking = True
        self.gain = gain
        self.u_gain = u_gain
        self.init_sigma=init_sigma
        self.init_mu = init_mu
        self.train_mu = train_mu
        self.train_sigma = train_sigma
        self.train_weights = train_weights
        self.normed = normed
        self.verbose = verbose
        self.add_mu_si = add_mu_si
        #self.incoming_width, self.incoming_height = incoming[:2]  # Get only last two dims
        self.sigma=None
        self.initer=initer
        self.distribution=distribution        
        self.input_output_scale_norm=input_output_scale_norm


    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]
        self.incoming=input_shape[1:]
        #self.incoming_width, self.incoming_height = input_shape[1:3]
        self.incoming_height, self.incoming_width = input_shape[1:3]
        print("units= ",self.units)
        self.input_spec = InputSpec(ndim=4, axes={-1: self.input_dim})
        #print("self.input_spec= ",self.input_spec.shape)
        try:
            mu_shape = (self.units+(2,))
            ## shape is formed of sigma for neuron and repeated for incoming-channels
            si_shape = (self.units+(2,))
        except TypeError:
            print("The argument `units` should be a number")
        except:
            print("Unexpected error:")
        
        
        print(input_shape)
        print(self.incoming_height,self.incoming_height)
        
        x_range = np.linspace(0.0, 1.0, self.incoming_width)
        y_range = np.linspace(0.0, 1.0, self.incoming_height)
        i_x, i_y =  np.meshgrid(x_range, y_range)
        
        idxs = np.concatenate([i_y[:,:,np.newaxis], i_x[:,:,np.newaxis]], axis=2)
        idxs_shape=idxs.shape
        
        # '''
        # mu, si = mu_si_initializer2D(self.init_mu,self.init_sigma, mu_shape,si_shape, self.input_dim,
        #                            self.units,self.incoming_width,self.incoming_height, verbose=self.verbose)
        # '''
        # x_range = np.linspace(0.2,.8,self.incoming_width)
        # y_range = np.linspace(0.2,.8,self.incoming_height)
        # #x_range = np.clip(x_range, x_range[1], x_range[-2])
        # #y_range = np.clip(y_range, y_range[1], y_range[-2])
        # mu_x, mu_y =  np.meshgrid(x_range,y_range)
        # mu = np.concatenate([mu_y[:,:,np.newaxis],mu_x[:,:,np.newaxis]], axis=2)
        # self.mu_mean = np.mean(mu)
        



        '''
        
        #idxs = np.zeros(idxs_shape,dtype='float32')
   
        c=0
        for x in np.arange(self.incoming_width)/((self.incoming_width-1)*1.0):
            for y in np.arange(self.incoming_height)/((self.incoming_height-1)*1.0):
                idxs[c,0]=y
                idxs[c,1]=x
                c+=1
        '''      
        idxs = idxs.astype(dtype='float32')
        #print("idxs= ",idxs)
        print("idxs.shape= ",idxs.shape)
        self.idxs = K.constant(value=idxs, shape=(idxs_shape),name="idxs")
  
        # assert self.init_sigma > 0. , "The argument 'init_sigma' should be bigger than 0."
        mu_init =  MUInitializer(self.init_mu, input_shape, self.units, self.init_sigma)
        si_init =  SIInitializer(self.init_sigma, input_shape, self.units)
  
       
        self.sigma = self.add_weight(shape=(si_shape), 
                                     initializer=si_init, 
                                     name="Sigma", 
                                     regularizer=self.si_regularizer,
                                     trainable=self.train_sigma)
        self.mu = self.add_weight(shape=(mu_shape),
                                  initializer=mu_init, 
                                  name="Mu", regularizer=None,
                                  trainable=self.train_mu)
        
        #print("Sigma:",self.sigma)
        #print("Mu:",self.mu)
        #input("wait")
        
        MIN_SI = 1e-5  # zero or below si will crashed calc_u
        MAX_SI = 10.0 
        
        # create shared vars.
        self.MIN_SI = np.float32(MIN_SI)#, dtype='float32')
        self.MAX_SI = np.float32(MAX_SI)#, dtype='float32')
        
        self.w_shape = self.incoming + (np.prod(self.units),)#not shared #3 yok
        
        print("self.w_shape= ",self.w_shape)
        #w_shape = (int(np.prod(self.incoming[-2:])), 1)  #shared
        if self.kernel_initializer==None:
            w_init = self.weight_initializer # this is function
        else:
            w_init = self.kernel_initializer
        #w_init=self.kernel_initializer
        #print("FOCUSING NEURON WEIGHT INIT", w_init)
        self.W = self.add_weight(shape=self.w_shape,
                                      initializer=w_init,
                                      name='Weights',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.train_weights)
        print("after weight initialize")
        if self.use_bias:
            self.bias = self.add_weight(shape=(np.prod(self.units),),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        self.built = True
        
    
    def calc_U(self,verbose=False):
       """
       function calculates focus coefficients. 
       normalizes and prunes if
       """
       #print("mu.shape= ",self.mu.shape)
       #print("self.idxs.shape= ",self.idxs.shape)

       idx_ = K.expand_dims(K.expand_dims(self.idxs,axis=2),axis=2)
       mu_ = K.expand_dims(K.expand_dims(self.mu,axis=0),axis=0)
       print(idx_.shape, mu_.shape)
       up_d=(idx_ - mu_)**2
      
       dwn_d = (2 * ( self.sigma)**2)
       #print("dwn.shape= ",dwn_d.shape)
       #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
       
       result = K.exp(- K.sum(up_d / dwn_d, axis=-1))
       #print("result= ",K.eval(result))
       print("result= ",result.shape)
       #input("HEreere")
       # PLOT ALLL
       '''
       for ix in range(0,result.shape[3],4):
           for iy in range(0,result.shape[2],4):
               plt.imshow(np.reshape(result[:,:,ix,iy],(32,32)))
               plt.colorbar()  
               plt.show()
       input("HEreere")
       '''
       result = K.reshape(result, (self.w_shape[0], self.w_shape[1], 1, self.w_shape[3]))
       if self.w_shape[2] > 1:
           result=K.repeat_elements(result, self.w_shape[2], 2)

       #print("K sum is on the neuron dims. correct",K.sum(K.square(result), axis=-1,keepdims=False))
       
       if self.normed==1:
           result /= K.sqrt(K.sum(K.square(result), axis=(0,1,2),keepdims=True))
       
       elif self.normed==2:
           result /= K.sqrt(K.sum(K.square(result), axis=(0,1,2),keepdims=True))
           result *= K.sqrt(K.constant(np.prod(result.shape[:3])))*self.u_gain #Second Layer= 28x28x5, #First Layer= 28x28x1

           if verbose:
               kernel= K.eval(result)
               print("RESULT after NORMED max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
           #
       '''
       for ix in range(0,result.shape[3],4):
               plt.imshow(np.reshape(result[:,:,0,ix],(result.shape[0],result.shape[1])))
               plt.colorbar()  
               plt.show()
       input("HEreere")
       
       '''
       #Normalize to get equal to WxW Filter
       #masks *= K.sqrt(K.constant(self.input_channels*self.kernel_size[0]*self.kernel_size[1]))
       # make norm sqrt(filterw x filterh x self.incoming_channel)
       # the reason for this is if you take U all ones(self.kernel_size[0],kernel_size[1], num_channels)
       # its norm will sqrt(wxhxc)
       #print("Vars: ",self.input_channels,self.kernel_size[0],self.kernel_size[1])
       return result
       #return K.transpose(result/2)  #/2 trying to understand why this /2 i
   
    
   
    def call(self, inputs):
        u = self.calc_U()
        #print("u= ",u)
        #u=K.repeat(u,1)
        
        #u=K.expand_dims(u,axis=0)
        #u=K.repeat_elements(u,1,axis=0)
        #print("inputs shape", inputs.shape)
            
        print("type(inputs)= ",type(inputs))
        #>inputs=K.batch_flatten(inputs)
        #inputs = K.reshape(inputs, (-1,np.prod(inputs.shape[1:])))
        if self.verbose:
            print("weights shape", self.W.shape)
        #w=K.repeat(self.W,np.prod(self.units),axis=1)  only for shared
        #print("u= ",u)
        #print("weights shape", self.W.shape)
        self.kernel = self.W * u
        

                                      
        #print("self.kernel shape", self.kernel.shape)
        
        #import matplotlib.pyplot as plt
        #tf.print(K.max(self.kernel[:,:,0,625]))
        #print(K.value(K.max(self.kernel[:,:,0,625])))
        
        #input("here")
        # Generates a random number between 
        # a given positive range 
        
        output = tf.tensordot(inputs, self.kernel, axes =[[1,2,3],[0,1,2]])#do(inputs,self.kernel)
            
        #print("output shape", output.shape)
        if self.use_bias:
            print("No bias")
            output = K.bias_add(output, self.bias, data_format='channels_last')

        output=K.reshape(output,shape=(-1,
                                     self.units[0],
                                     self.units[1],1))
        
        #print(output.shape, inputs.shape)
        #input("waiiiittt")
        #output = 
        if self.input_output_scale_norm:
          #input_var= K.mean(K.std(inputs,axis=-1,keepdims=True), axis=-1, keepdims=True)
          input_var = K.std(inputs)
          print("input_var.shape",input_var.shape)
          #input_var= K.std(inputs,axis=-1,keepdims=True)
                
          #output_var = K.mean(K.std(output,axis=-1, keepdims=True), axis=-1,keepdims=True)#+1e-5
          output_var = K.std(output)
          print("output_var.shape",output_var.shape)
          #output_var = K.std(output,axis=-1, keepdims=True)
          #output -= K.mean(output)
          output /= output_var
          output *=input_var

        if self.activation is not None:
            #print("output shape ", output.shape)
            output = self.activation(output)

        add_mu_si = self.add_mu_si
        if add_mu_si:
            #base = tf.ones_like(output)*(-0.5)
            base = tf.zeros_like(output)
            #mu = K.reshape(self.mu, (1,
            #                             self.units[0],
            #                             self.units[1],2))
            mu = K.expand_dims(self.mu, 0)
            #print("mu_mean= ",K.mean(mu))
            ##mu -= K.mean(mu)
            #print("mu_std= ",K.std(mu))
            ##mu /= 0.3
            #mu /= (K.std(mu)+1e-5)
            #print("Add mu-si, mu= ",mu)

            si = K.expand_dims(self.sigma, 0)
            #print("si_mean= ",K.mean(si))
            ##si -= K.mean(si)
            #print("si_std= ",K.std(si))
            ##si /= 0.3
            #si /= (K.std(si)+1e-5)
            #print("Add mu-si, si= ",si)
            mu = base + mu #tf.broadcast_to(mu,tf.shape(output))
            #mu = K.repeat_elements(mu, 256, axis=0)
            si = base + si
            
            print(mu.shape, si.shape)
            #input("HEre:")
            
            #si = K.repeat_elements(si, 256, axis=0)
            musi = K.concatenate((output, mu, si), axis=3)
            print(musi.shape)
                
                #print(musi.shape)
                #input('HEre')
            output = musi   
        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'si_regularizer': regularizers.serialize(self.si_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'gain': self.gain,
            'u_gain': self.u_gain,
            'init_mu': self.init_mu,
            'init_sigma': self.init_sigma,
            'initer': self.initer,
            'distribution': self.distribution,
            'init_bias': initializers.serialize(self.init_bias),
            'train_mu': self.train_mu,
            'train_sigma': self.train_sigma, 
            'train_weights': self.train_weights,
             'reg_bias': regularizers.serialize(self.reg_bias),
             'normed': self.normed,
             'verbose': self.verbose,
             'add_mu_si': self.add_mu_si,
             'input_output_scale_norm': self.input_output_scale_norm,    
        }
        base_config = super(FocusedLayer2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Assumes that the layer will be built
        to match that input shape provided.

        # Arguments
            input_shape: Shape tuple (tuple of integers)
                or list of shape tuples (one per output tensor of the layer).
                Shape tuples can include None for free dimensions,
                instead of an integer.

        # Returns
            An input shape tuple.
        """
        
        #o_s = (input_shape[0], self.units[0],self.units[1],input_shape[-1])
        o_s = (input_shape[0], self.units[0],self.units[1],1)
        print("input shape", input_shape,"outputshape",o_s)
        
        return o_s
    
    
    def weight_initializer(self,shape,verbose=False,dtype='float32'):
        #only implements channel last and HE uniform        
        kernel = K.eval(self.calc_U())
        
        #print("kernel= ",kernel)
        #print("kernel_shape= ",kernel.shape)
        W = np.zeros(shape=self.w_shape, dtype='float32')
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        plot_res = False
        import matplotlib.pyplot as plt
        if plot_res:
            
            for i in range(0,kernel.shape[3],10):
                plt.imshow(kernel[:,:,0,i])
                plt.colorbar()  
                plt.show()
        
        fan_out = np.prod(self.units)
        print("W_shape= ",W.shape)
        
        #print("kernek_shape= ",kernel.shape)
        for c in range(W.shape[3]):#For each output neuron c

            #print("kernel[:,c]= ",kernel[:,c].shape)
            #print("sum= ",np.sum((kernel[:,c])**2))
            fan_in = np.sum((kernel[:,:,:,c])**2)
            #sig=K.eval(self.sigma[c])
            
            #print("fan_in= ",fan_in)
            #print("fan_out= ",fan_out)
            #fan_in *= self.input_channels no need for this in repeated U. 
            if self.initer=='he':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
            elif self.initer=='glorot':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
            elif self.initer=='lecun':
                std = self.gain * sqrt32(1.0) / sqrt32(fan_in)
            else:
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)   
   
            if self.distribution=='uniform':
                std = std * sqrt32(3.0)
                std = np.float32(std)
                w_vec = np.random.uniform(low=-std, high=std, size=W.shape[:3])
                #print("w_vec.shape= ",w_vec.shape)
            else:
                std = std/ np.float32(.87962566103423978)           
                w_vec = np.random.normal(scale=std, size=(W.shape[:3],))
            # #Glorot
             #He Why it is sqrt32(6)
             #normal
            #print("W.shape= ",W.shape)
            #print("std= ",std)
            # uniform
            #std = np.float32(std)
            if (c == 0 or c==(kernel.shape[1]-1) or c==(kernel.shape[1]/2)) and verbose:
                print ("Inıtializing neuron", c, "WU: ", (kernel[:,c].repeat(self.incoming_channels)*w_vec)[:6])
                #print ("Initializing neuron weight std:, fan_in:, sig:", std,fan_in, sig)
            W[:,:,:,c] = w_vec.astype('float32')
            
        print("w.shape= ",W.shape)
        #print("w_vec.shape= ",w_vec.shape)
        #print(np.sum(W**2,axis=(0,1,2)))
        #input("HErE")
        
        return W

    def weight_initializer_fw_bg(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        initer = 'Glorot'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U())
        
        W = np.zeros(shape=shape, dtype=dtype)
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        sum_over_domain = np.sum(kernel**2,axis=1) # r base
        sum_over_neuron = np.sum(kernel**2,axis=0)
        for c in range(W.shape[1]): #1
            for r in range(W.shape[0]):
                fan_out = sum_over_domain[r]
                fan_in = sum_over_neuron[c]
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
                else:
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
                    print(r,",",c," Fan in ", fan_in, " Fan_out:", fan_out, W[r,c])
                    
                if distribution == 'uniform':
                    std = std * sqrt32(3.0)
                    std = np.float32(std)
                    w_vec = np.random.uniform(low=-std, high=std, size=1)
                elif distribution == 'normal':
                    std = std/ np.float32(.87962566103423978)           
                    w_vec = np.random.normal(scale=std, size=1)
                    
                W[r,c] = w_vec.astype('float32')
                
        return W
    
   

        
def mu_si_initializer2D(initMu, initSi,mu_shape,si_shape, num_incoming, num_units,in_width,in_height ,verbose=False):
    '''
    Initialize focus centers and sigmas with regards to initMu, initSi
    
    initMu: a string, a value, or a numpy.array for initialization
    initSi: a string, a value, or a numpy.array for initialization
    num_incoming: number of incoming inputs per neuron
    num_units: number of neurons in this layer
    '''
    total_units = np.prod(num_units)    
    if isinstance(initMu, str):
        if initMu == 'middle':
            #print(initMu)
             mu = 0.5*np.ones(mu_shape,dtype='float32')
             # On paper we have this initalization                
        elif initMu =='middle_random':
            mu = 0.5*np.ones(mu_shape,dtype='float32')
            r = np.array([(np.random.rand(total_units)-0.5)*(1.0/(float(20.0))),
                  (np.random.rand(total_units)-0.5)*(1.0/(float(20.0)))])  
            mu +=r.T             
            
        elif initMu == 'spread':
            #print(num_units[0])
            #paper results were taken with this. IT EFFECTS RESULTS!!!
            #mux = np.linspace(0.2, 0.8, num_units[0])  
            #muy = np.linspace(0.2, 0.8, num_units[1])  
            mux = np.linspace(0.2,0.8,num_units[0])
            muy = np.linspace(0.2,0.8,num_units[1])
            mu=np.zeros(mu_shape,dtype='float32')
            #print("mu.shape= ",mu.shape)
            #print("si.shape= ",si_shape)
            #print("m.shape= ",mu_shape)
            
            c=0
            for x in mux:
                for y in muy:
                    mu[c,0] = y
                    mu[c,1] = x
                    c+=1
            '''
            c=0
            for y in muy:  
                for x in mux:
                    mu[c,0] = y
                    mu[c,1] = 0.5
                    c+=1
            print("mu= ",mu)
            '''
            
            #mu = np.linspace(0.1, 0.9, num_units)
        elif initMu=='spread2D':
            in_wid = np.sqrt(num_incoming)
            ns =total_units
            
            in_wid= int(in_wid)
            in_hei = num_incoming//in_wid
            
            ns_hei = total_units//ns
            ns_min = min(ns,ns_hei)
            vert = np.linspace(in_hei*0.2,in_hei*0.8, ns_min) 
            hor = np.linspace(in_wid*0.2,in_wid*0.8, ns_min)

            mu_ = vert*in_wid
            mu= (mu_+ hor[:,np.newaxis]).reshape(in_wid*in_hei) / (num_incoming)
        else:
            print(initMu, "Not Implemented")
            
    elif isinstance(initMu, float):  #initialize it with the given scalar
        mu = np.repeat(initMu, total_units)  # 

    elif isinstance(initMu,np.ndarray):  #initialize it with the given array , must be same length of num_units
        if initMu.max() > 1.0:
            print("Mu must be [0,1.0] Normalizing initial Mu value")
            initMu /=(num_incoming - 1.0)
            mu = initMu        
        else:
            mu = initMu
    
    #Initialize sigma
    if isinstance(initSi,str):
        if initSi == 'random':
            si = 0.05*np.ones(si_shape,dtype='float32')  # On paper we have this initalization                
            r = np.random.uniform(low=0.01, high=0.05, size=np.prod(total_units))
            si+=r
        elif initSi == 'spread':
            print("Not Implemented")
        elif initSi == 'elips':
            si=np.ones(si_shape,dtype='float32')
            for i in range(0,si_shape[0]-1):
                si[i,0] = 0.1

    elif isinstance(initSi,float):  #initialize it with the given scalar
        print("here")
        si = initSi*np.ones(si_shape,dtype='float32')  # On paper we have this initalization  
        
    elif (isinstance(initSi,list) or isinstance(initSi,tuple))and len(initSi)==2: # set the values directly from array
        si = np.linspace(initSi[0], initSi[1], total_units)
    elif isinstance(initSi,np.ndarray):#initialize it with the given array , must be same length of num_units
        si=initSi
        
    # Convert Types for GPU
    mu = mu.astype(dtype='float32')
    si = si.astype(dtype='float32')

    if verbose:
        print("mu init:", mu)
        print("si init:", si)
        
    return mu, si



def U_numeric(idxs, mus, sis, scaler, normed=0):
    '''
    This function provides a numeric computed focus coefficient vector for
    
    idxs: the set of indexes (positions) to calculate Gaussian focus coefficients
    
    mus: a numpy array of focus centers
    
    sis: a numpy array of focus aperture sigmas
    
    scaler: a scalar value
    
    normed: apply sum normalization   
    '''
    #cal_u deki koda göre implement et
    mus=np.expand_dims(mus,1)
    sis=np.expand_dims(sis,1)
    up = np.sum((idxs - mus) ** 2,axis=2)
    down = (2 * (sis ** 2))
    ex = np.exp(-up / down)
    if normed==1:
        ex /= np.sqrt(np.sum(np.square(ex), axis=-1,keepdims=True))
    elif normed==2:
        ex /= np.sqrt(np.sum(np.square(ex), axis=-1,keepdims=True))
        ex *= np.sqrt(idxs.shape[0])

    return (np.transpose(ex.astype(dtype='float32')))

def calculate_fi_and_weights(layer_instance):
    ''' 
    This aux function calculates its focus functions, focused weights for a given
    a layer instance
    '''
    w = layer_instance.get_weights()
    si = w[0]
    mu = w[1]
    we = w[2]

    idxs_shape=(int(np.prod(layer_instance.input_shape[1:3])),2)
    idxs = np.zeros(idxs_shape,dtype='float32')
   
    c=0
    for x in np.arange(layer_instance.input_shape[1:3][0])/((layer_instance.input_shape[1:3][0]-1)*1.0):
        for y in np.arange(layer_instance.input_shape[1:3][1])/((layer_instance.input_shape[1:3][1]-1)*1.0):
            idxs[c,0]=x
            idxs[c,1]=y
            c+=1
                
    idxs = idxs.astype(dtype='float32')
    
    fi = U_numeric(idxs, mu,si, scaler=1.0, normed=layer_instance.normed)
    if layer_instance.input.shape[-1].value>1:
        fi=np.repeat(fi,3,0)
    fiwe =  fi*we    
    return fi, we, fiwe


def sqrt32(x):
    return np.sqrt(x,dtype='float32')

def lr_schedule(epoch,lr=1e-3):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """

    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr



# code by btek to find different weights
def set_pattern_find(name, keyset):
    ''' this function searchs keyset patterns in name. 
    if finds it, it returns that key. single match only'''
    #print(name, keyset)
    #print(name, keyset)
    for k in keyset:
        if name.find(k) >= 0:
            #print(name, k)
            return k   
    return 'all'

def create_residual_model(input_shape, num_classes=10, settings={}):
    print("num_classes= ",num_classes)
    from tensorflow.keras.models import  Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, AlphaDropout
    from tensorflow.keras.layers import Activation, LeakyReLU,Add ,SpatialDropout2D # MaxPool2D
    act_func = 'relu'
    #act_func = 'selu'
    #act_func = 'relu'
    #drp_out = AlphaDropout
    drp_out = Dropout
    #from keras.regularizers import l2
    
    node_in = Input(shape=input_shape, name='inputlayer')

    if settings['neuron']=='focus2d':
        print("Flatten removed.")
        node_ = Dropout(0.2)(node_in)
    else:
        node_fl = Flatten(data_format='channels_last')(node_in)
        node_ = Dropout(0.2)(node_fl)
    
    heu= initializers.he_uniform
    h = 1
    for nh in settings['nhidden']:
        node_residual = node_
        if settings['neuron']=='focus1d':
            init_mu = settings['focus_init_mu']
            node_ = FocusedLayer1D(units=nh,
                                   name='focus-'+str(h),
                                   activation=settings['activation'],
                                   init_sigma=settings['focus_init_sigma'], 
                                   kernel_initializer=settings['kernel_init'],
                                   init_mu=init_mu,
                                   train_sigma=settings['focus_train_si'], 
                                   train_weights=settings['focus_train_weights'],
                                   si_regularizer=settings['focus_sigma_reg'],
                                   train_mu = settings['focus_train_mu'],
                                   normed=settings['focus_norm_type'],
                                   gain=1.0)(node_)
        elif settings['neuron']=='focus2d':
            init_mu = settings['focus_init_mu']
            node_=FocusedLayer2D(units=nh, name='focus-'+str(h),initer=settings['initer'],
                                 distribution=settings['distribution'], 
                                 kernel_regularizer=settings['kernel_reg'],
                                 train_sigma=settings['focus_train_si'], 
                                 train_mu = settings['focus_train_mu'],
                                 train_weights=settings['focus_train_weights'],
                                 si_regularizer=settings['focus_sigma_reg'],
                                 kernel_initializer=settings['kernel_init'],
                                 normed=settings['focus_norm_type'],
                                 init_mu = init_mu, init_sigma=settings['focus_init_sigma'],#30
                                 activation=settings['activation'], verbose=False)(node_)
        elif settings['neuron']=='dense':
            node_ = Dense(nh,name='dense-'+str(h),activation='linear',
                          kernel_initializer=heu())(node_)
        
        node_ = BatchNormalization()(node_)
        #node_ = LeakyReLU()(node_)
        node_ = Add()([node_, node_residual])
        node_= Activation(act_func)(node_)
        
        node_ = Dropout(0.25)(node_)
        h = h + 1
    
    if settings['neuron']=='focus2d':
        node_ = Flatten(data_format='channels_last')(node_)
        
    node_fin = Dense(num_classes, name='output', activation='linear', 
                     kernel_initializer=initializers.he_uniform(),
                    kernel_regularizer=None)(node_)
    
    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model

def create_simple_model(input_shape, num_classes=10, settings={}):
    print("num_classes= ",num_classes)
    from tensorflow.keras.models import  Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, AlphaDropout
    from tensorflow.keras.layers import Activation, LeakyReLU,SpatialDropout2D,Lambda,Concatenate # MaxPool2D

    #act_func = 'relu'
    act_func = 'relu'
    #act_func = 'selu'
    #act_func = 'relu'
    #drp_out = AlphaDropout
    drp_out = Dropout
    #from keras.regularizers import l2
    
    node_in = Input(shape=input_shape, name='inputlayer')

    if settings['neuron']=='focus2d':
        print("Flatten removed.")
        node_ = Dropout(0.2)(node_in)
    else:
        node_fl = Flatten(data_format='channels_last')(node_in)
        node_ = Dropout(0.2)(node_fl)
    
    heu= initializers.he_uniform
    h = 1
    for nh in settings['nhidden']:
        if settings['neuron']=='focus1d':
            init_mu = settings['focus_init_mu']
            node_ = FocusedLayer1D(units=nh,
                                   name='focus-'+str(h),
                                   activation=settings['activation'],
                                   init_sigma=settings['focus_init_sigma'], 
                                   kernel_initializer=settings['kernel_init'],
                                   init_mu=init_mu,
                                   train_sigma=settings['focus_train_si'], 
                                   train_weights=settings['focus_train_weights'],
                                   si_regularizer=settings['focus_sigma_reg'],
                                   train_mu = settings['focus_train_mu'],
                                   normed=settings['focus_norm_type'],
                                   gain=1.0)(node_)
        elif settings['neuron']=='focus2d':
            init_mu = settings['focus_init_mu']
            node_=FocusedLayer2D(units=nh, name='focus-'+str(h),initer=settings['initer'],
                                 distribution=settings['distribution'], 
                                 kernel_regularizer=settings['kernel_reg'],
                                 train_sigma=settings['focus_train_si'], 
                                 train_mu = settings['focus_train_mu'],
                                 train_weights=settings['focus_train_weights'],
                                 si_regularizer=settings['focus_sigma_reg'],
                                 kernel_initializer=settings['kernel_init'],
                                 normed=settings['focus_norm_type'],
                                 init_mu = init_mu, init_sigma=settings['focus_init_sigma'],#30
                                 activation=settings['activation'], verbose=False)(node_)
        elif settings['neuron']=='dense':
            node_ = Dense(nh,name='dense-'+str(h),activation='linear',
                          kernel_initializer=heu())(node_)
        
        #node_ = LeakyReLU()(node_)
        split1, split2 = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': [1,4]})(node_)
        print(split1)
        print(split2)
        split1= Activation(act_func)(split1)
        split1 = BatchNormalization()(split1)    
        node_ = Concatenate()([split1, split2])
        #node_ -= K.mean(node_)
        #node_ /= K.std(node_)

        #node_ = Lambda(lambda x: K.stack([x[0], x[1]]))([split1, split2])
        print(node_)
        node_ = Dropout(0.25)(node_)
        h = h + 1
    
    if settings['neuron']=='focus2d':
        node_ = Flatten(data_format='channels_last')(node_)
        
    node_fin = Dense(num_classes, name='output', activation='linear', 
                     kernel_initializer=initializers.he_uniform(),
                    kernel_regularizer=None)(node_)
    
    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model
    

def create_cnn_model(input_shape,  num_classes=10, settings={}):
    from tensorflow.keras.models import  Model
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten,Conv2D, BatchNormalization
    from tensorflow.keras.layers import Activation, MaxPool2D
    from tensorflow.keras.layers import LeakyReLU,SpatialDropout2D,Lambda,Concatenate # MaxPool2D
    
    node_in = Input(shape=input_shape, name='inputlayer')
    
    node_conv1 = Conv2D(filters=settings['nfilters'][0],kernel_size=settings['kn_size'][0], padding='same',
                        activation='relu')(node_in)
    node_conv2 = Conv2D(filters=settings['nfilters'][1],kernel_size=settings['kn_size'][0], padding='same',
                        activation='relu')(node_conv1)
    #node_conv3 = Conv2D(filters=nfilters,kernel_size=kn_size, padding='same',
    #                    activation='relu')(node_conv2)

    node_pool = MaxPool2D((2,2))(node_conv2)
    #node_pool = MaxPool2D((4,4))(node_conv2) works good. 
    if settings['neuron']=='focus2d':
        print("Focus2d, Flatten is removed.")
        node_ = Dropout(0.5)(node_pool)
    else:
        node_fl = Flatten(data_format='channels_last')(node_pool)
        node_ = Dropout(0.5)(node_fl)
    #node_fl = Flatten(data_format='channels_last')(node_conv2)

    #node_fl = node_in
    # smaller initsigma does not work well. 
    heu= initializers.he_uniform
    act_func = 'relu'
    h = 1
    
    for nh in settings['nhidden']:
        if settings['neuron']=='focus1d':
            init_mu = settings['focus_init_mu']
            node_ = FocusedLayer1D(units=nh,
                                   name='focus-'+str(h),
                                   activation=settings['activation'],
                                   init_sigma=settings['focus_init_sigma'], 
                                   kernel_initializer=settings['kernel_init'],
                                   init_mu=init_mu,
                                   train_sigma=settings['focus_train_si'], 
                                   train_weights=settings['focus_train_weights'],
                                   si_regularizer=settings['focus_sigma_reg'],
                                   #si_regularizer=None,
                                   train_mu = settings['focus_train_mu'],
                                   normed=settings['focus_norm_type'])(node_)
                                   #si_regularizer=None,
        elif settings['neuron']=='focus2d':
            init_mu = settings['focus_init_mu']
            node_=FocusedLayer2D(units=nh, name='focus-'+str(h),initer=settings['initer'],
                                 distribution=settings['distribution'],
                                 kernel_initializer=settings['kernel_init'],
                                 kernel_regularizer=settings['kernel_reg'],
                                 train_sigma=settings['focus_train_si'], 
                                 train_mu = settings['focus_train_mu'],
                                 train_weights=settings['focus_train_weights'],
                                 si_regularizer=settings['focus_sigma_reg'],
                                 normed=settings['focus_norm_type'],
                                 init_mu = init_mu, init_sigma=settings['focus_init_sigma'],#30
                                 activation=settings['activation'], verbose=False)(node_)
            
            split1, split2 = Lambda(tf.split, arguments={'axis': 3, 'num_or_size_splits': [1,4]})(node_)
            print(split1)
            print(split2)
            split1= Activation(act_func)(split1)
            split1 = BatchNormalization()(split1)    
            node_ = Concatenate()([split1, split2])
                                   
        elif settings['neuron']=='dense':
            node_ = Dense(nh,name='dense-'+str(h),activation='linear',
                          kernel_initializer=heu())(node_)
    
        node_ = BatchNormalization()(node_)
        node_ = Activation('relu')(node_)
        node_ = Dropout(0.5)(node_)
        h = h + 1
        
    if settings['neuron']=='focus2d':
        node_ = Flatten(data_format='channels_last')(node_)
    
    node_fin = Dense(num_classes, name='softmax', activation='softmax', 
                     kernel_initializer=initializers.he_uniform(),
                     kernel_regularizer=None)(node_)

    #decay_check = lambda x: x==decay_epoch

    model = Model(inputs=node_in, outputs=[node_fin])
    
    return model
    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]        

def test_comp(settings,random_sid=9):
    import tensorflow.keras as keras
    #from keras.optimizers import SGD
    from tensorflow.keras.datasets import mnist,fashion_mnist, cifar10    
    #from skimage import filters
    from tensorflow.keras import backend as K
    #from keras_utils import WeightHistory as WeightHistory
    from keras_utils_tf2 import RecordVariable, RecordOutput, \
    PrintLayerVariableStats, SGDwithLR, eval_Kdict, standarize_image_025,\
    standarize_image_01, AdamwithClip, ClipCallback
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    K.clear_session()
    
    epochs = settings['Epochs']
    batch_size = settings['batch_size']

    sid = random_sid  
    np.random.seed(sid)
    tf.random.set_seed(sid)
    
    # MINIMUM SIGMA CAN EFFECT THE PERFORMANCE.
    # BECAUSE NEURON CAN GET SHRINK TOO MUCH IN INITIAL EPOCHS WITH LARGER GRADIENTS
    #, and GET STUCK!
    
            
    print("Loading dataset")
    if settings['dset']=='mnist':
        # input image dimensions 
        img_rows, img_cols = 28, 28  
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        n_channels=1
    
    elif settings['dset']=='cifar10':
        img_rows, img_cols = 32,32
        n_channels=3
        
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # works good as high as 77 for cnn-focus
        #decay_dict = {'all':0.9, 'focus-1/Sigma:0': 1.1,'focus-1/Mu:0':0.9,
        #          'focus-2/Sigma:0': 1.1,'focus-2/Mu:0': 0.9}
        #if cnn_model: batch_size=256 # this works better than 500 for cifar-10
        #decay_epochs =np.array([e_i*10], dtype='int64')
        
    elif settings['dset']=='fashion':
        img_rows, img_cols = 28,28
        n_channels=1

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
                     
    elif settings['dset']=='mnist-clut':
        
        img_rows, img_cols = 60, 60  
        # the data, split between train and test sets
        
        folder='/media/users/suayb/.keras/datasets/'
        data = np.load(folder+"mnist_cluttered_60x60_6distortions.npz")
    
        x_train, y_train = data['x_train'], np.argmax(data['y_train'],axis=-1)
        x_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'],axis=-1)
        x_test, y_test = data['x_test'], np.argmax(data['y_test'],axis=-1)
        x_train=np.vstack((x_train,x_valid))
        y_train=np.concatenate((y_train, y_valid))
        n_channels=1
        
            
    elif settings['dset']=='lfw_faces':
        from sklearn.datasets import fetch_lfw_people
        lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
        
        # introspect the images arrays to find the shapes (for plotting)
        n_samples, img_rows, img_cols = lfw_people.images.shape
        n_channels=1
        
        X = lfw_people.data
        n_features = X.shape[1]
        
        # the label to predict is the id of the person
        y = lfw_people.target
        target_names = lfw_people.target_names
        n_classes = target_names.shape[0]
        
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
        
        from sklearn.model_selection import train_test_split
        
        #X -= X.mean()
        #X /= X.std()
        #split into a training and testing set
        x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=sid) #sid
        
        plt.imshow(X[0].reshape((img_rows,img_cols)))
        plt.show()
    
    num_classes = np.unique(y_train).shape[0]
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], n_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], n_channels, img_rows, img_cols)
        input_shape = (n_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
        input_shape = (img_rows, img_cols, n_channels)
    if settings['dset']!='mnist-clut':
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        #x_train, _, x_test = standarize_image_01(x_train, tst=x_test)
        x_train, _, x_test = standarize_image_025(x_train, tst=x_test)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, n_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, n_channels)
    
    input_shape = (img_rows, img_cols, n_channels)    
    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    
    ### BT ADDING THIS AUG 2020 for MULTIPLE LEARNING RATE. 
    train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(x_train.shape[0],
                                reshuffle_each_iteration=True
                                ).batch(batch_size, 
                                        drop_remainder=True
                                        ).repeat(epochs)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    #from keras_data_tf2 import load_dataset
    #dset = settings['dset']
    #ld_data = load_dataset(dset,normalize_data=True,options=[])
    #x_train,y_train,x_test,y_test,input_shape,num_classes=ld_data
       
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    sigma_reg = settings['focus_sigma_reg']
    sigma_reg = tf.keras.regularizers.l2(sigma_reg) if sigma_reg is not None else sigma_reg
    settings['focus_sigma_reg'] = sigma_reg
    if settings['cnn_model']:
        model=create_cnn_model(input_shape,num_classes, settings=settings)
    else:
        model=create_simple_model(input_shape, num_classes, settings=settings)
    
 
    model.summary()
    #input("Continue")

    #opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, 
    #               decay_epochs,clipvalue=1.0)#, decay=None)
    #opt = AdamwithClip(clips=clip_dict)
    #opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, 
    #                decay_epochs,update_clip=UPDATE_Clip)#, decay=None)

    def lr_schedule_mul(epoch, lr_mul=1.0, decay_epoch=(100,150), decay=0.9):
        if epoch in decay_epoch: 
            return lr_mul*decay
        return lr_mul
    
    #opt1 = tf.keras.optimizers.Adam(lr=settings['lr_all'], momentum=0.9, clipvalue=1.0)
    #opt1 = tf.keras.optimizers.Adam(lr=settings['lr_all'], clipvalue=1.0)
    opt1 = tf.keras.optimizers.SGD(lr=settings['lr_all'],clipvalue=1.0,momentum=0.9)
    opt2 = tf.keras.optimizers.SGD(lr=settings['lr_mu'],clipvalue=1.0,momentum=0.9)
    opt3 = tf.keras.optimizers.SGD(lr=settings['lr_si'],clipvalue=1.0,momentum=0.9)
    #optim_dict={'all':opt1, 'Mu':opt2, 'Sigma':opt3,"gamma":opt2,"beta":opt2}
    optim_dict={'all':opt1, 'Mu':opt2, 'Sigma':opt3}
    MIN_SIG = 1.0/float(max(img_cols,img_rows))
    MAX_SIG = 1.0
    MIN_MU = 0.0
    MAX_MU = 1.0  
    clip_dict ={'Mu':[MIN_MU, MAX_MU], 'Sigma':[MIN_SIG, MAX_SIG]}
    
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)#false ise softmax aç

    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')
    #opt = tf.keras.optimizers.Adam(lr=1e-3, clipvalue=1.0)
                   
    #model.compile(loss=tf.keras.losses.categorical_crossentropy,
    #              optimizer=opt,
    #              metrics=['accuracy'])
    
 
    #callbacks = [tb]
    callbacks = []
       
    red_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='test_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=10, min_lr=1e-5)
    
    callbacks+=[red_lr]
    
    '''
    if  (settings['neuron']=='focus2d') or (settings['neuron']=='focus1d'):
        ccp1 = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
        ccp2 = ClipCallback('Mu',[MIN_MU,MAX_MU])
        #ccp = ClipCallback('Sigma',[MIN_SIG,MAX_SIG])
        
        stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
        stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
        
        pr_1 = PrintLayerVariableStats("focus-1","Weights:0",stat_func_list,stat_func_name)
        pr_2 = PrintLayerVariableStats("focus-1","Sigma:0",stat_func_list,stat_func_name)
        pr_3 = PrintLayerVariableStats("focus-1","Mu:0",stat_func_list,stat_func_name)
        
        #pr_4 = PrintLayerVariableStats("focus-2","Sigma:0",stat_func_list,stat_func_name)
        #red_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-5)
        #schedule = lambda epoch: np.exp()
        
        callbacks += [ccp1, ccp2]    
   
   def learning_rate_for_mu_sigma(lr, epoch,mod):
        if epoch < 5:
            return 27e-6
        elif epoch == 5:
            tf.print ("learning rate set to:",mod['lr_mu_sig'], output_stream=sys.stdout)            
            return mod['lr_mu_sig']
        elif epoch%40==0:  #1e-4, 
            tf.print ("learning rate reduced to:",lr*1e-2, output_stream=sys.stdout)
            return lr*1e-2
        else :
            #print ("lr",lr)
            return lr
        '''

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape(persistent=False) as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            #print(model.trainable_variables)
        gradients = tape.gradient(loss, model.trainable_variables)
        
        #opt1.apply_gradients(zip(gradients, model.trainable_variables))
            
        for var,grad in zip(model.trainable_variables,gradients):
            #print(var.name,"\n")
            opt_key = set_pattern_find(var.name,optim_dict)
            #gradient = tape.gradient(loss, var)
            opt = optim_dict[opt_key]
            print("OPTIM: ",var.name,":",opt_key, grad)
          
            opt.apply_gradients(zip([grad], [var]))
        
        
        #   # now check for clips
        for var in model.trainable_variables:
            opt_key = set_pattern_find(var.name, clip_dict)
            if opt_key !='all':
                print("Clip:", var, clip_dict[opt_key])
                clip_min, clip_max =clip_dict[opt_key]
                var.assign(tf.clip_by_value(var, clip_min, clip_max))
                ''' 
                if False:
                    stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
                    stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
                    stat_str = [n+str(s(var.numpy())) for s,n in zip(stat_func_list,stat_func_name)]
                    
                    print(opt_key,":", stat_str,"\n") # this will not print when tf.function is in progress 
                 '''
        #reduce_lr
        return (train_loss(loss),
        train_accuracy(labels, predictions))
        
        
    @tf.function
    def test_step(images, labels):
      # training=False is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = model(images, training=False)
      t_loss = loss_object(labels, predictions)
    
      test_loss(t_loss)
      test_accuracy(labels, predictions)
      
      
    ########################################################################
      
    print("Batch Norm Weights= ",model.get_layer("batch_normalization_1").trainable_weights)
    test_accuracy_results=[]
    test_loss_results=[]
    train_loss_results=[]
    train_acc_results=[]
    global mu_0,mu_100,mu_250,mu_500,mu_750,all_si_list
    mu_0=[]
    mu_100=[]
    mu_250=[]
    mu_500=[]
    mu_750=[]
    all_si_list =[]
    min_lr=0.005 #0.005
    max_lr=settings['max_lr_all']
    patience=15 #15
    i=0
    if not settings['augment']:

        for epoch in range(epochs):
          # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()
            
            progBar = tf.keras.utils.Progbar(x_train.shape[0])
            #train_ds.shuffle()
            print('Epoch:',epoch)        
            
            opt1.learning_rate.assign(min_lr+ max_lr*np.exp(-(epoch-epochs//2)**2/(epochs//2*0.4)**2))
            lr1 = opt1.learning_rate     
            print("lr1= ",lr1) 

            if epoch>11 and test_loss_results[epoch-1]>test_loss_results[epoch-10] and i>patience-1 :
                i=0
                #lr_mul = lr_schedule_mul(epoch, lr_mul)
                #lr1 = opt1.learning_rate
                lr2 = opt2.learning_rate
                lr3 = opt3.learning_rate
                #opt1.learning_rate.assign(lr1*0.9)           
                opt2.learning_rate.assign(lr2*0.9)
                opt3.learning_rate.assign(lr3*0.9)
                #print("lr1= ",lr1)
                print("lr2= ",lr2)
                print("lr3= ",lr3)
            else:
                i=i+1    

            for idx,(images,labels) in enumerate(iterate_minibatches(x_train, y_train, batch_size, shuffle=True)):
              #print(idx/x_train.shape[0], images.shape)
              #print(batch_size, labels[0:10])
              #input('ds')
              ls, ac = train_step(images, labels)
              values=[('train_loss',ls),('train_accuracy',ac)]
              progBar.update(idx*batch_size, values=values) 
                        
            for test_images, test_labels in test_ds:
              test_step(test_images, test_labels)
              
            template = ' Test Loss: {}, Test Accuracy: {}'

            print(template.format(test_loss.result(),
                                  test_accuracy.result() * 100))
            test_accuracy_results.append(test_accuracy.result().numpy() * 100)
            test_loss_results.append(test_loss.result().numpy())
            train_acc_results.append(train_accuracy.result().numpy()*100)
            train_loss_results.append(train_loss.result().numpy())
            all_si = model.get_layer("focus-1").get_weights()[0] #all mu save
            all_si_list.append(all_si)
            #mu_0.append(all_mu[0,:])
            #mu_100.append(all_mu[100,:])
            #mu_250.append(all_mu[250,:])
            #mu_500.append(all_mu[500,:])
            #mu_750.append(all_mu[750,:])
            def get_index(model,layname):
                names = [n.name for n in model.layers]
                return names.index(layname)
            
            try:
                ix_focus_1 = get_index(model,'focus-1')
                ix_focus_2 = get_index(model,'focus-2')
                ix_focus_3 = get_index(model,'focus-3')
            except:
                print("layers not found")
            
            if epoch%10==0 and False:
                if ix_focus_1 is not None:
                    fn1 = tf.keras.backend.function(model.input,model.layers[ix_focus_1].output)
                    outputs_1 = fn1(np.array([x_train[1000]]))
                if ix_focus_2 is not None:
                    fn3 = tf.keras.backend.function(model.input,model.layers[ix_focus_2].output)
                    outputs_3 = fn3(np.array([x_train[1000]]))
                

                

                from mpl_toolkits.axes_grid1 import make_axes_locatable
                fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1, 5,figsize=(20,15))
                #fig.title('Focus Layers Outputs')
                plot_1=ax1.imshow(np.array(outputs_1[0])[:,:,0].reshape(28,28))
                ax1.set_title("Focus-1 [0] Output")
                
                plot_2=ax2.imshow(np.array(outputs_1[0])[:,:,1].reshape(28,28))
                ax2.set_title("Focus-1 [1] Output")
                plot_3=ax3.imshow(np.array(outputs_1[0])[:,:,2].reshape(28,28))
                ax3.set_title("Focus-1 [2] Output")
                plot_4=ax4.imshow(np.array(outputs_1[0])[:,:,3].reshape(28,28))
                ax4.set_title("Focus-1 [3] Output")
                plot_5=ax5.imshow(np.array(outputs_1[0])[:,:,4].reshape(28,28))
                ax5.set_title("Focus-1 [3] Output")
                
                divider1 = make_axes_locatable(ax1)
                divider2 = make_axes_locatable(ax2)
                divider3 = make_axes_locatable(ax3)
                divider4 = make_axes_locatable(ax4)
                divider5 = make_axes_locatable(ax5)
                
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                cax3 = divider3.append_axes("right", size="5%", pad=0.05)
                cax4 = divider4.append_axes("right", size="5%", pad=0.05)
                cax5 = divider5.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(plot_1,cax=cax1)
                fig.colorbar(plot_2,cax=cax2)
                fig.colorbar(plot_3,cax=cax3)
                fig.colorbar(plot_4,cax=cax4)
                fig.colorbar(plot_5,cax=cax5)
                fig.tight_layout()
                plt.show()
                print("Focus 1 - 0 mean=",np.mean(np.array(outputs_1[0])[:,:,0]))
                print("Focus 1 - 0 max=",np.max(np.array(outputs_1[0])[:,:,0]))
                print("Focus 1 - 0 min=",np.min(np.array(outputs_1[0])[:,:,0]))
                print("Focus 1 - 0 var= ",np.var(np.array(outputs_1[0])[:,:,0]))

                print("Focus 1 - all var= ",np.var(np.array(outputs_1[0])))

                fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1, 5,figsize=(20,15))
                #fig.title('Focus Layers Outputs')
                plot_1=ax1.imshow(np.array(outputs_3[0])[:,:,0].reshape(28,28))
                ax1.set_title("Focus-2 [0] Output")
                plot_2=ax2.imshow(np.array(outputs_3[0])[:,:,1].reshape(28,28))
                ax2.set_title("Focus-2 [1] Output")
                plot_3=ax3.imshow(np.array(outputs_3[0])[:,:,2].reshape(28,28))
                ax3.set_title("Focus-2 [2] Output")
                plot_4=ax4.imshow(np.array(outputs_3[0])[:,:,3].reshape(28,28))
                ax4.set_title("Focus-2 [3] Output")
                plot_5=ax5.imshow(np.array(outputs_3[0])[:,:,4].reshape(28,28))
                ax5.set_title("Focus-2 [4] Output")
                
                divider2 = make_axes_locatable(ax2)
                divider3 = make_axes_locatable(ax3)
                divider4 = make_axes_locatable(ax4)
                divider5 = make_axes_locatable(ax5)
                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)

                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                cax3 = divider3.append_axes("right", size="5%", pad=0.05)
                cax4 = divider4.append_axes("right", size="5%", pad=0.05)
                cax5 = divider5.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(plot_1,cax=cax1)
                fig.colorbar(plot_2,cax=cax2)
                fig.colorbar(plot_3,cax=cax3)
                fig.colorbar(plot_4,cax=cax4)
                fig.colorbar(plot_5,cax=cax5)
                fig.tight_layout()
                plt.show()

                print("Focus 2 - 0 mean=",np.mean(np.array(outputs_3[0])[:,:,0]))
                print("Focus 2 - 0 max=",np.max(np.array(outputs_3[0])[:,:,0]))
                print("Focus 2 - 0 min=",np.min(np.array(outputs_3[0])[:,:,0]))
                print("Focus 2 - 0 var= ",np.var(np.array(outputs_3[0])[:,:,0]))      
            
                print("Focus 2 - all var= ",np.var(np.array(outputs_3[0])))
            
        
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False,
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format='channels_last',
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        for e in range(epochs):
          train_loss.reset_states()
          train_accuracy.reset_states()
          test_loss.reset_states()
          test_accuracy.reset_states()
          progBar = tf.keras.utils.Progbar(x_train.shape[0])
          print('Epoch', e)
          opt1.learning_rate.assign(min_lr+ max_lr*np.exp(-(e-epochs//2)**2/(epochs//2*0.2)**2))
          lr1 = opt1.learning_rate
          print("lr1= ",lr1)
          if e>11 and test_loss_results[e-1]>test_loss_results[e-10] and i>patience-1 :
                i=0
                #lr_mul = lr_schedule_mul(epoch, lr_mul)
                #lr1 = opt1.learning_rate
                lr2 = opt2.learning_rate
                lr3 = opt3.learning_rate
                #opt1.learning_rate.assign(lr1*0.9)#weight           
                opt2.learning_rate.assign(lr2*0.9)#mu
                opt3.learning_rate.assign(lr3*0.9)#sigma
                #print("lr1= ",lr1)
                print("lr2= ",lr2)
                print("lr3= ",lr3)
          else:
                i=i+1
          batches = 0
          for idx,(x_batch, y_batch) in enumerate(datagen.flow(x_train, y_train, batch_size=batch_size)):
            ls, ac = train_step(x_batch, y_batch)
            values=[('train_loss',ls),('train_accuracy',ac)]
            progBar.update(idx*batch_size, values=values) 
            batches += 1
            if batches >= len(x_train) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
              break

          for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
              
          template = ' Test Loss: {}, Test Accuracy: {}'

          print(template.format(test_loss.result(),
                                  test_accuracy.result() * 100))
          test_accuracy_results.append(test_accuracy.result().numpy() * 100)
          test_loss_results.append(test_loss.result().numpy())
          train_acc_results.append(train_accuracy.result().numpy()*100)
          train_loss_results.append(train_loss.result().numpy())
           
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    
    #print(model.get_layer("focus-1").weights)
    #print("test_accuracy",test_accuracy_results)
    score=[test_loss_results[-1],test_accuracy_results[-1]]
    return train_loss_results,train_acc_results,test_loss_results,test_accuracy_results, score ,model, callbacks


def repeated_trials(test_function=None, settings={}):
    
    list_scores =[]
    list_val_acc_histories =[]
    list_val_loss_histories =[]
    list_train_loss_histories=[]
    list_train_acc_histories=[]
    list_sigmas = []
    sigmas = settings['focus_sigma_reg']
    sigmas = [None] if sigmas is None or sigmas is [] else sigmas
    models = []
    
    import time 
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    from datetime import datetime
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    
    if settings['cnn_model']:
        filename = '/media/users/suayb/outputs/deneme/'+settings['dset']+'/'+timestr+'_'+settings['neuron']+'_cnn'+'_'+str(settings['batch_size'])+'.model_results.npz'
    else:
        filename = timestr+'_'+settings['neuron']+'_simple'+'_'+str(settings['batch_size'])+'.model_results.npz'
    #copyfile("Kfocusingtf2.py",filename+"code.py")
   
    for s in range(len(sigmas)): # sigmas loop, should be safe if it is empty
        for i in range(settings['repeats']):
            
            sigma_reg = sigmas[s] if sigmas else None
            print("REPEAT",i,"sigma regularization", sigma_reg)
            #run_settings = settings.copy()
            settings['focus_sigma_reg'] = sigma_reg
            hs_train_loss,hs_train_acc,hs_val_loss,hs_val_acc,sc, ms, cb = test_function(random_sid=i*78,settings=settings)
            list_scores.append(sc)
            list_val_acc_histories.append(hs_val_acc)
            list_val_loss_histories.append(hs_val_loss)
            list_train_loss_histories.append(hs_train_loss)
            list_train_acc_histories.append(hs_train_acc)
            models.append(ms)
            # record current regularizer and final sigma 
            '''
            if (settings['neuron']=='focus1d' or settings['neuron']=='focus2d') and sigma_reg:
                list_sigmas.append([sigma_reg, np.mean(cb[4].record[-1])])        
            '''
    print("Final scores", list_scores)
    mx_scores = [np.max(list_val_acc_histories[i]) for i in range(len(list_val_acc_histories))]
    histories = [{'val_acc':list_val_acc_histories,'val_loss':list_val_loss_histories,
                  'train_acc':list_train_acc_histories,'train_loss':list_train_loss_histories}]
    all_si_npz=[{'all_si':all_si_list}]
    print("Max scores", mx_scores)
    settings_a = [settings]
    np.savez_compressed(filename, mx_scores=mx_scores, list_scores=list_scores, 
                        modelz=histories, sigmas=list_sigmas, all_si=all_si_npz, settings=settings_a)
    return models

    
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES']="0"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
    print("Run as main")
    #test()
    delayed_start = 0*3600
    import time 
    print("Delayed start ",delayed_start)
    time.sleep(delayed_start)
    #dset='mnist'
    dset='cifar10'  # ~64,5 cifar is better with batch 256, init_sigma =0.01 
    #dset='mnist'
    #dset = 'mnist-clut'
    #set = 'fashion'
    #dset='lfw_faces' # ~78,use batch_size = 32, augment=True, init_sigm=0.025, init_mu=spread
    sigma_reg_set = [1e-10]
    kernel_reg=None
    nhidden = (784,784) #(784,784) #1225,784
    #nhidden = (256,)
    #sigma_reg_set = [1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    #fashion 90.9 batch_size 32, sigma_reg 1e-10,adam lr 0.0015,1600,sigma 0.08
  
    mod={'dset':dset, 'neuron':'focus2d', 'nhidden':nhidden, 'cnn_model':False,
         'nfilters':(28,28), 'kn_size':(5,5),'activation':'linear',
         'focus_init_sigma':[1./32,1/32.],'focus_init_mu':'spread','focus_train_mu':True, 
         'focus_train_si':True,'focus_train_weights':True,'focus_norm_type':2,
         'focus_sigma_reg':sigma_reg_set,'kernel_reg':kernel_reg,'augment':False, 
         'Epochs':200, 'batch_size':128,'repeats':5,
         'max_lr_all':0.05, # this was hard coded
         'lr_all':0.01,'lr_mu':0.001,'lr_si':0.001,'initer':'glorot','distribution':'uniform',
         'kernel_init':None}
         
         
    if mod['neuron']=='focus2d':
        temp=mod['nhidden']
        mod['nhidden']=(int(np.sqrt(mod['nhidden'][0])),)
        for i in range(1,len(temp)):
            mod['nhidden']+=(int(np.sqrt(temp[i])),)

    mod['kernel_reg'] = tf.keras.regularizers.l2(mod['kernel_reg']) if mod['kernel_reg'] is not None else mod['kernel_reg']
    
    f = test_comp
    res = repeated_trials(test_function=f, settings=mod)
    
# max_lr_all was hard coded 0.05 all this experiments. 
# fashion simple focus-2d lr 0.01, si=0.1, mu=new spread. hidden 784,84, sigmra_reg=1e-10
# Max scores [91.60000085830688, 91.43999814987183, 91.6700005531311, 91.82999730110168, 91.51999950408936]

# fashion CNN-focus mu train -false. new spread set. 
# Max scores [94.34000253677368, 94.40000057220459, 94.16999816894531, 94.06999945640564, 94.26000118255615]
# Max scores [94.2900002002716, 94.34000253677368, 94.2900002002716, 94.02999877929688, 94.30999755859375]

# mnist random si. focus new spread. 
# Max scores [99.22999739646912, 99.19000267982483, 99.22000169754028, 99.19000267982483, 99.16999936103821]
# mnist si=.1 mu_spread 2sigma,1-2sigma, lr0.01, lrmu0.001
# Max scores [99.19000267982483, 99.2900013923645, 99.30999875068665, 99.27999973297119, 99.2900013923645]

# cifar new spread, bs = 128 
# Max scores [66.36000275611877, 66.54000282287598, 67.00000166893005, 66.51999950408936, 65.79999923706055]
# lrmu,lrsi=0.001 BEST. 
# Max scores [67.41999983787537, 68.09999942779541, 67.41999983787537, 67.28000044822693, 67.11000204086304]
# new spread with ix[3],[-4]
# Max scores [67.98999905586243, 67.29999780654907, 67.5499975681305, 67.21000075340271, 67.33999848365784]
# this one is si=.1 mu_spread 1sigma,1-1sigma, lr0.01, lrmu0.001
#Max scores [67.21000075340271, 67.33999848365784, 67.330002784729, 67.65000224113464, 67.94999837875366]
# random si. 
# Max scores [63.73000144958496, 63.31999897956848, 63.55999708175659, 63.48999738693237, 63.669997453689575]
