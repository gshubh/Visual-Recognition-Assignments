from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import math


plt.rcParams['figure.figsize'] = (5.0, 4.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)
input_img = Input(shape=(28, 28, 1))   

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

print("shape of encoded", K.int_shape(encoded))

#Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

print("shape of encoded", K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))    
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='conv_autoencoder')], verbose=2)

decoded_imgs = autoencoder.predict(x_test)

# Output from keras implimentation
n = 20
plt.figure(figsize=(10, 4), dpi=100)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

plt.show()





def relu(Z):
    return np.maximum(Z,0)

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm

def zero_pad(X, pad):
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values=0)
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W)
    Z = np.sum(s)
    Z = Z+b
    return Z

def conv_forward(A_prev, W, b, hparameters):
    W, b = W
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = 1
    if(hparameters["padding"] == "same"):
        pad = int((f-1)/2)
    else:
        pad = 0
        
    n_H = math.floor((n_H_prev -f +2*pad)/stride + 1)
    n_W = math.floor((n_W_prev -f +2*pad)/stride + 1)
    
    Z = np.zeros((m,n_H,n_W,n_C))
    A = np.zeros((m,n_H,n_W,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):                               
        a_prev_pad = A_prev_pad[i]                              
        for h in range(n_H):                           
            for w in range(n_W):                       
                for c in range(n_C):                   
                    
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[c])
    
    if(hparameters["activation"] == 'relu'):
        A = np.maximum(0, Z)
    elif(hparameters["activation"] == 'sigmoid'):
        A = sigmoid(Z)
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    
    return A

def pool_forward(A_prev, hparameters, mode = "max"):
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    pool_width = hparameters['pool_size'][0]
    pool_height = hparameters['pool_size'][1]
    
    pad_width = n_W_prev % pool_width
    pad_height = n_H_prev % pool_height
    
    A_prev = np.pad(A_prev, [(0, 0), (0, pad_width), (0, pad_height), (0, 0)], 'constant',constant_values=0)
    
    f = pool_width
    stride = hparameters["strides"][0]
    stride = 2
    n_H = int(1 + (n_H_prev - f + 2*pad_height) / stride)
    n_W = int(1 + (n_W_prev - f + 2*pad_width) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         
        for h in range(n_H):                     
            for w in range(n_W):                 
                for c in range (n_C):            
                    
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w*stride
                    horiz_end = horiz_start + f
                    
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.average(a_prev_slice)
    
    
    cache = (A_prev, hparameters)
    
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A

# np.random.seed(1)
# A_prev = np.random.randn(10,4,4,3)
# W = np.random.randn(2,2,3,8)
# b = np.random.randn(1,1,1,8)
# hparameters = {"pad" : 2,
#                "stride": 2}

# A,Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
# print("Z's mean =", np.mean(Z))
# print("Z[3,2,1] =", Z[3,2,1])
# print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
# print("A == : ",A[3,2,1])

def upsampling(A_prev,shape=(2,2)):
    row = shape[0]
    col = shape[1]
    temp = np.repeat(A_prev, row, axis = 1)
    A = np.repeat(temp, col, axis = 2)
    return A

def forward_pass(x, layers):
    for single_layer in layers[1:]:
        name = single_layer.__class__.__name__
        parameters = single_layer.get_config()
        print(name)
        if(name == 'Conv2D'):
            x = conv_forward(x, single_layer.get_weights(),0, parameters)
        elif(name == 'MaxPooling2D'):
            x = pool_forward(x, parameters)
        elif(name == 'UpSampling2D'):
            x = upsampling(x)
        print("Hello")
        
    return x

images = 20
XX = x_test[0:images,:,:,:]
print(XX.shape)

y_predict = forward_pass(x_test, autoencoder.layers)
print(y_predict.shape)

plt.figure(figsize=(images, 2), dpi=100)
for i in range(images):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(y_predict[i].reshape(28, 28))
    plt.gray()
    ax.set_axis_off()

plt.show()

