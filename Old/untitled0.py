#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:41:42 2021

@author: cjrichier
"""

'''This is a pass at defining a series of functions that
are reproducible and deployable for using a 3dCNN on neuroimaging data.
All names and structures are tentative'''

##################################
#### Load in relevant modules ####
##################################

#general
import os
import numpy as np
import pandas as pd
import matplotlib as plt
#neuroimaging
import nibabel as nb
#sklearn
from sklearn.model_selection import train_test_split
#TensorFlow/keras
import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPool3D, BatchNormalization, Input
from tensorflow.keras import activations, regularlizers

######################################
#### Define some useful functions ####
######################################

def loadNifti(filename, NonSteadyState=0): 
    '''Takes a list of various types of 4 dimensional brain data
    (subject by X by Y by Z dimensions) and stacks them together in
    a fifth dimension, making it usable for TensorFlow 3DCNN's.
    Arguments:
        filename: name of the file
        NonSteadyState: ?
    Returns:
        tuple of 2d image, the dimensions, and the affine matrix
        '''
    n = nb.load(filename)
    naff   = n.affine
    img4d  = n.get_fdata();
    imgsiz = img4d.shape
    if len(imgsiz) == 4:
        img4d  = img4d[:,:,:,NonSteadyState:]
        imgsiz = img4d.shape
        img2d  = np.reshape(img4d, (np.prod(imgsiz[0:3]), imgsiz[-1]), order='F').T
    else:
        img2d  = np.reshape(img4d, (np.prod(imgsiz[0:3]), 1), order='F').T
    return img2d, imgsiz, naff

def saveNifti(filename, X, size, affine, mask):
    size = (*size[0:3], X.shape[0])
    if len(mask) == 0:
        mask = np.ones(X.shape[-1], dtype=bool)
    img = np.zeros((X.shape[0], len(mask)))
    img[:, mask] = X
    n = nb.Nifti1Image(np.reshape(np.transpose(img), size, order='F'), affine)
    nb.save(n, filename)
 
def loadGifti(fname, NonSteadyState=0, icres=7):
    gii = nb.load(fname)
    gii_data = [d.data[:,None] for d in gii.darrays]
    gii_data = np.concatenate(gii_data, axis=1).T
    nV = 4**icres*10 + 2
    gii_data = gii_data[:,0:nV]
    return gii_data[NonSteadyState:,:]

def stack_input_data(input_data):
    '''Takes a list of various types of 4 dimensional brain data
    (subject by X by Y by Z dimensions) and stacks them together in
    a fifth dimension, making it usable for TensorFlow 3DCNN's.
    Arguments:
        input_data: a list of 4D brain arrays as described above
    Returns:
        input_data: a 5D array of stacked image types
        (subject by X by Y by Z by image type)
    '''
    input_data = np.stack(data_list, axis=4)    
    return input_data


def brain_3DCNN(input_data, outputs, optimizer, loss_function, validation_set=False, output_type='continuous'):
    '''Creates a 3D CNN used by .........
    
    ¯\_(ツ)_/¯ 
    
    Arguments:
        input_data: 5d array of brain imaging data
        outputs: vector of predctions
        optimizer: method of optimization
        loss_function: choice of loss function
        output_type: specifies what type of activation function to use for final class. 
        defaults to continuous but also takes binary
    Returns:
        model: an model keras object to fit on data
    '''
    
    #######################################
    # First step: Make the test train split 
    #######################################
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(gm_and_wm_3d, covariates['Age'][:].astype(float)) #age needs to be a float for the model to run for some reason
    
    #if employing a validation set, we split up train to be some validation:
    if validation_set == True:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
    #But if not, we make the full X train the training set
    if validation_set == False:
        X_train = X_train_full
    
    ##########################  
    # Define the Keras Model #
    ##########################
    
    #First, we make the input layer. Takes the inputs but drops the batch size dimension to be compatible with Keras
    inputs = Input(shape=X_train[0, :, :, :, :], name='input')
 
    ######################################################################
    # Now we start the model. We move through phases of groups of layers #
    # These groups are a convolution layer, batch normalization,         #
    # convolution, batch norm, and a max pool                            #
    # We start with 8 filters, a 3x3x3 size kernel, same padding, and    #
    # regularize each layer with L2                                      #
    # We include detailed explanations below.
    ######################################################################   
    # first, we have a 3D convolution layer. This layer slides the 8 unique kernels along
    # the 3D image and convolutes the result. The weights are parameterized with the L2 regularizer.
    x = Conv3D(8, kernel_size=(3,3,3), activation='relu', padding='same', 
        kernel_regularizer=regularizers.l2(0.001), name='cnv_1')(inputs)  
    # Next, we have batch normalization (BN). BN zero-centers and normalizes each input 
    # to the next layer, then scales and shifts the resultant output. It is learning the scale 
    # and mean of each layers inputs. It takes the mean and SD over each batch. 
    x = BatchNormalization(name='bn_1')(x)
    # This pattern of Convolution and BN is repeated
    x = Conv3D(8, kernel_size=(3,3,3), activation='relu', padding='same',
        kernel_regularizer=regularizers.l2(0.001), name='cnv_2')(x)
    x = BatchNormalization(name='bn_2')(x)  
    # Lastly, our first set of layers includes a max pooling (MP) layer. 
    # The MP layer downsamples the image by taking the max value within the size of the kernel.
    # Whatever value is hihest in the 3x3x3 kernel gets passed on to the next convolutional layer.
    x = MaxPool3D(pool_size=(3, 3, 3),strides=(2,2,2), name='mxp_1')(x)
    ######################################################################
   
    
    #########################################################################
    # Now we move on to the next block. The size of the image has decreased #
    # We half the size of the image but double the number of filters        #
    # This pattern is consistent through the network                        #
    #########################################################################
    x = Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Conv3D(16, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_4')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = MaxPool3D(pool_size=(3, 3, 3),strides=(2,2,2), name='mxp_2')(x)
    #########################################################################
   
    ##############
    # next block #
    #####################################################################
    x = Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_5')(x)
    x = BatchNormalization(name='bn_5')(x)
    x = Conv3D(32, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_6')(x)
    x = BatchNormalization(name='bn_6')(x)
    x = MaxPool3D(pool_size=(3, 3, 3),strides=(2,2,2), name='mxp_3')(x)
    #####################################################################
 
    ##############
    # next block #
    #####################################################################
    x = Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_7')(x)
    x = BatchNormalization(name='bn_7')(x)
    x = Conv3D(64, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_8')(x)
    x = BatchNormalization(name='bn_8')(x)
    x = MaxPool3D(pool_size=(3, 3, 3),strides=(2,2,2), name='mxp_4')(x)
    #####################################################################

    ##############
    # next block #
    #####################################################################
    x = Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_7b')(x)
    x = BatchNormalization(name='bn_7b')(x)
    x = Conv3D(128, kernel_size=(3,3,3), activation='relu', padding='same',
          kernel_regularizer=regularizers.l2(0.001), name='cnv_8b')(x)
    x = BatchNormalization(name='bn_8b')(x)
    x = MaxPool3D(pool_size=(3, 3, 3),strides=(2,2,2), name='mxp_4b')(x) 
    #####################################################################
 
    ############################################################################################
    #now we flatten the network, so that we can predict the continuous singular outcome variable
    ############################################################################################
    x = Flatten(name='flt_1')(x)
    x = BatchNormalization(name='bn_9')(x)
    x = Dropout(0.5, name='dpt_1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='d_1')(x)
    x = tf.keras.layers.Dense(64, activation = 'relu')(x)
    x = Dropout(.5)(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dropout(.5)(x)
    ############################################################################################
   
   
    ##############################
    # Now for the output layers: #
    ##############################
   
    if output_type == 'continuous':
        outputs = Dense(1, activation = 'relu', name='output', dtype='float32')(x)
    if output_type == 'binary':
        outputs = Dense(1, activation = 'sigmoid', name='output', dtype='float32')(x)
        
        
        