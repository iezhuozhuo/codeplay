# -*- coding: utf-8 -*-
# @Author  : zhuo & zdy
# @github   : iezhuozhuo
import torch.nn as nn

def cnn_output_size(input_size,filter_size,stride,padding):
    '''
    Calculate the size of a conv2d_layer output or pooling2d_layer output tensor based on its hyperparameters. All hyperparameters should be 2-element tuple or array of length 2.
    '''
    return [int((input_size[i]-filter_size[i]+padding[i])/stride[i] +1 )for i in range(2)]

