from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns

class AutoEncoder:
    def __init__(self, data_shape):
        self.data_shape = data_shape
        
    
    def net_arch(self):
        ## input layer 
        self.input_layer = Input(shape=(self.data_shape,))

        ## encoding part
        encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(self.input_layer)
        encoded = Dense(50, activation='relu')(encoded)

        ## decoding part
        decoded = Dense(50, activation='tanh')(encoded)
        decoded = Dense(100, activation='tanh')(decoded)

        ## output layer
        self.output_layer = Dense(self.data_shape, activation='relu')(decoded)

    def net_model(self):
        #Compile the mdoel
        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer="adadelta", loss="mse")

        return autoencoder