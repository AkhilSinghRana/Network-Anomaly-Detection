import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
import seaborn as sns

import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class processData():
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)

        pass

    def save_df(self, predictions,num_samples, file_path):
        self.raw_data["label"] = predictions[:num_samples]
        self.raw_data.to_csv(file_path)

    def visualizeData(self):
        """
            Visualize the data,
        """
        pass

    def dataScaling(self, x):
        #let's do bit of data transformation for scaling
        x_scale = sklearn.preprocessing.MinMaxScaler().fit_transform(x.values)
        return x_scale

    def prepareTrainingData(self):

        # Based on Assumption I set all the data labeled as normal --> 0,
        # everything else as anomaly --> 1
        

        LABELS = self.raw_data["label"].unique() # Get Unique Values of label column
        LABELS = [label for label in LABELS if label != "normal"] #All labels other than Normal!

        #set inplace to True if you want to change the raw data
        self.raw_data["label"].replace(['normal'], 0, inplace=True)
        self.raw_data["label"].replace(LABELS, 1 , inplace=True)
        data = self.raw_data
        # fill all the empty hosts with None
        data = data.fillna("None") 

        #Now let's encode all the columns which are not float or int e.g. sourceIP, MAC to make it possible for model to interpret
        columnsToEncode = list(data.select_dtypes(include=['category', 'object']))  
                    
        le = LabelEncoder() # use label encoder from sklearn
        for feature in columnsToEncode:
            try:
                data[feature] = le.fit_transform(data[feature])
                #print(data[feature])
            except:
                print ('error' + feature)


        #Let's split the data to train and validation
        Train, Val = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=1, shuffle=True)
        
        #Let's now split the features and the target ground truth
        features = [feature for feature in Train.columns.tolist() if feature not in ["label"]]
        target = "label"

        # Define a random state 
        state = np.random.RandomState(42)
        X_train = Train[features]
        Y_train = Train[target]

        X_val = Val[features]
        Y_val = Val[target]

        X_outliers = state.uniform(low=0, high=1, size=(X_train.shape[0], X_train.shape[1]))

        print("Data processed ...")
        return (X_train, Y_train , X_val, Y_val)

    def prepareTestData(self):
            data = self.raw_data
            # fill all the empty hosts with None
            data = data.drop(data.columns[0], axis=1)
            data = data.fillna("None") 

            #Now let's encode all the columns which are not float or int e.g. sourceIP, MAC to make it possible for model to interpret
            columnsToEncode = list(data.select_dtypes(include=['category', 'object']))  
                        
            le = LabelEncoder() # use label encoder from sklearn
            for feature in columnsToEncode:
                try:
                    data[feature] = le.fit_transform(data[feature])
                    #print(data[feature])
                except:
                    print ('error' + feature)

            return data
    