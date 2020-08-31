import numpy as np
import os, random, math, sys, signal
from shutil import copyfile
from importlib import reload
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# for debugging
#import ptvsd
#ptvsd.enable_attach(log_dir= os.path.dirname(__file__))
#ptvsd.wait_for_attach(timeout=15)

from util import options, data_processing
from ml_models import autoencoder

def train(args):
    print(args.model_name)
    
    #Prepare the data for training
    process_data = data_processing.processData(args.data_path)
    X_train, Y_train, X_val, Y_val = process_data.prepareTrainingData()
    
    # Pre-train AutoEncoder with normal data
    model = autoencoder.AutoEncoder(X_train.shape[1])
    compiled_model = model.compile_model() 
    print("model compiled")
    
    # We need to rescale the data
    x_scale = process_data.dataScaling(X_train)
    x_norm, x_fraud = x_scale[Y_train == 0], x_scale[Y_train == 1]
    
    
    #Let's pretrain the compiled autoencoder with normal data, we don't need to train with every data point!
    compiled_model.fit(x_norm[0:8000], x_norm[0:8000], 
                batch_size = 256, epochs = 50, 
                shuffle = True, validation_split = 0.20);
    
    save_path = os.path.join(args.ckpt_path, args.model_name)
    model.save_load_models(path=save_path, model=compiled_model)
    del(compiled_model)
    compiled_model = model.save_load_models(path=save_path, mode="load")

    # Now Let's try to get latent representation of the trained model
    hidden_representation = model.getHiddenRepresentation(compiled_model)

    norm_hid_rep = hidden_representation.predict(x_norm[9000:15000])
    fraud_hid_rep = hidden_representation.predict(x_fraud[9000:15000])
    rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis = 0)
    y_n = np.zeros(norm_hid_rep.shape[0])
    y_f = np.ones(fraud_hid_rep.shape[0])
    rep_y = np.append(y_n, y_f)

    #Finally we can train a classifier on learnt representations

    train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, test_size=0.25)
    clf = LogisticRegression(solver="lbfgs").fit(train_x, train_y)
    pred_y = clf.predict(val_x)
    #Let's also save this classifier for future
    filename = os.path.join(save_path, "linear_regression_Classifier.pkl")
    s = pickle.dump(clf, open(filename, 'wb'))


    print("model is now pickled and saved for later")
    

    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))

    print ("")
    print ("Classification Report: ")
    print (classification_report(val_y, pred_y))

    print ("")
    print ("Accuracy Score: ", accuracy_score(val_y, pred_y))



def test(args):
    #Step1 Read the test data, you can use the helper funtions from util that were also used duting training
    process_data = data_processing.processData(args.data_path)
    X_test = process_data.prepareTestData()
    
    #Step2: Pass the data to same pre-processing that was used for training
    # # We need to rescale the data
    model = autoencoder.AutoEncoder(X_test.shape[1])
    x_scale = model.inputPipeline(X_test)

    #Step3: Let's load the trained autoencoder into memory
    # Pre-train AutoEncoder with normal data
    save_path = os.path.join(args.ckpt_path, args.model_name)
    compiled_model = model.compile_model() 
    compiled_model = model.save_load_models(path=save_path, mode="load")
    
    #Step4: Get latent representation of test data
    hidden_representation = model.getHiddenRepresentation(compiled_model)
    
    #step 5 Load the Classifier now
    # load the model from disk
    filename = os.path.join(save_path, "linear_regression_Classifier.pkl")
    loaded_model = pickle.load(open(filename, 'rb'))

    #Step6: Get predictions and save to the data
    batch_size = 5000 # increase this according to the memory space you have
    num_samples = len(x_scale)
    print(num_samples)
    predictions = []
    for idx in range(0, num_samples, batch_size):
        norm_hid_rep = hidden_representation.predict(x_scale[idx: idx+batch_size])
        pred_y = loaded_model.predict(norm_hid_rep)
        predictions.extend(pred_y)
        
    process_data.save_df(predictions,num_samples,"test_output.csv")

if __name__ == "__main__":
    args = options.ArgumentParser()

    if args.mode == "train":
            print("Training")
            # train()
            train(args)
    elif args.mode=="test":
            print("Testing")
            test(args)
    
    elif args.mode=="colab":
            pass
    else:
            raise Exception 