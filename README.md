# Network-Anomaly-Detection

This project aims to provide a setup for anomaly detection in Networking, specifically to detect DDoS attacks

### Introduction


Many IoT devices are becoming victims of hackers due to their lack of security and they are often turned into botnets conducting Distributed Denial of Service (DDoS) attacks. We aim to detect those attacks by analyzing their network traffic. 

When designing the model, one has to keep in mind that in a real life scenario, the attack detection is relevant only if it is conducted in a streaming/near real time way.

 

### Results

Results for testing on 2 different classes (ChirpJammer and NarrowBandSignal) is shown in fig below ![AutoEncoderResults](extra/autoencoder-results.png) 



### Setup Instructions:
#### 1. Requirements

To reproduce the results from this repository, it is recommended to use virtual python environment and python version 3.6 Tensorflow version 2.3 was used to build the models. The project is tested only on Linux

Follow these simple steps to setup the dependencies:

```shell
git clone https://github.com/AkhilSinghRana/Network-Anomaly-Detection.git

cd Network-Anomaly-Detection/ 

virtualenv env_name -p python3

source env_name/bin/activate #for linux


pip install -e .

 ```

Note*- The above code will setup all the required dependencies for you. Tested only on Linux


You are now ready to train the models. I recommend to also browse through notebooks folder to understand the workflow a bit better.


#### 2. Train on your own Dataset:

Instructions for training on your own Dataset is shown in the notebook below. 

Jupyter Notebook:

``` jupyter notebook notebooks/Model_Training.ipynb  ```
 
Running from a Terminal:

``` python main.py --help ```

Sample training command, I assume that the csv files are stored inside data folder of the root project directory & the command is run from root_dir

```shell

python main.py --data_path data/data.csv --mode train


```
         
  
#### 2. Testing/Loading model from checkpoint:


Running from a Terminal:

``` python main.py --help ```

Sample test command from root_dir 

```shell

python main.py --data_path data/test.csv --mode test

```

The above test command will take the test csv and pass through the autoencoder for creating latent representation of the test data, which then be passed through to the trained linear regression classifier
         
