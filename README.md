# Network-Anomaly-Detection

This project aims to provide a setup for anomaly detection in Networking, specifically to detect DDoS attacks

### Introduction


Many IoT devices are becoming victims of hackers due to their lack of security and they are often turned into botnets conducting Distributed Denial of Service (DDoS) attacks. We aim to detect those attacks by analyzing their network traffic. 

When designing the model, one has to keep in mind that in a real life scenario, the attack detection is relevant only if it is conducted in a streaming/near real time way.

We will create an autoencoder model in which we only show the model non-fraud cases. The model will try to learn the best representation of normal cases. The same model will be used to generate the representations of cases where a DDoS attack is done, and we expect them to be different from normal ones.

Create a network with one input layer and one output layer having identical dimentions ie. the shape of non-fraud cases. We will use keras package to craete our model.

![AutoEncoder](extra/autoencoder-net-arch.png) 

The beauty of this approach is that we do not need too many samples of data for learning the good representations. We will use only 8000 rows of normal cases to train the autoencoder. Additionally, We do not need to run this model for a large number of epochs, running it for 10 epochs was sufficient.

Explanation: The choice of small samples from the original dataset is based on the intuition that one class characteristics (normal) will differ from that of the other (DDoS-attack). To distinguish these characteristics we need to show the autoencoders only one class of data. This is because the autoencoder will try to learn only one class and automaticlly distinuish the other class.


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

Sample test command from root_dir 

```shell

python main.py --data_path data/test.csv --mode test

```

The above test command will take the test csv and pass through the autoencoder for creating latent representation of the test data, which then be passed through to the trained linear regression classifier
         
#### 3. Visualizing Data Relationship with TSNE:

T-SNE (t-Distributed Stochastic Neighbor Embedding) is a dataset decomposition technique which reduced the dimentions of data and produces only top n components with maximum information.

Every dot in the following represents a request. Normal transactions are represented as Green while potential attacks are represented as Red. The two axis are the components extracted by tsne.

| TSNE on Normal scaled data vs | TSNE on embedded Latent representation|
|![TSNE](extra/TSNE-1.png) | ![TSNE-1](extra/TSNE-embeddings.png)|