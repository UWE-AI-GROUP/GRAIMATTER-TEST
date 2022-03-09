import os
os.chdir('/home/ec2-user/GRAIMatter/WP1/models/Synthetic_data/models/

from data_generator import data_generator
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pate_gan import pategan
import seaborn as sns
import numpy as np

parameters = {'n_s': 1, 'batch_size': 64, 'k': 100, 
                'epsilon': 100, 'delta': 0.0001, 
                'lamda': 1}
      
train_data, test_data = data_generator(1000, 2, 1)

synth_train_data_temp = pategan(train_data, parameters)


sns.set_style('whitegrid'); bwx=0.1
sns.kdeplot(np.array(train_data[:,0]), bw=bwx,color="red")
sns.kdeplot(np.array(synth_train_data_temp[:,0]), bw=bwx,color="blue")


         
         
data = pd.read_csv('/home/ec2-user/GRAIMatter/WP1/models/Synthetic_data/models/creditcard.csv').to_numpy()
data = MinMaxScaler().fit_transform(data)
train_ratio = 0.5
train = np.random.rand(data.shape[0])<train_ratio 
train_data, test_data = data[train], data[~train]
data_dim = data.shape[1]
  