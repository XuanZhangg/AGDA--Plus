import torch
import pandas as pd
import pickle
from ALG.Utils import *
from ALG.dataclass import Creatdata
import scipy.io

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)
data = scipy.io.loadmat('./data/sido0/sido0_train.mat') 
data = getSparse(data['X'])
targets =  pd.read_csv('./data/sido0/sido0_train.targets', header=None)
targets = torch.tensor(targets.values.tolist(),dtype=torch.int64, device=device).squeeze(1)
train_set =  Creatdata(data,targets)
data_name = 'sido0'
file_name = './data/' + data_name + '/' + data_name
with open(file_name , "wb") as fp:  
    pickle.dump(train_set, fp)
