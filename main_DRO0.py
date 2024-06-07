from ALG.Optimizer import *
from ALG.Models import *
from ALG.Utils import *
from ALG.dataclass import *
torch.manual_seed(123)
torch.set_default_dtype(torch.float64)

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.get_device_name(0)
except AssertionError:
    device = 'cpu'

# Load the data
# gisette: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2
# sido0: https://www.causality.inf.ethz.ch/data/sido0_matlab.zip, please use the *_train.mat
# a9a: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a, please save it as txt
# FMINST,MNIST,CIFAR10 will be downloaded by itself.
data_name = 'gisette' # gisette, sido0, a9a, MNIST, F-MNIST, CIFAR10,
train_set = loaddata(data_name,device)
#data normalization
if data_name == 'gisette':
    train_set.data = (train_set.data - torch.min(train_set.data,dim=1,keepdim=True)[0])/(torch.max(train_set.data,dim=1,keepdim=True)[0] - torch.min(train_set.data,dim=1,keepdim=True)[0])
elif data_name == 'sido0':
    train_set.data = (train_set.data - torch.min(train_set.data,dim=1,keepdim=True)[0])/(torch.max(train_set.data,dim=1,keepdim=True)[0] - torch.min(train_set.data,dim=1,keepdim=True)[0])
    #train_set.data = train_set.data/torch.norm(train_set.data)
    
print(f'data:{data_name},number_of_data:{train_set.data.shape[0]},dim_features:{train_set.data.shape[1]}')

# set the model
# Q: anydata works since it wont use it
# DRO: for gisette, sido0,a9a or any binary classification task.
# FairCNN: for FMINST,MNIST,CIFAR10 or any image m-classification task. 

for mu_y in [0.01]:
    model_type = 'DRO' # Q: projection_y=False,DRO,FairCNN: projection_y=True
    sim_time=3
    max_iter=10000
    freq=50# print result by freq
    b = 6000
    sgd_b = b
    my_optimizer = ALG(train_set=train_set,data_name=data_name,mu_y=mu_y,
                        sim_time=sim_time,max_iter=max_iter, b=b,
                        freq=freq, is_show_result=True, is_save_data=True,
                        projection_y=True,projection_x=False, # Q: projection_y=False; DRO,FairCNN: projection_y=True
                        maxsolver_step=1/10/mu_y,maxsolver_tol=1e-4,maxsolver_b=40000, # this is the setting for find y*(x)
                        device=device,model_type=model_type)
    L = my_optimizer.start_model.estimate_L(train_set.data,train_set.targets,data_name,load=True,b=6000)
    kappa = L/mu_y
    lr_y = 1/L
    lr_x = 1/L/kappa**2
    print(f'L:{L}, mu:{mu_y}, kappa: {kappa}')
    print(f'GDA: lr_y={1/L},lr_x={1/16/(kappa+1)**2/L}')
    print(f'AGDA: lr_y={1/L},lr_x={1/3/L/(1+kappa)**2}')
    oc_tmp = 1
    my_optimizer.max_iter = max_iter
    gamma2 = 0.95
    gamma1 = gamma2**2
    result = my_optimizer.line_search_one_step(gamma1 = gamma1, gamma2 = gamma2, isMaxSolver=True, isRestart=False,b=b, verbose=False)
    # result = my_optimizer.line_search_one_step(gamma1 = gamma1, gamma2 = gamma2, isMaxSolver=False, isRestart=False,b=b, verbose=False)
    # result = my_optimizer.optimizer(lr_x=1, lr_y=1, method='TiAda', b=sgd_b)
    # result = my_optimizer.optimizer(lr_x=1/3/L/(1+kappa)**2,lr_y=1/L,method='AGDA',b=sgd_b)
    # result = my_optimizer.optimizer(lr_x=1/16/(kappa+1)**2/L,lr_y=1/L,method='GDA',b=sgd_b)
    # result = my_optimizer.line_search_one_step(gamma1 = gamma1, gamma2 = gamma2, isMaxSolver=True, isRestart=False,b=b, verbose=False)
    # L = ?
    # result = my_optimizer.optimizer(lr_x=1/3/L, lr_y=1/144/L,p=2*L,beta=1/144/L*mu_y/1600, method='Smooth-AGDA', b=sgd_b)

