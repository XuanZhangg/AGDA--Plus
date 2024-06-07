import numpy as np
from ALG.Models import Problem
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import copy
# def projection_simplex_sort(x, radius=1):
#     norm_x = np.linalg.norm(x)
#     if norm_x <= radius:
#         # x is already within the L2 ball
#         return x
#     else:
#         # project x onto the L2 ball
#         unit_x = x / norm_x
#         return radius * unit_x

def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    if np.isinf(v).any():
        return v
    func = lambda x: np.sum(np.maximum(v - x, 0)) - z
    lower = np.min(v) - z / len(v)
    upper = np.max(v)

    for it in range(max_iter):
        midpoint = (upper + lower) / 2.0
        value = func(midpoint)

        if abs(value) <= tau:
            break

        if value <= 0:
            upper = midpoint
        else:
            lower = midpoint

    return np.maximum(v - midpoint, 0)

def projection_simplex_sort(v, z=1):
    # if np.isinf(v).any():
    #     return v
    # return np.minimum(np.maximum(v, -10),10)

    if np.isinf(v).any():
        return v
    save_shape = v.shape
    v = v.flatten()
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    
    if len(ind[cond]) == 0:
        return np.full_like(v, np.inf).reshape(save_shape)
    
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    w = w.reshape(save_shape)
    e = np.abs(np.sum(w)-1)
    if e>1e-2: # to check projection onto simplex
        print(f'warning: numerical precision issues happen in projection onto simplex, the error is {e}')
    return w


def projection_simplex_sort2(v, z=0.8):
    # if np.isinf(v).any():
    #     return v
    # return np.minimum(np.maximum(v, -10),10)

    if np.isinf(v).any():
        return v
    save_shape = v.shape
    v = v.flatten()
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    
    if len(ind[cond]) == 0:
        return np.full_like(v, np.inf).reshape(save_shape)
    
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    w = w.reshape(save_shape)
    e = np.abs(np.sum(w)-z)
    if e>1e-2: # to check projection onto simplex
        print(f'warning: numerical precision issues happen in projection onto simplex, the error is {e}')
    
    return w + (1-z)/w.shape[0]



def projection_l1(v, z=1):
    if np.isinf(v).any():
        return v
    return np.minimum(np.maximum(v, -z),z)

def projection_l2(v, z=1):
    if np.isinf(v).any():
        return v

    if np.linalg.norm(v)>z:
        return v/np.linalg.norm(v)*z

    return v

def isPrime(x):
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True
    for i in range(2, int(np.sqrt(x)) + 1):
        if x % i == 0:
            return False

    return True

def getNthPrime(x):
    cnt = 0
    num = 2
    while True:
        if isPrime(num):
            cnt = cnt + 1
            if cnt == x:
                return num
        num = num + 1

def shadowplot(x: list, y: list, alg_name: str, center=0, alpha=0.5, label_input=None, is_var=False,is_log=False,is_step = False,is_speical=False, plot_part=None, is_last=False):
    temp = []
    colors = {'GS-GDA-B,N=1': 'y',
              'GS-GDA-B,N=5': 'g','GS-GDA-B,N=10': 'y','GS-GDA-B,N=2': 'y','primal_line_search_100': 'saddlebrown',\
              'LS-GS-GDA':'r', 'LS-GS-GDA-S':'g', 'LS-GS-GDA-R':'lightgreen', 'LS-GS-GDA-S-R':'r', \
              'J-GDA': 'saddlebrown', 'GS-GDA': 'b', 'TiAda': 'c', \
              'Smooth-AGDA':'m',
              }
    linestyle_input = {'GS-GDA-B,N=1': '-', 'J-GDA':'-','GS-GDA':'-' ,'TiAda':'-',\
                       'GS-GDA-B,N=5': '-','GS-GDA-B,N=10': '-','GS-GDA-B,N=2': '-','GS-GDA-B,N=100': '-', \
                       'LS-GS-GDA': '-','LS-GS-GDA-S': '-','LS-GS-GDA-R': '-','LS-GS-GDA-S-R': '-',\
                       'Smooth-AGDA':'-',
                       }
    if is_speical:
        label = {'GS-GDA_lr_y':r'AGDA,$\sigma=1/L$','GS-GDA_lr_x':r'AGDA,$\tau=\Theta(\frac{1}{L\kappa^2})$','GS-GDA_ratio':r'AGDA,$\sigma/\tau =\Theta(\kappa^2)$'}
    else:
        label = {'TiAda':'TiAda','J-GDA': 'GDA', 'GS-GDA': 'AGDA','Smooth-AGDA':'Sm-AGDA',\
            'LS-GS-GDA':'AGDA+','LS-GS-GDA-S': 'AGDA+ max','LS-GS-GDA-R': 'RAGDA+','LS-GS-GDA-S-R': 'ADGA+SR',\
            'GS-GDA-B,N=1':'SGDA-B'
                       }
    if is_last:
        for i in range(len(y)):
            for k in range(1,len(y[i])):
                y[i][k] = min(y[i][k],y[i][k-1])

    for i in range(len(y)):
        y[i] = [val-center for val in y[i]]

        if is_log:
            y[i]=np.log(y[i])
        temp.append(y[i])
    y = temp

    mid_line = [np.average(val) for val in zip(*y)]
    var =  [np.var(val) for val in zip(*y)]

    if is_var:
        lowline = [x-y for x,y in zip(mid_line,var)]
        highline = [x+y for x,y in zip(mid_line,var)]
    else:
        lowline = [np.average(val) - (np.average(val) - np.min(val)) for val in zip(*y)]
        highline = [np.average(val) + (-np.average(val) + np.max(val)) for val in zip(*y)]

    if plot_part =='z':
        mid_line = smooth_data(mid_line,100)
        lowline = smooth_data(lowline,100)
        highline = smooth_data(highline,100)
        
    if is_step:
        plt.scatter(x, mid_line, label = label[label_input], color=colors[alg_name])
        return
    
    print(f'plotting {alg_name}')
    if is_speical and plot_part:
        label_input = alg_name + '_' +plot_part
    if alg_name == 'LS-GS-GDA-S':
        plt.plot(x,mid_line, label = label[label_input], linestyle = linestyle_input[alg_name], color=colors[alg_name], linewidth=3,marker='^', markevery=500)
    else:
        plt.plot(x,mid_line, label = label[label_input], linestyle = linestyle_input[alg_name], color=colors[alg_name], linewidth=3)

    plt.fill_between(x, lowline, highline, alpha=alpha, facecolor=colors[alg_name])
    # plt.fill_between(x, lowline, highline, facecolor='green', alpha=0.2)
    return
    
def loaddata(data_name,device):
    try:
        file_name = './data/'+ data_name + '/' + data_name
        with open(file_name, "rb") as fp:   # Unpickling
            train_set = pickle.load(fp)
    except:
        import os
        
        print('start to download and create data...')
        if not os.path.exists('./data/'+ data_name):
            os.makedirs('./data/'+ data_name)

        script = './dataScripts/'+ data_name +'.py'
        exec(open(script).read())

        file_name = './data/'+ data_name + '/' + data_name
        with open(file_name, "rb") as fp:   # Unpickling
            train_set = pickle.load(fp)

    train_set.data = train_set.data.to(device).to(torch.float64)
    train_set.targets = train_set.targets.to(device)
    return train_set

def getSparse(sparse_matrix,sparse=False):
    if sparse:
        # convert the numpy sparse matrix to a PyTorch sparse tensor
        values = sparse_matrix.data
        indices = np.vstack((sparse_matrix.indices, sparse_matrix.indptr))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sparse_matrix.shape

        foo = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    else:
        # convert the sparse matrix to a dense numpy array
        dense_array = sparse_matrix.toarray()

        # convert the dense numpy array to a PyTorch tensor
        foo = torch.from_numpy(dense_array)

    return foo


def smooth_data(data, window_size):
    """
    对数组进行移动平均平滑处理。

    :param data: 要平滑的原始数组。
    :param window_size: 移动平均的窗口大小。
    :return: 平滑后的数组。
    """
    if window_size < 1:
        raise ValueError("窗口大小必须大于等于1")

    n = len(data)
    if window_size > n:
        raise ValueError("窗口大小不能大于数组长度")

    smoothed = [0] * n
    # 计算初始窗口的平均值
    sum_window = sum(data[:window_size])
    smoothed[window_size - 1] = sum_window / window_size

    # 滑动窗口计算其余的平均值
    for i in range(window_size, n):
        sum_window += data[i] - data[i - window_size]
        smoothed[i] = sum_window / window_size

    # 处理数组开头的元素
    for i in range(1, window_size):
        smoothed[i - 1] = sum(data[:i]) / i

    return smoothed
def normlize_data(input:list[list]):
    return [[ele[i]/(ele[0]+1e-10) for i in range(len(ele))] for ele in input]

def getxyFromStateModel(model:Problem, grad = False):
    x = []
    y = []
    if not grad:
        for name,param in model.named_parameters():
            if name == 'dual_y':
                y = torch.concat([y,param.data.flatten().clone()]) if len(y) else param.data.flatten().clone()
            else:
                x = torch.concat([x,param.data.flatten().clone()])  if len(x) else param.data.flatten().clone()
    else:
        for name,param in model.named_parameters():
            if name == 'dual_y':
                y = torch.concat([y,param.grad.data.flatten().clone()])  if len(y) else param.grad.data.flatten().clone()
            else:
                x = torch.concat([x,param.grad.data.flatten().clone()])  if len(x) else param.grad.data.flatten().clone()

    return x.unsqueeze(1).clone(),y.unsqueeze(1).clone()

def computeGrad(model:Problem, data_by_batch,target_by_batch, batch_index, b):
    model.zero_grad()
    if model.name[0] == 'Q':
        epsx = 2*(torch.rand_like(model.x)-0.5)*model.std_x
        epsy = 2*(torch.rand_like(model.dual_y)-0.5)*model.std_y
        loss = model.loss(data_by_batch, batch_index, target_by_batch) + 2*(torch.rand(1,device=model.device)-0.5)*model.std_x
        loss2 = loss + model.x.T @ epsx/b + model.dual_y.T @ epsy/b
        loss2.backward()
    else:
        loss = model.loss(data_by_batch, batch_index, target_by_batch)
        loss.backward()

    return loss.clone()