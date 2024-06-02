import torch
import torch.nn as nn
import numpy as np
from ALG.Utils import *
import torch.nn.functional as F

class Problem(nn.Module):
    def __init__(self, mu_y, device):
        super(Problem,self).__init__()
        self.device = device
        self.std_x = 0
        self.std_y = 0
        self.std = max(self.std_x,self.std_y)
        self.F_lower = 0
        self.mu_y = mu_y

    def forward(self):
        pass

    def predict(self):
        pass
    
    def loss(self,input,idx,target):
        foo = 0
        while i<= len(target)-1:
            foo += self.batch_loss(input[:i],idx[:i],target[:i])
            i += 6000
        return foo + self.regularizer() 

    def batch_loss(self, input,idx,target):
        pass
    
    def regularizer(self):
        pass

    def weight_init(self):
        for layer in self.parameters():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

class ProblemQ(Problem):
    def __init__(self, data_size, mu_y, device):
        super().__init__(mu_y, device)
        self.d_x, self.d_y = data_size
        assert self.d_x == self.d_y # the way to generate the model only support the same dim of x and y
        self.std_x = 0.01
        self.std_y = 0.01
        self.std = max(self.std_x,self.std_y)
        self.x = nn.Parameter(torch.rand(self.d_x,1),requires_grad=True)
        self.dual_y = nn.Parameter(torch.rand(self.d_y,1), requires_grad=True)
        self.weight_init()

        Lambda_A =  np.diag(np.random.uniform(low=-10, high=10, size=self.d_x))
        Lambda_Q = - Lambda_A**2 * self.mu_y + np.diag(np.random.uniform(low=-5, high=5, size=self.d_x))
        # generate a random n x n matrix
        foo = np.random.rand(self.d_x, self.d_y)
        # perform QR decomposition
        V, _ = np.linalg.qr(foo)
        A,Q = torch.from_numpy(V.T@Lambda_A@V), torch.from_numpy(V.T@Lambda_Q@V)
        self.register_buffer('Q',Q)
        self.register_buffer('A',A)

        self.to(self.device)

    def forward(self, *args):
        return 
    
    def predict(self,data):
        return torch.ones((data.shape[0],), device = data.device)
    
    def loss(self, *args):
        return self.x.T @ self.A @ self.dual_y + 1 / 2 * self.x.T @ self.Q @ self.x - self.mu_y / 2 * torch.norm(self.dual_y)**2 

class ProblemDRO(Problem):
    def __init__(self, data_size, mu_y, device):
        super().__init__(mu_y, device)
        self.d_x, self.d_y = data_size
        self.w1 = nn.Parameter(torch.zeros(self.d_x,120),requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros(120,84),requires_grad=True)
        self.w3 = nn.Parameter(torch.zeros(84,1),requires_grad=True)
        self.dual_y = nn.Parameter(1/self.d_y*torch.torch.ones(self.d_y,), requires_grad=True)
        self.ELU = torch.nn.ELU()
        self.w1 = torch.nn.init.xavier_uniform_(self.w1)
        self.w2 = torch.nn.init.xavier_uniform_(self.w2)
        self.w3 = torch.nn.init.xavier_uniform_(self.w3)

        self.std_x = 1000
        self.std_y = 1000
        self.std = max(self.std_x,self.std_y) # 10 is just a random number i set, becuase i did not measure what is the variance

        self.to(self.device)

    def forward(self, x):
        x = self.ELU(torch.matmul(x, self.w1))
        #x = nn.functional.batch_norm(x)
        x = self.ELU(torch.matmul(x, self.w2))
        x = torch.matmul(x, self.w3)
        return x
    
    def loss(self,input,idx,target): 
        input = self.forward(input)       
        #regularizer part
        regularizer_x = 0
        regularizer_y =  self.mu_y/2 * torch.sum((self.dual_y - 1/self.d_y)**2)

        #loss part
        bax = target.unsqueeze(1) *input #:ba is the log(1 + exp(-bax))
        logistic_loss = torch.zeros_like(bax, dtype = torch.float64)
        #case1:
        logistic_loss[bax <= -100.0] = -bax[bax <= -100.0]
        #case2:
        logistic_loss[bax > -100.0] = torch.log(1+torch.clamp(torch.exp(-bax[bax > -100.0]), min = 1e-12))
        weight_y = torch.index_select(self.dual_y,0,index=idx)

        return torch.sum(logistic_loss*weight_y) + regularizer_x - regularizer_y
   
    def predict(self, x):
        judge = self.forward(x)>=0
        temp = torch.ones_like(judge, dtype = torch.int64)
        temp[judge == False] = torch.tensor(-1,dtype = torch.int64)
        return torch.flatten(temp)

class FairCNN(Problem):
    def __init__(self, data_size,  mu_y, device):
        super().__init__(mu_y, device)
        self.channel,self.d_x = data_size[0]
        self.n = data_size[1]
        self.d_y = 3
        self.ELU = torch.nn.ELU()
        self.conv1 = nn.Conv2d(self.channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        demo = torch.zeros((1,self.channel,self.d_x,self.d_x))
        demo = self.cnn(demo)
        self.fc1 = nn.Linear(demo.shape[1], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.dual_y = nn.Parameter(torch.tensor([1/3,1/3,1/3]), requires_grad=True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        
        self.std_x = 1
        self.std_y = 1
        self.std = max(self.std_x,self.std_y)
        self.to(self.device)

    def cnn(self,x):
        x = self.pool(self.ELU(self.conv1(x)))
        x = self.pool(self.ELU(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return x

    def forward(self, x):
        x = self.cnn(x)
        x = self.ELU(self.fc1(x))
        x = self.ELU(self.fc2(x))
        x = self.fc3(x)
        softmax = torch.nn.Softmax(dim=1)
        return softmax(x)
    
    def loss(self, input, idx, target):
        target_transform = F.one_hot(target,3).double()
        input = self.forward(input)
        size = len(input)
        regularizer_y =  self.mu_y/2* torch.sum((self.dual_y - 1/self.d_y)**2)

        return -torch.matmul(
                            torch.log(torch.clamp(input[range(size), target],min=10**(-12))),
                            torch.matmul(target_transform,self.dual_y)
                            )/size - regularizer_y

    def predict(self, x):
        return self.forward(x).argmax(dim=1)

class ProblemTest(Problem):
    def __init__(self, data_size, mu_y, device):
        super().__init__(mu_y, device)
        self.d_x,self.d_y = data_size
        self.mu_y = mu_y
        self.dual_y = nn.Parameter(torch.rand(self.d_y,1), requires_grad=True)
        self.fc = nn.Sequential(
            nn.Linear(self.d_x, 120),
            nn.ELU(),
            nn.Linear(120, 84),
            nn.ELU(),
            nn.Linear(84, 1),
        )
        self.weight_init()
        self.std_x = 100
        self.std_y = 100
        self.std = max(self.std_x,self.std_y)

        self.to(self.device)

    def forward(self, input_data):
        return self.fc(input_data)
    
    def predict(self, input_data):
        return  torch.where(self.forward(input_data).flatten()>0, torch.ones(input_data.shape[0],device=self.device), -torch.ones(input_data.shape[0],device=self.device))
    
    def loss(self,input,y_idx,target):
        input = self.forward(input)
        #regularizer part
        regularizer_x = 0
        regularizer_y =  self.mu_y*1/2*1/(self.d_y)**2 * torch.sum((self.d_y*self.dual_y - 1)**2)

        #loss part
        foo = target.unsqueeze(1) *input #:ba is the log(1 + exp(-foo))
        logistic_loss = torch.zeros_like(foo, dtype = torch.float64)
        #case1:
        logistic_loss[foo <= -100.0] = -foo[foo <= -100.0] # if -foo>100, we use -foo to approximate log(1 + exp(-foo))
        #case2:
        logistic_loss[foo > -100.0] = torch.log(1+torch.exp(-foo[foo > -100.0]))

        return torch.sum(logistic_loss*self.dual_y[y_idx]) + regularizer_x - regularizer_y
    

    

