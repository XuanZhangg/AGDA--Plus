import torch
import torch.nn as nn
import numpy as np
from ALG.Utils import *
import torch.nn.functional as F

class Problem(nn.Module):
    def __init__(self, mu_y, kappa, device, injected_noise_x=0,injected_noise_y=0):
        super(Problem,self).__init__()
        self.name = 'NN'
        self.device = device
        self.std_x = injected_noise_x
        self.std_y = injected_noise_y
        self.std = max(self.std_x,self.std_y)
        self.F_lower = 0
        self.mu_y = mu_y
        self.kappa = kappa #only valid for Qmodel
        self.L = kappa*mu_y #only valid for Qmodel

    def forward(self):
        pass

    def predict(self):
        pass
    
    def loss(self,input,idx,target):
        # this function is to avoid memory issues
        foo,i = 0,0
        b = 6000
        while i<= len(target)-1:
            foo += self.batch_loss(input[i:i+b],idx[i:i+b],target[i:i+b])
            i += b
        if foo == 0:
            pass
            #print('Warning: bad batch was selected for DRO problem!!!')
        return 1/target.shape[0]*foo + self.regularizer() 

    def batch_loss(self, input,idx,target):
        pass
    
    def regularizer(self):
        pass

    def weight_init(self):
        for layer in self.parameters():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight,gain=1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def reset_init(self):
        self.weight_init()
        pass

class ProblemQ(Problem):
    def __init__(self, data_size, mu_y, kappa, device, injected_noise_x=0, injected_noise_y=0):
        super().__init__(mu_y, kappa, device,injected_noise_x,injected_noise_y)
        self.d_x, self.d_y = data_size
        self.name = f'Q_stdx_{self.std_x}_stdy_{self.std_y}'
        assert self.d_x == self.d_y # the way to generate the model only support the same dim of x and y
        self.reset_init()
        
        if self.d_x == 1:
            A = torch.tensor([[20.]])
            Q = torch.tensor([[-20.]])
        else:
            try:
                if self.std>0:
                    A,Q = torch.load(f'data/sQ/AQ_kappa_{self.kappa}_mu_{self.mu_y}.pt')
                else:
                    A,Q = torch.load(f'data/Q/AQ_kappa_{self.kappa}_mu_{self.mu_y}.pt')
            except:
                Lambda_Q = np.diag(np.random.uniform(low=-1, high=1, size=self.d_x))
                Lambda_Q = Lambda_Q/np.linalg.norm(Lambda_Q, ord=2)*self.L
                if self.std>0:
                    eps = 1e-1
                else:
                    eps = 1e-2
                Lambda_A = (np.abs(Lambda_Q) * self.mu_y)**(1/2) + np.diag(np.random.uniform(low=eps, high=eps, size=self.d_x))

                # generate a random n x n matrix
                foo = np.random.rand(self.d_x, self.d_y)
                # perform QR decomposition
                V, _ = np.linalg.qr(foo)
                A,Q = torch.from_numpy(V.T@Lambda_A@V), torch.from_numpy(V.T@Lambda_Q@V)
                if self.std>0:
                    torch.save([A,Q], f'data/sQ/AQ_kappa_{self.kappa}_mu_{self.mu_y}.pt')
                else:
                    torch.save([A,Q], f'data/Q/AQ_kappa_{self.kappa}_mu_{self.mu_y}.pt')

        assert np.abs(max([np.linalg.norm(A, ord=2), np.linalg.norm(Q, ord=2), self.mu_y])-self.L) <= 1
        self.register_buffer('Q', Q)
        self.register_buffer('A', A)

        self.to(self.device)

    def reset_init(self):
        if self.d_x == 1:
            self.x = nn.Parameter(torch.tensor([[1.]]),requires_grad=True)
            self.dual_y = nn.Parameter(torch.tensor([[0.01]]), requires_grad=True)
            return
        
        self.x = nn.Parameter(100+10*2*(torch.rand(self.d_x,1)-0.5),requires_grad=True)
        self.dual_y = nn.Parameter(100+10*2*(torch.rand(self.d_y,1)-0.5), requires_grad=True)

    def forward(self, *args):
        return 
    
    def predict(self,data):
        return torch.ones((data.shape[0],), device = data.device)
    
    def loss(self, *args):
        loss = self.x.T @ self.A @ self.dual_y + 1 / 2 * self.x.T @ self.Q @ self.x 
        return loss - self.mu_y / 2 * torch.norm(self.dual_y)**2

    def exact_y_opt(self,input=None,idx=None,target=None):
        # this function is to avoid memory issues
        return 1/self.mu_y*self.A.T@self.x


class FairCNN(Problem):
    def __init__(self, data_size,  mu_y, kappa, device, injected_noise_x=0 ,injected_noise_y=0):
        super().__init__(mu_y, kappa, device, injected_noise_x, injected_noise_y)
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
    
    def batch_loss(self, input, idx, target):
        target_transform = F.one_hot(target,3).double()
        input = self.forward(input)
        size = len(input)

        return -torch.matmul(
                            torch.log(torch.clamp(input[range(size), target],min=10**(-12))),
                            torch.matmul(target_transform,self.dual_y)
                            )/size

    def regularizer(self):
        return -self.mu_y/2 * torch.sum((self.dual_y - 1/self.d_y)**2)
    
    def predict(self, x):
        return self.forward(x).argmax(dim=1)
class ProblemDRO(Problem):
    def __init__(self, data_size, mu_y, kappa, device, injected_noise_x=0 ,injected_noise_y=0):
        super().__init__(mu_y, kappa, device, injected_noise_x ,injected_noise_y)
        self.d_x, self.d_y = data_size
        self.fc = nn.Sequential(
            nn.Linear(self.d_x, 120),
            nn.ELU(),
            nn.Linear(120, 84),
            nn.ELU(),
            nn.Linear(84, 1),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(self.d_x, 16),
        #     nn.ELU(),
        #     # nn.Linear(16, 32),
        #     # nn.ELU(),
        #     nn.Linear(32, 32),
        #     nn.ELU(),
        #     nn.Linear(32, 32),
        #     nn.ELU(),
        #     nn.Linear(32, 8),
        #     nn.ELU(),
        #     nn.Linear(8, 1),
        # )
        self.dual_y = nn.Parameter(1 / self.d_y * torch.torch.ones(self.d_y, ), requires_grad=True)
        self.weight_init()

        self.std_x = 0.000
        self.std_y = 0.000
        self.std = max(self.std_x,
                       self.std_y)  # 10 is just a random number i set, becuase i did not measure what is the variance

        self.to(self.device)

    def forward(self, x):
        return self.fc(x)

    def batch_loss(self, input, idx, target, return_ele_loss = False):
        input = self.forward(input)
        # loss part
        bax = target.unsqueeze(1) * input  #:ba is the log(1 + exp(-bax))
        logistic_loss = torch.zeros_like(bax, dtype=torch.float64)
        # case1:
        logistic_loss[bax <= -30.0] = -bax[bax <= -30.0]
        # case2:
        logistic_loss[bax > -30.0] = torch.log(1 + torch.exp(-bax[bax > -30.0]))
        weight_y = torch.index_select(self.dual_y, 0, index=idx)

        if return_ele_loss:
            return logistic_loss

        return torch.sum(logistic_loss * weight_y)

    def exact_y_opt(self,input,idx,target):
        # this function is to avoid memory issues
        i = 0
        b = 6000
        while i<= len(target)-1:
            if i == 0:
                foo = self.batch_loss(input[i:i+b],idx[i:i+b],target[i:i+b],return_ele_loss= True)
            else:
                foo = torch.cat([foo,self.batch_loss(input[i:i+b],idx[i:i+b],target[i:i+b],return_ele_loss= True)])
            i += b
        foo = foo.squeeze(1)
        foo = 1/self.mu_y*foo + 1/self.d_y
        return foo

    def regularizer(self):
        return -self.mu_y / 2 * torch.sum((self.dual_y - 1 / self.d_y) ** 2)

    def predict(self, x):
        judge = self.forward(x) >= 0
        temp = torch.ones_like(judge, dtype=torch.int64)
        temp[judge == False] = torch.tensor(-1, dtype=torch.int64)
        return torch.flatten(temp)

    def estimate_L(self, data: torch.tensor, target: torch.tensor, data_name, load, b=None):
        if load:
            cache = {'gisette': 0.053220038704550165, 'sido0': 0.5220159537252982}  # 3127.4873
            return cache[data_name]

        full_batch = torch.arange(data.shape[0]).to(self.device)
        if b == None:
            b = data.shape[0]
            x = self.forward(data).flatten()
        else:
            indices = torch.randperm(data.shape[0])[:b]
            data = data[indices]
            target = target[indices]
            x = self.forward(data).flatten()

        h = 1/b*torch.log(1 + torch.exp(torch.clamp(-target * x, max=100)))
        foo = -target * torch.exp(-target * h) / (1 + torch.exp(-target * h))
        length = 0
        for i in range(0,len(self.fc),2):
            length += self.fc[i].weight.data.flatten().shape[0] + self.fc[i].bias.data.flatten().shape[0]
        hessian = torch.zeros((target.shape[0], length))
        print(f'number of neurals:{length}')
        for i in range(target.shape[0]):
            self.zero_grad()
            h[i].backward(retain_graph=True)
            hessian[i, :] = torch.cat(
                [ torch.cat([self.fc[i].weight.grad.data.flatten(),self.fc[i].bias.grad.data.flatten()]).flatten()  \
                  for i in range(0,len(self.fc),2)] \
                ).unsqueeze(0).detach().cpu()
            h.grad = None
            if i % 500 == 0:
                print(f"First stage of estimating L has process: {100 * i / target.shape[0]:.2f}%")

        print('Computing huge hessian matrix start, good luck!!!')
        self.L_estimated = torch.norm(hessian, p=2) + self.mu_y
        print(f'L={self.L_estimated}')
        return self.L_estimated

