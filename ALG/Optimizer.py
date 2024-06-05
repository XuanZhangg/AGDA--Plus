from ctypes.wintypes import MAX_PATH
import random
from typing import final
import torch
import numpy as np
import copy
import pickle
import random
import math
from collections import OrderedDict
from importlib import reload
from ALG.Utils import projection_simplex_bisection as pj_y
#from ALG.Utils import projection_simplex_sort2 as pj_y
from ALG.Utils import projection_simplex_sort as pjy_y2
#from ALG.Utils import projection_l2 as pj_y
from ALG.Utils import projection_l2 as pj_x
from ALG.Utils import getxyFromStateModel, computeGrad
from  collections import  defaultdict

def call_model(model_type):
    if model_type == 'Q':
        from ALG.Models import ProblemQ as Model
    elif model_type == 'DRO':
        from ALG.Models import ProblemDRO as Model
    elif model_type == 'Test':
        from ALG.Models import ProblemTest as Model
    elif model_type =='FairCNN':
        from ALG.Models import FairCNN as Model
    return Model

class ALG():
    def __init__(self, train_set, data_name, mu_y, kappa = 10,
        sim_time = 3, max_epoch = 100, max_iter=100, b = 6000, 
        maxsolver_step=0.01, maxsolver_tol=1e-4, maxsolver_b = 6000,
        is_show_result = False, is_save_data = False, freq = 500,
        device = 'cuda',
        projection_y=False, projection_x=False,
        model_type = 'DRO', toymodel = False,
        inject_noise_x=0,
        inject_noise_y=0,
        isSameInitial = False, optimize_batch = False,
                 ) -> None:

        self.sim_time = sim_time    # sim_time means the times of running through the alg
        self.max_epoch = max_epoch  # an epoch means a complete pass of the data set
        self.max_iter = max_iter    # iter means the times of iteration within the alg
        self.b = b # b is the batchsize for stochastic alg
        self.is_show_result = is_show_result # whether display the training process
        self.freq = freq # the freq to show training process in terms of iter
        self.is_save_data = is_save_data # whether save data
        self.device = device
        self.maxsolver_step = maxsolver_step
        self.maxsolver_tol = maxsolver_tol
        self.maxsolver_b = maxsolver_b

        # problem parameters
        self.mu_y = mu_y
        self.kappa = kappa
        self.Dy = 1e6
        self.projection_y = projection_y # whether using projection for y
        self.projection_x = projection_x # whether using projection for x
        self.std_x = inject_noise_x
        self.std_y = inject_noise_y

        self.model_type = model_type
        self.data_name = data_name
        self.data = train_set.data.clone()
        self.targets = train_set.targets.clone()
        self.total_number_data = len(train_set.targets)

        self.isSameInitial = isSameInitial
        self.toymodel = toymodel
        self.optimize_batch = optimize_batch
        
        #initialize savers
        self.reset_all()

        #initilize start model
        if model_type == 'FairCNN':
            self.data_size = ((train_set.data.shape[1],train_set.data.shape[2]),len(train_set.targets))
            self.data_number_in_each_epoch = train_set.data.shape[0]
        elif model_type == 'Q':
            if toymodel:
                self.data_size = (1,1)
            else:
                self.data_size = (30,30)
            self.data_number_in_each_epoch = 1
        else:
            self.data_size = (len(train_set.data[0]),len(train_set.targets))
            self.data_number_in_each_epoch = self.total_number_data

        self.generate_initial_model()
        self.load_initial_model(0)
        
    def reset_initial_model(self):
        #initilize start model
        model_type = self.model_type
        mu_y = self.mu_y
        device = self.device
        kappa = self.kappa

        Model = call_model(model_type)
        self.start_model = Model(self.data_size, mu_y, kappa, device=device, injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
        self.y_opt = self.maximizer_solver(start=self.start_model,lr_y=self.maxsolver_step) # set y as y_opt for initial model
        self.start_model.dual_y.data = self.y_opt.clone()
        self.model_copy = Model(self.data_size, mu_y, kappa, device=device, injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
        self.model_bk = Model(self.data_size, mu_y, device=device, injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model

    def generate_initial_model(self):
        StartModelSet = {}
        for i in range(self.sim_time):
            #initilize start model
            model_type = self.model_type
            mu_y = self.mu_y
            device = self.device
            kappa = self.kappa

            Model = call_model(model_type)
            start_model = Model(self.data_size, mu_y, kappa, device=device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
            y_opt = self.maximizer_solver(start=start_model,lr_y=self.maxsolver_step) # set y as y_opt for initial model
            start_model.dual_y.data = y_opt.clone()
            model_copy = Model(self.data_size, mu_y, kappa, device=device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
            model_bk = Model(self.data_size, mu_y, kappa, device=device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(device) # initial model
            
            if self.isSameInitial:
                for i in range(self.sim_time):
                    StartModelSet[i] = [start_model, y_opt, model_copy, model_bk]
                break

            StartModelSet[i] = [start_model, y_opt, model_copy, model_bk]
            
        torch.save(StartModelSet, 'initial.pt')

    def load_initial_model(self, sim_time):
        start = torch.load('initial.pt')
        self.start_model,self.y_opt,self.model_copy,self.model_bk = start[sim_time]

    def reset_all(self):
        #the following saver will be reset for a new run with sim_times simulations
        self.record = {}
        self.record['loss'] = [[] for _ in range(self.sim_time)]
        self.record['primalF'] =  [[] for _ in range(self.sim_time)]
        self.record['acc'] = [[] for _ in range(self.sim_time)]

        self.record['norm_square_sto_grad_x'] = [[] for _ in range(self.sim_time)]
        self.record['norm_square_sto_grad_y'] = [[] for _ in range(self.sim_time)]
        self.record['norm_square_full_grad_x'] = [[] for _ in range(self.sim_time)]
        self.record['norm_square_full_grad_y'] = [[] for _ in range(self.sim_time)]   

        self.record['total_sample_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['total_oracle_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['total_iter'] = [0 for _ in range(self.sim_time)]
        self.record['total_epoch'] = [0 for _ in range(self.sim_time)]

        self.record['total_sample_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['total_oracle_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['total_iter_counter'] = [[] for _ in range(self.sim_time)]
        self.record['total_epoch_counter'] = [[] for _ in range(self.sim_time)]

        self.record['sample_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['oracle_complexity'] = [0 for _ in range(self.sim_time)]
        self.record['iter'] = [0 for _ in range(self.sim_time)]
        self.record['epoch'] = [0 for _ in range(self.sim_time)]

        self.record['sample_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['oracle_complexity_counter'] = [[] for _ in range(self.sim_time)]
        self.record['iter_counter'] = [[] for _ in range(self.sim_time)]
        self.record['epoch_counter'] = [[] for _ in range(self.sim_time)]

        self.record['contraction_times'] = [-1 for _ in range(self.sim_time)]
        self.record['config'] = [{} for _ in range(self.sim_time)]

        self.record['lr_x'] = [[] for _ in range(self.sim_time)]
        self.record['lr_y'] = [[] for _ in range(self.sim_time)]

        self.record['l(small)'] = [[] for _ in range(self.sim_time)]
        self.record['L(large)'] = [[] for _ in range(self.sim_time)]
        self.record['Deltak'] = [[] for _ in range(self.sim_time)]

        self.zt_Smooth_AGDA = {}

    def reset_contraction(self, s):
        #the following saver will be reset for a new contraction at the s-th simulation
        self.record['sample_complexity'][s] = 0
        self.record['oracle_complexity'][s] = 0
        self.record['iter'][s] = 0
        self.record['epoch'][s] = 0

        self.record['sample_complexity_counter'][s] = []
        self.record['oracle_complexity_counter'][s] = []
        self.record['iter_counter'][s] = []
        self.record['epoch_counter'][s] = []

        self.record['total_sample_complexity_counter'][s] = []
        self.record['total_oracle_complexity_counter'][s] = []
        self.record['total_iter_counter'][s] = []
        self.record['total_epoch_counter'][s] = []

        self.record['loss'][s] = []
        self.record['primalF'][s] = []
        self.record['acc'][s] = []
        self.record['norm_square_sto_grad_x'][s] = []
        self.record['norm_square_sto_grad_y'][s] = []
        self.record['norm_square_full_grad_x'][s] = []
        self.record['norm_square_full_grad_y'][s] = [] 
        self.record['config'][s] = {}

    def line_search_one_step(self, gamma1:float=0.9,gamma2=0.9, gamma3=1e-3, b:int=None, N:int=1, method:str='LS-GS-GDA', min_b:int=1, force_b:int=-1, kernal='AGDA',\
                             isMaxSolver = True, isRestart = True, mu_coeff = 1,\
                             max_epoch=None,start=None,max_iter=None,is_tell_gamma=False,verbose=False):
        self.reset_all()
        if is_tell_gamma:
            method += f'-g1-{int(100*gamma1)}-g2-{int(100*gamma2)}-g3-{int(1000*gamma3)}'
        Model = call_model(self.model_type)
        if not b:
            b = self.b
        if not max_epoch:
            max_epoch = self.max_epoch
        if not max_iter:
            max_iter = self.max_iter

        if isMaxSolver:
            method += '-S'

        if isRestart:
            method += '-R'

        # Generate full block
        N = 1
        flattened_x = torch.cat([param.flatten() for name,param in self.start_model.named_parameters() if name!='dual_y'])
        indices = np.arange(flattened_x.shape[0])
        full_block = copy.deepcopy(indices)

        for s in range(self.sim_time):
            self.load_initial_model(s)
            self.reset_contraction(s)
            self.record['contraction_times'][s] += 1
            self.record['config'][s] = {'b':b,'N':N,'K':self.max_iter,'std_x':self.start_model.std_x, 'std_y':self.start_model.std_y}
            self.record['config'][s]['method'] = method
            self.record['config'][s]['pjx'] = self.projection_x
            self.record['config'][s]['pjy'] = self.projection_y

            #load the start model
            start = self.start_model

            self.model_curr = Model(data_size=self.data_size,mu_y=self.mu_y, kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
            self.model_curr.load_state_dict(copy.deepcopy(start.state_dict()))

            if self.model_type == 'Q':
                self.model_curr.b = b
                self.data_number_in_each_epoch = b

            #initialize the data loader and full batch
            data_loader_dumb = self.batchselect()
            #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
            batch_start = 0

            tilde_l = self.mu_y/gamma2
            L, l = tilde_l, tilde_l
            upper_delta = 10
            deltak = min(self.Dy, torch.norm(self.y_opt - self.model_curr.dual_y) ** 2 + upper_delta)
            Deltak = deltak
            foo_i = 1
            self.new_stage = True
            reset_flag = True
            R_k,Lambda_k = 0,0
            bar_D_y = self.Dy
            eta = 1

            while True:
                #break the if beyond max iter 
                if self.record['total_iter'][s]>=self.max_iter:
                    break

                if len(self.record['acc'][s])>1 and self.record['acc'][s][-1] > self.record['acc'][s][-2]+0.1:
                    print(11)

                #select data by batch index
                if self.optimize_batch and torch.sum(self.model_curr.dual_y.flatten()[data_loader_dumb[batch_start:batch_start+b]])==0:
                    # skip if the batch are all invalid
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                elif batch_start+b <= len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                elif b >= len(data_loader_dumb):
                    batch_index = data_loader_dumb
                    batch_start = 0
                else:
                    #drop the incomplete data if they can not form a full batch
                    #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
                target_by_batch = torch.index_select(self.targets,0,index=batch_index)

                                      
                def backtrack():
                    nonlocal l,L,Deltak,upper_delta,reset_flag,R_k,Lambda_k
                    f,x,y,z,gx,gy,GMx,GMy = {},{},{},{},{},{},{},{}

                    self.model_curr.zero_grad()
                    computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)

                    self.model_bk.load_state_dict(self.model_curr.state_dict())
                    self.model_bk.zero_grad()
                    f['k'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                    z['k'] = copy.deepcopy(self.model_bk.state_dict())
                    x['k'],y['k'] = getxyFromStateModel(self.model_bk)
                    gx['k'], gy['k'] = getxyFromStateModel(self.model_bk, grad = True)
                    gz = {}
                    for name, param in self.model_bk.named_parameters():
                        gz[name] = param.grad.data.clone()

                    while True:
                        if isMaxSolver and reset_flag:
                            y_opt = self.maximizer_solver(start=self.model_curr,lr_y=self.maxsolver_step)
                            deltak = min(self.Dy, torch.norm(y_opt - self.model_curr.dual_y) ** 2 + upper_delta / foo_i ** 1.1)
                            Deltak = deltak
                            self.model_curr.dual_y.data = y_opt.clone()
                            f, x, y, z, gx, gy = {}, {}, {}, {}, {}, {}
                            self.model_curr.zero_grad()
                            computeGrad(self.model_curr, data_by_batch, target_by_batch, batch_index, b)
                            #Deltak = max(Deltak, torch.tensor(upper_delta)/ foo_i ** 1.1)
                        elif not isMaxSolver:
                            y_temp = self.model_curr.dual_y.data.clone()
                            def Y_Step(lr_y):
                                #update y only
                                for (name,param) in self.model_curr.named_parameters():
                                    if name == 'dual_y':
                                        if self.projection_y:
                                            projection_center =  param.data + lr_y*param.grad.data
                                            param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                                        else:
                                            param.data = param.data + lr_y*param.grad.data 
                            Y_Step(1/L)
                            dt = torch.norm(self.model_curr.dual_y.data.clone() - y_temp)
                            f, x, y, z, gx, gy = {}, {}, {}, {}, {}, {}
                            self.model_bk.zero_grad()
                            computeGrad(self.model_curr, data_by_batch, target_by_batch, batch_index, b)

                        while l<=L:
                            self.model_bk.load_state_dict(self.model_curr.state_dict())
                            self.model_bk.zero_grad()
                            f['k'] = computeGrad(self.model_bk, data_by_batch, target_by_batch, batch_index, b)
                            z['k'] = copy.deepcopy(self.model_bk.state_dict())
                            x['k'],y['k'] = getxyFromStateModel(self.model_bk)
                            gx['k'], gy['k'] = getxyFromStateModel(self.model_bk, grad = True)
                            gz = {}
                            for name, param in self.model_bk.named_parameters():
                                gz[name] = param.grad.data.clone()

                            lr_y = 1/l
                            lr_x = (1-gamma3)/l/(4+1/gamma2+4*(1-lr_y*self.mu_y)*(2-lr_y*self.mu_y)*(15*L-8*self.mu_y)*L**3/self.mu_y**4 )
                            # if lr_x<0.001:
                            #     print(lr_x)
                            # lr_x = (1-gamma3)/(l**2/2 + lr_y*l**2/2 + (1-lr_y*self.mu_y)*(2-lr_y*self.mu_y)*24* (L/self.mu_y)**4/lr_y)

                            def ASGDA_X(lr_x,lr_y):
                                #update x then
                                for (name,param) in self.model_bk.named_parameters():
                                    if name != 'dual_y':
                                        if self.projection_x:
                                            projection_center =  param.data - lr_x*param.grad.data
                                            param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                                        else:
                                            param.data = param.data - lr_x*param.grad.data

                                #compute the gradients of current model using batches, not that the batch does not change here
                                self.model_bk.zero_grad()
                                f['k+1,k'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                                z['k+1,k'] = copy.deepcopy(self.model_bk.state_dict())
                                x['k+1'], _ = getxyFromStateModel(self.model_bk)
                                gx['k+1,k'], gy['k+1,k'] = getxyFromStateModel(self.model_bk, grad = True)

                                #update y then
                                for (name,param) in self.model_bk.named_parameters():
                                    if name == 'dual_y':
                                        if self.projection_y:
                                            projection_center =  param.data + lr_y*param.grad.data
                                            param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                                        else:
                                            param.data = param.data + lr_y*param.grad.data 

                            ASGDA_X(lr_x,lr_y)

                            #compute the gradients of current model using batches, not that the batch does not change here
                            self.model_bk.zero_grad()
                            f['k+1'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                            z['k+1'] = copy.deepcopy(self.model_bk.state_dict())
                            x['k+1'],y['k+1'] = getxyFromStateModel(self.model_bk)
                            gx['k+1'], gy['k+1'] = getxyFromStateModel(self.model_bk, grad = True)

                            state_xk = OrderedDict(
                                (key, value) for key, value in z['k'].items() if 'dual_y' not in key
                            )

                            self.model_bk.load_state_dict(state_xk,strict=False)
                            self.model_bk.zero_grad()
                            f['k,k+1'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                            z['k,k+1'] = copy.deepcopy(self.model_bk.state_dict())
                            gx['k,k+1'], gy['k,k+1'] = getxyFromStateModel(self.model_bk, grad = True)

                            if reset_flag:
                                if isMaxSolver:
                                    Deltak = torch.tensor(2*eta*self.mu_y, device=self.device)
                                    Lambda_k = eta + np.sqrt(2*eta/self.mu_y)*L*torch.norm(y['k+1']-y['k'])
                                else:
                                    Deltak = min((1+2*L/self.mu_y)**2*dt, bar_D_y**2)
                                    Lambda_k = 2*dt*L*torch.norm(y['k']-y['k+1'])
                                    R_k = 0
                            # compute condition from here

                            GMx['k'],GMy['k'] = -(x['k+1']-x['k'])/lr_x, (y['k+1'] - y['k'])/lr_y
                            c0 = lr_x - (2+1/gamma2)*lr_x**2*l
                            c1 = c0*torch.norm(GMx['k'])**2 \
                                + lr_y**2*self.mu_y/2*torch.norm(GMy['k'])**2 \
                                    - f['k'] + f['k+1'] \
                                    -Lambda_k - R_k - 4*(3*l-2*self.mu_y)*Deltak
                            c2 = -f['k+1']+f['k+1,k'] + (y['k+1']-y['k']).T@gy['k+1,k']-l/2*torch.norm(y['k+1']-y['k'])**2
                            c3 = torch.norm(gy['k+1']-gy['k+1,k'])-l*torch.norm(y['k+1']-y['k'])
                            c4 = torch.norm(GMy['k'])**2 -2*(4*(1-lr_y*self.mu_y)/lr_y**2+2*l**2)*Deltak - 2*l**2*lr_x**2*torch.norm(GMx['k'])**2

                            self.record['total_sample_complexity'][s] += b
                            self.record['total_oracle_complexity'][s] += b / N
                            self.record['total_iter'][s] += 1
                            self.record['total_epoch'][s] += b / self.data_number_in_each_epoch

                            # set tolerance eps for numerical issues
                            if self.toymodel:
                                eps = 1e-2
                            elif self.std_x>0:
                                eps = self.std_x+self.std_y # variance guess for sto q problem
                            elif self.model_type == 'DRO':
                                if b<6000:
                                    eps = 0.005/lr_y # variance guess for sto dro problem
                                else:
                                    eps = 1e-2
                            else:
                                eps = 1e-2 #1e-12, -1e-12, larger means large variance but faster
                            

                            condition1 = (c1<=eps and c2<=eps and c3<=eps and c4<=eps)
                            

                            if verbose:
                                if not condition1:
                                    print(f'violation,L={L},l={l},eps={eps}')
                                    for i,c in enumerate([c1,c2,c3,c4]):
                                        if c>eps:
                                            print(f'c{i}:{c}')
                                    print(f'L={L},l={l},c0={c0},c1={c1.item()},c2={c2.item()},c3={c3.item()},c4={c4.item()}, eps={eps}')

                            if condition1:
                                #
                                # save and show the current data before updating
                                self.save_iterates_info(s,batch_index,lr_x,lr_y,full_block)
                                self.record['lr_x'][s].append(lr_x)
                                self.record['lr_y'][s].append(lr_y)
                                self.record['l(small)'][s].append(l)
                                self.record['L(large)'][s].append(L)
                                self.record['Deltak'][s].append(Deltak)

                                #L = max(self.mu_y, L*0.6)
                                # y_opt = self.maximizer_solver(start=self.model_curr,lr_y=self.maxsolver_step)
                                # Deltak = torch.norm(y_opt-self.model_curr.dual_y)**2 #+1e2

                                if self.is_show_result and self.record['iter'][s]%self.freq==0:
                                    self.show_result(s,batch_index, sim_done=False)

                                #     if self.record['iter'][s]>0:
                                #         print(-self.ff + f['k'])
                                # self.ff = f['k+1']

                                    #print('GMX',torch.norm(GMx['k']).item()**2,'xk+1-xk', torch.norm(x['k+1']-x['k']).item()**2,"grad x^2", torch.norm(gx['k']).item() ** 2, "grad y^2", torch.norm(gy['k']).item() ** 2)
                                    #print(condition1,condition2)
                                # update the model state to k+1
                                self.model_curr.zero_grad()
                                self.model_curr.load_state_dict(z['k+1'])
                                Ck = (1-self.mu_y*lr_y)*(2-self.mu_y*lr_y)/(self.mu_y*lr_y)*L**2/self.mu_y**2*lr_x**2
                                Bk = 1-self.mu_y*lr_y/2
                                Deltak_tmp = Deltak
                                Deltak = Bk*Deltak + Ck*torch.norm(GMx['k'])**2
                                Lambda_k = 6*l*(Deltak + 2*Deltak) - 8*self.mu_y*Deltak_tmp
                                R_k = 2*lr_x**2*l*torch.norm(GMx['k'])**2 - lr_y**2*self.mu_y*torch.norm(GMy['k'])**2
                                reset_flag = False
                                l = max(l*gamma2,tilde_l)
                                # if self.record['iter'][s]%10==0:
                                #     l = tilde_l

                                return lr_x,lr_y
                            else:
                                l = l/gamma2
                        L = L/gamma1
                        reset_flag = True
                backtrack()
                    
                # update the complexity and iterations
                self.record['sample_complexity'][s] += b
                self.record['oracle_complexity'][s] += b/N
                self.record['iter'][s] += 1
                self.record['epoch'][s] += b/self.data_number_in_each_epoch

            #show this simulation result
            self.show_result(s,batch_index, sim_done=False)
            if self.is_save_data:
                if self.model_type == 'Q':
                    foo = self.start_model.name
                    if self.toymodel:
                        foo += '_toy'
                    save_kappa = self.kappa
                else:
                    foo = self.data_name
                    save_kappa = 1
                import os
                folder_path = './result_data/' + foo + '_muy_' + str(self.mu_y) + '_kappa_' + str(save_kappa) + '_b_' + str(self.b)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_name =  folder_path + '/' + method 
                with open(file_name , "wb") as fp:  
                    pickle.dump(self.record, fp)

        return self.record

    def line_search_one_step_archive(self, gamma1:float=0.9,gamma2=0.9, gamma3=1e-3, b:int=None, N:int=1, method:str='LS-GS-GDA', min_b:int=1, force_b:int=-1, kernal='AGDA',\
                             isMaxSolver = True, isRestart = True, mu_coeff = 1,\
                             max_epoch=None,start=None,max_iter=None,is_tell_gamma=False):
        self.reset_all()
        if is_tell_gamma:
            method += f'-g1-{int(100*gamma1)}-g2-{int(100*gamma2)}-g3-{int(1000*gamma3)}'
        Model = call_model(self.model_type)
        if not b:
            b = self.b
        if not max_epoch:
            max_epoch = self.max_epoch
        if not max_iter:
            max_iter = self.max_iter

        if isMaxSolver:
            method += '-S'

        if isRestart:
            method += '-R'

        # Generate full block
        N = 1

        flattened_x = torch.cat([param.flatten() for name,param in self.start_model.named_parameters() if name!='dual_y'])
        indices = np.arange(flattened_x.shape[0])
        full_block = copy.deepcopy(indices)

        for s in range(self.sim_time):
            self.load_initial_model(s)
            self.reset_contraction(s)
            self.record['contraction_times'][s] += 1
            self.record['config'][s] = {'b':b,'N':N,'K':self.max_iter,'std_x':self.start_model.std_x, 'std_y':self.start_model.std_y}
            self.record['config'][s]['method'] = method
            self.record['config'][s]['pjx'] = self.projection_x
            self.record['config'][s]['pjy'] = self.projection_y

            #load the start model
            start = self.start_model

            self.model_curr = Model(data_size=self.data_size,mu_y=self.mu_y, kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
            self.model_curr.load_state_dict(copy.deepcopy(start.state_dict()))

            if self.model_type == 'Q':
                self.model_curr.b = b
                self.data_number_in_each_epoch = b

            #initialize the data loader and full batch
            data_loader_dumb = self.batchselect()
            #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
            batch_start = 0

            L = self.mu_y/gamma2
            upper_delta = 10
            deltak = min(self.Dy, torch.norm(self.y_opt - self.model_curr.dual_y) ** 2 + upper_delta)
            Deltak = deltak

            foo_i = 1
            self.new_stage = True

            while True:
                #break the if beyond max iter 
                if self.record['total_iter'][s]>=self.max_iter:
                    break

                if self.record['total_iter'][s]==0:
                    y_opt = self.maximizer_solver(start=self.model_curr,lr_y=self.maxsolver_step)
                    self.model_curr.dual_y.data = y_opt.clone()
                    deltak = min(self.Dy, torch.norm(self.y_opt - self.model_curr.dual_y) ** 2 + upper_delta)
                    Deltak = deltak

                if self.record['iter'][s]%int(50*foo_i) == 0 and self.record['iter'][s]>0 and isRestart:
                    L = self.mu_y
                    y_opt = self.maximizer_solver(start=self.model_curr,lr_y=self.maxsolver_step)
                    self.model_curr.dual_y.data = y_opt.clone()
                    deltak = min(self.Dy, torch.norm(y_opt-self.model_curr.dual_y)**2 + upper_delta/foo_i**1.1)
                    Deltak = deltak
                    foo_i += 1
                    self.new_stage = True

                if len(self.record['acc'][s])>1 and self.record['acc'][s][-1] > self.record['acc'][s][-2]+0.1:
                    print(11)

                #select data by batch index
                if self.optimize_batch and torch.sum(self.model_curr.dual_y.flatten()[data_loader_dumb[batch_start:batch_start+b]])==0:
                    # skip if the batch are all invalid
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                elif batch_start+b <= len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                elif b >= len(data_loader_dumb):
                    batch_index = data_loader_dumb
                    batch_start = 0
                else:
                    #drop the incomplete data if they can not form a full batch
                    #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
                target_by_batch = torch.index_select(self.targets,0,index=batch_index)
                                      
                def backtrack(gamma1,gamma2):
                    nonlocal L,Deltak,upper_delta
                    f,x,y,z,gx,gy,GMx,GMy = {},{},{},{},{},{},{},{}

                    self.model_curr.zero_grad()
                    computeGrad(self.model_curr,data_by_batch,target_by_batch, batch_index,b)

                    self.model_bk.load_state_dict(self.model_curr.state_dict())
                    self.model_bk.zero_grad()
                    f['k'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                    z['k'] = copy.deepcopy(self.model_bk.state_dict())
                    x['k'],y['k'] = getxyFromStateModel(self.model_bk)
                    gx['k'], gy['k'] = getxyFromStateModel(self.model_bk, grad = True)
                    gz = {}
                    for name, param in self.model_bk.named_parameters():
                        gz[name] = param.grad.data.clone()

                    if self.new_stage or self.toymodel:
                        initial_l =  mu_coeff * self.mu_y + 1e-6
                    else:
                        initial_l = max( mu_coeff * self.mu_y + 1e-6, self.record['l(small)'][s][-1]*gamma2)

                    l = initial_l

                    while True:
                        l = max(l*gamma2, initial_l)
                        if self.new_stage:
                            l = initial_l
                            self.new_stage = False

                        while l<=L:
                            self.model_bk.load_state_dict(z['k'])
                            for name, param in self.model_bk.named_parameters():
                                param.grad.data = gz[name].clone()

                            lr_y = 1/l
                            lr_x = (1-gamma3)/(l**2/2 + lr_y*l**2/2 + (1-lr_y*self.mu_y)*(2-lr_y*self.mu_y)*24* (L/self.mu_y)**4/lr_y)
                            Ck = (1-self.mu_y*lr_y)*(2-self.mu_y*lr_y)/(self.mu_y*lr_y)*L**2/self.mu_y**2*lr_x**2
                            Bk = 1-self.mu_y*lr_y/2

                            def ASGDA_X(lr_x,lr_y):
                                #update x then
                                for (name,param) in self.model_bk.named_parameters():
                                    if name != 'dual_y':
                                        if self.projection_x:
                                            projection_center =  param.data - lr_x*param.grad.data
                                            param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                                        else:
                                            param.data = param.data - lr_x*param.grad.data

                                #compute the gradients of current model using batches, not that the batch does not change here
                                self.model_bk.zero_grad()
                                f['k+1,k'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                                z['k+1,k'] = copy.deepcopy(self.model_bk.state_dict())
                                x['k+1'], _ = getxyFromStateModel(self.model_bk)
                                gx['k+1,k'], gy['k+1,k'] = getxyFromStateModel(self.model_bk, grad = True)

                                #update y then
                                for (name,param) in self.model_bk.named_parameters():
                                    if name == 'dual_y':
                                        if self.projection_y:
                                            projection_center =  param.data + lr_y*param.grad.data
                                            param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                                        else:
                                            param.data = param.data + lr_y*param.grad.data 

                            ASGDA_X(lr_x,lr_y)

                            #compute the gradients of current model using batches, not that the batch does not change here
                            self.model_bk.zero_grad()
                            f['k+1'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                            z['k+1'] = copy.deepcopy(self.model_bk.state_dict())
                            x['k+1'],y['k+1'] = getxyFromStateModel(self.model_bk)
                            gx['k+1'], gy['k+1'] = getxyFromStateModel(self.model_bk, grad = True)

                            state_xk = OrderedDict(
                                (key, value) for key, value in z['k'].items() if 'dual_y' not in key
                            )

                            self.model_bk.load_state_dict(state_xk,strict=False)
                            self.model_bk.zero_grad()
                            f['k,k+1'] = computeGrad(self.model_bk,data_by_batch,target_by_batch, batch_index,b)
                            z['k,k+1'] = copy.deepcopy(self.model_bk.state_dict())
                            gx['k,k+1'], gy['k,k+1'] = getxyFromStateModel(self.model_bk, grad = True)


                            # compute condition from here
                            GMx['k'],GMy['k'] = -(x['k+1']-x['k'])/lr_x, (y['k+1'] - y['k'])/lr_y
                            #c1 = f['k+1,k']-f['k']-torch.dot(x['k+1']-x['k'],gx['k'])-l/2*torch.norm(x['k+1']-x['k'])**2
                            c1 = f['k+1,k'] - f['k'] - (x['k+1'] - x['k']).T@gx['k'] - l / 2 * torch.norm(
                                x['k+1'] - x['k']) ** 2
                            c2 = torch.norm(gy['k+1,k']-gy['k'])-l*torch.norm(x['k+1']-x['k'])
                            c3 = -f['k+1']+f['k+1,k'] + (y['k+1']-y['k']).T@gy['k+1,k']-l/2*torch.norm(y['k+1']-y['k'])**2
                            c4 = torch.norm(gy['k,k+1']-gy['k'])-l*torch.norm(y['k+1']-y['k'])
                            c5 = 1/lr_y*torch.norm(y['k+1']-y['k'])-torch.sqrt((6/lr_y**2+2*l**2)*Deltak)
                            
                            c6 = lr_x-lr_x**2*l/2-l**2*lr_x**2*lr_y/2
                            
                            if self.record['iter'][s] == 0:
                                c7 = c6*torch.norm(GMx['k'])**2 + lr_y**2*self.mu_y/2*torch.norm(GMy['k'])**2 \
                                    - f['k'] + f['k+1'] + 1e-10\
                                    -lr_y/2*(4/lr_y**2 + 2*l**2)*Deltak
                            else:
                                lr_y_prev = self.record['lr_y'][s][-1]
                                l_prev = self.record['l(small)'][s][-1]
                                c7 = c6*torch.norm(GMx['k'])**2 \
                                    + lr_y**2*self.mu_y/2*torch.norm(GMy['k'])**2 \
                                    - f['k'] + f['k+1'] + 1e-10 \
                                    - lr_y / 2 * (4 / lr_y ** 2 + 2 * l ** 2) * Deltak\
                                    - (lr_y_prev/2 + lr_y_prev**3*l_prev**2/2)*(4/lr_y_prev**2 + 2*l_prev**2)*self.record['Deltak'][s][-1]\
                                    - (lr_y**2/2/lr_y_prev)*(4/lr_y**2 + 2*l**2)*Deltak

                            self.record['total_sample_complexity'][s] += b
                            self.record['total_oracle_complexity'][s] += b / N
                            self.record['total_iter'][s] += 1
                            self.record['total_epoch'][s] += b / self.data_number_in_each_epoch
                           
                            eps2 = 1e-12
                            if self.toymodel:
                                eps = -1e-12
                            elif self.std_x>0:
                                eps = self.std_x # variance guess for sto q problem
                            elif self.model_type == 'DRO' and b<6000:
                                eps = 0.005/lr_y # variance guess for sto dro problem
                            else:
                                eps = -1e-32 #1e-12, -1e-12, larger means large variance but faster
                            
                            
                            condition1 = (c1<=eps and c2<=eps and c3<=eps and c4<=eps and c5<=eps)
                            condition2 = c7<=eps and c6>=1e-32 and c3<=eps2 and c4<=eps2
                            if condition1 or condition2:
                                #print(f'L={L},l={l},c1={c1.item()},c2={c2.item()},c3={c3.item()},c4={c4.item()},c5={c5.item()}, c7={c7.item()}')
                                # save and show the current data before updating
                                self.save_iterates_info(s,batch_index,lr_x,lr_y,full_block)
                                self.record['lr_x'][s].append(lr_x)
                                self.record['lr_y'][s].append(lr_y)
                                self.record['l(small)'][s].append(l)
                                self.record['L(large)'][s].append(L)
                                self.record['Deltak'][s].append(Deltak)

                                #L = max(self.mu_y, L*0.6)
                                # y_opt = self.maximizer_solver(start=self.model_curr,lr_y=self.maxsolver_step)
                                # Deltak = torch.norm(y_opt-self.model_curr.dual_y)**2 #+1e2

                                if self.is_show_result and self.record['iter'][s]%self.freq==0:
                                    self.show_result(s,batch_index, sim_done=False)

                                #     if self.record['iter'][s]>0:
                                #         print(-self.ff + f['k'])
                                # self.ff = f['k+1']

                                    #print('GMX',torch.norm(GMx['k']).item()**2,'xk+1-xk', torch.norm(x['k+1']-x['k']).item()**2,"grad x^2", torch.norm(gx['k']).item() ** 2, "grad y^2", torch.norm(gy['k']).item() ** 2)
                                    #print(condition1,condition2)
                                # update the model state to k+1
                                self.model_curr.zero_grad()
                                self.model_curr.load_state_dict(z['k+1'])
                                Deltak = Bk*Deltak + Ck*torch.norm(GMx['k'])**2
                                return lr_x,lr_y

                            l = l/gamma2

                        L = L/gamma1

                        y_opt = self.maximizer_solver(start=self.model_curr,lr_y=self.maxsolver_step)
                        deltak = min(self.Dy, torch.norm(y_opt - self.model_curr.dual_y) ** 2 + upper_delta / foo_i ** 1.1)
                        Deltak = deltak

                        if isMaxSolver:
                            self.model_curr.dual_y.data = y_opt.clone()
                            f, x, y, z, gx, gy = {}, {}, {}, {}, {}, {}
                            self.model_bk.zero_grad()
                            computeGrad(self.model_curr, data_by_batch, target_by_batch, batch_index, b)
                            self.model_bk.load_state_dict(self.model_curr.state_dict())
                            self.model_bk.zero_grad()
                            f['k'] = computeGrad(self.model_bk, data_by_batch, target_by_batch, batch_index, b)
                            z['k'] = copy.deepcopy(self.model_bk.state_dict())
                            x['k'], y['k'] = getxyFromStateModel(self.model_bk)
                            gx['k'], gy['k'] = getxyFromStateModel(self.model_bk, grad=True)
                            #Deltak = max(Deltak, torch.tensor(upper_delta)/ foo_i ** 1.1)

                lr_x,lr_y = backtrack(gamma1,gamma2)
                    
                # update the complexity and iterations
                self.record['sample_complexity'][s] += b
                self.record['oracle_complexity'][s] += b/N
                self.record['iter'][s] += 1
                self.record['epoch'][s] += b/self.data_number_in_each_epoch

            #show this simulation result
            self.show_result(s,batch_index, sim_done=False)
            if self.is_save_data:
                if self.model_type == 'Q':
                    foo = self.start_model.name
                    if self.toymodel:
                        foo += '_toy'
                    save_kappa = self.kappa
                else:
                    foo = self.data_name
                    save_kappa = 1
                import os
                folder_path = './result_data/' + foo + '_muy_' + str(self.mu_y) + '_kappa_' + str(save_kappa) + '_b_' + str(self.b)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_name =  folder_path + '/' + method 
                with open(file_name , "wb") as fp:  
                    pickle.dump(self.record, fp)

        return self.record

    def maximizer_solver(self,start,lr_y=None,b=None,tol=None,max_iter = 1e5):
        from torch import optim
        import time
        s = time.time()
        if not lr_y:
            lr_y=min(1,self.maxsolver_step)
        if not b:
            b = self.maxsolver_b
        if not tol:
            tol = self.maxsolver_tol

        #print(f'maximizer solver start, the stepsize is {lr_y}')
        Model = call_model(self.model_type)
        #load the start model
        model_tmp = Model(data_size=self.data_size,mu_y=self.mu_y,kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
        model_tmp.load_state_dict(copy.deepcopy(start.state_dict()))
        debug = copy.deepcopy(start.state_dict())
        optimizer = optim.Adam([model_tmp.dual_y], lr=lr_y)
        #initialize the data loader
        data_loader_dumb = self.batchselect(model=model_tmp).to(self.device)
        #initialize the batch counters
        batch_start = 0
        time_counter = 1

        if torch.norm(model_tmp.dual_y.data.clone() )**2 >1e100 or torch.isnan(torch.norm(model_tmp.dual_y.data.clone() )**2):
            print('maximizer intial point not valid!!! Please debug from maximizer_solver!!!')
        
        for iter in range(int(max_iter)):
            #select data by batch index
            if self.model_type!='Q':
                #select data by batch index
                if torch.sum(model_tmp.dual_y.flatten()[batch_start:batch_start+b])==0:
                    # skip if the batch are all invalid
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                elif batch_start+b <= len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                elif b >= len(data_loader_dumb):
                    batch_index = data_loader_dumb
                    batch_start = 0
                else:
                    #drop the incomplete data if they can not form a full batch
                    #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
                    data_loader_dumb = self.batchselect(model=model_tmp).to(self.device)
                    batch_start = 0
                    continue
            else:
                batch_index = torch.tensor([],device=self.device)

            data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
            target_by_batch = torch.index_select(self.targets,0,index=batch_index)

            if self.model_type == 'DRO' and 0:
                y = model_tmp.exact_y_opt(data_by_batch,batch_index, target_by_batch)
                y = torch.tensor(pjy_y2(y.cpu().detach().numpy()),device=self.device)
                return y
            elif self.model_type == 'Q':
                y = model_tmp.exact_y_opt(data_by_batch,batch_index, target_by_batch).clone()
                if self.projection_y:
                    y = torch.tensor(pj_y(y.cpu().detach().numpy()), device=self.device)
                return y


            #compute the gradients of current model using batches
            for (name,param) in model_tmp.named_parameters():
                if name != 'dual_y':
                    param.requires_grad_(False)
            model_tmp.zero_grad()
            loss_tmp = - model_tmp.loss(data_by_batch,batch_index, target_by_batch) #we take -loss as gradient so that it is gradient descent now.
            loss_tmp.backward()
            y_prev = model_tmp.dual_y.data.clone()

            #update the model by method
            def SGDA(lr_y,start_debug):
                for (name,param) in model_tmp.named_parameters():
                    if name == 'dual_y':
                        if self.projection_y:
                            # debug = param.data.clone()
                            # assert torch.abs(torch.sum(param.data)-1)<1e-2
                            projection_center =  param.data - lr_y*param.grad.data
                            param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                        else:
                            param.data = param.data - lr_y*param.grad.data 


            SGDA(lr_y,start)
            #optimizer.step()
            iter += 1
            y_gap = torch.norm((model_tmp.dual_y.data.clone() - y_prev)/lr_y)**2

            if 0:
                y1 = model_tmp.exact_y_opt(data_by_batch,batch_index, target_by_batch)
                y1 = torch.tensor(pjy_y2(y1.cpu().detach().numpy()),device=self.device)
                if  torch.norm(model_tmp.dual_y.data - y1)>0.1:
                    print('error',torch.norm(model_tmp.dual_y.data - y1))
                
            #model_tmp.loss(data_by_batch,batch_index, target_by_batch)
            #model_tmp.dual_y.data=y1
            #model_tmp.loss(data_by_batch,batch_index, target_by_batch)

            if (time.time()-s)//30>=time_counter:
                time_counter += 1
                print(f"Warning!!! The maximizer solve has cost {(time.time()-s)//30*0.5} minutes!!!Consider to adjust maximizer solver!!!The gap is {y_gap}")
            if y_gap<tol:
                return model_tmp.dual_y.data.clone()
                    
        assert 'Fail to find the optimal y, please adjust the parameters'

    def optimizer(self, method:str, lr_x=None, lr_y=None,alpha=0.6,beta=0.4,p=0, b=None, max_epoch=None,start=None,max_iter=None):
        self.reset_all()
        Model = call_model(self.model_type)
        if not b:
            b = self.b
        if not max_epoch:
            max_epoch = self.max_epoch
        if not max_iter:
            max_iter = self.max_iter

        # Generate full block
        N = 1
        flattened_x = torch.cat([param.flatten() for name,param in self.start_model.named_parameters() if name!='dual_y'])
        indices = np.arange(flattened_x.shape[0])
        full_block = copy.deepcopy(indices)

        for s in range(self.sim_time):
            self.load_initial_model(s)
            self.reset_contraction(s)
            self.record['contraction_times'][s] += 1
            self.record['config'][s] = {'b':b,'N':N,'K':self.max_iter,'std_x':self.start_model.std_x, 'std_y':self.start_model.std_y}
            self.record['config'][s]['method'] = method
            self.record['config'][s]['pjx'] = self.projection_x
            self.record['config'][s]['pjy'] = self.projection_y

            #load the start model
            start = self.start_model

            self.model_curr = Model(data_size=self.data_size, mu_y=self.mu_y, kappa=self.kappa, device=self.device,injected_noise_x=self.std_x, injected_noise_y=self.std_y).to(self.device)
            self.model_curr.load_state_dict(copy.deepcopy(start.state_dict()))

            if self.model_type == 'Q':
                self.model_curr.b = b
                self.data_number_in_each_epoch = b

            #initialize the data loader and full batch
            full_batch = torch.arange(self.total_number_data).to(self.device)
            data_loader_dumb = self.batchselect() # torch.randperm(self.total_number_data).to(self.device)
            batch_start = 0

            #initialize the TiAda
            self.vx,self.vy = 1,1

            if method == 'TiAda':
                lr_x_TiAda = lr_x / math.pow(np.maximum(self.vx, self.vy), alpha)
                lr_y_TiAda = lr_y / math.pow(self.vy, beta)

            if method == 'Smooth-AGDA':
                for name, param in self.model_curr.named_parameters():
                    if name != 'dual_y':
                        self.zt_Smooth_AGDA[name] = 0# param.data.clone()
     
            while True:
                #select data by batch index
                #select data by batch index
                if self.optimize_batch and torch.sum(self.model_curr.dual_y.flatten()[data_loader_dumb[batch_start:batch_start+b]])==0:
                    # skip if the batch are all invalid
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                elif batch_start+b <= len(data_loader_dumb):
                    batch_index = data_loader_dumb[batch_start:batch_start+b]
                    batch_start += b
                elif b >= len(data_loader_dumb):
                    batch_index = data_loader_dumb
                    batch_start = 0
                else:
                    #drop the incomplete data if they can not form a full batch
                    #data_loader_dumb = torch.randperm(self.total_number_data).to(self.device)
                    data_loader_dumb = self.batchselect()
                    batch_start = 0
                    continue
                data_by_batch = torch.index_select(self.data,0,index=batch_index) #unseueeze is to make [64,28,28] to [64,1,28,28]
                target_by_batch = torch.index_select(self.targets,0,index=batch_index)

                # compute the gradients of current model using batches
                self.model_curr.zero_grad()
                if self.model_type == 'Q':
                    # we inject noise to the gradient of quadratic function
                    epsx = torch.rand_like(self.model_curr.x)*self.model_curr.std_x
                    epsy = torch.rand_like(self.model_curr.dual_y)*self.model_curr.std_y
                    loss = self.model_curr.loss(data_by_batch, batch_index, target_by_batch) + self.model_curr.x.T @ epsx/b + self.model_curr.dual_y.T @ epsy/b
                    loss.backward()
                else:
                    self.model_curr.loss(data_by_batch, batch_index, target_by_batch).backward()

                # save and show the current data before updating
                if method == 'TiAda':
                    self.save_iterates_info(s, batch_index, lr_x_TiAda, lr_y_TiAda, full_block)
                else:
                    self.save_iterates_info(s, batch_index, lr_x, lr_y, full_block)

                # update the model by method
                def SGDA(lr_x,lr_y):
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data
                        else:
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data 
                    
                #update the model by method
                def ASGDA(lr_x,lr_y):
                    #update y first
                    for (name,param) in self.model_curr.named_parameters():
                        if name == 'dual_y':
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data 

                    #compute the gradients of current model using batches, not that the batch does not change here
                    self.model_curr.zero_grad()
                    if self.model_type == 'Q':
                        # we inject noise to the gradient of quadratic function
                        epsx = torch.rand_like(self.model_curr.x)*self.model_curr.std_x
                        epsy = torch.rand_like(self.model_curr.dual_y)*self.model_curr.std_y
                        loss = self.model_curr.loss(data_by_batch, batch_index, target_by_batch) + self.model_curr.x.T @ epsx/b + self.model_curr.dual_y.T @ epsy/b
                        loss.backward()
                    else:
                        self.model_curr.loss(data_by_batch, batch_index, target_by_batch).backward()

                    #update x then
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data
                def ASGDA_X(lr_x,lr_y):
                    #update x then
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data

                    #compute the gradients of current model using batches, not that the batch does not change here
                    self.model_curr.zero_grad()
                    if self.model_type == 'Q':
                        # we inject noise to the gradient of quadratic function
                        epsx = torch.rand_like(self.model_curr.x)*self.model_curr.std_x
                        epsy = torch.rand_like(self.model_curr.dual_y)*self.model_curr.std_y
                        loss = self.model_curr.loss(data_by_batch, batch_index, target_by_batch) + self.model_curr.x.T @ epsx/b + self.model_curr.dual_y.T @ epsy/b
                        loss.backward()
                    else:
                        self.model_curr.loss(data_by_batch, batch_index, target_by_batch).backward()

                    #update y then
                    for (name,param) in self.model_curr.named_parameters():
                        if name == 'dual_y':
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data 

                #update the model by method
                def TiAda(lr_x,lr_y,alpha,beta):
                    assert alpha>beta
                    self.vx += self.record['norm_square_sto_grad_x'][s][-1]
                    self.vy += self.record['norm_square_sto_grad_y'][s][-1]
                    lr_x_TiAda = lr_x/math.pow(np.maximum(self.vx,self.vy),alpha)
                    lr_y_TiADA = lr_y/math.pow(self.vy,beta)
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x/math.pow(np.maximum(self.vx,self.vy),alpha)*param.grad.data 
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x/math.pow(np.maximum(self.vx,self.vy),alpha)*param.grad.data 
                        else:
                            if self.projection_y:
                                projection_center =  param.data + lr_y/math.pow(self.vy,beta)*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y/math.pow(self.vy,beta)*param.grad.data
                    return lr_x_TiAda,lr_y_TiADA

                def Smooth_AGDA(lr_x,lr_y,p,beta):
                    #update x first
                    for (name,param) in self.model_curr.named_parameters():
                        if name != 'dual_y':
                            if self.projection_x:
                                projection_center =  param.data - lr_x*param.grad.data - lr_x*p*(param.data - self.zt_Smooth_AGDA[name])
                                param.data = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device).clone()
                            else:
                                param.data = param.data - lr_x*param.grad.data - lr_x*p*(param.data - self.zt_Smooth_AGDA[name])

                            temp = self.zt_Smooth_AGDA[name] + beta*(param.data - self.zt_Smooth_AGDA[name])
                            self.zt_Smooth_AGDA[name] = temp.clone()

                    #compute the gradients of current model using batches, not that the batch does not change here
                    self.model_curr.zero_grad()
                    if self.model_type == 'Q':
                        # we inject noise to the gradient of quadratic function
                        epsx = torch.rand_like(self.model_curr.x)*self.model_curr.std_x
                        epsy = torch.rand_like(self.model_curr.dual_y)*self.model_curr.std_y
                        loss = self.model_curr.loss(data_by_batch, batch_index, target_by_batch) + self.model_curr.x.T @ epsx/b + self.model_curr.dual_y.T @ epsy/b
                        loss.backward()
                    else:
                        self.model_curr.loss(data_by_batch, batch_index, target_by_batch).backward()
                    #update y then
                    for (name,param) in self.model_curr.named_parameters():
                        if name == 'dual_y':
                            if self.projection_y:
                                projection_center =  param.data + lr_y*param.grad.data
                                param.data = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64).to(self.device)
                            else:
                                param.data = param.data + lr_y*param.grad.data
                    return lr_x,lr_y



                if method == 'GDA':
                    SGDA(lr_x,lr_y)
                elif method == 'AGDA':
                    ASGDA(lr_x,lr_y)
                elif method == 'AGDA_X':
                    ASGDA_X(lr_x,lr_y)
                elif method == 'SAPD':
                    print('method error!!!')
                elif method == 'TiAda':
                    lr_x_TiAda,lr_y_TiAda = TiAda(lr_x,lr_y,alpha,beta)
                elif method == 'Smooth-AGDA':
                    Smooth_AGDA(lr_x, lr_y, p, beta)


                if method == 'TiAda':
                    self.record['lr_x'][s].append(lr_x_TiAda)
                    self.record['lr_y'][s].append(lr_y_TiAda)
                else:
                    self.record['lr_x'][s].append(lr_x)
                    self.record['lr_y'][s].append(lr_y)

                #break the if beyond max iter
                if self.record['iter'][s]>=self.max_iter:
                    break

                if self.is_show_result and self.record['iter'][s]%self.freq==0:
                    self.show_result(s,batch_index, sim_done=False)

                # update the complexity and iterations
                self.record['total_sample_complexity'][s] += b
                self.record['total_oracle_complexity'][s] += b/N
                self.record['total_iter'][s] += 1
                self.record['total_epoch'][s] += b/self.data_number_in_each_epoch

                self.record['sample_complexity'][s] += b
                self.record['oracle_complexity'][s] += b/N
                self.record['iter'][s] += 1
                self.record['epoch'][s] += b/self.data_number_in_each_epoch

            #show this simulation result
            self.show_result(s,batch_index,sim_done=True)
            if self.is_save_data:
                if self.model_type == 'Q':
                    foo = self.start_model.name
                    if self.toymodel:
                        foo += '_toy'
                    save_kappa = self.kappa
                else:
                    foo = self.data_name
                    save_kappa = 1
                import os
                folder_path = './result_data/' + foo + '_muy_' + str(self.mu_y) + '_kappa_' + str(save_kappa) + '_b_' + str(self.b)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_name =  folder_path + '/' + method 
                with open(file_name , "wb") as fp:  
                    pickle.dump(self.record, fp)
    
    def batchselect(self, model=None):
        from random import shuffle
        if not self.optimize_batch:
            return torch.randperm(self.total_number_data).to(self.device)
        if not model:
            model = self.model_curr

        x = model.dual_y.flatten()
        positive_indices = torch.nonzero(x > 0).view(-1)
        foo = torch.randperm(positive_indices.size(0)).to(self.device) #s
        positive_indices = positive_indices[foo]
        
        other_indices = torch.nonzero(x <= 0).view(-1)
        l = positive_indices.shape[0] - positive_indices.shape[0]//10
        l = positive_indices.shape[0] - positive_indices.shape[0]//30
        combined_indices = torch.cat((positive_indices[:l], other_indices,positive_indices[l:]))

        sorted_indices = torch.argsort(model.dual_y.data.flatten(),descending=True)
        
        return combined_indices
    
    def save_iterates_info(self, s, batch, lr_x, lr_y, chosen_block=None, gx=None, gy=None):
        #record the iter or complexity or epoch for plotting x-axis
        for name in ['iter','epoch','oracle_complexity','sample_complexity']:
            self.record[name+'_counter'][s].append(self.record[name][s])
            self.record['total_' + name + '_counter'][s].append(self.record['total_' + name][s])
        
        #compute and save the current norm square of sto gradients for plotting y-axis
        if gx!=None and gy!=None:
            grad_x_tmp,grad_y_tmp = gx**2,gy**2
        else:
            grad_x_tmp = 0
            grad_y_tmp = 0
            flattened_x = torch.cat([param.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])
            flattened_x_grad = torch.cat([param.grad.data.clone().flatten() for name,param in self.model_curr.named_parameters() if name!='dual_y'])

            if self.projection_x:
                projection_center = flattened_x[chosen_block] - lr_x*flattened_x_grad[chosen_block]
                flattened_x_new = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64,device =self.device)
                grad_x_tmp = torch.norm(1/lr_x*(flattened_x_new - flattened_x[chosen_block])).item()**2
            else:
                grad_x_tmp = torch.norm(flattened_x_grad[chosen_block]).item()**2

            for name,param in self.model_curr.named_parameters():
                if name == 'dual_y':
                    if self.projection_y:
                        projection_center =  param.data.clone() + lr_y*param.grad.data.clone()
                        y_new = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64, device=self.device)
                        y_new = 1/lr_y*(y_new.clone() - param.data.clone())
                        grad_y_tmp += torch.norm(y_new).item()**2
                    else:
                        grad_y_tmp += torch.norm(param.grad.data).item()**2
        self.record['norm_square_sto_grad_x'][s].append(grad_x_tmp)
        self.record['norm_square_sto_grad_y'][s].append(grad_y_tmp)

        #compute the current loss and acc
        if gx!=None and gy!=None:
            grad_x_tmp, grad_y_tmp = gx**2, gy**2
        else:
            full_batch = torch.arange(self.total_number_data).to(self.device)

            self.model_copy.load_state_dict(self.model_curr.state_dict())
            self.model_copy.dual_y.data = self.maximizer_solver(start=self.model_copy,lr_y=self.maxsolver_step)
            primalF = self.model_copy.loss(self.data,full_batch,self.targets)
            self.record['primalF'][s].append(
                primalF.item())

            self.model_copy.load_state_dict(self.model_curr.state_dict())
            self.model_copy.zero_grad()
            loss = self.model_copy.loss(self.data,full_batch,self.targets)
            loss.backward()
            self.record['loss'][s].append(
                loss.item())
            self.record['acc'][s].append(
                torch.sum(self.model_copy.predict(self.data)==self.targets).item()/self.total_number_data
                )
            

            #compute and save the current norm square of full gradients for plotting y-axis
            grad_x_tmp = 0
            grad_y_tmp = 0
            flattened_x = torch.cat([param.data.clone().flatten() for name,param in self.model_copy.named_parameters() if name!='dual_y'])
            flattened_grad = torch.cat([param.grad.data.clone().flatten() for name,param in self.model_copy.named_parameters() if name!='dual_y'])

            if self.projection_x:
                projection_center =  flattened_x[chosen_block] - lr_x*flattened_grad[chosen_block]
                flattened_x_new = torch.tensor(pj_x(projection_center.cpu().detach().numpy()),dtype=torch.float64,device =self.device)
                grad_x_tmp = torch.norm(1 / lr_x * (flattened_x_new - flattened_x[chosen_block])).item() ** 2
            else:
                grad_x_tmp = torch.norm(flattened_grad[chosen_block]).item()**2

            for name,param in self.model_copy.named_parameters():
                if name == 'dual_y':
                    if self.projection_y:
                        projection_center =  param.data.clone() + lr_y*param.grad.data.clone()
                        y_new = torch.tensor(pj_y(projection_center.cpu().detach().numpy()),dtype=torch.float64, device=self.device)
                        y_new = 1/lr_y*(y_new.clone() - param.data.clone())
                        grad_y_tmp += torch.norm(y_new).item()**2
                    else:
                        grad_y_tmp += torch.norm(param.grad.data).item()**2
        self.record['norm_square_full_grad_x'][s].append(grad_x_tmp)
        self.record['norm_square_full_grad_y'][s].append(grad_y_tmp)

    def show_result(self, s, batch_index, sim_done = False):
        method = self.record['config'][s]['method']
        if sim_done:
            foo = self.record['contraction_times'][s]
            print(f'=================================================================================================')
            print(f'================={s+1}th sim is done!!! method:{method}. contraction:{foo}=================')
        else:
            print(f'-----------------------------------------------------------------------------------------------')
        print('Total Complexity:')
        print('iter:', self.record['total_iter'][s], 'epoch:', self.record['total_epoch'][s], 'OC:',  self.record['total_oracle_complexity'][s], 'SC:',  self.record['total_sample_complexity'][s])
        print('Current Complexity:')
        print('iter:', self.record['iter'][s], 'epoch:', self.record['epoch'][s], 'OC:',  self.record['oracle_complexity'][s], 'SC:',  self.record['sample_complexity'][s])
        print('Iterates Info:')
        print('acc:', self.record['acc'][s][-1], ', loss:', self.record['loss'][s][-1], ', primalF:', self.record['primalF'][s][-1])
        print('||sto grad_x||^2:', self.record['norm_square_sto_grad_x'][s][-1],'||sto grad_y||^2:', self.record['norm_square_sto_grad_y'][s][-1])
        print('||true grad_x||^2:', self.record['norm_square_full_grad_x'][s][-1], '||true grad_y||^2:',
              self.record['norm_square_full_grad_y'][s][-1])
        print('lr_x:',self.record['lr_x'][s][-1], ', lr_y:',self.record['lr_y'][s][-1])
        print('mu:', self.mu_y, ', kappa:', self.kappa)
        if 'LS' in method:
            print('l(small):', self.record['l(small)'][s][-1], \
                ', L(large):', self.record['L(large)'][s][-1],\
                  ', Deltak:', self.record['Deltak'][s][-1].item()
                  )
        print(self.record['config'][s])
        if self.model_type =='DRO':
            print('positive number of yi(batch):',torch.sum(self.model_curr.dual_y[batch_index]>0).item(), \
                    'max y_i:' , torch.max(self.model_curr.dual_y[batch_index]).item(),\
                    'min y_i:' , torch.min(self.model_curr.dual_y[batch_index]).item())
            print('positive number of yi(total):',torch.sum(self.model_curr.dual_y>0).item(), \
                    'max y_i:' , torch.max(self.model_curr.dual_y).item(),\
                    'min y_i:' , torch.min(self.model_curr.dual_y).item())
        if sim_done:
            print(f'=================================================================================================\n\n')
        else:
            print(f'-----------------------------------------------------------------------------------------------\n\n')