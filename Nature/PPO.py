import torch
from torch import nn
import sympy as sp
import numpy as np
from copy import deepcopy
from attNet import netAttention
import torch.nn.functional as F

class ouNoiseGenerator:
    def __init__(self, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None

    def reset(self):
        self.state = None

    def sample(self, size):
        size = int(size[0])
        if self.state is None:
            self.state = np.ones(size) * self.mu
        x = self.state + self.theta * (self.mu - self.state) + \
            self.sigma * np.random.randn(size)
        self.state = x
        return torch.tensor(x, dtype=torch.float32)

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, task_num, provider_num, net,device="cpu", D=None): 
        '''
            input_dim = provider_num
            output_dim = task_num*provider_num
        '''
        super().__init__()
        self.D = D.unsqueeze(0).to(device)
        self.DT = D.T.unsqueeze(0).to(device)
        # attention machine
        self.qlinear = nn.Sequential(
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
        )
        
        
        self.klinear = nn.Sequential(
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
        )
        
        self.vlinear = nn.Sequential(
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
            nn.Linear(input_dim,input_dim).to(device),
        )
        
        # T attention machine
        self.Tqlinear = nn.Sequential(
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
        )
        
        self.Tklinear = nn.Sequential(
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
        )
        
        self.Tvlinear = nn.Sequential(
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
            nn.Linear(task_num,task_num).to(device),
        )
        
        # first linear layer
        self.linear = nn.Sequential(
            nn.Linear(provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,provider_num*task_num),
            nn.ReLU()
        )
        

        # degorate layer
        self.degorate = nn.Sequential(
            nn.Linear(provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,provider_num),
        )
        self.derograteT = nn.Sequential(
            nn.Linear(provider_num*task_num,2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num,task_num),
        )
        self.decision = nn.Sequential(
            nn.Linear(provider_num+task_num*2,(provider_num+task_num)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+task_num)*3//2,provider_num),
        )
        
        # final linear layer with current position added
        self.final = nn.Sequential(
            nn.Linear(provider_num+output_dim,output_dim),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(provider_num+output_dim, (provider_num+output_dim)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+output_dim)*3//2, 1),
            nn.ReLU()
        )
        # lstm components
        self.hidden_state = [torch.zeros(output_dim).unsqueeze(0).to(device)]
        self.cell_state = [torch.zeros(output_dim).unsqueeze(0).to(device)]
        self.forgetDoor = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        )
        self.inputDoor1 = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        )
        self.inputDoor2 = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Tanh()
        )
        self.outputDoor = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.taskL = []
        for i in range(task_num):
            new_net = nn.Sequential(
                nn.Linear(provider_num+output_dim, provider_num+1).to(device),
                nn.Softmax(dim=1)
            )
            self.taskL.append(new_net)
        
        self.criteria = nn.MSELoss().to(device)
        self.task_num = task_num
        self.provider_num = provider_num
        
        # init weights
        for m in self.linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.degorate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.final.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.forgetDoor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.inputDoor1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.inputDoor2.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.outputDoor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.klinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.vlinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.qlinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.Tklinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.Tvlinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.Tqlinear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.expL = [] # format: (X, pos, prehc, decision, value, reward)
        self.ouNoise = ouNoiseGenerator(mu=0, theta=0.15, sigma=0.2)

    def attention(self, x):
        for ql, kl, vl in zip(self.qlinear, self.klinear, self.vlinear):
            k = kl(x).T.unsqueeze(0)
            v = vl(x).unsqueeze(0)
            q = ql(x).unsqueeze(0)
            x = torch.div(torch.bmm(q, k), np.sqrt(k.shape[1]))
            x = torch.bmm(x, self.D.to(x.device))
            x = F.softmax(x, dim=-1)
            x = torch.bmm(x, v).squeeze(0)
        return x
    
    def Tattention(self, x):
        x = x.T
        for ql, kl, vl in zip(self.Tqlinear, self.Tklinear, self.Tvlinear):
            k = kl(x).T.unsqueeze(0)
            v = vl(x).unsqueeze(0)
            q = ql(x).unsqueeze(0)
            x = torch.div(torch.bmm(q, k), np.sqrt(k.shape[1]))
            x = F.softmax(x, dim=-1)
            x = torch.bmm(x, v).squeeze(0)
        x = x.T
        return x
    
    def forward(self,x,pos,prehc, AP):
        x = self.attention(x)
        x_t = self.derograteT(x.flatten())
        x = self.Tattention(x)
        x = x.flatten()
        x = x.unsqueeze(0) 
        
        x = self.linear(x)
        
        x_p = self.degorate(x)
        x_t = x_t.unsqueeze(0)    
        
        x = self.decision(torch.cat([x_t,x_p,pos],dim=1))
        
        self.hidden_state.append(None)
        self.cell_state.append(None)
        
        remember = self.forgetDoor(torch.cat([x,self.hidden_state[prehc]],dim=1))
        input1 = self.inputDoor1(torch.cat([x,self.hidden_state[prehc]],dim=1))
        input2 = self.inputDoor2(torch.cat([x,self.hidden_state[prehc]],dim=1))
        output = self.outputDoor(torch.cat([x,self.hidden_state[prehc]],dim=1))
        
        self.cell_state[-1] = (remember * self.cell_state[prehc] + input1 * input2).clone().detach()
        out = output * torch.tanh(self.cell_state[-1]).clone().detach()
        self.hidden_state[-1] = out.clone().detach()
        hcV = len(self.hidden_state)-1
        prex = x.clone().detach()
        ansV = torch.zeros((1,(self.provider_num+1)*self.task_num)).to(x.device)
        for ap in AP:
            ans = self.taskL[ap](torch.cat([x,out],dim=1))
            ansV[0][ap*(self.provider_num+1):(ap+1)*(self.provider_num+1)] = ans[0]
        x = ansV
        value = self.value(torch.cat([x,prex],dim=1))
        return x , hcV, value
    
    def train_forward(self,x,pos,prehc, AP):
        x = self.attention(x)
        x_t = self.derograteT(x.flatten())
        x = self.Tattention(x)
        x = x.flatten()
        x = x.unsqueeze(0) 
        
        x = self.linear(x)
        
        x_p = self.degorate(x)
        x_t = x_t.unsqueeze(0)    
        
        x = self.decision(torch.cat([x_t,x_p,pos],dim=1))
        
        self.hidden_state.append(None)
        self.cell_state.append(None)
        
        remember = self.forgetDoor(torch.cat([x,self.hidden_state[prehc]],dim=1))
        input1 = self.inputDoor1(torch.cat([x,self.hidden_state[prehc]],dim=1))
        input2 = self.inputDoor2(torch.cat([x,self.hidden_state[prehc]],dim=1))
        output = self.outputDoor(torch.cat([x,self.hidden_state[prehc]],dim=1))
        
        self.cell_state[-1] = (remember * self.cell_state[prehc] + input1 * input2).clone().detach()
        out = output * torch.tanh(self.cell_state[-1]).clone().detach()
        self.hidden_state[-1] = out.clone().detach()
        hcV = len(self.hidden_state)-1
        prex = x.clone().detach()
        ansV = torch.zeros((1,(self.provider_num+1)*self.task_num)).to(x.device)
        for ap in AP:
            ans = self.taskL[ap](torch.cat([x,out],dim=1))
            ansV[0][ap*(self.provider_num+1):(ap+1)*(self.provider_num+1)] = ans[0]
        x = ansV + self.ouNoise.sample(ansV.shape[1:]).unsqueeze(0).to(x.device)  # add noise for exploration
        value = self.value(torch.cat([x,prex],dim=1))
        return x , hcV, value

    def forward_based_on_given_hc(self, x, pos, hiddenstate, cellstate, AP):
        x = x.squeeze(0)
        x = self.attention(x)
        x_t = self.derograteT(x.flatten())
        x = self.Tattention(x)
        x = x.flatten()
        x = x.unsqueeze(0)

        x = self.linear(x)

        x_p = self.degorate(x)
        x_t = x_t.unsqueeze(0)

        x = self.decision(torch.cat([x_t, x_p, pos[0]], dim=1))

        hiddenstate = hiddenstate.squeeze(0)
        cell_state = cellstate.squeeze(0)

        remember = self.forgetDoor(torch.cat([x, hiddenstate], dim=1))
        input1 = self.inputDoor1(torch.cat([x, hiddenstate], dim=1))
        input2 = self.inputDoor2(torch.cat([x, hiddenstate], dim=1))
        output = self.outputDoor(torch.cat([x, hiddenstate], dim=1))

        cell_state = (remember * cellstate + input1 * input2).clone().detach()
        out = output * torch.tanh(cell_state).clone().detach()

        ansV = torch.zeros((1, (self.provider_num + 1) * self.task_num)).to(x.device)
        for ap in AP:
            ans = self.taskL[ap](torch.cat([x, out[0]], dim=1))
            ansV[0][ap * (self.provider_num + 1):(ap + 1) * (self.provider_num + 1)] = ans[0]

        return ansV, out, cell_state
    
    def choose_action(self, y):
        args = y[0].argmax()
        sense = "And"
        arg = y[self.provider_num * self.task_num:self.provider_num * self.task_num*2]
        return sense, arg
        
    
    def training_step(self,x,pos,y):
        y_hat = self(x,pos)
        loss = self.criteria(y_hat, y)
        return loss
    
    def configure_optimizers(self,optimizer,scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
        return [self.optimizer], [self.scheduler]
    
    def add_expL(self, X, pos, prehc, decision, value, reward, AP):
        self.expL.append((X.clone(), pos.clone(), prehc, decision.clone(), value.clone(), reward.clone(), AP))
        return
    
    def PPOtrain(self, batchsize=5):
        '''
        PPO training
        '''
        # transfer into batch
        loss_sum = 0
        loss = torch.tensor(0.0).to(self.expL[0][0].device)
        self.expL = self.expL[-5000:]
        for i in range(len(self.expL)):
            X, pos, prehc, decision, value, reward, availablep = self.expL[i]
            
            y_hat, hcV, v = self.forward(X.clone().detach(), pos, prehc, availablep)
            
            ratio = (decision)/(y_hat + 1e-10)
            surr1 = ratio * (value-reward).detach()
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * value

            loss += -torch.min(surr1, surr2).mean()

            loss += self.criteria(v, value)

            self.expL[i] = (X.clone().detach(), pos.clone().detach(), prehc, y_hat.clone().detach(), v.clone().detach(), reward.clone(), availablep)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        loss_sum += loss.item()
        return torch.tensor(loss_sum / (len(self.expL)))


class point:
    def __init__(self):
        self.parent = []
        self.children = []
        self.sense = "And" # AND, OR sense
        self.loc = 0
        self.provider = 0
        self.task = 0
        self.finished = False
        self.time = None


class network:
    def __init__(self, deadlines, budgets, Rs, abilities, cost,\
        input_dim, output_dim, provider_num, task_num, edges, device="cpu"):
        self.name = "xtmPPO"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.points = []
        for j in range(task_num):
            p = self.add_point(T=10)
            p.deadline = deadlines[j]
            p.budget = budgets[j]
            p.R = Rs[j]
            p.ability = abilities[j]
            p.cost = cost[j]
        self.D = torch.zeros((task_num, task_num)).to(device) # Dependency matrix
        for e in edges:
            self.add_edge(self.points[e[0]], self.points[e[1]], 1)
            self.D[e[0]][e[1]] = 1
        self.edges = edges
        self.model = Net(input_dim, output_dim, task_num, provider_num, self,device, self.D).to(device)
        self.provider_num = provider_num
        self.task_num = task_num
        self.constrains = []
        self.device = device
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        self.model.configure_optimizers(optimizer, scheduler)
        self.critical_path = -1
        self.paths = [0 for i in range(task_num)]
        self.lam = 0.1
        self.lam = torch.tensor(self.lam).to(device)
        self.omega = [0.2,0.2,0.2,0.2,0.2]
        self.provider = torch.tensor([0 for i in range(provider_num)]).to(device)
        self.providerL = []
        # self.alpha = 0.5/self.task_num
        self.alpha = 0.1
        self.alphaC = 0.1
        self.alpha = torch.tensor(self.alpha).to(device)
        self.alphaC = torch.tensor(self.alphaC).to(device)
        self.providerW = torch.tensor([0 for i in range(provider_num)]).to(device)
        self.pathe = [0 for i in range(task_num)]
        self.edges = edges
        self.M = 1000000

    def add_point(self,T):
        p = point()
        self.points.append(p)
        p.T = T
        p.loc = len(self.points) - 1
        return p
    
    def set_beginning(self, point):
        self.beginning = point
        return point
    
    def available_P(self, p):
        ans = []
        for c in p.children:
            ans.append(c.loc)
        return ans
    
    def forward(self, x, point, p, prehc):
        y, hc, v = self.model.forward(x, point, prehc, self.available_P(p))
        # self.temp.append((x.clone(), point.clone(), prehc, y.clone(), v.clone()))
        return y, hc, v
    
    def train_forward(self, x, point, p, prehc): # calculate the choice of this step
        y, hc, v = self.model.train_forward(x, point, prehc, self.available_P(p))
        
        return y, hc, v
    
    def proceed(self, x, point): # calculate the choice of this step
        '''
        x: current state of the task match with providers
        point: current task
        '''
        if point.loc != -1:
            pos = torch.zeros(self.task_num)
            # pos[point.loc] = 1
            pos = pos.unsqueeze(0).to(self.device)
        else:
            pos = torch.zeros(self.task_num).unsqueeze(0).to(self.device)
        y, hcV, v = self.forward(x, pos, point, point.hc)
        sense, args = self.model.choose_action(y)
        self.temp.append((x.clone(), pos.clone(), point.hc, y.clone(), v.clone()))
        provider = []
        task = []
        points = []
        choose_V = 0
        provider = []
        task = []
        points = []
        choose_V = []
        if self.andor[point.loc] == "or":
            arg = -1
            maxv = -1
            for c in point.children:
                i = c.loc
                for j in range(self.provider_num):
                    if maxv < y[0][i*(self.provider_num+1)+j]:
                        arg = i*(self.provider_num+1)+j
                        maxv = y[0][i*(self.provider_num+1)+j]
                        choose_V = j
            t = arg//(self.provider_num+1)
            provid = arg%(self.provider_num+1)
            self.provider[provid] += int(self.points[t].R)
            self.providerW[provid] += 1
            self.points[t].finished = True
            self.points[t].provider = provid
            self.points[t].sense = "Or"
            self.points[t].hc = hcV
            self.temp.append((x.clone(), pos.clone(), self.points[t].hc, y.clone(), v.clone()))
            task.append(t)
            points.append(self.points[t])
        else:
            for c in point.children:
                if c.finished:
                    continue
                # for p in c.parent:
                #     if not p.finished:
                #         continue
                maxV = -1
                provid = -1
                if self.andor[point.loc] == "and":
                    for i in range(self.provider_num):
                        if maxV < y[0][c.loc*(self.provider_num+1)+i]:
                            maxV = y[0][c.loc*(self.provider_num+1)+i]
                            provid = i
                else:
                    for i in range(self.provider_num+1):
                        if maxV < y[0][c.loc*(self.provider_num+1)+i]:
                            maxV = y[0][c.loc*(self.provider_num+1)+i]
                            provid = i
                if maxV == -1:
                    continue
                if int(provid) < int(self.provider_num):
                    provider.append(provid)
                    self.provider[provider[-1]] += int(c.R)
                    self.providerW[provider[-1]] += 1
                    c.finished = True
                    task.append(c.loc)
                    points.append(c)
                    c.provider = provid
                    c.sense = "And"
                    c.hc = hcV
                    
                    self.temp.append((x.clone(), pos.clone(), c.hc, y.clone(), v.clone()))
        if provider == []:
            return None, None, None, None, None
        return provider, task, points, sense, choose_V
    
    def train_proceed(self, x, point): # calculate the choice of this step
        '''
        x: current state of the task match with providers
        point: current task
        '''
        if point.loc != -1:
            pos = torch.zeros(self.task_num)
            # pos[point.loc] = 1
            pos = pos.unsqueeze(0).to(self.device)
        else:
            pos = torch.zeros(self.task_num).unsqueeze(0).to(self.device)
        y, hcV, v = self.train_forward(x, pos, point, point.hc)
        self.temp.append((x.clone(), pos.clone(), point.hc, y.clone(), v.clone()))
        sense, args = self.model.choose_action(y)
        provider = []
        task = []
        points = []
        choose_V = 0
        provider = []
        task = []
        points = []
        choose_V = []
        if self.andor[point.loc] == "or":
            arg = -1
            maxv = -1
            for c in point.children:
                i = c.loc
                for j in range(self.provider_num):
                    if maxv < y[0][i*(self.provider_num+1)+j]:
                        arg = i*(self.provider_num+1)+j
                        maxv = y[0][i*(self.provider_num+1)+j]
                        choose_V = j
            t = arg//(self.provider_num+1)
            provid = arg%(self.provider_num+1)
            self.provider[provid] += int(self.points[t].R)
            self.providerW[provid] += 1
            self.points[t].finished = True
            self.points[t].provider = provid
            self.points[t].sense = "Or"
            self.points[t].hc = hcV
            self.temp.append((x.clone(), pos.clone(), self.points[t].hc, y.clone(), v.clone()))
            task.append(t)
            points.append(self.points[t])
        else:
            for c in point.children:
                if c.finished:
                    continue
                # for p in c.parent:
                #     if not p.finished:
                #         continue
                maxV = -1
                provid = -1
                if self.andor[point.loc] == "and":
                    for i in range(self.provider_num):
                        if maxV < y[0][c.loc*self.provider_num+i]:
                            maxV = y[0][c.loc*self.provider_num+i]
                            provid = i
                else:
                    for i in range(self.provider_num+1):
                        if maxV < y[0][c.loc*(self.provider_num+1)+i]:
                            maxV = y[0][c.loc*(self.provider_num+1)+i]
                            provid = i
        
                if maxV == -1:
                    continue
                if int(provid) < int(self.provider_num):
                    provider.append(provid)
                    self.provider[provider[-1]] += int(c.R)
                    self.providerW[provider[-1]] += 1
                    c.finished = True
                    task.append(c.loc)
                    points.append(c)
                    c.provider = provid
                    c.sense = "And"
                    c.hc = hcV
                    
                    self.temp.append((x.clone(), pos.clone(), c.hc, y.clone(), v.clone()))
        if provider == []:
            return None, None, None, None, None
        return provider, task, points, sense, choose_V
    
    def search(self, startpoint): # search the whole network, choose with proceed()
        '''
        startpoint: the beginning point of the network, which is ahead of the point 0
        '''
        
        points = [startpoint]
        calcuated = []
        x_list = []
        x = torch.zeros(self.task_num, self.provider_num).to(self.device)
        x_list.append(x)
        temp = torch.clone(x)
        
        while len(points) > 0:
            p = points.pop(0)
            if p.children == []:
                continue
            provider, task, point, sense, choose = self.proceed(x_list[p.L], p)
            if provider == None:
                continue
            temp = torch.clone(x)
            temp = temp.clone()
            for pr, t, po in zip(provider, task, point):
                temp = temp.clone()
                if int(pr) == int(self.provider_num):
                    po.finished = False
                else:
                    po.finished = True
                    temp[t, pr] = 1
                
                    points.append(po)
                    po.L = len(x_list)
                    po.sense = "And"
                    po.provider = pr
                    
                    x_list.append(temp)
                    calcuated.append(choose)
            
        return calcuated, x_list

    def search_train(self, startpoint, epislon=0.5): # search the whole network, choose with train_proceed()
        '''
        startpoint: the beginning point of the network, which is ahead of the point 0
        '''
        points = [startpoint]
        calcuated = []
        x_list = []
        x = torch.zeros(self.task_num, self.provider_num).to(self.device)
        x_list.append(x)
        temp = torch.clone(x)
        
        while len(points) > 0:
            p = points.pop(0)
            if p.children == []:
                continue
            provider, task, point, sense, choose = self.train_proceed(x_list[p.L], p)
            if provider == None:
                continue
            temp = torch.clone(x)
            temp = temp.clone()
            for pr, t, po in zip(provider, task, point):
                temp = temp.clone()
                if int(pr) == int(self.provider_num):
                    po.finished = False
                else:
                    po.finished = True
                    temp[t, pr] = 1
                
                    points.append(po)
                    po.L = len(x_list)
                    po.sense = "And"
                    po.provider = pr
                    
                    x_list.append(temp)
                    calcuated.append(choose)
            
        return calcuated, x_list

    def criticalpath(self, p, time): # calculate the critical path
        pT = 0
        if not p.finished:
            self.critical_path = max(self.critical_path, time)
            return
        if p.loc != -1:
            if self.paths[p.loc] > time:
                self.paths[p.loc] = time
            # cR = (1-torch.exp(self.alpha*1)) if self.providerW[p.provider] > 1 else 1
            # pT = self.taskTime[p.provider][p.loc]*cR
            pT = self.taskTime[p.provider][p.loc]
            if self.pathe[p.loc] < time+pT:
                self.pathe[p.loc] = time+pT
        else:
            pT = 0
        for c in p.children:
            if c.finished:
                self.criticalpath(c, time+pT)
        if len(p.children) == 0:
            if self.critical_path < time+pT:
                self.critical_path = time+pT
        return
    
    def traceCriticalPath(self, p, time, critical_path): # trace the critical path
        pT = 0
        
        if p.loc != -1:
            if self.paths[p.loc] < time:
                self.paths[p.loc] = time
            # cR = (1-torch.exp(self.alpha*1)) if self.providerW[p.provider] > 1 else 1
            # pT = self.taskTime[p.provider][p.loc]*cR
            pT = self.taskTime[p.provider][p.loc]
            if self.pathe[p.loc] < time+pT:
                self.pathe[p.loc] = time+pT
        else:
            pT = 0
        critical_path.append((p.loc, p.provider, time, pT))
        if len(p.children) == 0:
            if self.critical_path < time+pT:
                self.critical_path = time+pT
                self.crPath = critical_path.copy()
        for c in p.children:
            if c.finished:
                self.traceCriticalPath(c, time+pT, critical_path)

        critical_path.pop(-1)
        return      
    
    def objv(self, x): # calculate the objective value
        '''
        x: the final state of the network
        '''
        # calculate length of critical path
        self.critical_path = -1
        self.pathe = [0 for i in range(self.task_num)]
        self.paths = [0 for i in range(self.task_num)]
        self.criticalpath(self.beginning, 0)
        budget = self.budget
        cost = 0
        for p in self.points:
            if p.finished:
                cost += self.providerPrice[p.provider][p.loc]
        if cost == 0:
            return torch.tensor(-100).float(), torch.tensor([-100, -100]).float()  
        self.cost = cost
        satisfaction = torch.tensor((budget - cost) / budget).to(self.device)
        if self.deadline > self.critical_path:
            satisfaction += torch.exp(torch.tensor(-self.lam * (self.deadline-self.critical_path))).to(self.device)
        satisfaction /= 2
        
        sat1 = satisfaction.clone()

        for p in self.points:
            if p.finished:
                sT = 0
                if self.deadlines[p.loc]>self.pathe[p.loc]:
                    sT = self.deadlines[p.loc]/self.pathe[p.loc]
                else:
                    sT = 1
                sP = self.budgets[p.loc]/self.providerPrice[p.provider][p.loc] if p.budget>self.providerPrice[p.provider][p.loc] else 0
                satisfaction += self.omega[0] * sT + self.omega[1] * sP + self.omega[2] * (self.providerL[p.provider]-self.provider[p.provider])/self.providerL[p.provider]\
                    + self.omega[3] * self.rep[p.provider] + self.omega[4] *  (self.providerReliability[p.provider] + self.providerEnergyCost[p.provider])/2
                p.up = self.omega[0] * sT + self.omega[1] * sP + self.omega[2] * (self.providerL[p.provider]-self.provider[p.provider])/self.providerL[p.provider]\
                    + self.omega[3] * self.rep[p.provider] + self.omega[4] *  (self.providerReliability[p.provider] + self.providerEnergyCost[p.provider])/2
                satisfaction += ((self.providerPrice[p.provider][p.loc]-p.cost)/self.providerPrice[p.provider][p.loc] + self.param[p.provider][0]+\
                    self.param[p.provider][1]+ self.param[p.provider][2]+self.param[p.provider][3])/5
                p.us = ((self.providerPrice[p.provider][p.loc]-p.cost)/self.providerPrice[p.provider][p.loc] + self.param[p.provider][0]+\
                    self.param[p.provider][1]+ self.param[p.provider][2]+self.param[p.provider][3])/5
        
        satisfaction = self.add_punishment_to_objv(satisfaction, x)
        
        satisfactionS = satisfaction
        return satisfactionS, (sat1, satisfaction-sat1)

    def compute_step_reward(self, task_idx, prov_idx):
        """
        xtmqn-style real-time step reward based on task/provider choice.
        """
        # Budget saving ratio
        r_cost = (self.budgets[task_idx] - self.providerPrice[prov_idx][task_idx]) / self.budgets[task_idx]

        # Time satisfaction reward
        if self.pathe[task_idx] <= self.deadlines[task_idx]:
            r_time = 1.0
        else:
            r_time = -1.0

        # Resource utilization reward
        r_resource = (self.providerL[prov_idx] - self.provider[prov_idx]) / self.providerL[prov_idx]

        # Reputation reward
        r_reputation = self.rep[prov_idx]

        # Reliability and energy cost reward
        r_reliability = self.providerReliability[prov_idx]
        r_energy = self.providerEnergyCost[prov_idx]

        reward = r_cost + r_time + r_resource + r_reputation + r_reliability + r_energy
        return torch.tensor(reward, dtype=torch.float32, device=self.device)

    def add_punishment_to_objv(self, objv, x):
        x = torch.tensor(x).to(self.device)
        cost = 0
        minpV = float("inf")
        minpP = None
        minsV = float("inf")
        minsP = None
        for p in self.points:
            if p.finished:
                if getattr(p, 'up', float('inf')) < minpV:
                    minpV = p.up
                    minpP = p
                if getattr(p, 'us', float('inf')) < minsV:
                    minsV = p.us
                    minsP = p
        if minpP == minsP:
            objv -= self.M * (minpV + minsV)/2
        for p in self.points:
            if not p.finished:
                continue
            if self.critical_path > self.deadline:
                objv -= self.M * (self.critical_path - self.deadline) / max(1, len(self.points))
            cost += self.providerPrice[p.provider][p.loc]
        return objv

    def monteCarloRewardList(self, objv, len, epi = 0.9):
        '''
        objv: the objective value of the network
        len: the length of the reward list
        '''
        rewardList = []
        base = objv.item()/(sum([epi**i for i in range(len)])+1e-10)
        for i in range(len):
            rewardList.append((base * epi**i+self.temp[i][4][0].item()*(1-epi**i)).item())
        return rewardList
    
    def training_step(self, startpoint, state=0):
        """
        Roll out one trajectory with exploration, compute xtmqn-style real-time rewards
        at each step, push to PPO buffer, then run one PPO update.
        """
        self.temp = []
        epislon = np.exp(-state/1000000)

        # reset network state
        for p in self.points:
            p.finished = False
            p.provider = -1
            p.sense = "And"
            p.time = None
        self.provider = torch.tensor([0 for _ in range(self.provider_num)]).to(self.device)
        self.paths = [0 for _ in range(self.task_num)]
        self.pathe = [0 for _ in range(self.task_num)]
        self.providerW = [0 for _ in range(self.provider_num)]
        self.critical_path = 0
        self.model.hidden_state = [torch.zeros(self.output_dim).unsqueeze(0).to(self.device)]
        self.model.cell_state = [torch.zeros(self.output_dim).unsqueeze(0).to(self.device)]

        # -------- rollout with train_proceed (adds OU exploration) --------
        points = [startpoint]
        x_list = [torch.zeros(self.task_num, self.provider_num).to(self.device)]

        while points:
            p = points.pop(0)
            if not p.children:
                continue

            provider, task, pts, sense, choose = self.train_proceed(x_list[p.L], p)
            if provider is None:
                continue

            # For each chosen (task, provider), compute immediate reward and push to buffer
            k = len(provider)
            for idx, (pr, t, pt) in enumerate(zip(provider, task, pts)):
                reward = self.compute_step_reward(t, pr)
                # the corresponding (X,pos,prehc,y,v) is the last k entries appended by train_proceed
                X, pos, prehc, y, v = self.temp[-k + idx]
                self.model.add_expL(X, pos, prehc, y, v, reward, self.available_P(pt))

            # advance environment/state
            cur_x = x_list[p.L]
            temp = cur_x.clone()
            for pr, t, pt in zip(provider, task, pts):
                if int(pr) == int(self.provider_num):
                    pt.finished = False
                else:
                    pt.finished = True
                    temp[t, pr] = 1
                    points.append(pt)
                    pt.L = len(x_list)
                    pt.sense = "And"
                    pt.provider = pr
                    x_list.append(temp.clone())

        # compute final objective once (not used as reward now, but useful for logging)
        if not hasattr(self, 'beginning'):
            startpoint.finished = True
            startpoint.sense = "Or"
        _objv, _ = self.objv(x_list[-1])

        # -------- one PPO update over accumulated buffer --------
        loss = self.model.PPOtrain()
        return loss, torch.tensor(0.0)
        
    
    def add_restrition(self, new_constrainf):
        self.constrains.append(new_constrainf)
        return

    def add_punishment_to_objv(self, objv, x):
        x = torch.tensor(x).to(self.device)
        # if sum(self.provider > self.providerL):
        #     print("Provider Resource Exceeded;", end=" ")
        #     objv -= self.M * torch.sum(self.provider - self.providerL)
        cost = 0
        for p in self.points:
            # if p.ability > self.providerAbility[p.provider]:
            #     objv -= self.M * (p.ability-self.providerAbility[p.provider])/len(self.points)
            if not p.finished:
                continue
            if p.budget < self.providerPrice[p.provider][p.loc] and self.paths[p.loc] > self.deadline:
                print("Budget Exceeded;", end=" ")
                objv -= self.M/len(self.points)
            if self.critical_path > self.deadline:
                print("Deadline Exceeded;", end=" ")
                objv -= self.M * (self.critical_path-self.deadline)/len(self.points)
            cost += self.providerPrice[p.provider][p.loc]
            if p.children == []:
                continue
        print("-"*20)
        return objv
    
    def add_edge(self, parent, child, edge):
        parent.children.append(child)
        child.parent.append(parent)
        return parent, child
    
    def return_children(self, parent):
        return parent.children, parent.children_edge
    
    def return_parent(self, child):
        return child.parent, child.parent_edge
    
    def init_points(self):
        for p in self.points:
            p.finished = False
            p.provider = -1
            p.sense = "And"
            p.time = None
        return

    def init_net(self):
        self.init_points()
        self.provider = torch.tensor([0 for i in range(self.provider_num)]).to(self.device)
        self.paths = [0 for i in range(self.task_num)]
        self.pathe = [0 for i in range(self.task_num)]
        self.providerW = [0 for i in range(self.provider_num)]
        self.critical_path = 0
        self.model.hidden_state = [torch.zeros(self.output_dim).unsqueeze(0).to(self.device)]
        self.model.cell_state = [torch.zeros(self.output_dim).unsqueeze(0).to(self.device)]
        
        return

device = "cuda"
datafile = "C:/Users/Kaka/PycharmProjects/dsm-main (1)/30_15/data_30_15_174.json"
def process_bar(i, epoch, loss, taskN, objv):
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("|"*int(i/epoch*100)+" "*int((epoch-i)/epoch*100)+"\r{}/{} loss:{}".format(i, epoch, loss/taskN))

if __name__ == '__main__':
    torch.manual_seed(4244)
    np.random.seed(4244)
    # torch.autograd.set_detect_anomaly(True)
    import numpy as np
    import json

        