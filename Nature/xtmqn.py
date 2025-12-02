import torch
from torch import nn
import sympy as sp
import numpy as np
from copy import deepcopy
from attNet import netAttention
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, task_num, provider_num, net,device="cuda", D=None): 
        '''
            input_dim = provider_num
            output_dim = task_num*provider_num
        '''
        super().__init__()

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
            )
            self.taskL.append(new_net)
        
        self.criteria = nn.MSELoss().to(device)
        self.task_num = task_num
        self.provider_num = provider_num
        
        self.D = D.unsqueeze(0).to(device)
        self.DT = D.T.unsqueeze(0).to(device)
        
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

        
    def attention(self, x):
        """
        x: (B, task_num, provider_num) or (task_num, provider_num)
        Returns: (B, task_num, provider_num) or (task_num, provider_num)
        """
        is_batch = True
        if x.dim() == 2:
            x = x.unsqueeze(0)
            is_batch = False
        B = x.shape[0]
        for ql, kl, vl in zip(self.qlinear, self.klinear, self.vlinear):
            # q, k, v: (B, task_num, provider_num)
            q = ql(x)
            k = kl(x)
            v = vl(x)
            # k: (B, task_num, provider_num) -> (B, provider_num, task_num)
            k_t = k.transpose(1, 2)
            # q: (B, task_num, provider_num)
            # k_t: (B, provider_num, task_num)
            # Attention score: (B, task_num, provider_num) @ (B, provider_num, task_num) -> (B, task_num, task_num)
            attn_scores = torch.bmm(q, k_t) / np.sqrt(k.shape[-1])
            # D: (task_num, task_num), expand to (B, task_num, task_num)
            D_exp = self.D.expand(B, -1, -1).to(x.device)
            attn_scores = torch.bmm(attn_scores, D_exp)
            attn_probs = F.softmax(attn_scores, dim=-1)
            # v: (B, task_num, provider_num)
            # attn_probs: (B, task_num, task_num)
            x = torch.bmm(attn_probs, v)
        if not is_batch:
            x = x.squeeze(0)
        return x
    
    def Tattention(self, x):
        """
        x: (B, task_num, provider_num) or (task_num, provider_num)
        Returns: (B, task_num, provider_num) or (task_num, provider_num)
        """
        is_batch = True
        if x.dim() == 2:
            x = x.unsqueeze(0)
            is_batch = False
        # Transpose to (B, provider_num, task_num)
        x = x.transpose(1, 2)
        for ql, kl, vl in zip(self.Tqlinear, self.Tklinear, self.Tvlinear):
            # q, k, v: (B, provider_num, task_num)
            q = ql(x)
            k = kl(x)
            v = vl(x)
            # k: (B, provider_num, task_num) -> (B, task_num, provider_num)
            k_t = k.transpose(1, 2)
            # q: (B, provider_num, task_num)
            # k_t: (B, task_num, provider_num)
            # Attention score: (B, provider_num, task_num) @ (B, task_num, provider_num) -> (B, provider_num, provider_num)
            attn_scores = torch.bmm(q, k_t) / np.sqrt(k.shape[-1])
            attn_probs = F.softmax(attn_scores, dim=-1)
            # v: (B, provider_num, task_num)
            # attn_probs: (B, provider_num, provider_num)
            x = torch.bmm(attn_probs, v)
        # Transpose back to (B, task_num, provider_num)
        x = x.transpose(1, 2)
        if not is_batch:
            x = x.squeeze(0)
        return x
    
    def forward(self,x,pos,p,prehc, AP):
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
        ansV = torch.zeros((1,(self.provider_num+1)*self.task_num)).to(x.device)
        for ap in AP:
            ans = self.taskL[ap](torch.cat([x,out],dim=1))
            ansV[0][ap*(self.provider_num+1):(ap+1)*(self.provider_num+1)] = ans[0]
        x = ansV
        
        return x , hcV, self.hidden_state[-1], self.cell_state[-1]
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
        
        x = self.decision(torch.cat([x_t,x_p,pos[0]],dim=1))
        
        hiddenstate = hiddenstate.squeeze(0)
        cell_state = cellstate.squeeze(0)
        
        remember = self.forgetDoor(torch.cat([x,hiddenstate],dim=1))
        input1 = self.inputDoor1(torch.cat([x,hiddenstate],dim=1))
        input2 = self.inputDoor2(torch.cat([x,hiddenstate],dim=1))
        output = self.outputDoor(torch.cat([x,hiddenstate],dim=1))
        
        cell_state = (remember * cellstate + input1 * input2).clone().detach()
        out = output * torch.tanh(cell_state).clone().detach()
        
        ansV = torch.zeros((1,(self.provider_num+1)*self.task_num)).to(x.device)
        for ap in AP:
            ans = self.taskL[ap](torch.cat([x,out[0]],dim=1))
            ansV[0][ap*(self.provider_num+1):(ap+1)*(self.provider_num+1)] = ans[0]
        
        return ansV, out, cell_state

    def batch_forward_based_on_given_hc(self, x, pos, hiddenstate, cellstate, APs):
        """
        Batch version of forward_based_on_given_hc.
        x: (B, ...) input features
        pos: (B, ...) position tensor
        hiddenstate: (B, output_dim)
        cellstate: (B, output_dim)
        APs: list of lists, len(APs)=B, each APs[i] is a list of ap indices for sample i
        Returns:
            ansV: (B, (provider_num+1)*task_num)
            out: (B, output_dim)
            cellstate_new: (B, output_dim)
        """
        B = x.size(0)
        x = self.attention(x)
        x_t = self.derograteT(x.flatten(1))
        x = self.Tattention(x)
        x = x.flatten(1)
        x = self.linear(x)
        x_p = self.degorate(x)
        # x_t: (B, task_num), x_p: (B, provider_num), pos: (B, ?)
        x = self.decision(torch.cat([x_t, x_p, pos], dim=1))
        remember = self.forgetDoor(torch.cat([x, hiddenstate], dim=1))
        input1 = self.inputDoor1(torch.cat([x, hiddenstate], dim=1))
        input2 = self.inputDoor2(torch.cat([x, hiddenstate], dim=1))
        output = self.outputDoor(torch.cat([x, hiddenstate], dim=1))
        cellstate_new = (remember * cellstate + input1 * input2).detach()
        out = (output * torch.tanh(cellstate_new)).detach()
        ansV = torch.zeros((B, (self.provider_num+1)*self.task_num), device=x.device)
        for i in range(B):
            for ap in APs[i]:
                ans = self.taskL[ap](torch.cat([x[i].unsqueeze(0), out[i].unsqueeze(0)], dim=1))
                ansV[i, ap*(self.provider_num+1):(ap+1)*(self.provider_num+1)] = ans[0]
        return ansV, out, cellstate_new
    
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
        self.name = "xtmDQN"
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
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
        self.exp = []
        self.tempexp = []
        
        self.M = 100000

        # Experience replay buffer for RL
        self.replay_buffer = []
        self.max_buffer_size = 50000

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
        AP = self.available_P(p)
        y, hc, h, c = self.model.forward(x, point, p, prehc, AP)
        return y, hc, h, c, AP
    def _inner_step(self, x, point, use_random=False):
        """
        use_random=True → 训练时随机选择
        use_random=False → 用网络输出贪心
        """
        if point.loc != -1:
            pos = torch.zeros(self.task_num).unsqueeze(0).to(self.device)
        else:
            pos = torch.zeros(self.task_num).unsqueeze(0).to(self.device)

        y, hcV, h, cell, AP = self.forward(x, pos, point, point.hc)
        provider, task, points, choose = [], [], [], []

        if self.andor[point.loc] == "or":
            avaT = [c.loc for c in point.children]
            t = np.random.choice(avaT) if use_random else avaT[np.argmax(
                [y[0][c.loc * (self.provider_num + 1): (c.loc + 1) * (self.provider_num + 1)].max().item()
                for c in point.children])]
            pr = np.random.choice(self.provider_num) if use_random else \
                y[0][t * (self.provider_num + 1): (t + 1) * (self.provider_num + 1)][:self.provider_num].argmax().item()

            provider.append(pr)
            task.append(t)
            pt = self.points[t]
            points.append(pt)
            pt.finished = True
            pt.provider = pr
            pt.sense = "Or"
            pt.hc = hcV
            choose.append(y[0][t * (self.provider_num + 1) + pr])
        else:
            for c in point.children:
                if c.finished:
                    continue
                pr = np.random.choice(self.provider_num) if use_random else \
                    y[0][c.loc * (self.provider_num + 1): (c.loc + 1) * (self.provider_num + 1)][:self.provider_num].argmax().item()
                provider.append(pr)
                task.append(c.loc)
                c.finished = True
                c.provider = pr
                c.sense = "And"
                c.hc = hcV
                points.append(c)
                choose.append(y[0][c.loc * (self.provider_num + 1) + pr])

        return provider, task, points, "And" if self.andor[point.loc] == "and" else "Or", choose
    def proceed(self, x, point):
        provider, task, points, sense, choose = self._inner_step(x, point, use_random=False)
        return provider, task, points, sense, choose

    def train_proceed(self, x, point, epsilon=0.3):
        provider, task, points, sense, choose = self._inner_step(x, point, use_random=True)
        return provider, task, points, sense, choose


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
        self.tempexp = []
        return calcuated, x_list

    def search_train(self, startpoint, epislon=0.3): # search the whole network, choose with train_proceed()
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
            if np.random.rand() < epislon:
                provider, task, point, sense, choose = self.train_proceed(x_list[p.L], p, epislon=epislon)
            else:
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
        Based on the task and provider index, compute the reward for the step
        """
        # Budget saving ratio
        r_cost = (self.budgets[task_idx] -
                  self.providerPrice[prov_idx][task_idx]) / self.budgets[task_idx]

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
        
        reward = (r_cost + r_time + r_resource + r_reputation + r_reliability + r_energy)/6
        return torch.tensor(reward, dtype=torch.float32, device=self.device)
    
    def training_step(self, startpoint, state=0, epoches=100):
        """
        Each rollout a complete trajectory, then do experience replay training
        """
        gamma = 0.9
        batch_size = 512
        self.init_net()                       # Clear the state
        trajectory = []                     # Save (x,pos,h,c,AP,action,reward)

        # ---------- 1. rollout a trajectory ----------
        points = [startpoint]
        x_list = [torch.zeros(self.task_num, self.provider_num).to(self.device)]
        losslist = []
        l1 = len(self.replay_buffer)
        while points:
            p = points.pop(0)
            if not p.children:
                continue

            # 1.1 Forward
            x = x_list[p.L]
            pos = torch.zeros(self.task_num).unsqueeze(0).to(self.device)
            y, hcV, h, cell, AP = self.forward(x, pos, p, p.hc)

            # 1.2 Choose action (use ε-greedy or proceed)
            if torch.rand(1).item() < 0.9:           # ε-greedy can be changed
                provider, task, pts, sense, choose = self.proceed(x, p)
            else:
                provider, task, pts, sense, choose = self.train_proceed(x, p, 0.5)

            if provider is None:
                continue

            # 1.3 Store transitions in replay buffer
            for pr, t, pt in zip(provider, task, pts):
                reward = self.compute_step_reward(t, pr)
                action = t * (self.provider_num + 1) + pr
                reward = reward
                transition = (
                    x.clone().detach(),
                    pos.clone().detach(),
                    h.clone().detach(),
                    cell.clone().detach(),
                    AP,
                    action,
                    reward.item()
                )
                self.replay_buffer.append(transition)
                # Maintain buffer size
                if len(self.replay_buffer) > self.max_buffer_size:
                    self.replay_buffer.pop(0)

            # Update the state
            temp = x.clone()
            temp[t, pr] = 1
            x_list.append(temp)
            pt.L = len(x_list) - 1
            points.append(pt)
        objv, objv1 = self.objv(x_list[-1])
        self.replay_buffer = self.replay_buffer[-batch_size:]
        # Use all buffer as batch
        xs = []
        poss = []
        hs = []
        cs = []
        APs = []
        actions = []
        rewards = []
        sig = 0.9
        for i in range(l1, len(self.replay_buffer)):
            reward = self.replay_buffer[i][-1]
            reward = reward*sig + objv1[0]*(1-sig)
            sig = 0.8*sig
            self.replay_buffer[i] = (self.replay_buffer[i][0], self.replay_buffer[i][1], self.replay_buffer[i][2], self.replay_buffer[i][3], self.replay_buffer[i][4], self.replay_buffer[i][5], reward)
        for x, pos, h, c, AP, action, reward in self.replay_buffer:
            xs.append(x.unsqueeze(0))
            poss.append(pos)
            hs.append(h)
            cs.append(c)
            APs.append(AP)
            actions.append(action)
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        xs = torch.cat(xs, dim=0)
        poss = torch.cat(poss, dim=0)
        hs = torch.cat(hs, dim=0)
        cs = torch.cat(cs, dim=0)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        loss_sum = 0
        
        for epoch in range(epoches):
            y_hat, _, _ = self.model.batch_forward_based_on_given_hc(xs, poss, hs, cs, APs)
            
            q_values = y_hat.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            rewards_tensor = torch.stack(rewards, dim=0)
            loss = self.model.criteria(q_values, rewards_tensor)
            loss.backward()
            self.model.optimizer.step()
            loss_sum += loss
            losslist.append(loss.item())
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss {loss.item():.4f}")

        return loss_sum, losslist
        
    
    def add_restrition(self, new_constrainf):
        self.constrains.append(new_constrainf)
        return

    def add_punishment_to_objv(self, objv, x):
        x = torch.tensor(x).to(self.device)
        # if sum(self.provider > self.providerL):
        #     print("Provider Resource Exceeded;", end=" ")
        #     objv -= self.M * torch.sum(self.provider - self.providerL)
        cost = 0
        minpV = float("inf")
        minpP = None
        minsV = float("inf")
        minsP = None
        for p in self.points:
            if p.finished:
                if p.up < minpV:
                    minpV = p.up
                    minpP = p
                if p.us < minsV:
                    minsV = p.us
                    minsP = p
        if minpP == minsP and minpP != None:
            objv -= self.M * (minpV + minsV)/2
        for p in self.points:
            # if p.ability > self.providerAbility[p.provider]:
            #     objv -= self.M * (p.ability-self.providerAbility[p.provider])/len(self.points)
            if not p.finished:
                continue
            if self.critical_path > self.deadline:
                print("Deadline Exceeded;", end=" ")
                objv -= self.M * (self.critical_path-self.deadline)/len(self.points)
            cost += self.providerPrice[p.provider][p.loc]
            if p.children == []:
                continue
            # if self.andor[p.loc] == "and":
            #     num = 0
            #     for c in p.children:
            #         if c.finished:
            #             num += 1
            #     if num < len(p.children):
            #         print("And Condition Not Satisfied;", end=" ")
            #         objv -= self.M
            # else:
            #     num = 0
            #     for c in p.children:
            #         if c.finished:
            #             num += 1
            #     if num != 1:
            #         print("Or Condition Not Satisfied;", end=" ")
            #         objv -= self.M
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

        