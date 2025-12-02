import pickle
from PPO import network, point # ppo method
import numpy as np
t = 5
p = 3
typ = "ppo"

pointdir = f"points_{t}_{p}_{typ}.pkl"
datadir = f"data_{t}_{p}.json"
datadir = "5_3/data_5_3_0.json"
device = "mps"

points = pickle.load(open(pointdir, "rb"))

import torch
import json

class network:
    def __init__(self, deadlines, budgets, Rs, abilities, cost,\
        input_dim, output_dim, provider_num, task_num, edges, device="cpu"):
        self.name = "xtmPPO"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.points = []
        self.D = torch.zeros((task_num, task_num)).to(device) # Dependency matrix
        self.edges = edges
        self.provider_num = provider_num
        self.task_num = task_num
        self.constrains = []
        self.device = device
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
    def criticalpath(self, p, time): # calculate the critical path
        pT = 0
        p.cobjv = True
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
            global typ
            if (typ == "ppo" or p.cobjv)and p.finished:
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

if __name__ == "__main__":
    torch.manual_seed(4244)
    np.random.seed(4244)
    # torch.autograd.set_detect_anomaly(True)
    import numpy as np
    import json
    det = json.load(open(datadir,"r"))
    taskN = det["taskNum"]
    providerN = det["providerNum"]
    edges = det["edges"]
    
    deadlines = np.array(det["taskdeadlines"])
    budgets = np.array(det["taskbudgets"])
    Rs = np.array(det["taskResources"])
    abilities = np.array(det["taskabilities"])
    providerRep = np.array(det["providerRep"])
    cost = np.array(det["taskCost"])

    epoch = 1000
    netsNum = 1
    # netsNum = 0
    cnt = 10
    # net.model.load_state_dict(torch.load("test.pth")[0])
    nets = []
    draw = {
    }
    bestSolution = {
    }
    net_new = network(deadlines, budgets, Rs, abilities, cost,\
                        providerN,(providerN+1)*taskN,providerN,taskN, edges,device=device)
    net_new.taskTime = det["taskTime"]
    net_new.rep = det["providerRep"]
    net_new.deadline = det["deadline"]
    net_new.deadlines = det["taskdeadlines"]
    net_new.providerAbility = det["providerAbility"]
    net_new.providerL = torch.tensor(det["providerL"]).to(device)
    net_new.providerPrice = det["providerPrice"]
    net_new.providerReliability = det["providerReliability"]
    net_new.providerEnergyCost = det["providerEnergyCost"]
    net_new.budget = det["budget"]
    net_new.param = det["providerParam"]
    net_new.andor = det["andor"]
    net_new.budgets = det["taskbudgets"]
    net_new.critical_path = -1
    net_new.paths = [0 for i in range(taskN)]
    # net_new.model.load_state_dict(torch.load("test.pth")[0])
    

    net_new.points = points
    for p in net_new.points:
        p.cobjv=False
    startp = point()
    startp.loc = -1
    startp.children = [net_new.points[0]]
    startp.finished = True
    startp.L = 0
    startp.hc = 0
    net_new.beginning = startp
    
    
    net_new.critical_path = -1
    net_new.criticalpath(startp, 0)
    
    print("Critical Path:", net_new.critical_path)
    
    objv, objs = net_new.objv([1 for i in range(taskN*providerN)])
    print("Objective Value:", objv.item())
    print("Objectives:", [o.item() for o in objs])
    
    if taskN == 5 and providerN == 3:
        net_new.points[0].provid = 2
        net_new.points[1].provid = 0
        net_new.points[2].provid = 0
        net_new.points[3].provid = 1
        net_new.points[4].provid = 1
        net_new.critical_path = -1
        net_new.criticalpath(startp, 0)
        
        print("Gurobi Answer:")
        print("Critical Path:", net_new.critical_path)
        
        objv, objs = net_new.objv([1 for i in range(taskN*providerN)])
        print("Objective Value:", objv.item())
        print("Objectives:", [o.item() for o in objs])