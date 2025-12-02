import torch
from torch import nn
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

class BasicDQN(nn.Module):
    def __init__(self, input_dim, output_dim, task_num, provider_num, device="cpu"):
        """
        基础DQN网络 - 使用简单的MLP
        input_dim: 输入维度 (provider_num)
        output_dim: 输出维度 (task_num * (provider_num + 1))
        """
        super().__init__()
        
        self.task_num = task_num
        self.provider_num = provider_num
        self.device = device
        
        # 计算flattened输入维度
        # 状态矩阵: task_num * provider_num
        # 位置编码: task_num
        # 提供者状态: provider_num
        self.flattened_input_dim = task_num * provider_num + task_num + provider_num
        
        # 简单的MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_input_dim, task_num*provider_num*2),
            nn.ReLU(),
            nn.Linear(task_num*provider_num*2, task_num*provider_num*2),
            nn.ReLU(),
            nn.Linear(task_num*provider_num*2, task_num*provider_num*2),
            nn.ReLU(),
            nn.Linear(task_num*provider_num*2, task_num*provider_num*2),
            nn.ReLU(),
            nn.Linear(task_num*provider_num*2, task_num*provider_num*2),
            nn.ReLU(),
            nn.Linear(task_num*provider_num*2, task_num*provider_num*2),
            nn.ReLU(),
            nn.Linear(task_num*provider_num*2, task_num*provider_num*2),
            nn.ReLU(),
            nn.Linear(task_num*provider_num*2, output_dim)
        ).to(device)
        
        # 损失函数
        self.criteria = nn.MSELoss().to(device)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, pos, provider_state):
        """
        前向传播
        x: 任务-提供者匹配状态矩阵 [task_num, provider_num]
        pos: 当前位置编码 [task_num]
        provider_state: 提供者状态 [provider_num]
        """
        # 将所有输入flatten并拼接
        x_flat = x.flatten()  # [task_num * provider_num]
        pos_flat = pos.flatten()  # [task_num]
        provider_flat = provider_state.flatten()  # [provider_num]
        
        # 拼接所有输入
        input_vector = torch.cat([x_flat, pos_flat, provider_flat], dim=0)
        input_vector = input_vector.unsqueeze(0)  # 添加batch维度
        
        # 通过MLP
        q_values = self.mlp(input_vector)
        
        return q_values
    
    def choose_action(self, q_values, available_tasks, epsilon=0.1):
        """
        选择动作 (epsilon-greedy策略)
        q_values: Q值 [1, output_dim]
        available_tasks: 可用任务列表
        epsilon: 探索率
        """
        if np.random.random() < epsilon:
            # 随机选择
            if len(available_tasks) == 0:
                return None, None
            task = np.random.choice(available_tasks)
            provider = np.random.randint(0, self.provider_num + 1)
        else:
            # 贪婪选择
            best_q = -float('inf')
            best_task = None
            best_provider = None
            
            for task in available_tasks:
                for provider in range(self.provider_num + 1):
                    idx = task * (self.provider_num + 1) + provider
                    if q_values[0, idx] > best_q:
                        best_q = q_values[0, idx]
                        best_task = task
                        best_provider = provider
        
        return best_task, best_provider


class point:
    def __init__(self):
        self.parent = []
        self.children = []
        self.sense = "And"  # AND, OR sense
        self.loc = 0
        self.provider = 0
        self.task = 0
        self.finished = False
        self.time = None
        self.L = 0  # 层级


class network:
    def __init__(self, deadlines, budgets, Rs, abilities, cost,
                 input_dim, output_dim, provider_num, task_num, edges, device="cpu"):
        self.name = "basicDQN"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.points = []
        
        # 初始化任务点
        for j in range(task_num):
            p = self.add_point(T=10)
            p.deadline = deadlines[j]
            p.budget = budgets[j]
            p.R = Rs[j]
            p.ability = abilities[j]
            p.cost = cost[j]
        
        # 建立依赖关系
        self.edges = edges
        for e in edges:
            self.add_edge(self.points[e[0]], self.points[e[1]], 1)
        
        # 初始化网络
        self.model = BasicDQN(input_dim, output_dim, task_num, provider_num, device).to(device)
        self.provider_num = provider_num
        self.task_num = task_num
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # 其他参数
        self.critical_path = -1
        self.paths = [0 for i in range(task_num)]
        self.lam = 0.1
        self.lam = torch.tensor(self.lam).to(device)
        self.omega = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.provider = torch.tensor([0 for i in range(provider_num)]).to(device)
        self.providerL = []
        self.alpha = 0.1
        self.alphaC = 0.1
        self.alpha = torch.tensor(self.alpha).to(device)
        self.alphaC = torch.tensor(self.alphaC).to(device)
        self.providerW = torch.tensor([0 for i in range(provider_num)]).to(device)
        self.pathe = [0 for i in range(task_num)]
        self.M = 1000000
    
    def add_point(self, T):
        p = point()
        self.points.append(p)
        p.T = T
        p.loc = len(self.points) - 1
        return p
    
    def set_beginning(self, point):
        self.beginning = point
        return point
    
    def available_tasks(self, current_point):
        """获取可执行的任务"""
        available = []
        for c in current_point.children:
            if not c.finished:
                # 检查前置任务是否完成
                ready = True
                for parent in c.parent:
                    if not parent.finished and parent.loc != -1:
                        ready = False
                        break
                if ready:
                    available.append(c.loc)
        return available
    
    def proceed(self, x, point, epsilon=0.1):
        """
        执行一步决策
        x: 当前状态矩阵
        point: 当前点
        epsilon: 探索率
        """
        # 准备输入
        pos = torch.zeros(self.task_num).to(self.device)
        if point.loc != -1:
            pos[point.loc] = 1
        
        provider_state = self.provider.float()
        
        # 前向传播
        q_values = self.model.forward(x, pos, provider_state)
        
        # 获取可用任务
        available = self.available_tasks(point)
        if not available:
            return None, None, None, None, None
        
        # 选择动作
        selected_providers = []
        selected_tasks = []
        selected_points = []
        selected_q_values = []
        
        for child in point.children:
            if child.finished:
                continue
                
            # 检查前置条件
            ready = True
            for parent in child.parent:
                if not parent.finished and parent.loc != -1:
                    ready = False
                    break
            if not ready:
                continue
            
            # 为每个可执行任务选择提供者
            best_q = -float('inf')
            best_provider = None
            
            for provider in range(self.provider_num + 1):
                idx = child.loc * (self.provider_num + 1) + provider
                if q_values[0, idx] > best_q:
                    best_q = q_values[0, idx]
                    best_provider = provider
            
            if best_provider is not None:
                if int(best_provider) < int(self.provider_num):
                    self.provider[best_provider] += int(child.R)
                    self.providerW[best_provider] += 1
                    child.finished = True
                else:
                    child.finished = False
                
                child.provider = best_provider
                child.sense = "And"
                
                selected_providers.append(best_provider)
                selected_tasks.append(child.loc)
                selected_points.append(child)
                selected_q_values.append(best_q)
        
        if not selected_providers:
            return None, None, None, None, None
        
        return selected_providers, selected_tasks, selected_points, "And", selected_q_values
    
    def search(self, startpoint):
        """搜索整个网络"""
        points = [startpoint]
        calculated = []
        x_list = []
        x = torch.zeros(self.task_num, self.provider_num).to(self.device)
        x_list.append(x)
        
        while len(points) > 0:
            p = points.pop(0)
            if p.children == []:
                continue
            
            provider, task, point, sense, q_values = self.proceed(x_list[p.L], p)
            if provider is None:
                continue
            
            temp = torch.clone(x_list[p.L])
            for pr, t, po, q_val in zip(provider, task, point, q_values):
                if int(pr) == int(self.provider_num):
                    po.finished = False
                else:
                    po.finished = True
                    temp[t, pr] = 1
                    points.append(po)
                    po.L = len(x_list)
                    x_list.append(temp.clone())
                    calculated.append(q_val)
        
        return calculated, x_list
    
    def criticalpath(self, p, time):
        """计算关键路径"""
        pT = 0
        if not p.finished:
            self.critical_path = max(self.critical_path, time)
            return
        
        if p.loc != -1:
            if self.paths[p.loc] > time:
                self.paths[p.loc] = time
            pT = self.taskTime[p.provider][p.loc]
            if self.pathe[p.loc] < time + pT:
                self.pathe[p.loc] = time + pT
        else:
            pT = 0
        
        for c in p.children:
            if c.finished:
                self.criticalpath(c, time + pT)
        
        if len(p.children) == 0:
            if self.critical_path < time + pT:
                self.critical_path = time + pT
        return
    
    def objv(self, x):
        """计算目标值"""
        # 计算关键路径长度
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
            satisfaction += torch.exp(torch.tensor(-self.lam * (self.deadline - self.critical_path))).to(self.device)
        satisfaction /= 2
        
        sat1 = satisfaction.clone()
        
        for p in self.points:
            if p.finished:
                sT = 0
                if self.deadlines[p.loc] > self.pathe[p.loc]:
                    sT = self.deadlines[p.loc] / self.pathe[p.loc]
                else:
                    sT = 1
                sP = self.budgets[p.loc] / self.providerPrice[p.provider][p.loc] if p.budget > self.providerPrice[p.provider][p.loc] else 0
                satisfaction += self.omega[0] * sT + self.omega[1] * sP + self.omega[2] * (self.providerL[p.provider] - self.provider[p.provider]) / self.providerL[p.provider] \
                              + self.omega[3] * self.rep[p.provider] + self.omega[4] * (self.providerReliability[p.provider] + self.providerEnergyCost[p.provider]) / 2
                p.up = self.omega[0] * sT + self.omega[1] * sP + self.omega[2] * (self.providerL[p.provider] - self.provider[p.provider]) / self.providerL[p.provider] \
                              + self.omega[3] * self.rep[p.provider] + self.omega[4] * (self.providerReliability[p.provider] + self.providerEnergyCost[p.provider]) / 2
                satisfaction += ((self.providerPrice[p.provider][p.loc] - p.cost) / self.providerPrice[p.provider][p.loc] + self.param[p.provider][0] +
                                self.param[p.provider][1] + self.param[p.provider][2] + self.param[p.provider][3]) / 5
                p.us = ((self.providerPrice[p.provider][p.loc] - p.cost) / self.providerPrice[p.provider][p.loc] + self.param[p.provider][0] +
                                self.param[p.provider][1] + self.param[p.provider][2] + self.param[p.provider][3]) / 5
        
        satisfaction = self.add_punishment_to_objv(satisfaction, x)
        
        satisfactionS = satisfaction
        return satisfactionS, (sat1, satisfaction - sat1)
    
    def training_step(self, startpoint, state=0):
        """训练步骤"""
        for p in self.points:
            p.finished = False
            p.provider = -1
            p.sense = "And"
            p.time = None
        
        calculated, x = self.search(startpoint)
        objv, obj = self.objv(x[-1])
        
        startpoint.finished = True
        startpoint.sense = "Or"
        loss_sum = 0
        
        self.optimizer.zero_grad()
        for q_val in calculated:
            target = objv.detach()
            loss = self.model.criteria(q_val, target)
            loss_sum += loss
        
        if loss_sum == 0:
            return torch.tensor(0.0).to(self.device), obj
        
        loss_sum.backward(retain_graph=True)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss_sum, obj
    
    def add_punishment_to_objv(self, objv, x):
        """添加约束惩罚"""
        cost = 0
        for p in self.points:
            if not p.finished:
                continue
            if p.budget < self.providerPrice[p.provider][p.loc] and self.paths[p.loc] > self.deadline:
                print("Budget Exceeded;", end=" ")
                objv -= self.M / len(self.points)
            if self.critical_path > self.deadline:
                print("Deadline Exceeded;", end=" ")
                objv -= self.M * (self.critical_path - self.deadline) / len(self.points)
            cost += self.providerPrice[p.provider][p.loc]
        print("-" * 20)
        return objv
    
    def add_edge(self, parent, child, edge):
        parent.children.append(child)
        child.parent.append(parent)
        return parent, child
    
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
        return


if __name__ == '__main__':
    # 测试代码
    torch.manual_seed(4244)
    np.random.seed(4244)
    
    print("Basic DQN implementation ready!")
    print("Network architecture: Simple MLP with flattened inputs")
    print("Features:")
    print("- Flattened input: task_matrix + position_encoding + provider_state")
    print("- Simple MLP: Input -> 512 -> 512 -> 256 -> 256 -> Output")
    print("- Epsilon-greedy action selection")
    print("- MSE loss for Q-value updates")
