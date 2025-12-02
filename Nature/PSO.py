from generate_net import netw
from xtmqn import point
from torch import nn
import torch, json
import numpy as np
import random
from collections import defaultdict


def _summarize_matching(points_snapshot, task_time_matrix):
    """
    根据点信息统计任务-服务匹配与总耗时/成本。
    """
    if not points_snapshot or task_time_matrix is None:
        return [], 0.0, 0.0

    matches = []
    total_time = 0.0
    total_cost = 0.0
    provider_cnt = len(task_time_matrix)

    for p in points_snapshot:
        if getattr(p, "loc", -1) < 0 or not getattr(p, "finished", False):
            continue
        provider = getattr(p, "provider", -1)
        if provider is None or provider < 0 or provider >= provider_cnt:
            continue
        task_idx = int(p.loc)
        provider_idx = int(provider)
        try:
            task_time = float(task_time_matrix[provider_idx][task_idx])
        except (IndexError, TypeError):
            continue
        task_cost = float(getattr(p, "cost", 0.0))
        matches.append({
            "task": task_idx,
            "provider": provider_idx,
            "taskTime": task_time,
            "taskCost": task_cost
        })
        total_time += task_time
        total_cost += task_cost

    return matches, total_time, total_cost


def _print_matching_summary(points_snapshot, task_time_matrix):
    matches, total_time, total_cost = _summarize_matching(points_snapshot, task_time_matrix)
    print("任务-服务匹配结果：")
    if not matches:
        print("  未找到有效匹配。")
        return
    for item in matches:
        print(f"  任务 {item['task']} -> 服务提供商 {item['provider']} | taskTime={item['taskTime']:.4f} | taskCost={item['taskCost']:.4f}")
    print(f"  总taskTime: {total_time:.4f}")
    print(f"  总taskCost: {total_cost:.4f}")


datafile = "data_60_30.json"
device = "cuda"
class network(netw):
    def __init__(self, deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, providerNum, taskNum, edges, device,\
                taskTime, rep, deadline, providerAbility,\
                providerL, providerPrice, providerReliability,\
                    providerEnergyCost, budget, param, omega, lam, andorInfo):
        super(network, self).__init__()
        self.device = device
        self.deadlines = deadlines
        self.budgets = budgets
        self.Rs = Rs
        self.abilities = abilities
        self.cost = cost
        self.providerNum = providerNum
        self.taskNum = taskNum
        self.edges = edges
        self.taskTime = taskTime
        self.rep = rep
        self.deadline = deadline
        self.providerAbility = providerAbility
        self.providerL = providerL
        self.providerPrice = providerPrice
        self.providerReliability = providerReliability
        self.providerEnergyCost = providerEnergyCost
        self.providerN = providerN
        self.taskN = taskN
        self.budget = budget
        self.param = param
        self.omega = omega
        self.lam = lam
        self.andor = andorInfo
        self.M = 1000000  # punishment factor
        self.points = []
        self.x = torch.zeros((self.taskNum, self.providerNum + 1)).to(self.device)
        self.pathe = [0 for i in range(taskNum)]
        # 初始化provider状态
        self.provider = np.zeros((self.providerNum,))
    
    def init_points(self):
        self.points = []
        for i in range(self.taskNum):
            p = point()
            p.loc = i
            p.children = []
            p.finished = False
            p.deadline = self.deadlines[i]
            p.budget = self.budgets[i]
            p.Rs = self.Rs[i]
            p.abilities = self.abilities[i]
            p.cost = self.cost[i]
            p.provider = -1
            p.providerL = 0
            p.providerPrice = 0
            p.cobjv = False
            self.points.append(p)
        for path in self.edges:
            if path[0] == -1:
                continue
            self.points[path[0]].children.append(self.points[path[1]])
        
        return
    def add_punishment_to_objv(self, objv):
        # for p, lp in zip(self.provider, self.providerL):
        #     if p > lp:
        #         objv -= self.M * (p - lp)/len(self.points)
        if objv == float('inf'):
            print("Objective value is infinity, applying punishment.")
        cost = 0
        for p in self.points:
            # if p.ability > self.providerAbility[p.provider]:
            #     objv -= self.M * (p.ability-self.providerAbility[p.provider])/len(self.points)
            if not p.finished:
                continue
            if p.budget < self.providerPrice[p.provider][p.loc] and self.paths[p.loc] > self.deadline:
                objv -= self.M/len(self.points)
            if self.critical_path > self.deadline:
                objv -= self.M * (self.critical_path-self.deadline)/len(self.points)
            cost += self.providerPrice[p.provider][p.loc]
            if p.children == []:
                continue
            if self.andor[p.loc] == "and":
                num = 0
                for c in p.children:
                    if c.finished:
                        num += 1
                if num < len(p.children):
                    # print("And Condition Not Satisfied;", end=" ")
                    objv -= self.M
            elif self.andor[p.loc] == "or":
                num = 0
                for c in p.children:
                    if c.finished:
                        num += 1
                if num != 1:
                    # print("Or Condition Not Satisfied;", end=" ")
                    objv -= self.M
        if objv == float('inf'):
            objv = -self.M*10
        return objv
    
    def objv(self): # calculate the objective value
        '''
        x: the final state of the network
        '''

        if not hasattr(self, 'provider'):
            self.provider = np.zeros((self.providerNum,))
        
        # 重新计算provider资源分配
        self.provider = np.zeros((self.providerNum,))
        for p in self.points:
            if p.finished and p.provider != -1:
                self.provider[p.provider] += self.Rs[p.loc]
        
        # calculate length of critical path
        self.critical_path = -1
        self.paths = [0 for i in range(self.taskNum)]
        self.pathe = [0 for i in range(self.taskNum)]
        self.critical_path = 0
        
        for p in self.points:
            p.cobjv = False
        self.criticalpath(self.beginning, 0)
            
        budget = self.budget
        cost = 0
        for p in self.points:
            if p.cobjv:
                cost += self.providerPrice[p.provider][p.loc]
        if cost == 0:
            return torch.tensor(-200000000).float(), torch.tensor([-10000000000, -10000000000]).float()  
        # self.cost = cost
        satisfaction = torch.tensor((budget - cost) / budget).to(self.device)
        if self.deadline > self.critical_path:
            satisfaction += torch.exp(torch.tensor(-self.lam * (self.deadline-self.critical_path))).to(self.device)
        satisfaction /= 2
        
        sat1 = satisfaction.clone()
        if sat1 == float('inf'):
            print("Objective value is infinity, applying punishment.")
        for p in self.points:
            if p.cobjv and p.loc != -1:
                sT = 0
                if self.deadlines[p.loc]>self.pathe[p.loc]:
                    if self.pathe[p.loc] == 0:
                        pass
                    sT = self.deadlines[p.loc]/(self.pathe[p.loc])
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
        satisfaction = self.add_punishment_to_objv(satisfaction)
        
        satisfactionS = torch.tensor(satisfaction)
        
        return satisfactionS, (sat1, satisfaction-sat1)
    
    def process(self):
        """Process the network to update points based on assignments"""
        self.provider = np.zeros(self.providerNum)
        for i in range(self.taskNum):
            p = self.points[i]
            p.provider = -1
            p.providerL = 0
            p.providerPrice = 0
            p.finished = False
            p.cobjv = False

            assigned_provider = -1
            if torch.sum(self.x[i]) > 0:
                assigned_provider = torch.argmax(self.x[i]).item()

            if assigned_provider != -1 and assigned_provider < self.providerNum:
                p.provider = assigned_provider
                self.provider[p.provider] += self.Rs[i]
                p.providerL = self.providerL[p.provider].item()
                p.providerPrice = self.providerPrice[p.provider].copy()
                p.finished = True
                p.cobjv = True

        # Update paths and critical path
        self.critical_path = -1
        self.paths = [0 for _ in range(self.taskNum)]
        if hasattr(self, 'beginning') and self.beginning is not None:
            self.criticalpath(self.beginning, 0)
        else:
            self.criticalpath(self.points[0], 0)

class PSO:
    def __init__(self, taskNum, providerNum, device, 
                 deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, edges,\
                 taskTime, rep, deadline, providerAbility,\
                 providerL, providerPrice, providerReliability,\
                 providerEnergyCost, budget, param, omega, lam, andorInfo,
                 population_size=100, w=0.5, c1=1.5, c2=1.5):
        self.taskNum = taskNum
        self.providerNum = providerNum
        self.device = device
        self.population_size = population_size
        self.w = w  # inertia
        self.c1 = c1  # cognitive
        self.c2 = c2  # social

        self.deadlines = deadlines
        self.budgets = budgets
        self.Rs = Rs
        self.abilities = abilities
        self.cost = cost
        self.providerN = providerN
        self.taskN = taskN
        self.edges = edges
        self.taskTime = taskTime
        self.rep = rep
        self.deadline = deadline
        self.providerAbility = providerAbility
        self.providerL = providerL
        self.providerPrice = providerPrice
        self.providerReliability = providerReliability
        self.providerEnergyCost = providerEnergyCost
        self.budget = budget
        self.param = param
        self.omega = omega
        self.lam = lam
        self.andorInfo = andorInfo

        self.particles = []
        self.velocities = []
        self.pBest = []
        self.pBest_scores = []
        self.gBest = None
        self.gBest_score = -float('inf')
        self.bestPoints = None
    
    def rebuild_points_from_assignment(self, assignment_matrix):
        """
        基于给定分配矩阵重建网络节点状态，便于统计taskTime/taskCost。
        """
        helper_net = self.create_individual()
        helper_net.x = assignment_matrix.clone().to(self.device)
        helper_net.process()
        return helper_net.points

    def create_individual(self):
        net = network(self.deadlines, self.budgets, self.Rs, self.abilities, self.cost,
                      self.providerN, self.taskN, self.providerNum, self.taskNum, self.edges, self.device,
                      self.taskTime, self.rep, self.deadline, self.providerAbility,
                      self.providerL, self.providerPrice, self.providerReliability,
                      self.providerEnergyCost, self.budget, self.param, self.omega, self.lam, self.andorInfo)
        net.init_points()
        net.beginning = point()
        net.beginning.loc = -1  # beginning的loc应该是-1
        net.beginning.children = [net.points[0]]
        net.beginning.finished = True
        net.beginning.provider = -1  # beginning不需要provider
        return net

    def initialize(self):
        self.particles = []
        self.velocities = []
        self.pBest = []
        self.pBest_scores = []
        
        attempts = 0
        max_attempts = self.population_size * 100

        while len(self.particles) < self.population_size and attempts < max_attempts:
            individual = self.create_individual()
            x = torch.zeros((self.taskNum, self.providerNum + 1), device=self.device)
            task_provider = torch.randint(0, self.providerNum + 1, (self.taskNum,)).to(self.device)
            for i, p in enumerate(task_provider):
                x[i, p] = 1
            
            v = torch.randn((self.taskNum, self.providerNum + 1), device=self.device) * 0.1

            individual.x = x
            individual.process()
            score, _ = individual.objv()

            if score > 0 and score != -float('inf') and score != float('inf'):
                self.particles.append(individual)
                self.velocities.append(v)
                self.pBest.append(x.clone())
                self.pBest_scores.append(score)
                if score > self.gBest_score:
                    self.gBest = x.clone()
                    self.gBest_score = score
                    self.bestPoints = individual.points

            attempts += 1
        
        # Fill up population if not enough valid individuals found
        while len(self.particles) < self.population_size:
            individual = self.create_individual()
            x = torch.zeros((self.taskNum, self.providerNum + 1), device=self.device)
            task_provider = torch.randint(0, self.providerNum + 1, (self.taskNum,)).to(self.device)
            for i, p in enumerate(task_provider):
                x[i, p] = 1
            
            v = torch.randn((self.taskNum, self.providerNum + 1), device=self.device) * 0.1
            individual.x = x
            individual.process()
            score, _ = individual.objv()
            self.particles.append(individual)
            self.velocities.append(v)
            self.pBest.append(x.clone())
            self.pBest_scores.append(score)
            if score > self.gBest_score:
                self.gBest = x.clone()
                self.gBest_score = score
                self.bestPoints = individual.points


    def update_velocity_position(self):
        for i, individual in enumerate(self.particles):
            r1 = torch.rand((self.taskNum, self.providerNum + 1), device=self.device)
            r2 = torch.rand((self.taskNum, self.providerNum + 1), device=self.device)
            v = self.velocities[i]
            x = individual.x

            cognitive = self.c1 * r1 * (self.pBest[i] - x)
            social = self.c2 * r2 * (self.gBest - x)
            new_v = self.w * v + cognitive + social
            
            # Position is treated as probabilities, then select the best one
            # No sigmoid needed if we use argmax directly
            
            # Discretize position to one-hot encoding
            new_x_discrete = torch.zeros_like(new_v)
            best_provider_indices = torch.argmax(new_v, dim=1)
            new_x_discrete[torch.arange(self.taskNum), best_provider_indices] = 1

            individual.x = new_x_discrete
            self.velocities[i] = new_v

    def process(self):
        for pop in self.particles:
            pop.process()

    def evaluate(self):
        for i, individual in enumerate(self.particles):
            individual.init_points()
            individual.beginning = point()
            individual.beginning.loc = 0
            individual.beginning.children = [individual.points[0]]
            individual.beginning.finished = True
            individual.x = individual.x  # Ensure x is set
            individual.process()  # Add process call

            score, _ = individual.objv()
            if score > self.pBest_scores[i]:
                self.pBest[i] = individual.x.clone()
                self.pBest_scores[i] = score
                if score > self.gBest_score:
                    self.gBest = individual.x.clone()
                    self.gBest_score = score
                    self.bestPoints = individual.points

    def run(self, iterations):
        print(f"开始PSO算法，共{iterations}轮，种群大小：{self.population_size}")
        print(f"任务数：{self.taskNum}，服务提供商数：{self.providerNum}")
        print(f"参数设置 - w:{self.w}, c1:{self.c1}, c2:{self.c2}")
        print("-" * 60)
        
        self.initialize()
        prev_best = -float('inf')
        
        for it in range(iterations):
            self.update_velocity_position()
            self.evaluate()
            
            # 保存当前最佳points
            import pickle
            if self.bestPoints is not None:
                pickle.dump(self.bestPoints, open(f"points_{self.taskNum}_{self.providerNum}_pso.pkl", "wb"))
            
            # 计算当前代的适应度统计
            fitness_values = []
            for particle in self.particles:
                fit, _ = particle.objv()
                fitness_values.append(fit)
            
            current_best = max(fitness_values)
            current_worst = min(fitness_values)
            current_avg = sum(fitness_values) / len(fitness_values)
            
            # 记录日志
            print(f"第{it+1}轮:")
            print(f"  当前最佳适应度: {current_best:.6f}")
            print(f"  当前最差适应度: {current_worst:.6f}")
            print(f"  当前平均适应度: {current_avg:.6f}")
            print(f"  全局最佳适应度: {self.gBest_score:.6f}")
            
            if it > 0 and current_best > prev_best:
                print(f"  *** 发现更好解！适应度从 {prev_best:.6f} 提升到 {current_best:.6f} ***")
            
            prev_best = current_best
            print("-" * 60)
        
        print(f"PSO算法完成！最终最佳适应度: {self.gBest_score:.6f}")
        return self.gBest, self.gBest_score

if __name__ == "__main__":
    torch.manual_seed(4244)
    np.random.seed(4244)
    
    print("=" * 80)
    print("                    PSO算法求解网络优化问题")
    print("=" * 80)

    det = json.load(open(datafile, "r"))
    taskN = det["taskNum"]
    providerN = det["providerNum"]
    edges = det["edges"]
    
    print(f"数据加载完成:")
    print(f"  任务数量: {taskN}")
    print(f"  服务提供商数量: {providerN}")
    print(f"  边数量: {len(edges)}")
    
    deadlines = np.array(det["taskdeadlines"])
    budgets = np.array(det["taskbudgets"])
    Rs = np.array(det["taskResources"])
    abilities = np.array(det["taskabilities"])
    providerRep = np.array(det["providerRep"])
    cost = np.array(det["taskCost"])

    taskTime = det["taskTime"]
    rep = det["providerRep"]
    deadline = det["deadline"]
    providerAbility = det["providerAbility"]
    providerL = np.array(det["providerL"])
    providerPrice = det["providerPrice"]
    providerReliability = det["providerReliability"]
    providerEnergyCost = det["providerEnergyCost"]
    budget = det["budget"]
    param = det["providerParam"]
    andorInfo = det["andor"]
    omega = [0.2, 0.2, 0.2, 0.2, 0.2]
    lam = 0.5

    print(f"参数设置:")
    print(f"  预算: {budget}")
    print(f"  截止时间: {deadline}")
    print(f"  权重omega: {omega}")
    print(f"  lambda参数: {lam}")
    print()

    model = PSO(taskN, providerN, device,
                deadlines, budgets, Rs, abilities, cost,
                providerN, taskN, edges,
                taskTime, rep, deadline, providerAbility,
                providerL, providerPrice, providerReliability,
                providerEnergyCost, budget, param, omega, lam, andorInfo)

    print(f"PSO模型初始化完成:")
    print(f"  种群大小: {model.population_size}")
    print(f"  惯性权重w: {model.w}")
    print(f"  认知系数c1: {model.c1}")
    print(f"  社会系数c2: {model.c2}")
    print()

    print("开始优化求解...")
    best_x, best_fitness = model.run(800)
    
    print("=" * 80)
    print("                         优化结果")
    print("=" * 80)
    print("最佳解决方案:")
    for i in range(taskN):
        provider = torch.argmax(best_x[i]).item()
        if provider == providerN:
            print(f"  任务 {i} -> 不执行")
        else:
            print(f"  任务 {i} -> 服务提供商 {provider}")
    
    print(f"\n最佳适应度: {best_fitness:.6f}")
    print("=" * 80)

    if model.bestPoints is not None:
        summary_points = model.bestPoints
    else:
        summary_points = model.rebuild_points_from_assignment(best_x)
    _print_matching_summary(summary_points, taskTime)