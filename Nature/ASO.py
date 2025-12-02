from generate_net import netw
from xtmqn import point
from torch import nn
import torch
import json
import numpy as np
import random
import pickle
from collections import defaultdict


def _summarize_matching(points_snapshot, task_time_matrix):
    """
    根据点集合统计任务-服务匹配，并汇总耗时与成本。
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


datafile = "C:/Users/Kaka/PycharmProjects/dsm-main (1)/dsm-main/data_30_15_16181.json"
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
                    objv -= self.M
            elif self.andor[p.loc] == "or":
                num = 0
                for c in p.children:
                    if c.finished:
                        num += 1
                if num != 1:
                    objv -= self.M

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
        
        # Reset cobjv flag for all points
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

        for p in self.points:
            if p.cobjv and p.loc != -1:
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
                satisfaction += ((self.providerPrice[p.provider][p.loc] - p.cost) / self.providerPrice[p.provider][p.loc] + self.param[p.provider][0] + \
                    self.param[p.provider][1] + self.param[p.provider][2] + self.param[p.provider][3]) / 5
                p.us = ((self.providerPrice[p.provider][p.loc] - p.cost) / self.providerPrice[p.provider][p.loc] + self.param[p.provider][0] + \
                    self.param[p.provider][1] + self.param[p.provider][2] + self.param[p.provider][3]) / 5
        
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
            for j, v in enumerate(self.x[i]):
                if v == 1:
                    assigned_provider = j
                    break
            
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

class ACO:
    def __init__(self, taskNum, providerNum, device, alpha=1.0, beta=2.0, rho=0.1, Q=1.0):
        self.device = device
        self.taskNum = taskNum
        self.providerNum = providerNum
        self.alpha = alpha  # pheromone importance
        self.beta = beta    # heuristic importance
        self.rho = rho      # evaporation rate
        self.Q = Q          # pheromone deposit factor

        self.population = []
        self.size = 100
        self.pheromone = torch.ones((taskNum, providerNum + 1)).to(device)
        self.heuristic = torch.ones((taskNum, providerNum + 1)).to(device)
        self.bestSolution = None
        self.bestFitness = -float('inf')
        self.bestPoints = None

    def create_individual(self, deadlines, budgets, Rs, abilities, cost,
                          providerN, taskN, providerNum, taskNum, edges, device,
                          taskTime, rep, deadline, providerAbility,
                          providerL, providerPrice, providerReliability,
                          providerEnergyCost, budget, param, omega, lam, andorInfo):
        tempn = network(deadlines, budgets, Rs, abilities, cost,
                        providerN, taskN, providerNum, taskNum, edges, device,
                        taskTime, rep, deadline, providerAbility,
                        providerL, providerPrice, providerReliability,
                        providerEnergyCost, budget, param, omega, lam, andorInfo)
        tempn.init_points()
        # 正确设置beginning
        tempn.beginning = point()
        tempn.beginning.loc = -1  # beginning的loc应该是-1
        tempn.beginning.children = [tempn.points[0]]
        tempn.beginning.finished = True
        tempn.beginning.provider = -1  # beginning不需要provider
        return tempn

    def rebuild_points_from_assignment(self, template_network, assignment_matrix):
        """
        基于给定解恢复节点状态，便于统计taskTime和taskCost。
        """
        helper = self.create_individual(
            template_network.deadlines, template_network.budgets, template_network.Rs, template_network.abilities,
            template_network.cost, template_network.providerNum, template_network.taskNum,
            template_network.providerNum, template_network.taskNum, template_network.edges, self.device,
            template_network.taskTime, template_network.rep, template_network.deadline, template_network.providerAbility,
            template_network.providerL, template_network.providerPrice, template_network.providerReliability,
            template_network.providerEnergyCost, template_network.budget, template_network.param,
            template_network.omega, template_network.lam, template_network.andor
        )
        helper.x = assignment_matrix.clone().to(self.device)
        helper.process()
        return helper.points

    def construct_solution(self, template_network):
        individual = self.create_individual(
            template_network.deadlines, template_network.budgets, template_network.Rs, template_network.abilities,
            template_network.cost, template_network.providerNum, template_network.taskNum,
            template_network.providerNum, template_network.taskNum, template_network.edges, self.device,
            template_network.taskTime, template_network.rep, template_network.deadline, template_network.providerAbility,
            template_network.providerL, template_network.providerPrice, template_network.providerReliability,
            template_network.providerEnergyCost, template_network.budget, template_network.param,
            template_network.omega, template_network.lam, template_network.andor
        )
        x = torch.zeros((self.taskNum, self.providerNum + 1)).to(self.device)

        for task in range(self.taskNum):
            # 计算概率，添加数值稳定性检查
            pheromone_part = torch.clamp(self.pheromone[task], min=1e-10, max=1e10) ** self.alpha
            heuristic_part = torch.clamp(self.heuristic[task], min=1e-10, max=1e10) ** self.beta
            probabilities = pheromone_part * heuristic_part
            
            # 检查概率是否有效
            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                # 如果出现NaN或inf，使用均匀分布
                probabilities = torch.ones_like(probabilities)
            
            # 确保概率和不为0
            prob_sum = probabilities.sum()
            if prob_sum <= 0:
                probabilities = torch.ones_like(probabilities)
                prob_sum = probabilities.sum()
            
            probabilities = probabilities / prob_sum
            
            # 再次检查概率是否有效
            if torch.isnan(probabilities).any() or (probabilities < 0).any():
                # 如果仍有问题，使用均匀分布
                probabilities = torch.ones_like(probabilities) / len(probabilities)
            
            provider = torch.multinomial(probabilities, 1).item()
            x[task][provider] = 1

        individual.x = x
        return individual

    def process(self):
        for pop in self.population:
            pop.process()

    def fitness(self, individual):
        return individual.objv()

    def update_pheromone(self):
        # 信息素挥发
        self.pheromone *= (1 - self.rho)
        
        # 添加新的信息素
        for individual in self.population:
            fit, _ = self.fitness(individual)
            # 只有当适应度为正数时才添加信息素
            if fit > 0:
                for task in range(self.taskNum):
                    provider = torch.argmax(individual.x[task]).item()
                    self.pheromone[task][provider] += self.Q * fit
        
        # 限制信息素的范围，避免过大或过小
        self.pheromone = torch.clamp(self.pheromone, min=1e-10, max=1e10)

    def initialize_heuristic(self, template_network):
        """根据任务和提供商的特征初始化启发式信息"""
        for task in range(self.taskNum):
            for provider in range(self.providerNum + 1):
                if provider == self.providerNum:  # "do not execute" option
                    self.heuristic[task][provider] = 1e-6  # A small heuristic value
                    continue
                # 基于成本效益比计算启发式值
                task_cost = template_network.cost[task]
                provider_price = template_network.providerPrice[provider][task] if task < len(template_network.providerPrice[provider]) else 1.0
                
                # 避免除零错误
                if provider_price <= 0:
                    provider_price = 1.0
                
                # 启发式值与成本效益成反比
                heuristic_value = 1.0 / (provider_price + 1e-10)
                
                # 考虑提供商的能力和可靠性
                reliability = template_network.providerReliability[provider] if provider < len(template_network.providerReliability) else 0.5
                heuristic_value *= (1 + reliability)
                
                self.heuristic[task][provider] = heuristic_value
        
        # 标准化启发式值
        self.heuristic = torch.clamp(self.heuristic, min=1e-10, max=1e10)

    def run(self, generations, template_network):
        print(f"开始ACO算法，共{generations}轮，种群大小：{self.size}")
        print(f"任务数：{self.taskNum}，服务提供商数：{self.providerNum}")
        print("-" * 60)
        
        # 初始化启发式信息
        self.initialize_heuristic(template_network)
        print("启发式信息初始化完成")
        
        # 初始化种群
        self.population = []
        attempts = 0
        max_attempts = self.size * 100
        while len(self.population) < self.size and attempts < max_attempts:
            ant = self.construct_solution(template_network)
            ant.process()
            fit, _ = self.fitness(ant)
            if fit > 0:
                self.population.append(ant)
            attempts += 1
        
        while len(self.population) < self.size:
            ant = self.construct_solution(template_network)
            self.population.append(ant)

        for gen in range(generations):
            # 在每一代的开始，根据信息素重新构建解决方案
            current_population = []
            for _ in range(self.size):
                ant = self.construct_solution(template_network)
                current_population.append(ant)
            self.population = current_population

            self.process()

            # 计算当前代的适应度统计
            fitness_values = []
            valid_ants = []
            for ant in self.population:
                fit, _ = self.fitness(ant)
                # 移除无效解的检查，因为适应度计算现在更鲁棒
                fitness_values.append(fit)
                valid_ants.append(ant)
            
            if not fitness_values:
                print(f"第{gen+1}轮: 没有找到有效的解决方案。")
                # 重新初始化部分种群以增加多样性
                new_ants = []
                for _ in range(self.size // 2):
                    new_ant = self.construct_solution(template_network)
                    new_ant.process()
                    new_ants.append(new_ant)
                self.population = self.population[:self.size // 2] + new_ants
                continue

            self.population = valid_ants
            
            current_best = max(fitness_values) if fitness_values else -1
            current_worst = min(fitness_values) if fitness_values else -1
            current_avg = sum(fitness_values) / len(fitness_values) if fitness_values else -1
            
            # 记录日志
            print(f"第{gen+1}轮:")
            print(f"  当前最佳适应度: {current_best:.6f}")
            print(f"  当前最差适应度: {current_worst:.6f}")
            print(f"  当前平均适应度: {current_avg:.6f}")
            
            self.update_pheromone()
            
            # Find best solution
            generation_improved = False
            for ant in self.population:
                fit, _ = self.fitness(ant)
                if fit > self.bestFitness:
                    self.bestFitness = fit
                    self.bestSolution = ant.x.clone()
                    self.bestPoints = ant.points
                    generation_improved = True
            
            # 保存当前最佳points
            if self.bestPoints is not None:
                pickle.dump(self.bestPoints, open(f"points_{self.taskNum}_{self.providerNum}_aco.pkl", "wb"))
            
            if generation_improved:
                print(f"  *** 发现更好解！全局最佳适应度更新为: {self.bestFitness:.6f} ***")
            else:
                print(f"  全局最佳适应度: {self.bestFitness:.6f}")
            
            # 显示信息素浓度统计
            pheromone_max = torch.max(self.pheromone).item()
            pheromone_min = torch.min(self.pheromone).item()
            pheromone_avg = torch.mean(self.pheromone).item()
            print(f"  信息素浓度 - 最大: {pheromone_max:.4f}, 最小: {pheromone_min:.4f}, 平均: {pheromone_avg:.4f}")
            print("-" * 60)

        print(f"ACO算法完成！最终最佳适应度: {self.bestFitness:.6f}")
        return self.bestSolution, self.bestFitness

if __name__ == "__main__":
    torch.manual_seed(4244)
    np.random.seed(4244)
    
    print("=" * 80)
    print("                    ACO算法求解网络优化问题")
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

    model = ACO(taskN, providerN, device)
    print(f"ACO模型初始化完成:")
    print(f"  alpha (信息素重要性): {model.alpha}")
    print(f"  beta (启发式重要性): {model.beta}")
    print(f"  rho (挥发率): {model.rho}")
    print(f"  Q (信息素沉积因子): {model.Q}")
    print()
    
    template_network = model.create_individual(deadlines, budgets, Rs, abilities, cost,
        providerN, taskN, providerN, taskN, edges, device,
        taskTime, rep, deadline, providerAbility,
        providerL, providerPrice, providerReliability,
        providerEnergyCost, budget, param, omega, lam, andorInfo)

    model.initialize_heuristic(template_network)

    print("开始优化求解...")
    solution, fitness = model.run(800, template_network)
    
    # 添加调试：显示最终状态
    print("\n" + "="*50)
    print("调试：检查ACO最终状态")
    print("="*50)
    
    print("=" * 80)
    print("                         优化结果")
    print("=" * 80)
    if solution is not None:
        print("最佳解决方案:")
        for i in range(taskN):
            provider = torch.argmax(solution[i]).item()
            if provider == providerN:
                print(f"  任务 {i} -> 不执行")
            else:
                print(f"  任务 {i} -> 服务提供商 {provider}")
        
        print(f"\n最佳适应度: {fitness:.6f}")
    else:
        print("未找到有效解决方案")
    print("=" * 80)

    summary_points = None
    if model.bestPoints is not None:
        summary_points = model.bestPoints
    elif solution is not None:
        summary_points = model.rebuild_points_from_assignment(template_network, solution)
    if summary_points is not None:
        _print_matching_summary(summary_points, taskTime)
