from generate_net import netw
from xtmqn import point
from torch import nn
import torch, json
import numpy as np
import time

datafile = "C:/Users/Kaka/PycharmProjects/dsm-main (1)/dsm-main/data_60_30.json"
# datafile = "5_3/data_5_3_0.json"
# datafile = "30_20/data_30_20_174.json"
device = "cuda"

def _summarize_matching(points_snapshot, task_time_matrix):
    """
    根据点集信息整理任务-服务匹配，并计算总taskTime与taskCost。
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
        task_time = float(task_time_matrix[provider_idx][task_idx])
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

def _print_matching_summary(title, points_snapshot, task_time_matrix):
    matches, total_time, total_cost = _summarize_matching(points_snapshot, task_time_matrix)
    print(title)
    if not matches:
        print("  未找到有效的任务-服务匹配。")
        return
    for item in matches:
        print(f"  任务 {item['task']} -> 服务 {item['provider']} | taskTime={item['taskTime']:.4f} | taskCost={item['taskCost']:.4f}")
    print(f"  总taskTime: {total_time:.4f}")
    print(f"  总taskCost: {total_cost:.4f}")

class network(netw):
    def __init__(self, deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, providerNum, taskNum, edges, device,\
                taskTime, rep, deadline, providerAbility,\
                providerL, providerPrice, providerReliability,\
                    providerEnergyCost, budget, param, omega, lam, andorInfo):
        # 正确调用父类构造函数
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
        self.budget = budget
        self.param = param
        self.omega = omega
        self.lam = lam
        self.points = []
        self.paths = [0 for i in range(taskNum)]
        self.x = torch.zeros((self.taskNum, self.providerNum + 1)).to(self.device)
        self.pathe = [0 for i in range(taskNum)]
        self.andor = andorInfo
        self.M = 1000000  # punishment factor
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
        # 检查provider资源超载
        for i in range(self.providerNum):
            if self.provider[i] > self.providerL[i]:
                penalty = self.M * (self.provider[i] - self.providerL[i]) / max(1, self.providerL[i])
                if isinstance(objv, torch.Tensor):
                    objv = objv - torch.tensor(penalty).to(objv.device)
                else:
                    objv -= penalty
                
        # 检查关键路径是否超过截止时间（全局检查，只检查一次）
        if self.critical_path > self.deadline:
            penalty = self.M * (self.critical_path - self.deadline) / max(1, len(self.points))
            if isinstance(objv, torch.Tensor):
                objv = objv - torch.tensor(penalty).to(objv.device)
            else:
                objv -= penalty
        
        # 检查每个点的约束条件
        for p in self.points:
            if not p.finished:
                continue
            
            # 检查任务能力是否超过提供商能力
            if p.loc != -1 and p.provider != -1:
                if hasattr(p, 'abilities') and hasattr(self, 'providerAbility'):
                    if p.abilities > self.providerAbility[p.provider]:
                        penalty = self.M * (p.abilities - self.providerAbility[p.provider]) / max(1, len(self.points))
                        if isinstance(objv, torch.Tensor):
                            objv = objv - torch.tensor(penalty).to(objv.device)
                        else:
                            objv -= penalty
            
            # 检查AND/OR约束条件
            if p.children == []:
                continue
            if self.andor[p.loc] == "and":
                num = 0
                for c in p.children:
                    if c.finished:
                        num += 1
                if num < len(p.children):
                    # print("And Condition Not Satisfied;", end=" ")
                    if isinstance(objv, torch.Tensor):
                        objv = objv - torch.tensor(self.M).to(objv.device)
                    else:
                        objv -= self.M
            elif self.andor[p.loc] == "or":
                num = 0
                for c in p.children:
                    if c.finished:
                        num += 1
                if num != 1:
                    # print("Or Condition Not Satisfied;", end=" ")
                    if isinstance(objv, torch.Tensor):
                        objv = objv - torch.tensor(self.M).to(objv.device)
                    else:
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
            if p.cobjv:
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
        
        satisfaction = self.add_punishment_to_objv(satisfaction)
        
        satisfactionS = torch.tensor(satisfaction)
        
        return satisfactionS, (sat1, satisfaction-sat1)

class geneticAlgorithm():
    def __init__(self, taskNum, providerNum, device):
        self.device = device
        self.taskNum = taskNum
        self.providerNum = providerNum
        self.population = []
        self.bestSolution = None
        self.size = 10
        self.finess = torch.zeros((self.size,)).to(self.device)
    
    def create_individual(self, deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, providerNum, taskNum, edges, device,\
                     taskTime, rep, deadline, providerAbility,\
                     providerL, providerPrice, providerReliability,\
                     providerEnergyCost, budget, param, omega, lam, andor):
        tempn = network(deadlines, budgets, Rs, abilities, cost,\
                    providerN, taskN, providerNum, taskNum, edges, device,\
                    taskTime, rep, deadline, providerAbility,\
                    providerL, providerPrice, providerReliability,\
                        providerEnergyCost, budget, param, omega, lam, andor)
        tempn.init_points()
        # 正确设置beginning
        tempn.beginning = point()
        tempn.beginning.loc = -1  # beginning的loc应该是-1
        tempn.beginning.children = [tempn.points[0]]
        tempn.beginning.finished = True
        tempn.beginning.provider = -1  # beginning不需要provider
        return tempn
    
    def initialize_population(self, size, deadlines, budgets, Rs, abilities, cost,\
                 providerN, taskN, providerNum, taskNum, edges, device,\
                     taskTime, rep, deadline, providerAbility,\
                     providerL, providerPrice, providerReliability,\
                     providerEnergyCost, budget, param, omega, lam, andor):
        
        self.population = []
        attempts = 0
        max_attempts = size * 100  # Set a limit to prevent infinite loops

        while len(self.population) < size and attempts < max_attempts and False:
            individual = self.create_individual(deadlines, budgets, Rs, abilities, cost,\
                    providerN, taskN, providerNum, taskNum, edges, device,\
                    taskTime, rep, deadline, providerAbility,\
                    providerL, providerPrice, providerReliability,\
                    providerEnergyCost, budget, param, omega, lam, andor)
            
            # The task assignment matrix now includes the "do not execute" option
            individual.x = torch.zeros((self.taskNum, self.providerNum + 1)).to(self.device)
            task_provider = torch.randint(0, self.providerNum + 1, (self.taskNum,)).to(self.device)
            for i, p in enumerate(task_provider):
                individual.x[i][p] = 1

            # Process the individual to update its state based on x
            # The provider load array only tracks actual providers
            individual.provider = np.zeros((self.providerNum,))
            for p in individual.points:
                p.finished = False
                p.provider = -1
                # Find which provider is assigned to the task
                assigned_provider = -1
                for i, v in enumerate(individual.x[p.loc]):
                    if v == 1:
                        assigned_provider = i
                        break
                
                if assigned_provider != -1 and assigned_provider < individual.providerNum:
                    p.provider = assigned_provider
                    p.finished = True
                    # Update the load for the assigned provider
                    individual.provider[assigned_provider] += individual.Rs[p.loc]
            
            # Check fitness
            fitness_val, _ = self.fitness(individual)
            if fitness_val > 0:
                print('find a feasible solution')
                self.population.append(individual)
            
            attempts += 1
        pass
        for i in range(len(self.population), size):
            individual = self.create_individual(deadlines, budgets, Rs, abilities, cost,\
                    providerN, taskN, providerNum, taskNum, edges, device,\
                    taskTime, rep, deadline, providerAbility,\
                    providerL, providerPrice, providerReliability,\
                    providerEnergyCost, budget, param, omega, lam, andor)

            # The task assignment matrix now includes the "do not execute" option
            individual.x = torch.zeros((self.taskNum, self.providerNum + 1)).to(self.device)
            task_provider = torch.randint(0, self.providerNum + 1, (self.taskNum,)).to(self.device)
            for i, p in enumerate(task_provider):
                individual.x[i][p] = 1

            # Process the individual to update its state based on x
            # The provider load array only tracks actual providers
            individual.provider = np.zeros((self.providerNum,))
            for p in individual.points:
                p.finished = False
                p.provider = -1
                # Find which provider is assigned to the task
                assigned_provider = -1
                for i, v in enumerate(individual.x[p.loc]):
                    if v == 1:
                        assigned_provider = i
                        break

                if assigned_provider != -1 and assigned_provider < individual.providerNum:
                    p.provider = assigned_provider
                    p.finished = True
                    # Update the load for the assigned provider
                    individual.provider[assigned_provider] += individual.Rs[p.loc]
            self.population.append(individual)

    def fitness(self, individual):
        # Calculate the fitness of an individual
        # This is a placeholder, replace with actual fitness calculation
        result = individual.objv()
        
        # 添加调试输出
        if hasattr(individual, 'debug_mode') and individual.debug_mode:
            print(f"GA Debug - Critical path: {individual.critical_path}")
            # cost = sum([self.providerPrice[p.provider][p.loc] for p in individual.points if p.finished and p.provider != -1])
            print(f"GA Debug - Budget: {individual.budget}, Cost: {cost}")
            print(f"GA Debug - Provider state: {individual.provider}")
            print(f"GA Debug - Finished points: {[p.loc for p in individual.points if p.finished]}")
            print(f"GA Debug - Point providers: {[(p.loc, p.provider) for p in individual.points if p.finished]}")
            print(f"GA Debug - Objective value: {result[0]}")
            print("-" * 50)
        
        return result

    def process(self):
        for pop in self.population:
            pop.provider = np.zeros((self.providerNum,))
            for p in pop.points:
                p.finished = False  # 先重置状态
                p.provider = -1
                
                # 检查任务分配
                for i, v in enumerate(pop.x[p.loc]):
                    if v == 1:
                        if i < self.providerNum:
                            p.provider = i
                            p.finished = True
                            pop.provider[i] += pop.Rs[p.loc]
                        # 如果 i == self.providerNum，则表示不执行该任务
                        break  # 一个任务只能分配给一个provider
    
    def calAllFitness(self):
        fitness_values = []
        for individual in self.population:
            fitness_values.append(self.fitness(individual))
        return fitness_values

    def selection(self):
        # Select individuals based on their fitness
        fitness_values = self.calAllFitness()
        alValue = zip(self.population, fitness_values)
        alValue = sorted(alValue, key=lambda x: x[1][0], reverse=True)
        selected = alValue[:self.size]
        self.population = [ind[0] for ind in selected]
        self.bestSolution = selected[0][0].x
        self.bestFitness = selected[0][1]
        self.bestpoints = selected[0][0].points
        self.best_task_time = getattr(selected[0][0], "taskTime", None)
        return self.bestSolution, self.bestFitness
    
    def calculate_total_cost(self, individual):
        """计算个体的总成本"""
        total_cost = 0
        for p in individual.points:
            if p.finished and p.provider != -1 and p.loc != -1:
                total_cost += individual.providerPrice[p.provider][p.loc]
        return total_cost
    
    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a child
        x1 = parent1.x
        x2 = parent2.x
        cross_point = np.random.randint(0, self.taskNum)
        al = np.random.rand()
        
        x1_new = torch.cat((x1[:cross_point], x2[cross_point:]), dim=0)
        x2_new = torch.cat((x2[:cross_point], x1[cross_point:]), dim=0)

        child1 = self.create_individual(parent1.deadlines, parent1.budgets, parent1.Rs, parent1.abilities, parent1.cost,\
                    parent1.providerNum, parent1.taskNum, parent1.providerNum, parent1.taskNum, parent1.edges, self.device,\
                    parent1.taskTime, parent1.rep, parent1.deadline, parent1.providerAbility,\
                    parent1.providerL, parent1.providerPrice, parent1.providerReliability,\
                    parent1.providerEnergyCost, parent1.budget, parent1.param, parent1.omega, parent1.lam, parent1.andor)
        child2 = self.create_individual(parent2.deadlines, parent2.budgets, parent2.Rs, parent2.abilities, parent2.cost,\
                    parent2.providerNum, parent2.taskNum, parent2.providerNum, parent2.taskNum, parent2.edges, self.device,\
                    parent2.taskTime, parent2.rep, parent2.deadline, parent2.providerAbility,\
                    parent2.providerL, parent2.providerPrice, parent2.providerReliability,\
                    parent2.providerEnergyCost, parent2.budget, parent2.param, parent2.omega, parent2.lam, parent2.andor)
        child1.x = x1_new
        child2.x = x2_new
        return child1, child2
    
    def mutation(self, individual):
        # Perform mutation on an individual
        mutation_rate = 0.01
        for i in range(self.taskNum):
            if np.random.rand() < mutation_rate:
                individual.x[i,:] = 0
                individual.x[i, np.random.randint(0, self.providerNum + 1)] = 1
    
    def init(self):
        for pop in self.population:
            pop.paths = [0 for i in range(self.taskNum)]
            pop.init_points()
            pop.critical_path = -1
            pop.beginning = pop.points[0]
            pop.pathe = [0 for i in range(self.taskNum)]
            for p in pop.points:
                p.provider = -1
                p.finished = False
                p.providerL = 0
                p.providerPrice = 0
                p.cobjv = False
    def run(self, generations):
        total_time = 0  # 总运行时间
        for generation in range(generations):
            epoch_start_time = time.time()  # 记录epoch开始时间
            
            # self.init()
            self.process()
            self.selection()
            new_population = []
            import pickle
            pickle.dump(self.bestpoints, open(f"points_{self.population[0].taskNum}_{self.population[0].providerNum}_ga.pkl", "wb"))
            for i in range(0, len(self.population), 2):
                for j in range(i , len(self.population)):
                    if i == j:
                        continue
                    if np.random.rand() < (2 - pow(2,(((i+j)/2)/self.size))):
                        parent1 = self.population[i]
                        parent2 = self.population[j]
                        child1, child2 = self.crossover(parent1, parent2)
                        new_population.append(child1)
                        new_population.append(child2)
            
            # 计算最佳个体的总成本
            best_individual = self.population[0] if len(self.population) > 0 else None
            total_cost = 0
            if best_individual is not None:
                total_cost = self.calculate_total_cost(best_individual)
            
            epoch_end_time = time.time()  # 记录epoch结束时间
            epoch_time = epoch_end_time - epoch_start_time  # 计算epoch耗时
            total_time += epoch_time  # 累加总时间
            
            # 输出每个epoch的信息
            print(f"Generation {generation+1}/{generations}, Best Fitness: {self.bestFitness[0] if isinstance(self.bestFitness, tuple) else self.bestFitness}, "
                  f"Total Cost: {total_cost:.2f}, Epoch Time: {epoch_time:.4f}s")
            
            for individual in new_population:
                self.mutation(individual)
            self.population = new_population
        
        # 输出总运行时间
        print(f"\n总运行时间: {total_time:.4f}s ({total_time/60:.2f}分钟)")
        return total_time

    def debug_compare_with_ppo(self, individual_idx=0):
        """
        调试函数：比较GA和PPO的初始状态
        """
        if individual_idx >= len(self.population):
            print(f"个体索引 {individual_idx} 超出范围，总共有 {len(self.population)} 个个体")
            return
            
        individual = self.population[individual_idx]
        print("=" * 60)
        print(f"GA调试信息 - 个体 {individual_idx}")
        print("=" * 60)
        
        # 显示任务分配情况
        print("任务分配情况:")
        for i, p in enumerate(individual.points):
            provider_allocation = [j for j, v in enumerate(individual.x[i]) if v == 1]
            print(f"  任务 {i}: provider={p.provider}, finished={p.finished}, x分配={provider_allocation}")
        
        # 显示provider资源使用情况
        print(f"\nProvider资源使用情况:")
        for i in range(individual.providerNum):
            print(f"  Provider {i}: 已用资源={individual.provider[i]}, 总容量={individual.providerL[i]}")
        
        # 显示关键路径信息
        print(f"\n关键路径信息:")
        print(f"  Critical path: {individual.critical_path}")
        print(f"  Deadline: {individual.deadline}")
        print(f"  Beginning存在: {hasattr(individual, 'beginning') and individual.beginning is not None}")
        
        # 计算成本信息
        total_cost = sum([individual.providerPrice[p.provider][p.loc] for p in individual.points if p.finished and p.provider != -1])
        print(f"\n成本信息:")
        print(f"  总预算: {individual.budget}")
        print(f"  实际成本: {total_cost}")
        print(f"  成本比例: {total_cost/individual.budget if individual.budget > 0 else 'N/A'}")
        
        # 计算目标函数
        objv_result = individual.objv()
        print(f"\n目标函数:")
        print(f"  主目标值: {objv_result[0]}")
        print(f"  分解值: {objv_result[1]}")
        print("=" * 60)
        
        return objv_result

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("                    GA算法求解网络优化问题")
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
    
    epoch = 800
    netsNum = 1
    # netsNum = 0

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

    model = geneticAlgorithm(taskN, providerN, device)
    
    print(f"GA模型初始化完成:")
    print(f"  任务数：{taskN}，服务提供商数：{providerN}")
    print()
    
    print("开始优化求解...")
    model.initialize_population(1000, deadlines, budgets, Rs, abilities, cost,
                    providerN, taskN, providerN, taskN, edges, device,
                    taskTime, rep, deadline, providerAbility,
                    providerL, providerPrice, providerReliability,
                    providerEnergyCost, budget, param, omega, lam, andorInfo)
    
    # 添加调试：在开始优化前检查初始状态
    print("\n" + "="*50)
    print("调试：检查GA初始状态")
    print("="*50)
    model.init()
    model.process()
    
    # 启用第一个个体的调试模式
    if len(model.population) > 0:
        model.population[0].debug_mode = True
        model.debug_compare_with_ppo(0)
    
    print("\n开始GA优化...")
    total_time = model.run(epoch)
    
    print("=" * 80)
    print("                         优化结果")
    print("=" * 80)
    print("最佳解决方案:")
    for i in range(taskN):
        provider = torch.argmax(model.bestSolution[i]).item()
        print(f"  任务 {i} -> 服务提供商 {provider}")
    
    # 输出匹配详情与总taskTime、taskCost
    _print_matching_summary("\n任务-服务匹配详情（GA）:", getattr(model, "bestpoints", None), getattr(model, "best_task_time", None))
    
    
    print(f"\n最佳适应度: {model.bestFitness[0] if isinstance(model.bestFitness, tuple) else model.bestFitness}")
    print(f"总运行时间: {total_time:.4f}s ({total_time/60:.2f}分钟)")
    print("=" * 80)
