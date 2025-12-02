from generate_net import netw
from xtmqn import point
from torch import nn
import torch, json
import numpy as np
import random
import pickle
from collections import defaultdict

datafile = "data_30_15_16181.json"
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
        
        satisfaction = torch.tensor((budget - cost) / budget).to(self.device)
        if self.deadline > self.critical_path:
            satisfaction += torch.exp(torch.tensor(-self.lam * (self.deadline-self.critical_path))).to(self.device)
        satisfaction /= 2
        
        sat1 = satisfaction.clone()

        for p in self.points:
            if p.cobjv and p.loc != -1:
                sT = 0
                if self.deadlines[p.loc]>self.pathe[p.loc]:
                    if self.pathe[p.loc] == 0:
                        pass
                    else:
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
        
class ALNS:
    def __init__(self, taskNum, providerNum, device):
        self.device = device
        self.taskNum = taskNum
        self.providerNum = providerNum
        self.current_solution = None
        self.best_solution = None
        self.best_objective = -float('inf')
        self.best_points = None
        
        # ALNS parameters
        self.iterations = 500
        self.adaptive_period = 50
        self.start_temperature = 1.0
        self.cooling_rate = 0.995
        self.destroy_degree = 0.2
        
        # Operator management
        self.destroy_ops = [
            self.random_removal,
            self.worst_removal,
            self.shaw_removal
        ]
        self.repair_ops = [
            self.random_repair,
            self.greedy_repair,
            self.regret_repair
        ]
        self.operator_weights = {
            'destroy': [1.0] * len(self.destroy_ops),
            'repair': [1.0] * len(self.repair_ops)
        }
        self.operator_scores = {
            'destroy': [0.0] * len(self.destroy_ops),
            'repair': [0.0] * len(self.repair_ops)
        }
        self.operator_counts = {
            'destroy': [0] * len(self.destroy_ops),
            'repair': [0] * len(self.repair_ops)
        }

    
    def initialize_solution(self, deadlines, budgets, Rs, abilities, cost,
                           providerN, taskN, providerNum, taskNum, edges, device,
                           taskTime, rep, deadline, providerAbility,
                           providerL, providerPrice, providerReliability,
                           providerEnergyCost, budget, param, omega, lam, andorInfo):
        """Create initial random solution"""
        net = network(deadlines, budgets, Rs, abilities, cost,
                      providerN, taskN, providerNum, taskNum, edges, device,
                      taskTime, rep, deadline, providerAbility,
                      providerL, providerPrice, providerReliability,
                      providerEnergyCost, budget, param, omega, lam, andorInfo)
        net.init_points()
        net.beginning = point()
        net.beginning.loc = -1
        net.beginning.children = [net.points[0]]
        net.beginning.finished = True
        net.beginning.provider = -1
        
        # Initial random assignment
        net.x = torch.zeros((self.taskNum, self.providerNum + 1))
        for i in range(self.taskNum):
            provider_choice = np.random.randint(0, self.providerNum + 1)
            net.x[i, provider_choice] = 1
        
        net.process()
        return net
    
    def random_removal(self, solution):
        """Randomly remove tasks from solution"""
        num_to_remove = max(1, int(self.destroy_degree * self.taskNum))
        removed_tasks = random.sample(range(self.taskNum), num_to_remove)
        
        # Store current assignments
        removed_assignments = {}
        for task_idx in removed_tasks:
            removed_assignments[task_idx] = solution.x[task_idx].clone()
            solution.x[task_idx] = torch.zeros(self.providerNum + 1) # Reset assignment
            solution.x[task_idx][-1] = 1 # Set to "do not execute"
            
        solution.process()
        return solution, removed_assignments
    
    def worst_removal(self, solution):
        original_objv, _ = solution.objv()
        removed_assignments = {}
        
        # Calculate contribution of each task to the objective function
        contributions = []
        for task_idx in range(self.taskNum):
            original_x = solution.x[task_idx].clone()
            
            # Temporarily remove the task
            temp_x = torch.zeros(self.providerNum + 1)
            temp_x[-1] = 1
            solution.x[task_idx] = temp_x
            solution.process()
            new_objv, _ = solution.objv()
            
            contributions.append(original_objv - new_objv)
            
            # Restore original assignment
            solution.x[task_idx] = original_x
            solution.process()

        num_to_remove = np.random.randint(1, max(2, self.taskNum // 2))
        # Sort tasks by their negative contribution (worst first)
        tasks_to_remove = np.argsort(contributions)[:num_to_remove]
        
        for task_idx in tasks_to_remove:
            removed_assignments[task_idx] = solution.x[task_idx].clone()
            solution.x[task_idx] = torch.zeros(self.providerNum + 1)
            solution.x[task_idx][-1] = 1
            
        solution.process()
        return solution, removed_assignments
    
    def shaw_removal(self, solution):
        removed_assignments = {}
        
        # Calculate similarity between tasks
        # This is a simplified similarity metric. A more complex one could be used.
        similarity = np.zeros((self.taskNum, self.taskNum))
        for i in range(self.taskNum):
            for j in range(i + 1, self.taskNum):
                sim = 1 / (1 + np.linalg.norm(solution.x[i].numpy() - solution.x[j].numpy()))
                similarity[i, j] = similarity[j, i] = sim

        num_to_remove = np.random.randint(1, max(2, self.taskNum // 2))
        
        # Pick a random task to start with
        start_task = np.random.randint(0, self.taskNum)
        tasks_to_remove = {start_task}
        
        while len(tasks_to_remove) < num_to_remove:
            # Find task most similar to the already selected tasks
            best_candidate = -1
            max_sim = -1
            for candidate in range(self.taskNum):
                if candidate not in tasks_to_remove:
                    current_sim = sum(similarity[candidate, t] for t in tasks_to_remove)
                    if current_sim > max_sim:
                        max_sim = current_sim
                        best_candidate = candidate
            if best_candidate != -1:
                tasks_to_remove.add(best_candidate)
            else:
                break # No more tasks to add
        
        for task_idx in tasks_to_remove:
            removed_assignments[task_idx] = solution.x[task_idx].clone()
            solution.x[task_idx] = torch.zeros(self.providerNum + 1)
            solution.x[task_idx][-1] = 1
            
        solution.process()
        return solution, removed_assignments
    
    def random_repair(self, solution, removed_assignments):
        """Randomly assign removed tasks"""
        for task_idx in removed_assignments.keys():
            provider_choice = np.random.randint(0, self.providerNum + 1)
            solution.x[task_idx] = torch.zeros(self.providerNum + 1)
            solution.x[task_idx][provider_choice] = 1
        
        solution.process()
        return solution
    
    def greedy_repair(self, solution, removed_assignments):
        """Assign tasks to best available provider"""
        for task_idx in removed_assignments.keys():
            best_provider = -1
            best_objv = -float('inf')
            
            # Try assigning to each provider and find the best outcome
            for provider_idx in range(self.providerNum + 1):
                original_x = solution.x[task_idx].clone()
                
                temp_x = torch.zeros(self.providerNum + 1)
                temp_x[provider_idx] = 1
                solution.x[task_idx] = temp_x
                solution.process()
                
                current_objv, _ = solution.objv()
                
                if current_objv > best_objv:
                    best_objv = current_objv
                    best_provider = provider_idx
                
                # Restore for next iteration
                solution.x[task_idx] = original_x
                solution.process()

            # Final assignment
            final_x = torch.zeros(self.providerNum + 1)
            final_x[best_provider] = 1
            solution.x[task_idx] = final_x
            solution.process()
            
        return solution
    
    def regret_repair(self, solution, removed_assignments):
        """Assign task with highest regret first"""
        
        unassigned_tasks = list(removed_assignments.keys())
        
        while unassigned_tasks:
            regrets = {}
            insert_costs = defaultdict(list)

            # Calculate insertion costs for all unassigned tasks
            for task_idx in unassigned_tasks:
                for provider_idx in range(self.providerNum + 1):
                    original_x = solution.x[task_idx].clone()
                    
                    temp_x = torch.zeros(self.providerNum + 1)
                    temp_x[provider_idx] = 1
                    solution.x[task_idx] = temp_x
                    solution.process()
                    
                    current_objv, _ = solution.objv()
                    insert_costs[task_idx].append(current_objv)
                    
                    # Restore
                    solution.x[task_idx] = original_x
                    solution.process()

                # Sort costs in descending order
                insert_costs[task_idx].sort(reverse=True)
                
                # Calculate regret (difference between best and second best)
                if len(insert_costs[task_idx]) > 1:
                    regrets[task_idx] = insert_costs[task_idx][0] - insert_costs[task_idx][1]
                else:
                    regrets[task_idx] = insert_costs[task_idx][0]

            # Choose task with highest regret
            task_to_insert = max(regrets, key=regrets.get)
            
            # Find the best provider for this task
            best_provider = np.argmax(insert_costs[task_to_insert])
            
            # Assign the task
            final_x = torch.zeros(self.providerNum + 1)
            final_x[best_provider] = 1
            solution.x[task_to_insert] = final_x
            solution.process()
            
            unassigned_tasks.remove(task_to_insert)
            
        return solution
    
    def select_operator(self, op_type):
        """Select operator using roulette wheel selection"""
        weights = self.operator_weights[op_type]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= r:
                return i
        
        return len(weights) - 1
    
    def update_operator_weights(self):
        """Update operator weights based on performance"""
        for op_type in ['destroy', 'repair']:
            for i in range(len(self.operator_weights[op_type])):
                if self.operator_counts[op_type][i] > 0:
                    avg_score = self.operator_scores[op_type][i] / self.operator_counts[op_type][i]
                    self.operator_weights[op_type][i] = max(0.1, self.operator_weights[op_type][i] * 0.8 + avg_score * 0.2)
                
                # Reset counters
                self.operator_scores[op_type][i] = 0
                self.operator_counts[op_type][i] = 0
    
    def accept_solution(self, current_objv, new_objv, temperature):
        """Simulated annealing acceptance criterion"""
        if new_objv > current_objv:
            return True
        else:
            prob = np.exp((new_objv - current_objv) / temperature).item()
            return random.random() < prob
    
    def run(self, deadlines, budgets, Rs, abilities, cost,
            providerN, taskN, providerNum, taskNum, edges, device,
            taskTime, rep, deadline, providerAbility,
            providerL, providerPrice, providerReliability,
            providerEnergyCost, budget, param, omega, lam, andorInfo):
        
        # Initialize with a valid solution
        attempts = 0
        max_attempts = 1000
        current_solution = self.initialize_solution(deadlines, budgets, Rs, abilities, cost,
                                                   providerN, taskN, providerNum, taskNum, edges, device,
                                                   taskTime, rep, deadline, providerAbility,
                                                   providerL, providerPrice, providerReliability,
                                                   providerEnergyCost, budget, param, omega, lam, andorInfo)
        current_objv, _ = current_solution.objv()
        while current_objv <= 0 and attempts < max_attempts:
            current_solution = self.initialize_solution(deadlines, budgets, Rs, abilities, cost,
                                                       providerN, taskN, providerNum, taskNum, edges, device,
                                                       taskTime, rep, deadline, providerAbility,
                                                       providerL, providerPrice, providerReliability,
                                                       providerEnergyCost, budget, param, omega, lam, andorInfo)
            current_objv, _ = current_solution.objv()
            attempts += 1

        if current_objv <= 0:
            print("Warning: Could not find a valid initial solution.")

        # Initialize solution assignment
        self.current_solution = current_solution
        best_solution = self.copy_solution(current_solution)
        self.best_objective, _ = best_solution.objv()
        self.best_points = best_solution.points
        
        temperature = self.start_temperature
        
        print(f"开始ALNS算法，共{self.iterations}轮")
        print(f"任务数：{self.taskNum}，服务提供商数：{self.providerNum}")
        print(f"初始目标函数值: {self.best_objective:.6f}")
        print("-" * 60)
        
        # Main ALNS loop
        for iteration in range(self.iterations):
            # Select operators
            destroy_idx = self.select_operator('destroy')
            repair_idx = self.select_operator('repair')
            
            # Copy current solution
            new_solution = self.copy_solution(self.current_solution)
            
            # Destroy phase
            destroy_op = self.destroy_ops[destroy_idx]
            new_solution, removed_assignments = destroy_op(new_solution)
            
            # Repair phase
            repair_op = self.repair_ops[repair_idx]
            new_solution = repair_op(new_solution, removed_assignments)
            
            # Evaluate new solution
            new_objv, _ = new_solution.objv()
            
            # Acceptance criterion
            if self.accept_solution(current_objv, new_objv, temperature):
                self.current_solution = new_solution
                current_objv = new_objv
                
                # Update operator scores (higher for better solutions)
                self.operator_scores['destroy'][destroy_idx] += 1.2 if new_objv > self.best_objective else 0.8
                self.operator_scores['repair'][repair_idx] += 1.2 if new_objv > self.best_objective else 0.8
                
                # Update best solution
                if new_objv > self.best_objective:
                    self.best_solution = new_solution
                    self.best_objective = new_objv
                    self.best_points = new_solution.points
                    print(f"第{iteration+1}轮: 发现更好解！目标函数值 = {self.best_objective:.6f}")
                    
                    # 保存当前最佳points
                    pickle.dump(self.best_points, open(f"points_{self.taskNum}_{self.providerNum}_alns.pkl", "wb"))
            else:
                # Moderate reward for trying
                self.operator_scores['destroy'][destroy_idx] += 0.3
                self.operator_scores['repair'][repair_idx] += 0.3
            
            # Update operator counts
            self.operator_counts['destroy'][destroy_idx] += 1
            self.operator_counts['repair'][repair_idx] += 1
            
            # Adaptive update
            if iteration % self.adaptive_period == 0 and iteration > 0:
                self.update_operator_weights()
                print(f"第{iteration+1}轮: 当前最佳目标函数值 = {self.best_objective:.6f}, 温度 = {temperature:.4f}")
            
            # Cool temperature
            temperature *= self.cooling_rate
        
        print(f"ALNS算法完成！最终最佳目标函数值: {self.best_objective:.6f}")
        return self.best_solution
    
    def debug_compare_with_others(self, solution_idx=0):
        """
        调试函数：显示ALNS算法的状态信息
        """
        if self.current_solution is None:
            print("当前没有解决方案")
            return
            
        solution = self.current_solution
        print("=" * 60)
        print(f"ALNS调试信息")
        print("=" * 60)
        
        # 显示任务分配情况
        print("任务分配情况:")
        for i in range(self.taskNum):
            provider_allocation = torch.argmax(solution.x[i]).item()
            if provider_allocation == self.providerNum:
                print(f"  任务 {i}: 不执行")
            else:
                print(f"  任务 {i}: 分配给服务提供商 {provider_allocation}")
        
        # 显示provider资源使用情况
        print(f"\nProvider资源使用情况:")
        for i in range(solution.providerNum):
            print(f"  Provider {i}: 已用资源={solution.provider[i]:.2f}, 总容量={solution.providerL[i]}")
        
        # 显示关键路径信息
        print(f"\n关键路径信息:")
        print(f"  Critical path: {solution.critical_path}")
        print(f"  Deadline: {solution.deadline}")
        
        # 计算成本信息
        total_cost = sum([solution.providerPrice[solution.points[i].provider][i] 
                         for i in range(self.taskNum) 
                         if solution.points[i].finished and solution.points[i].provider != -1])
        print(f"\n成本信息:")
        print(f"  总预算: {solution.budget}")
        print(f"  实际成本: {total_cost}")
        print(f"  成本比例: {total_cost/solution.budget if solution.budget > 0 else 'N/A'}")
        
        # 计算目标函数
        objv_result = solution.objv()
        print(f"\n目标函数:")
        print(f"  主目标值: {objv_result[0]}")
        print(f"  分解值: {objv_result[1]}")
        print("=" * 60)
        
        return objv_result
    
    def copy_solution(self, original):
        new_solution = self.initialize_solution(
            original.deadlines, original.budgets, original.Rs, original.abilities, original.cost,
            original.providerN, original.taskN, original.providerNum, original.taskNum, original.edges, self.device,
            original.taskTime, original.rep, original.deadline, original.providerAbility,
            original.providerL, original.providerPrice, original.providerReliability,
            original.providerEnergyCost, original.budget, original.param, original.omega, original.lam, original.andor
        )
        new_solution.x = original.x.clone()
        new_solution.process()
        return new_solution

if __name__ == "__main__":
    torch.manual_seed(4244)
    np.random.seed(4244)
    
    print("=" * 80)
    print("                    ALNS算法求解网络优化问题")
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

    alns = ALNS(taskN, providerN, device)
    
    print(f"ALNS模型初始化完成:")
    print(f"  任务数：{taskN}，服务提供商数：{providerN}")
    print(f"  迭代次数: {alns.iterations}")
    print(f"  自适应周期: {alns.adaptive_period}")
    print(f"  破坏程度: {alns.destroy_degree}")
    print()
    
    print("开始优化求解...")
    best_solution = alns.run(
        deadlines, budgets, Rs, abilities, cost,
        providerN, taskN, providerN, taskN, edges, device,
        taskTime, rep, deadline, providerAbility,
        providerL, providerPrice, providerReliability,
        providerEnergyCost, budget, param, omega, lam, andorInfo
    )
    
    # 添加调试：显示最终状态
    print("\n" + "="*50)
    print("调试：检查ALNS最终状态")
    print("="*50)
    if best_solution is not None:
        alns.debug_compare_with_others()
    
    print("=" * 80)
    print("                         优化结果")
    print("=" * 80)
    if best_solution is not None:
        print("最佳解决方案:")
        for i in range(taskN):
            provider = torch.argmax(best_solution.x[i]).item()
            if provider == providerN:
                print(f"  任务 {i} -> 不执行")
            else:
                print(f"  任务 {i} -> 服务提供商 {provider}")
        
        print(f"\n最佳适应度: {alns.best_objective:.6f}")
    else:
        print("未找到有效解决方案")
    print("=" * 80)