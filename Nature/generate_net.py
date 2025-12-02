import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json
import torch
import threading
locker = threading.Lock()

iffinded = False

# ========================== 数据定义 ==========================
def main(shared_dict, t, p, data, seed):
    tasks = range(data["taskNum"])
    providers = range(data["providerNum"])
    TB = sum(data["taskbudgets"])
    
    # ========================== 预计算 A[i,j] ==========================
    # A[i,j] = 0.2*(time_sat + cost_sat + providerRep + (providerReliability+providerEnergyCost)/2)
    #         + 0.5 * (平均(providerParam))
    A = {}
    for i in tasks:
        d = data["taskdeadlines"][i]
        b = data["taskbudgets"][i]
        for j in providers:
            t = data["taskTime"][j][i]
            c = data["providerPrice"][j][i]
            time_sat = 1 if t <= d else d / t
            cost_sat = b / c if c <= b else 0
            rep = data["providerRep"][j]
            rel_eng = (data["providerReliability"][j] + data["providerEnergyCost"][j]) / 2
            param_sat = np.mean(data["providerParam"][j])
            A[(i, j)] = 0.2 * (time_sat + cost_sat + rep + rel_eng) + 0.5 * param_sat

    # ========================== 构造模型 ==========================
    model = gp.Model("TaskAssignment_MultiObj")

    # 允许非凸（用于分段线性约束）
    model.Params.NonConvex = 2

    # 决策变量： x[i,j] = 1 表示任务 i 分配给服务商 j
    x = model.addVars(tasks, providers, vtype=GRB.BINARY, name="x")
    
    # 常量c[i] 表示第i个任务的子节点数量
    c = {i: sum(1 for edge in data["edges"] if edge[0] == i) for i in tasks}

    # 每个任务仅分配给一个服务商
    # model.addConstrs((gp.quicksum(x[i, j] for j in providers) == 1 for i in tasks), name="AssignTask")

    # 服务商资源上限约束
    # model.addConstrs(
    #     (gp.quicksum(x[i, j] * data["taskResources"][i] for i in tasks) <= data["providerL"][j]
    #      for j in providers),
    #     name="Capacity"
    # )

    # 必须完成前面的任务才能做后面的
    for i in range(data["taskNum"]):
        for j in range(i + 1, data["taskNum"]):
            if (i, j) in data["edges"]:
                model.addConstr(sum(x[i, k] for k in providers) >= sum(x[j, k] for k in providers), name="Precedence")

    # ========================== AND/OR网络约束 ==========================
    # 根据andor信息添加约束
    if "andor" in data and data["andor"] is not None:
        # 创建辅助变量表示任务是否被执行
        y = model.addVars(tasks, vtype=GRB.BINARY, name="y")
        
        # 连接x和y变量：如果任务i被分配给任何服务商，则y[i] = 1
        model.addConstrs((y[i] == gp.quicksum(x[i, j] for j in providers) for i in tasks), name="TaskExecuted")
        
        # 构建任务之间的父子关系
        task_children = {}  # task_id -> [child_task_ids]
        for i, (parent, child) in enumerate(data["edges"]):
            if parent not in task_children:
                task_children[parent] = []
            task_children[parent].append(child)
        
        # 添加AND/OR约束
        for task_id in range(data["taskNum"]):
            if task_id in task_children and data["andor"][task_id] is not None:
                children = task_children[task_id]
                if len(children) > 0:
                    if data["andor"][task_id] == "and":
                        # AND约束：如果父任务执行，则所有子任务都必须执行
                        model.addConstr(
                            y[task_id] == c[task_id],
                            name=f"AND_{task_id}_all"
                        )
                    
                    elif data["andor"][task_id] == "or":
                        # OR约束：如果父任务执行，则至少一个子任务必须执行
                        if len(children) > 1:
                            model.addConstr(
                                gp.quicksum(y[child] for child in children) == 1,
                                name=f"OR_{task_id}_at_most_one"
                            )
    else:
        # 如果没有andor信息，创建y变量但不添加特殊约束
        y = model.addVars(tasks, vtype=GRB.BINARY, name="y")
        model.addConstrs((y[i] == gp.quicksum(x[i, j] for j in providers) for i in tasks), name="TaskExecuted")

    # ========================== 定义满意度表达式 ==========================
    # 分配部分满意度
    assign_sat_expr = gp.quicksum(A[(i, j)] * x[i, j] for i in tasks for j in providers)

    # 服务商剩余资源满意度：0.2 * (1 - (已分配资源 / providerL))
    res_sat_expr = gp.quicksum(
        0.2 * (1 - (gp.quicksum(x[i, j] * data["taskResources"][i] for i in tasks) / data["providerL"][j]))
        for j in providers
    )

    total_satisfaction = assign_sat_expr + res_sat_expr

    # ========================== 时间成本满意度 ==========================
    # 总任务完成时间与总成本
    T_expr = gp.quicksum(x[i, j] * data["taskTime"][j][i] for i in tasks for j in providers)
    C_expr = gp.quicksum(x[i, j] * data["providerPrice"][j][i] for i in tasks for j in providers)

    # 新增变量 T，并添加约束使 T == T_expr，方便分段线性近似
    T = model.addVar(name="T")
    model.addConstr(T == T_expr, name="TotalTimeDef")

    # 使用分段线性近似 st = exp(-0.1*(deadline - T))（当 T<=deadline）
    f = model.addVar(lb=0, ub=1, name="f")
    T_min = sum(min(data["taskTime"][j][i] for j in providers) for i in tasks)
    T_max = data["deadline"]
    # 确保断点是排序的
    breakpoints_T = sorted([T_min, (T_min+T_max)/3, (T_min+T_max)*2/3, T_max])
    breakpoints_f = [np.exp(-0.1*(data["deadline"]-t)) for t in breakpoints_T]
    model.addGenConstrPWL(T, f, breakpoints_T, breakpoints_f, name="PWL_st")

    # 成本折扣
    sc_expr = (TB - C_expr) / TB

    satisfaction_tc = (f + sc_expr) / 2

    # ========================== 设置多目标 ==========================
    # 目标1: 最大化 total_satisfaction
    model.ModelSense = GRB.MAXIMIZE
    model.setObjectiveN(total_satisfaction, index=0, priority=2, name="TotalSatisfaction")

    # 目标2: 最大化 satisfaction_tc
    model.setObjectiveN(satisfaction_tc, index=1, priority=1, name="TimeCostSatisfaction")

    
    # 设置求解器参数
    model.Params.OutputFlag = 0  # 禁止输出
    model.Params.TimeLimit = 300  # 设置时间限制为5分钟
    model.Params.MIPGap = 0.01   # 设置相对误差容忍度
    model.Params.Presolve = 2    # 启用预处理
    model.Params.Cuts = 2        # 启用切割平面
    
    # ========================== 求解 ==========================
    model.optimize()

    # ========================== 输出结果 ==========================
    print(f"求解状态: {model.status}")
    if model.status == GRB.Status.OPTIMAL:
        print("找到最优解！")
        solution_found = True

        import os
        if not os.path.exists(f"{task_num}_{provider_num}"):
            os.makedirs(f"{task_num}_{provider_num}")
        with open(f"{task_num}_{provider_num}/data_{task_num}_{provider_num}_{seed}.json", "w") as f:
            json.dump(data, f, indent=4)
    elif model.status == GRB.Status.SUBOPTIMAL:
        print("找到次优解")
        solution_found = True

        import os
        if not os.path.exists(f"{task_num}_{provider_num}"):
            os.makedirs(f"{task_num}_{provider_num}")
        with open(f"{task_num}_{provider_num}/data_{task_num}_{provider_num}_{seed}.json", "w") as f:
            json.dump(data, f, indent=4)
    elif model.status == GRB.Status.INFEASIBLE:
        print("问题无可行解")
        # 尝试计算不可行解的信息
        model.computeIIS()
        # print("不可行约束集合:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f"约束 {c.constrName} 导致不可行")
        solution_found = False
    elif model.status == GRB.Status.UNBOUNDED:
        print("问题无界")
        solution_found = False
    elif model.status == GRB.Status.TIME_LIMIT:
        print("超时，但可能找到了可行解")
        solution_found = model.SolCount > 0
    else:
        print(f"其他状态: {model.status}")
        solution_found = False
    
    if solution_found:

        with locker:
            try:
                shared_dict['dat']=data
                shared_dict['iffinded'] = True

            except Exception as e:
                print(f"保存数据时出错: {e}")
        print("最优任务分配方案：")
        for i in tasks:
            for j in providers:
                if x[i, j].X > 0.5:
                    print(f"任务 {i} 分配给服务商 {j}")
        print("\n目标函数值：")
        print(f"Total Satisfaction = {total_satisfaction.getValue()}")
        print(f"TimeCost Satisfaction = {satisfaction_tc.getValue()}")
        print(f"Total Time = {T_expr.getValue()}, Total Cost = {C_expr.getValue()}")
        return True
    else:
        print("未找到可用解")
        return False



from xtmqn import point

import numpy as np

finded = []
to_be_finded = []

class netw:
    def __init__(self):
        self.points = []
    
    def add_point(self):
        p = point()
        self.points.append(p)
        p.loc = len(self.points)-1
        p.parents = []
        return p
    
    def add_edge(self, p1, p2):
        p1.children.append(p2)
        p2.parents.append(p1)
    
    
    
    def criticalpath(self, p, time): # calculate the critical path
        pT = 0
        p.cobjv=True
        if p.loc==12:
            pass
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
            
    def set_and_or_relation(self, p):
        if len(p.children) == 0:
            return
        
        # 如果只有一个子节点，直接设置为AND（AND和OR没区别）
        if len(p.children) == 1:
            p.type = "and"
            self.andor[p.loc] = "and"
        else:
            # 多个子节点时，尝试不同的AND/OR设置
            possible_types = ["and", "or", None]
            
            # 测试每种类型的可行性
            self.andor[p.loc] = np.random.choice(possible_types)

        # 递归设置子节点
        for c in p.children:
            self.set_and_or_relation(c)
        return
    
def dfs(p, finded, to_be_finded):
    num = 0
    finded.append(p)
    if p.children == []:
        return 1
    for c in p.children:
        num += dfs(c, finded, to_be_finded)
        
    return num+1

def find_end(p):
    if len(p.children) == 0:
        return p, 0
    for c in p.children:
        to_be_finded.append(c)
        return find_end(c)[0], find_end(c)[1]+1

def graph_generation(point_num):
    net = netw()
    for i in range(point_num):
        t = net.add_point()
    
    edges = []
    
    global finded, to_be_finded
    finded = []
    to_be_finded = []
    p = net.points[0]
    
    while dfs(p, finded, to_be_finded) < point_num:
        t1 = np.random.choice(finded)
        t2 = np.random.choice(net.points)
        while t2 in finded:
            t2 = np.random.choice(net.points)
        if t1 == t2:
            continue
        finded.append(t2)
        net.points.pop(net.points.index(t2))
        net.add_edge(t1, t2)
        edges.append((t1.loc, t2.loc))
        finded = []
        to_be_finded = []
    end_point, l = find_end(net.points[0])
    net.andor = [None for i in range(point_num)]
    net.set_and_or_relation(net.points[0])
    return edges, end_point, l, net



def gmain(task_num=90, provider_num=45, seed=None):
    # 使用不同的随机种子，如果未指定则使用当前时间
    if seed is None:
        import time
        seed = int(time.time() * 1000) % 1000000
    np.random.seed(seed)
    # print(f"使用随机种子: {seed}")
    
    edges, _, _ , n= graph_generation(task_num)
    # n.critical_path()
    
    andorInfo = n.andor
    
    # 增加数据范围的合理性
    deadlines = np.random.randint(20, 150, task_num)  # 增加截止时间范围
    budgets = np.random.randint(100, 200, task_num)   # 增加预算范围
    Rs = np.random.randint(1, 5, task_num)            # 降低资源需求
    abilities = np.random.randint(10, 100, task_num)
    providerRep = np.random.randint(500, 1000, provider_num)/1000  # 提高声誉基准
    
    deadline = 300  # 增加总体截止时间
    providerAbility = np.random.randint(10, 100, provider_num)
    providerL = np.random.randint(100, 800, provider_num)  # 增加资源限制
    taskTime = np.random.randint(1, 8, (provider_num, task_num))   # 减少任务时间
    taskCost = np.random.randint(20, 70, task_num)
    
    providerPrice = np.random.randint(10, 60, (provider_num, task_num))  # 降低价格
    providerTaskNum = np.random.randint(1, 10, provider_num)
    
    # 修复：使用0-1之间的随机数而不是只有0和1
    providerReliability = np.random.random(provider_num)  # 0到1之间的随机数
    providerEnergyCost = np.random.randint(0, 1000, provider_num)/1000
    providerParam = np.random.randint(0, 1000, (provider_num,4))/1000
    
    budget = budgets.sum()
    
    det = {
        "taskNum": task_num, # Number of tasks
        "providerNum": provider_num, # Number of providers
        "providerRep": providerRep.tolist(), # Reputation of providers
        "providerAbility": providerAbility.tolist(), # Ability of providers
        "providerPrice": providerPrice.tolist(), # Price of providers
        "providerL": providerL.tolist(), # Resource Limitation of providers
        "providerReliability": providerReliability.tolist(), # Reliability of providers
        "providerEnergyCost": providerEnergyCost.tolist(), # Energy cost of providers
        "providerParam": providerParam.tolist(),
        "budget": int(budget), # Budget of the whole task network
        # '''
        # First column: Pc
        # Second column: Pt
        # Third column: Ce
        # Fourth column: Ig
        # '''
        "taskCost": taskCost.tolist(), # Cost of tasks
        "taskdeadlines": deadlines.tolist(), # Deadline of tasks
        "taskbudgets": budgets.tolist(), # Budget of tasks
        "taskResources": Rs.tolist(), # Resource occupied by each task
        "taskabilities": abilities.tolist(), # Ability requirements of each task
        "taskTime": taskTime.tolist(), # Time required for each provider to complete each task
        "edges": edges, # Edges of the task network
        "deadline": deadline, # Deadline of the whole task network
        "andor": andorInfo, # And/Or relation of the task network
    }
    
    return det

def try_generate_data(shared_dict, task_num=90, provider_num=45, minseed=0, maxseed=100):
    
    max_attempts = 1000  # 限制尝试次数
    attempt = 0
    
    for seedd in range(minseed, maxseed):
        # if shared_dict['iffinded']:
        #     print("已找到可求解的数据，停止尝试。")
        #     break
        attempt += 1
        print(f"尝试第 {attempt} 次生成可求解的数据...")
        
        # 每次尝试使用不同的随机种子
        data = gmain(task_num, provider_num, seed=seedd)

        if main(shared_dict, task_num, provider_num, data, seed=seedd):
            print("成功找到可求解的数据！")
dat = []
if __name__ == "__main__":
    from multiprocessing import Pool, Manager
    import time
    
    task_num = 90
    provider_num = 45
    min_seed = 0
    max_seed = 2**32
    thread_num = 32
    
    
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['iffinded'] = False
    shared_dict['dat'] = []
    try_generate_data(shared_dict=shared_dict, task_num=task_num, provider_num=provider_num, minseed=min_seed, maxseed=max_seed)

    pool = Pool(processes=thread_num)
    
    # 提交所有任务
    results = []
    for i in range(thread_num):
        start_seed = i * (max_seed - min_seed) // thread_num + min_seed
        end_seed = (i + 1) * (max_seed - min_seed) // thread_num + min_seed
        result = pool.apply_async(try_generate_data, args=(shared_dict, task_num, provider_num, start_seed, end_seed))
        # results.append(result)
    
    # 等待直到找到解或所有进程完成
    # 检查是否有进程完成
    completed = [r.ready() for r in results]
    if all(completed):
        print("所有进程都已完成，但未找到解")
        
    if shared_dict['iffinded']:
        print("找到解，终止其他进程...")
        pool.terminate()  # 终止所有进程
    from time import sleep
    sleep(1)
    pool.close()
    pool.join()
    print(f"最终结果: {shared_dict['iffinded']}")