import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json

# ========================== 数据定义 ==========================
def main(t,p):
    data = json.load(open(f"data_{t}_{p}.json", "r"))
    shared_dict = {}
    tasks = range(data["taskNum"])
    providers = range(data["providerNum"])
    TB = sum(data["taskbudgets"])
    
    # ========================== 预计算 A[i,j] ==========================
    # 根据GA/PPO/XTMQN的objv函数修正满意度计算
    # omega权重参数
    omega = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    A = {}
    for i in tasks:
        d = data["taskdeadlines"][i]
        b = data["taskbudgets"][i]
        for j in providers:
            t = data["taskTime"][j][i]
            c = data["providerPrice"][j][i]
            
            # 时间满意度 (sT)
            time_sat = 1 if d > t else d / t
            
            # 成本满意度 (sP)  
            cost_sat = b / c if b > c else 0
            
            # 资源满意度 - 这里先用平均值，后面会在约束中处理
            resource_sat = 1.0  # 占位符，实际值在res_sat_expr中计算
            
            # 信誉满意度
            rep_sat = data["providerRep"][j]
            
            # 可靠性和能耗满意度
            rel_eng_sat = (data["providerReliability"][j] + data["providerEnergyCost"][j]) / 2
            
            # 参数满意度（对应原代码中的param部分）
            param_sat = np.mean(data["providerParam"][j])
            
            # 按照原算法的加权方式计算
            A[(i, j)] = (omega[0] * time_sat + omega[1] * cost_sat + omega[2] * resource_sat +
                        omega[3] * rep_sat + omega[4] * rel_eng_sat + param_sat)

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
    # 任务分配满意度（对应原算法中对每个完成任务的满意度计算）
    assign_sat_expr = gp.quicksum(A[(i, j)] * x[i, j] for i in tasks for j in providers)

    # 服务商资源满意度：omega[2] * (providerL - used_resource) / providerL
    res_sat_expr = gp.quicksum(
        omega[2] * (data["providerL"][j] - gp.quicksum(x[i, j] * data["taskResources"][i] for i in tasks)) / data["providerL"][j]
        for j in providers
    )

    # 任务级别的满意度（除了已经在A中计算的部分）
    task_satisfaction = assign_sat_expr + res_sat_expr

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

    # 成本满意度：(budget - cost) / budget（对应原算法中的主要满意度）
    sc_expr = (TB - C_expr) / TB

    # 时间成本组合满意度（对应原算法中的sat1部分）
    satisfaction_tc = (sc_expr + f) / 2
    
    # 总满意度组合（对应原算法最终的satisfaction）
    total_satisfaction = satisfaction_tc + task_satisfaction

    # ========================== 设置多目标 ==========================
    # 目标1: 最大化 total_satisfaction
    model.ModelSense = GRB.MAXIMIZE
    model.setObjectiveN(total_satisfaction, index=0, priority=2, name="TotalSatisfaction")

    # 目标2: 最大化 satisfaction_tc
    model.setObjectiveN(satisfaction_tc, index=1, priority=1, name="TimeCostSatisfaction")

    
    # 设置求解器参数
    model.Params.OutputFlag = 1  # 启用输出以查看求解过程
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
    elif model.status == GRB.Status.SUBOPTIMAL:
        print("找到次优解")
        solution_found = True
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
if __name__ == "__main__":
    t = 5
    p = 3
    main(t, p)