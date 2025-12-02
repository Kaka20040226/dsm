import torch
from torch import nn
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

class CustomGCNLayer(nn.Module):
    """自定义图卷积层"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x, adj_matrix):
        """
        x: 节点特征 [num_nodes, input_dim]
        adj_matrix: 邻接矩阵 [num_nodes, num_nodes]
        """
        # 添加自环
        adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0)).to(adj_matrix.device)
        
        # 度矩阵的逆平方根
        degree = torch.sum(adj_matrix, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        degree_matrix = torch.diag(degree_inv_sqrt)
        
        # 归一化邻接矩阵
        normalized_adj = torch.mm(torch.mm(degree_matrix, adj_matrix), degree_matrix)
        
        # 图卷积操作
        x = self.linear(x)
        x = torch.mm(normalized_adj, x)
        x = self.activation(x)
        
        return x

class CustomGraphConvLayer(nn.Module):
    """自定义图卷积层（适用于二分图）"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_self = nn.Linear(input_dim, output_dim)
        self.linear_neighbor = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x, edge_index):
        """
        x: 节点特征 [num_nodes, input_dim]
        edge_index: 边索引 [2, num_edges]
        """
        num_nodes = x.size(0)
        
        # 自连接
        x_self = self.linear_self(x)
        
        # 邻居聚合
        x_neighbor = torch.zeros_like(x_self)
        
        if edge_index.size(1) > 0:  # 如果有边
            for i in range(num_nodes):
                # 找到节点i的所有邻居
                neighbors = edge_index[1][edge_index[0] == i]
                if len(neighbors) > 0:
                    # 聚合邻居特征
                    neighbor_features = x[neighbors]
                    aggregated = torch.mean(neighbor_features, dim=0)
                    x_neighbor[i] = self.linear_neighbor(aggregated)
        
        # 组合自连接和邻居信息
        output = x_self + x_neighbor
        output = self.activation(output)
        
        return output

class GraphNeuralNetwork(nn.Module):
    """自定义图神经网络模块"""
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=2, gnn_type='GCN'):
        super().__init__()
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        # 选择GNN层类型
        if gnn_type == 'GCN':
            LayerClass = CustomGCNLayer
        elif gnn_type == 'GraphConv':
            LayerClass = CustomGraphConvLayer
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # 构建GNN层
        self.gnn_layers = nn.ModuleList()
        
        # 第一层
        self.gnn_layers.append(LayerClass(node_features, hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.gnn_layers.append(LayerClass(hidden_dim, hidden_dim))
        
        # 最后一层
        if num_layers > 1:
            self.gnn_layers.append(LayerClass(hidden_dim, output_dim))
        else:
            # 如果只有一层，直接输出
            self.gnn_layers[0] = LayerClass(node_features, output_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, graph_structure):
        """
        前向传播
        x: 节点特征 [num_nodes, node_features]
        graph_structure: 图结构（邻接矩阵或边索引）
        """
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == 'GCN':
                x = layer(x, graph_structure)  # 使用邻接矩阵
            else:
                x = layer(x, graph_structure)  # 使用边索引
            
            if i < len(self.gnn_layers) - 1:  # 最后一层不使用dropout
                x = self.dropout(x)
        
        return x


class GNNEnhancedNet(nn.Module):
    def __init__(self, input_dim, output_dim, task_num, provider_num, net, device="cpu", D=None): 
        '''
            基于no_attention但加入GNN结构的网络
            input_dim = provider_num
            output_dim = task_num*provider_num
        '''
        super().__init__()
        
        self.task_num = task_num
        self.provider_num = provider_num
        self.device = device
        
        # GNN配置
        self.gnn_hidden_dim = 64
        self.gnn_output_dim = 32
        self.node_feature_dim = 16  # 每个节点的特征维度
        
        # 节点特征投影层
        self.task_feature_proj = nn.Linear(provider_num, self.node_feature_dim).to(device)
        self.provider_feature_proj = nn.Linear(task_num, self.node_feature_dim).to(device)
        
        # 任务图GNN (任务之间的依赖关系)
        self.task_gnn = GraphNeuralNetwork(
            node_features=self.node_feature_dim,
            hidden_dim=self.gnn_hidden_dim,
            output_dim=self.gnn_output_dim,
            num_layers=3,
            gnn_type='GCN'
        ).to(device)
        
        # 二分图GNN (任务-提供者匹配关系)
        self.bipartite_gnn = GraphNeuralNetwork(
            node_features=self.node_feature_dim,
            hidden_dim=self.gnn_hidden_dim,
            output_dim=self.gnn_output_dim,
            num_layers=2,
            gnn_type='GraphConv'
        ).to(device)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.gnn_output_dim * 2, provider_num * task_num),
            nn.ReLU(),
            nn.Linear(provider_num * task_num, provider_num * task_num),
            nn.ReLU()
        ).to(device)
        
        # 输入处理器 (替代注意力机制)
        self.input_processor = nn.Sequential(
            nn.Linear(input_dim * task_num, 2 * input_dim * task_num),
            nn.ReLU(),
            nn.Linear(2 * input_dim * task_num, input_dim * task_num),
            nn.ReLU()
        ).to(device)
        
        # first linear layer (保留原始结构)
        self.linear = nn.Sequential(
            nn.Linear(provider_num*task_num*2, 2*provider_num*task_num),  # 输入维度翻倍，因为加入了GNN特征
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num, 2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num, provider_num*task_num),
            nn.ReLU()
        ).to(device)
        
        # degorate layer (保留原始结构)
        self.degorate = nn.Sequential(
            nn.Linear(provider_num*task_num, 2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num, 2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num, provider_num),
        ).to(device)
        
        self.derograteT = nn.Sequential(
            nn.Linear(provider_num*task_num*2, 2*provider_num*task_num),  # 输入维度翻倍
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num, 2*provider_num*task_num),
            nn.ReLU(),
            nn.Linear(2*provider_num*task_num, task_num),
        ).to(device)
        
        # decision layer (保留原始结构)
        self.decision = nn.Sequential(
            nn.Linear(provider_num+task_num*2, (provider_num+task_num)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+task_num)*3//2, (provider_num+task_num)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+task_num)*3//2, (provider_num+task_num)*3//2),
            nn.ReLU(),
            nn.Linear((provider_num+task_num)*3//2, provider_num),
        ).to(device)
        
        # final linear layer with current position added (保留原始结构)
        self.final = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
        ).to(device)
        
        # lstm components (保留原始LSTM结构)
        self.hidden_state = [torch.zeros(output_dim).unsqueeze(0).to(device)]
        self.cell_state = [torch.zeros(output_dim).unsqueeze(0).to(device)]
        
        self.forgetDoor = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        ).to(device)
        
        self.inputDoor1 = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        ).to(device)
        
        self.inputDoor2 = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Tanh()
        ).to(device)
        
        self.outputDoor = nn.Sequential(
            nn.Linear(provider_num+output_dim, output_dim),
            nn.Sigmoid()
        ).to(device)

        # task-specific layers (保留原始结构)
        self.taskL = []
        for i in range(task_num):
            new_net = nn.Sequential(
                nn.Linear(provider_num+output_dim, provider_num+1).to(device),
            )
            self.taskL.append(new_net)
        
        self.criteria = nn.MSELoss().to(device)
        
        # 保存依赖矩阵和图结构
        if D is not None:
            self.D = D.to(device)
            # 构建任务依赖图的邻接矩阵
            self.task_adj_matrix = self._build_task_adjacency_matrix(D).to(device)
        else:
            self.D = None
            self.task_adj_matrix = None
        
        # 初始化权重
        self._init_weights()
        
    def _build_task_adjacency_matrix(self, D):
        """根据依赖矩阵构建任务图的邻接矩阵"""
        adj_matrix = D.clone().float()
        # 使任务图无向（如果需要的话）
        adj_matrix = adj_matrix + adj_matrix.t()
        adj_matrix = torch.clamp(adj_matrix, 0, 1)  # 确保值在0-1之间
        return adj_matrix
    
    def _build_bipartite_edges(self, task_num, provider_num):
        """构建任务-提供者二分图的边索引"""
        edges = []
        # 为每个任务连接所有提供者
        for task_id in range(task_num):
            for provider_id in range(provider_num):
                # 任务节点索引: 0 到 task_num-1
                # 提供者节点索引: task_num 到 task_num+provider_num-1
                edges.append([task_id, task_num + provider_id])
                edges.append([task_num + provider_id, task_id])  # 双向边
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            # 如果没有边，创建空的边索引
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return edge_index.to(self.device)
    
    def _init_weights(self):
        """初始化所有网络层的权重"""
        modules_to_init = [
            self.task_feature_proj, self.provider_feature_proj, self.feature_fusion,
            self.input_processor, self.linear, self.degorate, self.derograteT,
            self.decision, self.final, self.forgetDoor, self.inputDoor1, 
            self.inputDoor2, self.outputDoor
        ]
        
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        
        # 初始化taskL
        for task_net in self.taskL:
            for m in task_net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def process_with_gnn(self, x):
        """使用GNN处理输入特征"""
        # 1. 处理任务依赖关系 (任务图)
        if self.task_adj_matrix is not None:
            # 为每个任务创建特征（基于其与提供者的匹配状态）
            task_features = self.task_feature_proj(x)  # [task_num, node_feature_dim]
            
            # 使用任务GNN处理任务间依赖
            task_gnn_output = self.task_gnn(task_features, self.task_adj_matrix)  # [task_num, gnn_output_dim]
        else:
            # 如果没有依赖关系，直接投影
            task_features = self.task_feature_proj(x)
            task_gnn_output = torch.zeros(self.task_num, self.gnn_output_dim).to(self.device)
        
        # 2. 处理任务-提供者二分图
        # 创建提供者特征（基于其与任务的匹配状态）
        provider_features = self.provider_feature_proj(x.t())  # [provider_num, node_feature_dim]
        
        # 构建二分图节点特征 (任务节点 + 提供者节点)
        bipartite_node_features = torch.cat([task_features, provider_features], dim=0)  # [task_num+provider_num, node_feature_dim]
        
        # 构建二分图边
        bipartite_edge_index = self._build_bipartite_edges(self.task_num, self.provider_num)
        
        # 使用二分图GNN
        bipartite_gnn_output = self.bipartite_gnn(bipartite_node_features, bipartite_edge_index)  # [task_num+provider_num, gnn_output_dim]
        
        # 分离任务和提供者的GNN输出
        task_bipartite_output = bipartite_gnn_output[:self.task_num]  # [task_num, gnn_output_dim]
        provider_bipartite_output = bipartite_gnn_output[self.task_num:]  # [provider_num, gnn_output_dim]
        
        # 3. 融合特征
        # 将任务的两种GNN输出拼接
        fused_task_features = torch.cat([task_gnn_output, task_bipartite_output], dim=1)  # [task_num, gnn_output_dim*2]
        
        # 重构为矩阵形式并展平
        gnn_enhanced_features = self.feature_fusion(fused_task_features.flatten())  # [provider_num*task_num]
        
        return gnn_enhanced_features.unsqueeze(0)  # [1, provider_num*task_num]
    
    def forward(self, x, pos, p, prehc, AP):
        """
        前向传播 - 加入GNN处理
        x: 输入状态矩阵
        pos: 位置编码
        p: 当前点
        prehc: 前一个隐藏状态索引
        AP: 可用任务列表
        """
        # 1. 使用GNN处理输入
        gnn_features = self.process_with_gnn(x)
        
        # 2. 传统处理流程
        x_flat = x.flatten()
        x_processed = self.input_processor(x_flat.unsqueeze(0))
        
        # 3. 融合GNN特征和传统特征
        combined_features = torch.cat([x_processed, gnn_features], dim=1)
        
        # 4. 通过线性层
        x = self.linear(combined_features)
        
        # 5. 提取provider和task特征
        x_p = self.degorate(x)
        x_t = self.derograteT(combined_features).unsqueeze(0)
        
        # 6. 决策层
        x = self.decision(torch.cat([x_t, x_p, pos], dim=1))
        
        # 7. LSTM部分 (保持原始结构)
        self.hidden_state.append(None)
        self.cell_state.append(None)
        
        remember = self.forgetDoor(torch.cat([x, self.hidden_state[prehc]], dim=1))
        input1 = self.inputDoor1(torch.cat([x, self.hidden_state[prehc]], dim=1))
        input2 = self.inputDoor2(torch.cat([x, self.hidden_state[prehc]], dim=1))
        output = self.outputDoor(torch.cat([x, self.hidden_state[prehc]], dim=1))
        
        self.cell_state[-1] = (remember * self.cell_state[prehc] + input1 * input2).clone().detach()
        out = output * torch.tanh(self.cell_state[-1]).clone().detach()
        self.hidden_state[-1] = out.clone().detach()
        hcV = len(self.hidden_state) - 1
        
        # 8. 任务特定输出
        ansV = torch.zeros((1, (self.provider_num+1)*self.task_num)).to(x.device)
        for ap in AP:
            ans = self.taskL[ap](torch.cat([x, out], dim=1))
            ansV[0][ap*(self.provider_num+1):(ap+1)*(self.provider_num+1)] = ans[0]
        
        return ansV, hcV
    
    def choose_action(self, y):
        """动作选择 (保持原始逻辑)"""
        args = y[0].argmax()
        sense = "And"
        arg = y[self.provider_num * self.task_num:self.provider_num * self.task_num*2]
        return sense, arg
    
    def training_step(self, x, pos, y):
        """训练步骤"""
        y_hat = self(x, pos)
        loss = self.criteria(y_hat, y)
        return loss
    
    def configure_optimizers(self, optimizer, scheduler):
        """配置优化器"""
        self.optimizer = optimizer
        self.scheduler = scheduler
        return [self.optimizer], [self.scheduler]


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
        self.hc = 0  # 隐藏状态索引


class network:
    def __init__(self, deadlines, budgets, Rs, abilities, cost,
                 input_dim, output_dim, provider_num, task_num, edges, device="cpu"):
        self.name = "gnnEnhancedDQN"
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
        self.D = torch.zeros((task_num, task_num)).to(device)  # Dependency matrix
        for e in edges:
            self.add_edge(self.points[e[0]], self.points[e[1]], 1)
            self.D[e[0]][e[1]] = 1
        self.edges = edges
        
        # 初始化GNN增强的网络模型
        self.model = GNNEnhancedNet(input_dim, output_dim, task_num, provider_num, self, device, self.D).to(device)
        self.provider_num = provider_num
        self.task_num = task_num
        self.constrains = []
        self.device = device
        
        # 优化器配置
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        self.model.configure_optimizers(optimizer, scheduler)
        
        # 其他参数设置 (保持与原始一致)
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
        self.edges = edges
        self.M = 1000000

    # 以下方法保持与no_attention_dqn相同
    def add_point(self, T):
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
        y, hc = self.model.forward(x, point, p, prehc, self.available_P(p))
        return y, hc
    
    def proceed(self, x, point):
        """执行一步决策 (保持原始逻辑)"""
        if point.loc != -1:
            pos = torch.zeros(self.task_num)
            pos = pos.unsqueeze(0).to(self.device)
        else:
            pos = torch.zeros(self.task_num).unsqueeze(0).to(self.device)
        
        y, hcV = self.forward(x, pos, point, point.hc)
        sense, args = self.model.choose_action(y)
        
        provider = []
        task = []
        points = []
        choose_V = []
        
        for c in point.children:
            if c.finished:
                continue
            
            maxV = -1
            provid = 0
            for i in range(self.provider_num+1):
                if maxV < y[0][c.loc*self.provider_num+i]:
                    maxV = y[0][c.loc*self.provider_num+i]
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
            c.provider = provider[-1] if provider else provid
            c.sense = "And"
            c.hc = hcV
            
            choose_V.append(y[0][(self.provider_num+1) * task[-1] + (provider[-1] if provider else provid)].clone())
        
        if not provider and not task:
            return None, None, None, None, None
        
        return provider, task, points, sense, choose_V

    def train_proceed(self, x, point, epislon=0.9):
        """训练时的决策过程 (保持原始逻辑)"""
        pos = torch.zeros(self.task_num).to(self.device).unsqueeze(0)
        if point.loc != -1:
            pos[0][point.loc] = 1
        
        y, hcV = self.forward(x, pos, point, point.hc)
        sense, _ = self.model.choose_action(y)
        
        if np.random.rand() < epislon:
            orAnd = np.random.choice(["Or", "And"])
        else:
            orAnd = sense
        
        tasks = []
        providers = []
        choose = []
        points = []
        
        for c in point.children:
            pros = list(range(self.provider_num+1))
            if pros == []:
                continue
            
            provider = np.random.choice(pros)
            if int(provider) < int(self.provider_num):
                self.provider[provider] += int(c.R)
                self.providerW[provider] += 1
                c.finished = True
            
            c.provider = provider
            c.sense = "And"
            c.hc = hcV
            tasks.append(c.loc)
            providers.append(provider)
            points.append(c)
            choose.append(y[0][c.loc*self.provider_num+provider])
        
        if tasks == []:
            return None, None, None, None, None
        
        return providers, tasks, points, orAnd, choose

    # 其他方法与no_attention_dqn完全相同，这里省略以节省空间
    # 包括: search, search_train, criticalpath, traceCriticalPath, objv, training_step,
    # add_punishment_to_objv, add_edge, init_points, init_net 等

    def search(self, startpoint):
        """搜索整个网络 (保持原始逻辑)"""
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

    def search_train(self, startpoint, epislon=0.5):
        """训练搜索 (保持原始逻辑)"""
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
            
            provider, task, point, sense, choose = self.train_proceed(x_list[p.L], p, epislon=epislon)
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

    def criticalpath(self, p, time):
        pT = 0
        if not p.finished:
            self.critical_path = max(self.critical_path, time)
            return
        
        if p.loc != -1:
            if self.paths[p.loc] > time:
                self.paths[p.loc] = time
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

    def traceCriticalPath(self, p, time, critical_path):
        pT = 0
        
        if p.loc != -1:
            if self.paths[p.loc] < time:
                self.paths[p.loc] = time
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

    def objv(self, x):
        """计算目标值 (保持原始逻辑)"""
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

    def training_step(self, startpoint, state=0):
        """训练步骤 (保持原始逻辑)"""
        epislon = np.exp(-state/1000000)
        for p in self.points:
            p.finished = False
            p.provider = -1
            p.sense = "And"
            p.time = None
        
        calcuated, x = self.search_train(startpoint, epislon)
        objv, obj = self.objv(x[-1])
        
        startpoint.finished = True
        startpoint.sense = "Or"
        loss_sum = 0
        
        self.model.optimizer.zero_grad()
        for cs in calcuated:
            l = torch.tensor(len(cs)).to(self.device)
            for c in cs:
                single_loss = torch.tensor(0).to(self.device)
                single_loss = self.model.criteria(c.to(self.device), objv.to(self.device))
                loss_sum += single_loss
        
        if loss_sum == 0:
            return torch.tensor(0.0).to(self.device), obj
        
        loss_sum.backward(retain_graph=True)
        self.model.optimizer.step()
        self.model.scheduler.step()

        return loss_sum, obj

    def add_restrition(self, new_constrainf):
        self.constrains.append(new_constrainf)
        return

    def add_punishment_to_objv(self, objv, x):
        x = torch.tensor(x).to(self.device)
        cost = 0
        for p in self.points:
            if not p.finished:
                continue
            if p.budget < self.providerPrice[p.provider][p.loc] and self.paths[p.loc] > self.deadline:
                print("Budget Exceeded;", end=" ")
                objv -= self.M/len(self.points)
            if self.critical_path > self.deadline:
                print("Deadline Exceeded;", end=" ")
                objv -= self.M * (self.critical_path-self.deadline)/len(self.points)
            cost += self.providerPrice[p.provider][p.loc]
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


if __name__ == '__main__':
    # 测试代码
    torch.manual_seed(4244)
    np.random.seed(4244)
    
    print("GNN-Enhanced DQN implementation ready!")
    print("Network architecture: Based on no-attention DQN but with GNN structure")
    print("Features:")
    print("- Added: Task dependency GNN (processes task relationships)")
    print("- Added: Task-Provider bipartite GNN (processes matching relationships)")
    print("- Added: Feature fusion layers")
    print("- Kept: LSTM, decision layers, task-specific outputs from original")
    print("- GNN Types: GCN for task dependencies, GraphConv for bipartite matching")
