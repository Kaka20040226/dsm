import torch

class netAttention(torch.nn.Module):
    def __init__(self, net, pd, size, kernel_size=3, find_time = 3, device = "cpu"):
        # make the forward() being used when using netAttention()()
        # super() is used to call the constructor of the parent class
        # torch.nn.Module is the parent class
        '''
        q: query
        k: key
        v: value
        net: network of the model
        size: size of kv matrix, assumed to be a square matrix
        '''
        super(netAttention, self).__init__()
        # define variable that can be optimized
        self.device = device
        self.netParm = torch.nn.Parameter(torch.randn(len(net.edges))).to(self.device)
        self.couponMatrix = torch.zeros(size).to(self.device)
        self.ss = size
        self.net = net
        self.ft = find_time
        self.calculation_seq = torch.tensor([[[0 for i in range(size[0])] for j in range(size[0])] for k in range(size[0])]).to(self.device)
        for i,m in enumerate(self.calculation_seq):
            for row in self.calculation_seq[i][1:]:
                row = torch.tensor([1 for i in range(size[0])])
        
    def dfs(self, pc): 
        plist = [(pc, 1)]
        find_time = [0]*self.ss[0]
        finded = [False for i in range(self.ss[0])]
        while plist != []:
            p, V = plist.pop(0)
            find_time[p.loc] += 1
            # if find_time[p.loc] >=self.ft:
            #     continue
            self.couponMatrix[pc.loc][p.loc] = V
            for c in p.children:
                if finded[c.loc]:
                    continue
                finded[c.loc] = True
                edge_Num = self.net.edges.index([p.loc, c.loc])
                plist.append((c, V*self.netParm[edge_Num]))

        return        
    
    def __calculateGraphSeq__(self):
        self.couponMatrix = torch.zeros(self.ss).to(self.device)
        for p in self.net.points:
            self.dfs(p)
        return
    
    def __attentionModule__(self, k, v):
        '''
        k: key
        v: value
        '''
        # get attention matrix
        att = torch.zeros(self.ss)
        att = torch.matmul(k, v.T)
        att = torch.nn.Softmax(dim=0)(att)        
        return att
        
    def __call__(self, q, k, v):
        '''
        q: query
        k: key
        v: value
        '''
        self.finded = {tuple(e):False for e in self.net.edges}
        self.__calculateGraphSeq__()
        att = self.__attentionModule__(k, v)
        att = att * self.couponMatrix
        att = att.T
        ans = torch.zeros(q.shape).to(self.device)
        # for i in range(self.ss[0]):
        #     row = att[i]
        #     temp = torch.zeros(self.ss[0], q.shape[1]).to(self.device)
        #     temp = row.unsqueeze(1) * q
        #     ans[i] = torch.sum(temp, dim=0)
        ans = torch.matmul(att, q)
            
        return ans, att