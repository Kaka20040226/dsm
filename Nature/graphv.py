from graphviz import Digraph

# 创建有向图，设置全局样式
dot = Digraph('Algorithm Flow', format='png',
             graph_attr={'rankdir':'LR', 'splines':'ortho', 'nodesep':'0.5'},
             node_attr={'shape':'record', 'style':'filled', 'fillcolor':'#F0F8FF'})

# ========== 定义节点 ==========
# 输入层
dot.node('input', '''<
<table border="0" cellborder="0">
  <tr><td bgcolor="#FFE4B5" colspan="2"><b>Input Matrix</b></td></tr>
  <tr>
    <td align="left">Tasks</td>
    <td align="right">Providers</td>
  </tr>
  <tr>
    <td colspan="2" border="1" cellpadding="5">(task_num × provider_num)</td>
  </tr>
</table>>''')

# 注意力模块
with dot.subgraph(name='cluster_attention') as c:
    c.attr(style='filled', color='lightgrey', label='Dual Attention Mechanism')
    c.node('attn1', '''<
    <table border="0">
        <tr><td><b>MultiheadAttention</b></td></tr>
        <tr><td>embed_dim=provider_num</td></tr>
        <tr><td>num_heads=1</td></tr>
    </table>>''')
    c.node('k_proj', 'klinear\n(Linear×3)')
    c.node('v_proj', 'vlinear\n(Linear×3)')

# 转置注意力
with dot.subgraph(name='cluster_t_attention') as c:
    c.attr(style='filled', color='lightgrey', label='Transposed Attention')
    c.node('t_attn', '''<
    <table border="0">
        <tr><td><b>MultiheadAttention</b></td></tr>
        <tr><td>embed_dim=task_num</td></tr>
        <tr><td>num_heads=1</td></tr>
    </table>>''')

# 展平与线性层
dot.node('flatten', '''<
<table border="0">
    <tr><td><b>Flatten</b></td></tr>
    <tr><td>(task_num×provider_num → 1×t·p)</td></tr>
</table>>''')

dot.node('linear', '''<
<table border="0">
    <tr><td><b>Linear Layers</b></td></tr>
    <tr><td>t·p → 2t·p → t·p</td></tr>
</table>>''')

# LSTM模块
with dot.subgraph(name='cluster_lstm') as c:
    c.attr(style='filled', color='lightgrey', label='LSTM State Update')
    c.node('forget', 'Forget Gate\nσ(W·[x, h_prev]+b)')
    c.node('input_gate', 'Input Gate\nσ(W·[x, h_prev]+b)')
    c.node('cell_update', 'Cell State\nc_t = f⊙c_{t-1} + i⊙g')
    c.node('output_gate', 'Output Gate\no = σ(W·[x, h_prev]+b)')
    c.node('hidden_state', 'New Hidden State\nh_t = o⊙tanh(c_t)')

# 决策层
dot.node('output', '''<
<table border="0">
    <tr><td><b>Decision Layer</b></td></tr>
    <tr><td>Concat[pos, h_t, c_t]</td></tr>
    <tr><td>Output: 2×t·p logits</td></tr>
</table>>''')

# ========== 连接节点 ==========
# 主数据流
dot.edges([
    ('input', 'attn1'),
    ('attn1', 't_attn'),
    ('t_attn', 'flatten'),
    ('flatten', 'linear'),
    ('linear', 'forget'),
    ('linear', 'input_gate'),
    ('linear', 'output_gate')
])

# LSTM内部连接
dot.edge('forget', 'cell_update', label='f⊙c_{t-1}')
dot.edge('input_gate', 'cell_update', label='i⊙g')
dot.edge('cell_update', 'hidden_state')
dot.edge('output_gate', 'hidden_state', label='o⊙tanh(c_t)')
dot.edge('hidden_state', 'output')

# 状态回传
dot.edge('hidden_state', 'forget', label='h_prev', style='dashed', color='#666666')
dot.edge('hidden_state', 'input_gate', label='h_prev', style='dashed', color='#666666')
dot.edge('hidden_state', 'output_gate', label='h_prev', style='dashed', color='#666666')

# 投影层连接
dot.edge('k_proj', 'attn1', label='Key Projection', style='dotted')
dot.edge('v_proj', 'attn1', label='Value Projection', style='dotted')

# ========== 维度标注 ==========
with dot.subgraph(name='cluster_annotations') as c:
    c.attr(color='none')
    c.node('dim1', '''<
    <table border="0" cellborder="0">
        <tr><td>◀─ Input Dimensions ─▶</td></tr>
        <tr><td>Task Axis: {task_num}</td></tr>
        <tr><td>Provider Axis: {provider_num}</td></tr>
    </table>>''', shape='none')
    
    c.node('dim2', '''<
    <table border="0" cellborder="0">
        <tr><td>State Dimensions:</td></tr>
        <tr><td>hidden_dim = {output_dim}</td></tr>
        <tr><td>cell_dim = {output_dim}</td></tr>
    </table>>''', shape='none')

# 保存graphviz文件
dot.save('algorithm_flow.dot')
# ========== 渲染输出 ==========
# dot.render('algorithm_flow', view=True, cleanup=True)
