import re
import jieba
import json
from tqdm import tqdm
import os
import itertools
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt



# 加载别名表
file_path = 'userdict.json'
with open(file_path, 'r', encoding='utf-8') as f:  # 替换为正确的编码方式
    names = json.load(f)
for name in names:
    for alias_name in names[name]:
        # 结巴分词可以动态加载词语，在这里加载人物别名，以免错误的分词
        # nr 在结巴分词中代表人名
        jieba.add_word(alias_name, tag="nr")

def load_stopwords(file_path):
    # 加载停用词函数
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip('\n') for line in f.readlines()]
    return stopwords

def process(file_path):
    # 对一个章节进行分词
    stopwords = load_stopwords('stopwords.txt')
    with open(file_path, 'rb') as f:
        content = f.read()
        para = content.decode("utf-8", 'ignore').split('\n')
        para = para[:-1]
        result = []
        for p in para:
            words = []
            # 对每一个段落进行分词
            seg_list = list(jieba.cut(p, cut_all=False))
            for x in seg_list:
                if len(x) <= 1:
                    continue
                if x in stopwords:
                    continue
                words.append(x)
            # 返回的是一个二维 list: [[段1结果], [段2结果], ]
            result.append(words)
    return result

result = {}

# 遍历目录
for file_name in tqdm(os.listdir('红楼梦')):
    # 获取章节的坐标，前面爬取的数据，是以 1_章节名 命名
    index = int(file_name.split('_')[0])
    # 分词，保存到一个字典，章节坐标对应分词结果
    result[index] = process(os.path.join('红楼梦', file_name))

#词频统计----------------------------------------

artical_word_dict = {}  # 整篇文章分词结果
chapters_word_dict = {}  # 每个章节的分词结果，对应到文章索引

# 遍历上文的分词结果
for index in tqdm(result):
    # 将一个章节的二维 list 压缩为一维，方便遍历
    chapter_word_list = list(itertools.chain.from_iterable(result[index]))
    chapter_word_dict = {}
    # 遍历
    for word in chapter_word_list:
        # 当词语不存在时 dict 初始化为 0
        if word not in chapter_word_dict:
            chapter_word_dict[word] = 0
        # 当词语不存在时 dict 初始化为 0
        if word not in artical_word_dict:
            artical_word_dict[word] = 0
        # 相应结果加一
        chapter_word_dict[word] += 1
        artical_word_dict[word] += 1
    chapters_word_dict[index] = chapter_word_dict

# 对整篇文章的词频统计结果进行排序，由高到低
sorted_word_list = [x[1] for x in sorted(
    zip(artical_word_dict.values(), artical_word_dict.keys()), reverse=True)]
sorted_word_list

#数据分析及可视化----------------------------
#分析是否为曹雪芹所作

all_names = []  # 所有名字的 list
for name in names:
    for alias_name in names[name]:
        all_names.append(alias_name)

# 待删除的名字
to_delete = []
for x in tqdm(sorted_word_list):
    for name in all_names:
        # 利用正则表达式判断一个词语是否是名字
        pattern = re.compile("(.*){}(.*)".format(name))
        # 正则查找匹配，如果匹配的结果不为 None 则该词语为名字
        t = re.search(pattern, x)
        # 如果一个词语是名字，则删除掉
        if t:
            to_delete.append(x)
            break

# 删除操作
for x in to_delete:
    sorted_word_list.remove(x)

V = 2000  # 选择前 V 个词语的词频作为一章节的特征
word_list = sorted_word_list[:V]
word_list



# 在这里生成一个 120*2000 的 0 矩阵，每一行对应一章节的特征。
features = np.zeros((len(chapters_word_dict), V))
# 遍历每个章节的 word dict
for index in chapters_word_dict:
    # 章节 word dict
    chapter_word_dict = chapters_word_dict[index]
    # 遍历每个词语，并改变相应位置的值为该词语出现次数
    for j, word in enumerate(word_list):
        try:
            features[index, j] = chapter_word_dict[word]
        except:
            # 该章节未出现这个词语
            pass
features.shape, features

features /= features.max(axis=0)  # 按列归一化到 [0, 1]
features.shape, features




# 特征降维
features = decomposition.PCA(n_components=2).fit_transform(features)

# 绘制前八十回和后四十回散点图
plt.plot(features[:80, 0].flatten(), features[:80, 1].flatten(), '.')
plt.plot(features[80:, 0].flatten(), features[80:, 1].flatten(), '.')
plt.legend(['First 80', 'Last 40'])



#人物关系绘制--------------
# 统计一起出现的次数
count = {}
for comb in itertools.product(names.keys(), repeat=2):
    count['_'.join(comb)] = 0

# 遍历章节词语
for para_word_list in tqdm(list(itertools.chain.from_iterable(result.values()))):
    para_name_list = []
    for name in names:
        tmp = set(names[name]).intersection(para_word_list)
        if len(tmp)>0:
            para_name_list.append(name)
    # 去重
    para_name_list = list(set(para_name_list))
    # 统计一段中共同出现的次数
    for comb in itertools.combinations(para_name_list, 2):
        count['_'.join(comb)] += 1
        

from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go
import networkx as nx
# 激活 Plotly notebook 模式，方便离线绘图
init_notebook_mode(connected=True)
G = nx.Graph()  # 创建 Graph
nodes = []  # 所有节点
edges = []  # 所有边
# 统计所有名字的组合
for comb in itertools.combinations(names.keys(), 2):
    # 两个名字一起出现的次数
    total = count['{}_{}'.format(comb[0], comb[1])] + \
        count['{}_{}'.format(comb[1], comb[0])]
    # 如果一起出现了，才保存
    if total > 0:
        edges.append((comb[0], comb[1], {'weight': total}))
        nodes.append(comb[0])
        nodes.append(comb[1])

nodes = list(set(nodes))  # 去除重复节点
G.add_nodes_from(nodes)  # 添加节点和边
G.add_edges_from(edges)

# 节点对应到图上的坐标 (x, y)
pos = nx.spring_layout(G)
pos

# 绘制边的轨迹
edge_trace = go.Scatter(x=[], y=[], line=dict(
    width=0.5, color='#888'), hoverinfo='none', mode='lines')

# 遍历每一条边，根据上文获得的每一个节点的坐标，连接两个节点
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

# 绘制点
node_trace = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
                        marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True,
                                    color=[], size=10,
                                    colorbar=dict(thickness=15, title='Node Connections',
                                                  xanchor='left', titleside='right'),
                                    line=dict(width=2)))
# 遍历每一个点
for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    
# 遍历邻接列表，对应了 节点 → 一条边的另一个节点和权重
for node, adjacencies in enumerate(G.adjacency()):
    # 把所有边的次数累加，据此对每一个点添加颜色，出现次数越多颜色越深（除以 5000 防止超出范围）
    node_trace['marker']['color'] += tuple(
        [sum([adjacencies[1][name]['weight'] for name in adjacencies[1]])/5000])
    node_info = adjacencies[0]
    # 每一个节点添加标注信息（名字）
    node_trace['text'] += tuple([node_info])
    
fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                width=1000,  # 设置图形宽度为800像素
                height=1000,  # 设置图层安高为600像素
                title='红楼梦人物关系网络图', titlefont=dict(size=16),
                showlegend=False, hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text="楼+ 数据分析与挖掘实战", showarrow=False,
                    xref="paper", yref="paper", x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False,
                           showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

# 绘制图像

iplot(fig)
    
# 显示图形
plt.show()