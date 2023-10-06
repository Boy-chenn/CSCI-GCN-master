import sys
import numpy as np
import networkx as nx
import pickle as pkl  # 数据写入到文件和从文件读取用到的模块
import scipy.sparse as sp
from scipy.sparse.linalg import norm as sparsenorm


def parse_index_file(filename):
    """解析索引文件"""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def load_data(dataset_str):
    """加载数据"""
    # step 1: 读取 x, y, tx, ty, allx, ally, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):  # 分别读取文件
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):  # python版本大于3.0
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
                """#objects
                    x:(140, 1433) 训练集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (140, 1433)
                    tx:(1000, 1433) 测试集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (1000, 1433)
                    allx: (1708, 1433) 包含有标签和无标签的训练节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，
                    实际展开后大小为：(1708, 1433)，可以理解为除测试集以外的其他节点特征集合，训练集是它的子集
                    y:(140,7) one-hot表示的训练节点的标签，保存对象为：numpy.ndarray,7维的独热编码[0 0 0 1 0 0 0][0 0 0 0 1 0 0] 
                    ty:(1000, 7) one-hot表示的测试节点的标签 
                    ally: (1708, 7)  one-hot表示的ind.cora.allx对应的标签     
                """
    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # step 2: 读取测试集索引
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    # 获取整个图的所有节点特征
    features = sp.vstack((allx, tx)).tolil()  # (2708,1433)
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # features = preprocess_features(features) 根据自己需要归一化特征

    # 获取整个图的邻接矩阵
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # (2708,2708)
    # adj = preprocess_adj(adj) 根据自己需要归一化邻接矩阵

    # 获取所有节点标签
    labels = np.vstack((ally, ty))  # (2708,7)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # 划分训练集、验证集、测试集索引
    idx_train = range(len(ally) - 500)  # [0,1208]
    idx_val = range(len(ally) - 500, len(ally))  # (1208,1708)
    idx_test = test_idx_range.tolist()  # [1708,2708]

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)  # [2708,2708]
    y_val = np.zeros(labels.shape)  # [2708,2708]
    y_test = np.zeros(labels.shape)  # [2708,2708]
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # print("-------adj--------")
    # print(adj.toarray())
    # print("-------features--------")
    # print(features.toarray())
    # print("-------y_train--------")
    # print(y_train)
    # print("-------train_mask--------")
    # print(train_mask)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


"""将稀疏矩阵转换为元组表示，以便特征预处理处理"""


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


"""将特征矩阵行规一化后并转换为元组表示"""


def preprocess_features(features):
    rowsum = np.array(features.sum(1))  # 每个节点行求和(词频总数)
    r_inv = np.power(rowsum, -1).flatten()  # rowsum的倒数
    r_inv[np.isinf(r_inv)] = 0.  # rowsum的倒数可能不存在(即无穷大)置为0
    r_mat_inv = sp.diags(r_inv)  # 将一维数组r_inv转换为一个对角矩阵r_mat_inv，其中对角线上的元素为r_inv中的元素
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)
    # return features


def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    """返回的是csr稀疏矩阵"""
    rowsum = np.array(features.sum(1))
    rowsum[rowsum == 0] = 1e-10
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """对称归一化邻接矩阵"""
    # sp.coo_matrix(adj) 将稠密邻接矩阵(adj) 转换成稀疏矩阵格式(coo_matrix)
    adj = sp.coo_matrix(adj)
    # adj.sum(1) 对邻接矩阵(adj) 的每一行求和，得到一个列向量，表示每个节点的出度
    # np.array() 将列向量转化为 NumPy 数组
    rowsum = np.array(adj.sum(1))
    # np.power(rowsum, -0.5) 对每个节点的出度(rowsum) 取 -0.5 次方，实现了对每个节点的度数进行归一化
    # flatten() 将二维数组展开成一维数组，得到长度为n的一维数组。其中，n 是节点数。
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # np.isinf(d_inv_sqrt) 返回一个布尔类型的数组，表示输入数组中哪些元素是无穷大的（inf）
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 将无穷大的元素置零，避免后面出现除以零的情况
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # sp.diags(d_inv_sqrt) 将得到的一维数组包装成一个对角矩阵(d_mat_inv_sqrt)
    # 该矩阵的对角线上依次为每个节点的度数的平方根的倒数，而非对角线上均为零。
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # adj_appr = np.array(sp.csr_matrix.todense(adj))
    # # adj_appr = dense_lanczos(adj_appr, 100)
    # adj_appr = dense_RandomSVD(adj_appr, 100)
    # if adj_appr.sum(1).min()<0:
    #     adj_appr = adj_appr- (adj_appr.sum(1).min()-0.5)*sp.eye(adj_appr.shape[0])
    # else:
    #     adj_appr = adj_appr + sp.eye(adj_appr.shape[0])
    # adj_normalized = normalize_adj(adj_appr)

    # adj_normalized = normalize_adj(adj+sp.eye(adj.shape[0]))
    # adj_appr = np.array(sp.coo_matrix.todense(adj_normalized))
    # # adj_normalized = dense_RandomSVD(adj_appr,100)
    # adj_normalized = dense_lanczos(adj_appr, 100)

    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)  # 考虑了节点自身的关系，所以给邻接矩阵加上了一个单位矩阵A+I
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, supports, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: supports[i] for i in range(len(supports))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def column_prop_L1(adj):
    column_norm = sparsenorm(adj, ord=1, axis=0)
    norm_sum = sum(column_norm)
    return column_norm / norm_sum


def column_prop_L2(adj):
    #column_norm = sparsenorm(adj, axis=0)
    column_norm = pow(sparsenorm(adj, axis=0), 2)
    norm_sum = sum(column_norm)
    return column_norm / norm_sum


# from scipy.sparse import diags
# from scipy.sparse.linalg import norm
# def column_prop_test(adj):
#     column_norm = norm(diags(1.0 / adj.sum(axis=0).A.ravel()) * adj, axis=0)
#     return column_norm / column_norm.sum()

def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape})
    return feed_dict


# shuffle是一个布尔值，表示是否对数据进行随机化处理
def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    # 输入数据的总样本数
    numSamples = inputs[0].shape[0]
    # 如果shuffle为True，则对输入数据的样本进行随机化处理，并生成一个索引列表indices
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    # 在每次迭代中，代码使用start_idx作为起始索引，从inputs中提取一个大小为batchsize的小批次数据
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        # 在每次迭代中，代码从inputs中提取一个小批次，并使用yield关键字将其返回。
        # 这个函数是一个生成器函数，它可以在需要时生成新的批次
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


"""::::::::::::::::::::::::::::::::::::::MiniAngleGCN::::::::::::::::::::::::::::::::::::::::::"""
"""数据加载并预处理"""


def data_prepare(adj, y_train, y_val, y_test, train_mask, val_mask, test_mask):
    """cora数据集：2708"""
    """训练集：1208[0~1207]"""
    train_index = np.where(train_mask)[0]  # 拿到参与训练的节点索引
    # adj[train_index, :]表示选取邻接矩阵中train_index行的所有元素
    # adj[train_index, :][:, train_index]表示在上一步的基础上，再选取其中的train_index列的所有元素
    # 这段代码的作用是从邻接矩阵中选取出训练集中节点之间的连接情况，生成一个新的邻接矩阵
    adj_train = adj[train_index, :][:, train_index]  # 根据节点索引获取这些节点构成的子图
    y_train = y_train[train_index]  # 上面训练节点对应的标签
    """验证集：500[1208~1707]"""
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]
    """测试集：1000[1708~2707]"""
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]

    train_val_index = np.concatenate([train_index, val_index], axis=0)
    train_test_idnex = np.concatenate([train_index, test_index], axis=0)

    # print("numNode", numNode)
    """"""
    return adj_train, y_train, y_val, y_test, test_mask, train_index, train_val_index, train_test_idnex


def adj_features_preprocess(adj, adj_train, features, train_index, train_val_index, train_test_index):
    """邻接矩阵预处理"""
    norm_adj_train = nontuple_preprocess_adj(adj_train)  # 训练集节点归一化并以三元组形式返回
    norm_adj_val = nontuple_preprocess_adj(adj[train_val_index, :][:, train_val_index])
    norm_adj_test = nontuple_preprocess_adj(adj[train_test_index, :][:, train_test_index])
    """特征预处理"""
    """特征处理,特征矩阵：[2708*1433]"""
    features = nontuple_preprocess_features(features).todense()  # 特征归一化处理后并以稀疏三元组形式返回后转化成稠密矩阵
    """训练集特征：[1208*1433]"""
    train_features = norm_adj_train.dot(features[train_index])
    """验证集特征：[1708*1433]"""
    val_features = norm_adj_val.dot(features[train_val_index])
    """测试集特征：[2208*1433]"""
    test_features = norm_adj_test.dot(features[train_test_index])

    return norm_adj_train, norm_adj_val, norm_adj_test, train_features, val_features, test_features


"""#####################################AS_train################################################"""


def prepare_pubmed(dataset, max_degree):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

    train_index = np.where(train_mask)[0]
    adj_train = adj[train_index, :][:, train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]

    num_train = adj_train.shape[0]
    input_dim = features.shape[1]

    features = nontuple_preprocess_features(features).todense()
    train_features = features[train_index]

    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)

    if dataset == 'pubmed':
        norm_adj = 1 * sp.diags(np.ones(norm_adj.shape[0])) + norm_adj
        norm_adj_train = 1 * sp.diags(np.ones(num_train)) + norm_adj_train

    # adj_train, adj_val_train = norm_adj_train, norm_adj_train
    adj_train, adj_val_train = compute_adjlist(norm_adj_train, max_degree)
    train_features = np.concatenate((train_features, np.zeros((1, input_dim))))

    return norm_adj, adj_train, adj_val_train, features, train_features, y_train, y_test, test_index


def compute_adjlist(sp_adj, max_degree):
    """Transfer sparse adjacent matrix to adj-list format"""
    num_data = sp_adj.shape[0]
    adj = num_data + np.zeros((num_data + 1, max_degree), dtype=np.int32)
    adj_val = np.zeros((num_data + 1, max_degree), dtype=np.float32)

    for v in range(num_data):
        neighbors = np.nonzero(sp_adj[v, :])[1]
        len_neighbors = len(neighbors)
        if len_neighbors > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
            adj[v] = neighbors
            adj_val[v, :len_neighbors] = sp_adj[v, neighbors].toarray()
        else:
            adj[v, :len_neighbors] = neighbors
            adj_val[v, :len_neighbors] = sp_adj[v, neighbors].toarray()

    return adj, adj_val


def construct_feed_dict_with_prob(features_inputs, supports, probs, labels, labels_mask, placeholders):
    """Construct feed dictionary with adding sampling prob."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features_inputs'][i]: features_inputs[i] for i in range(len(features_inputs))})
    feed_dict.update({placeholders['support'][i]: supports[i] for i in range(len(supports))})
    feed_dict.update({placeholders['prob'][i]: probs[i] for i in range(len(probs))})
    # feed_dict.update({placeholders['prob_norm'][i]: probs_norm[i] for i in range(len(probs_norm))})
    feed_dict.update({placeholders['num_features_nonzero']: features_inputs[1].shape})
    return feed_dict


"""#####################################################################################"""
