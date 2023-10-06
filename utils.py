import sys
import numpy as np
import networkx as nx
import pickle as pkl
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
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)

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

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # (2708,2708)
    labels = np.vstack((ally, ty))  # (2708,7)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_train = range(len(ally) - 500)
    idx_val = range(len(ally) - 500, len(ally))
    idx_test = test_idx_range.tolist()

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
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


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
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
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized.tocsr()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""

    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)  # 考虑了节点自身的关系，所以给邻接矩阵加上了一个单位矩阵A+I

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

def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape})
    return feed_dict


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
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
    train_index = np.where(train_mask)[0]
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

