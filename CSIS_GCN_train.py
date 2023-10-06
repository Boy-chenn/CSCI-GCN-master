from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from scipy.sparse import csr_matrix
from sampler import get_min_angle_similarity, jaccard_similarity
from utils import *
from models import MiniAngleGCN

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_mini_angle', 'Model string.')
flags.DEFINE_float('learning_rate', 0.03, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')  # cora:145 #citeseer:180 #pubmed:
flags.DEFINE_integer('rank1', 200, 'Number of samples')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-5, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


# def main(rank1, weight_p0, weight_p1):
def main(rank1, k):
# def main(rank1):
    """加载原始数据 """
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    """数据准备"""
    adj_train, y_train, y_val, y_test, test_mask, train_index, train_val_index, train_test_index = data_prepare(adj,
                                                                                                                y_train,
                                                                                                                y_val,
                                                                                                                y_test,
                                                                                                                train_mask,
                                                                                                                val_mask,
                                                                                                                test_mask)

    """预处理"""
    norm_adj_train, norm_adj_val, norm_adj_test, train_features, val_features, test_features = adj_features_preprocess(
        adj,
        adj_train,
        features,
        train_index,
        train_val_index,
        train_test_index)
    """选择模型"""
    if FLAGS.model == 'gcn_mini_angle':
        model_func = MiniAngleGCN
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    """定义占位符"""
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32),
        'AXfeatures': tf.placeholder(tf.float32, shape=(None, train_features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    """创建模型"""
    model = model_func(placeholders, input_dim=train_features.shape[-1], logging=True)  # features.shape[-1]特征维度

    # Initialize session
    sess = tf.Session()

    """定义模型评估函数"""

    def evaluate(features, support, labels, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feeddict_forMixlayers(features, support, labels, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    """创建一个会话(Session)对象"""
    sess.run(tf.global_variables_initializer())

    cost_val = []

    """多维加权计算节点的整体影响力分数"""
    norm_adj_train_todense = norm_adj_train.todense()

    p0 = csr_matrix(get_min_angle_similarity(norm_adj_train_todense))
    p0 = column_prop_L2(p0)
    p1 = column_prop_L2(norm_adj_train)

    # 计算Jaccard相似度
    # adjacency_matrix = adj_train.toarray()
    # jaccard_sim = jaccard_s、imilarity(adjacency_matrix)
    # csr_jaccard_sim = csr_matrix(jaccard_sim)
    # jaccard_sim = column_prop_L1(csr_jaccard_sim)

    # 设置重要性比例
    weight_p0 = 0.5
    weight_p1 = 0.5
    # weight_p2 = 2
    # 计算加权平均
    P = np.array([(weight_p0 * p0[i] + weight_p1 * p1[i]) / (
            weight_p0 + weight_p1) for i in range(len(p0))])
    # P = np.array([(weight_p0 * p0[i] + weight_p1 * p1[i] + weight_p2 * jaccard_sim[i]) / (
    #         weight_p0 + weight_p1 + weight_p2) for i in range(len(p0))])

    # print(P)
    valSupport = sparse_to_tuple(norm_adj_val[len(train_index):, :])
    testSupport = sparse_to_tuple(norm_adj_test[len(train_index):, :])

    t = time.time()
    # print(norm_adj_train.shape[0])
    adj_train_len = norm_adj_train.shape[0]
    """训练模型"""
    # for epoch in range(FLAGS.epochs):
    for epoch in range(k):
        t1 = time.time()

        # 使用了迭代的方式来训练模型，每次从训练数据集中随机选择一个batch的数据进行训练
        for batch in iterate_minibatches_listinputs([norm_adj_train, y_train], batchsize=512, shuffle=True):
            [norm_adj_batch, y_train_batch] = batch
            norm_adj_train_todense = norm_adj_batch.todense()

            # p0 = column_prop(norm_adj_batch)
            # p1 = csr_matrix(get_min_angle_similarity(norm_adj_train_todense))
            # p1 = column_prop(p1)
            # p1_resized = np.pad(p1, (0, adj_train_len - len(p1)), mode='constant', constant_values=0)[:adj_train_len]
            # P = (p0+p1_resized)/2
            # P=p1
            if rank1 is None:
                support1 = sparse_to_tuple(norm_adj_batch)
                features_inputs = train_features
            else:
                # np.nonzero函数返回度数不为零的节点的索引数组，即邻接矩阵的度分布
                distr = np.nonzero(np.sum(norm_adj_batch, axis=0))[1]
                # rank(采样数量)
                if rank1 > len(distr):
                    q1 = distr
                else:
                    q1 = np.random.choice(distr, rank1, replace=False, p=P[distr] / sum(P[distr]))
                support1 = sparse_to_tuple(norm_adj_batch[:, q1].dot(sp.diags(1.0 / (P[q1] * rank1))))
                if len(support1[1]) == 0:
                    continue

                features_inputs = train_features[q1, :]

            feed_dict = construct_feeddict_forMixlayers(features_inputs, support1, y_train_batch,
                                                        placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(val_features, valSupport, y_val, placeholders)
        cost_val.append(cost)

        # Print results
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        #       "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t1))

    # print("Train Finished!")
    # print("#################Result###################")
    train_duration = time.time() - t
    test_cost, test_acc, test_duration = evaluate(test_features, testSupport, y_test,
                                                  placeholders)
    # print("rank1 = {}".format(rank1), "cost=",
    #       "{:.5f}".format(test_cost),
    #       "accuracy=", "{:.5f}".format(test_acc), "training time per epoch=",
    #       "{:.5f}".format(train_duration / (epoch + 1)),
    #       "test time=", "{:.5f}".format(test_duration))
    print("epoch = {}".format(k), "cost=",
          "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "training time per epoch=",
          "{:.5f}".format(train_duration / (epoch + 1)),
          "test time=", "{:.5f}".format(test_duration))
    # print("w_0=", "{:.5f}".format(weight_p0), "rank1 = {}".format(rank1), "cost=",
    #       "{:.5f}".format(test_cost),
    #       "accuracy=", "{:.5f}".format(test_acc), "training time per epoch=",
    #       "{:.5f}".format(train_duration / (epoch + 1)),
    #       "test time=", "{:.5f}".format(test_duration))


if __name__ == "__main__":
    # print("DATASET:", FLAGS.dataset)
    # main(FLAGS.rank1)
    print("CSCI-GCN:epoch text")
    #
    for epochk in [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,100,
                   105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200,
                   205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300]:
        main(30, epochk)
    # for rank1 in [8, 16, 32, 64, 128, 256, 512]:
    #     main(rank1, 0.5, 0.5)
    # for weight_p0 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     main(FLAGS.rank1, weight_p0, 1 - weight_p0)
