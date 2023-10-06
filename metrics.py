import tensorflow as tf


"""
用tensorflow实现的softmax交叉熵损失函数，并且添加了一个掩码（mask）以过滤掉无效的样本。
在深度学习中，有时候需要过滤掉某些样本，比如序列数据中的填充项，这些填充项不应该对模型的训练产生影响。
在这种情况下，可以使用一个掩码来表示哪些样本是有效的，哪些是无效的。
具体来说，如果一个样本是无效的，那么在掩码中对应的位置就是0，在计算损失函数的时候，将其权重设置为0就可以了。
而对于有效的样本，其对应位置的掩码值是1，损失函数会对其进行计算。
因此，在这个函数中，将loss乘以掩码，就可以去除无效样本对损失函数的影响。
最后，返回所有有效样本的平均损失。
"""
"""
preds是模型的输出，代表每个类别的预测概率。
labels是真实的标签，用于计算损失。
"""


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)  # 对损失进行归一化，以避免掩码对损失的影响
    loss *= mask  # 将损失张量乘以掩码张量，以过滤掉不需要计算损失的样本
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    # tf.argmax() 函数返回矩阵中最大值的索引，第二个参数 1 表示在行维度上进行取值。
    # tf.argmax(preds, 1) 取得预测结果中每个样本的预测类别索引,tf.argmax(labels, 1) 取得真实标签中每个样本的类别索引
    # tf.equal() 函数对这两个索引进行比较，返回布尔型张量，表示预测是否正确。
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    # tf.cast() 函数将布尔型张量转换为浮点型张量，其中正确预测的位置为 1.0，错误预测的位置为 0.0。
    # 最后，对所有样本的准确率进行平均，即可得到该批次数据的准确率。
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    # 首先，代码将变量 mask 转换为 float32 数据类型，使用 tf.cast() 函数
    mask = tf.cast(mask, dtype=tf.float32)
    # 然后，它使用 tf.reduce_mean() 对 mask 中的值求平均数，并将 mask 除以该平均值。
    # 这一步实际上将 mask 规范化，使其值范围在0到1之间。
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


"""
这段代码定义了一个 softmax 交叉熵损失函数，用于计算神经网络的预测值和标签之间的差异。
其中，preds 是神经网络的预测值，labels 是对应的真实标签。

在函数内部，使用 TensorFlow 的 tf.nn.softmax_cross_entropy_with_logits 函数计算 softmax 交叉熵损失。
这个函数会自动将神经网络的预测值进行 softmax 处理，并计算其与标签之间的交叉熵。

最后，使用 tf.reduce_mean 函数对所有样本的损失值进行平均，得到最终的损失值。
这个平均操作是为了使损失函数对不同样本的数量保持不变，从而更好地比较不同模型的性能。
"""


def softmax_cross_entropy(preds, labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


# 这段代码是 TensorFlow 中用于计算分类模型准确率的代码。其中，preds 是模型预测的结果，labels 是真实标签。
def accuracy(preds, labels):
    # tf.argmax() 函数返回矩阵中最大值的索引，第二个参数 1 表示在行维度上进行取值。
    # tf.argmax(preds, 1) 取得预测结果中每个样本的预测类别索引,tf.argmax(labels, 1) 取得真实标签中每个样本的类别索引
    # tf.equal() 函数对这两个索引进行比较，返回布尔型张量，表示预测是否正确。
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    # tf.cast() 函数将布尔型张量转换为浮点型张量，其中正确预测的位置为 1.0，错误预测的位置为 0.0。
    # 最后，对所有样本的准确率进行平均，即可得到该批次数据的准确率。
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)



