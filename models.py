from layers import *
from metrics import *


# 定义模型抽象类
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            # 判断参数名是否在允许的参数列表中，如果不在则抛出异常
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    # 定义抽象方法，子类必须实现该方法
    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # 将每层的输出(激活值)都添加到列表，用于追踪每层的输出结果
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        # 整个神经网络的最后一层激活值就是模型的输出结果
        self.outputs = self.activations[-1]

        # 获取神经网络中的变量(variables)并将其存储在vars字典中。
        # 在TensorFlow中，变量是用来存储模型参数的对象，可以在训练过程中被优化更新。
        # 在神经网络中，变量通常用来存储权重(weights)和偏置(biases)等参数。
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        """Build metrics"""
        # 神经网络的损失(loss)函数
        # 用来衡量模型预测结果与实际结果之间的差异的函数。
        # 通过最小化损失函数，可以使模型的预测结果更加接近实际结果。
        self._loss()
        # 神经网络的准确率(accuracy)用来衡量模型预测结果与实际结果之间的一致性的指标。
        # 通过计算准确率，可以评估模型的性能，并进行模型选择和调整。
        self._accuracy()
        # 这段代码的作用是定义一个优化操作(opt_op)，用于最小化神经网络的损失函数(loss)。
        # 在TensorFlow中，可以使用优化器(optimizer)来自动计算并更新神经网络中的变量，以使损失函数的值最小化。
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    # 将训练好的神经网络模型保存到磁盘中，以便后续使用。
    # 在TensorFlow中，可以使用saver对象来保存和恢复神经网络的参数
    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        # self.inputs 表示模型的输入特征，通过 placeholders['features'] 获得。
        self.inputs = placeholders['features']
        # self.input_dim 表示输入特征的维度，通过 input_dim 参数传入。
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        # self.placeholders 是一个字典，包含了模型的所有 placeholder，它们可以在模型训练时填充数据。
        self.placeholders = placeholders
        # self.optimizer 表示模型的优化器，这里使用 Adam 优化器，学习率为 FLAGS.learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # self.build() 函数用于建立模型的计算图。
        self.build()

    def _loss(self):
        # 权重衰减函数
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # 交叉熵误差度量
        """
        outputs是模型的预测结果，labels是真实的标签，labels_mask是一个二进制掩码,用于指示哪些位置是有效的标签。
        在计算准确率时，只有掩码中对应位置为1的标签才会被考虑
        """
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,  # 不对模型的输出进行任何变换,用于回归问题或者在输出层使用softmax函数时，因为这些情况下需要保留模型输出的原始数值，而不是进行变换。
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class MiniAngleGCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MiniAngleGCN, self).__init__(**kwargs)
        self.inputs = placeholders['AXfeatures']  # A*X for the bottom layer, not original feature X
        # 模型输入(特征)维度:1433
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        # 模型最终输出维度：7
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.support = placeholders['support']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()  # self.build()方法会根据输入数据的形状和模型的结构，自动创建并初始化模型的权重和偏置。

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    """
    self.layers是一个列表类型的变量，表示神经网络模型中的所有层次结构。
    Dense是一个全连接层，表示神经网络模型中的一个层次结构，其参数包括输入维度、输出维度、激活函数、是否使用dropout等。
    input_dim是输入维度，表示该层次结构的输入特征的维度。
    output_dim是输出维度，表示该层次结构的输出特征的维度。
    FLAGS.hidden1是一个整数类型的变量，表示隐藏层的维度。
    placeholders是一个字典类型的变量，表示输入数据的占位符，用于在神经网络模型中传递数据。
    act是激活函数，表示该层次结构的输出特征经过激活函数之后的值。
    dropout表示是否使用dropout，用于防止过拟合。
    sparse_inputs表示输入特征是否为稀疏矩阵。
    logging表示是否输出日志信息。
    """

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,  # 第一层隐藏层单元：16
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,  # sparse_inputs表示输入特征是否为稀疏矩阵
                                 logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.support,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


"""###################################FastGCN########################################"""


class GCN_APPRO_Mix(Model):  # mixture of dense and gcn
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_APPRO_Mix, self).__init__(**kwargs)
        self.inputs = placeholders['AXfeatures']  # A*X for the bottom layer, not original feature X
        # 模型输入(特征)维度:1433
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        # 模型最终输出维度：7
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.support = placeholders['support']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()  # self.build()方法会根据输入数据的形状和模型的结构，自动创建并初始化模型的权重和偏置。

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += softmax_cross_entropy(self.outputs, self.placeholders['labels'])

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs, self.placeholders['labels'])

    """
    self.layers是一个列表类型的变量，表示神经网络模型中的所有层次结构。
    Dense是一个全连接层，表示神经网络模型中的一个层次结构，其参数包括输入维度、输出维度、激活函数、是否使用dropout等。
    input_dim是输入维度，表示该层次结构的输入特征的维度。
    output_dim是输出维度，表示该层次结构的输出特征的维度。
    FLAGS.hidden1是一个整数类型的变量，表示隐藏层的维度。
    placeholders是一个字典类型的变量，表示输入数据的占位符，用于在神经网络模型中传递数据。
    act是激活函数，表示该层次结构的输出特征经过激活函数之后的值。
    dropout表示是否使用dropout，用于防止过拟合。
    sparse_inputs表示输入特征是否为稀疏矩阵。
    logging表示是否输出日志信息。
    """

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,  # 第一层隐藏层单元：16
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=False,  # sparse_inputs表示输入特征是否为稀疏矩阵
                                 logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.support,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
