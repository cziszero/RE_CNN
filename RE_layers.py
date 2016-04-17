# -*- encoding: utf-8 -*-
# Created by Han on 2016/3/9
import theano
import theano.tensor as T
import numpy
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


class TransLayer(object):
    """
    将索引形式的输入转换为对应的词向量，作为卷积层的输入
    """
    def __init__(self, sent, pos1, pos2, wordidx_matrix, posidx_matrix):
        self.wordidx_matrix = theano.shared(
            value=wordidx_matrix,
            name="wordidx_matrix",
            borrow=True
        )
        self.posidx_matrix = theano.shared(
            value=posidx_matrix,
            name="posidx_matrix",
            borrow=True
        )

        sentvec = self.wordidx_matrix[T.cast(sent.flatten(), dtype="int32")].reshape((sent.shape[0], sent.shape[1], sent.shape[2], wordidx_matrix.shape[1]))
        posvec0 = self.posidx_matrix[T.cast(pos1.flatten(), dtype="int32")].reshape((pos1.shape[0], pos1.shape[1], pos1.shape[2], posidx_matrix.shape[1]))
        posvec1 = self.posidx_matrix[T.cast(pos2.flatten(), dtype="int32")].reshape((pos2.shape[0], pos2.shape[1], pos2.shape[2], posidx_matrix.shape[1]))

        self.output = T.concatenate([sentvec, posvec0, posvec1], axis=3)
        self.params = [self.wordidx_matrix, self.posidx_matrix]

        # x句子中词对应的索引号，x.flatten将x将为1维，cast获取一个与x值相同但数据类型不同的变量，
        # 最后通过索引在Words中取出相应的行
        # layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))


class FConvPoolLayer(object):
    """
    手工完成卷积操作，目前未用。
    """
    def __init__(self, n0, sent, pos1, pos2, batchsize, swidx, ewidx, wordidx_matrix, posidx_matrix):
        self.wordidx_matrix = theano.shared(
            value=wordidx_matrix,
            name="wordidx_matrix",
            borrow=True
        )
        self.posidx_matrix = theano.shared(
            value=posidx_matrix,
            name="posidx_matrix",
            borrow=True
        )

        # index of start symbol
        self.swidx = numpy.array([swidx] * batchsize, dtype=theano.config.floatX).reshape((batchsize, 1, 1, 1))
        # index of end symbol
        self.ewidx = numpy.array([ewidx] * batchsize, dtype=theano.config.floatX).reshape((batchsize, 1, 1, 1))

        self.concidx = T.concatenate([self.swidx, sent, self.ewidx], axis=2)

        S = self.wordidx_matrix[T.cast(self.concidx.flatten(), dtype="int32")]\
            .reshape((self.concidx.shape[0], self.concidx.shape[1], self.concidx.shape[2], wordidx_matrix.shape[1]))
        P1 = self.posidx_matrix[T.cast(pos1.flatten(), dtype="int32")].reshape((pos1.shape[0], pos1.shape[1], pos1.shape[2], posidx_matrix.shape[1]))
        P2 = self.posidx_matrix[T.cast(pos2.flatten(), dtype="int32")].reshape((pos2.shape[0], pos2.shape[1], pos2.shape[2], posidx_matrix.shape[1]))

        S_sub1 = S[:][:][:-2]
        S_sub2 = S[:][:][1:-1]
        S_sub3 = S[:][:][2:]
        X_S = T.concatenate([S_sub1, S_sub2, S_sub3], axis=3)

        X = T.concatenate([X_S, P1, P2], axis=3)

        self.W = theano.shared(
            value=numpy.zeros((wordidx_matrix.shape[1] + 2 * posidx_matrix.shape[1], n0),
                              dtype=theano.config.floatX),
            name='WC',
            borrow=True
        )

        Tmp = T.dot(X, self.W)
        self.output = T.max(Tmp, axis=3).reshape(X.shape[0], X.shape[1], 1, X.shape[3])


class JointLayer(object):
    """
    将句子特征与词法特征结合
    """
    def __init__(self, sent_feature, l1, l2, l3l, l3r, l4l, l4r, l51, l52, wordidx_matrix, dim_word=50):
        self.sent_featue = sent_feature
        l1 = wordidx_matrix[T.cast(l1.flatten(), dtype="int32")].reshape((l1.shape[0], l1.shape[1], l1.shape[2], dim_word))
        l2 = wordidx_matrix[T.cast(l2.flatten(), dtype="int32")].reshape((l2.shape[0], l2.shape[1], l1.shape[2], dim_word))
        l3l = wordidx_matrix[T.cast(l3l.flatten(), dtype="int32")].reshape((l3l.shape[0], l3l.shape[1], l3l.shape[2], dim_word))
        l3r = wordidx_matrix[T.cast(l3r.flatten(), dtype="int32")].reshape((l3r.shape[0], l3r.shape[1], l3r.shape[2], dim_word))
        l4l = wordidx_matrix[T.cast(l4l.flatten(), dtype="int32")].reshape((l4l.shape[0], l4l.shape[1], l4l.shape[2], dim_word))
        l4r = wordidx_matrix[T.cast(l4r.flatten(), dtype="int32")].reshape((l4r.shape[0], l4r.shape[1], l4r.shape[2], dim_word))
        l51 = wordidx_matrix[T.cast(l51.flatten(), dtype="int32")].reshape((l51.shape[0], l51.shape[1], l51.shape[2], dim_word))
        l52 = wordidx_matrix[T.cast(l52.flatten(), dtype="int32")].reshape((l52.shape[0], l52.shape[1], l52.shape[2], dim_word))

        self.l1 = l1
        self.l2 = l2
        self.l3l = l3l
        self.l3r = l3r
        self.l4l = l4l
        self.l4r = l4r
        self.l51 = l51
        self.l52 = l52

        # select features
        self.Features = [sent_feature, l1, l2, l3l, l3r, l4l, l4r, l51, l52]

        self.output = T.concatenate(self.Features, axis=3).flatten(2)


class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one  minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def model_prediction(self):
        return self.y_pred

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()