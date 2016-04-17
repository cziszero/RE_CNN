# -*- encoding: utf-8 -*-
# Created by Han on 2016/3/9
import os
import sys
import time
import re
import cPickle
from RE_layers import *
from collections import OrderedDict


def confusion_matrix_generator(answer, pred):
    """
    生成两个confusion_matrix，分别为不考虑方向和考虑方向
    :param answer:
    实际值
    :param pred:
    预测值
    :return:
    干扰矩阵，行为真实关系，列为预测关系。
    """
    answer = list(answer)
    pred = list(pred)
    confusion_matrix = [numpy.zeros([10, 10]), numpy.zeros([19, 19])]

    rel_d_dict = dict()
    rel_ud_dict = dict()

    for x in range(16):
        rel_d_dict[x] = x/2
        rel_ud_dict[x] = x
    rel_d_dict[16], rel_d_dict[17], rel_d_dict[18] = 9, 8, 8
    rel_ud_dict[16], rel_ud_dict[17], rel_ud_dict[18] = 18, 16, 17

    answer_d = map(lambda i: rel_d_dict[i], answer)
    pred_d = map(lambda i: rel_d_dict[i], pred)
    answer_ud = map(lambda i: rel_ud_dict[i], answer)
    pred_ud = map(lambda i: rel_ud_dict[i], pred)

    for res_d in zip(answer_d, pred_d):
        confusion_matrix[0][res_d[0], res_d[1]] += 1
    for res_ud in zip(answer_ud, pred_ud):
        confusion_matrix[1][res_ud[0], res_ud[1]] += 1

    return confusion_matrix


def static_cm(confusion_matrix, rel_num=9, method="MACRO"):
    """
    计算各个类别的准确率、召回率、F1值，计算总的准确率、召回率、F1值（分别计算各类求平均, 宏平均Macro-Average，考虑方向）
    :param confusion_matrix:
    干扰矩阵
    :param rel_num:
    关系数目，默认为9，不包含"other"分类。
    :return:
    列表，第i个元素代表第i类关系。
    第i个元素为[属于第i类关系的实例数，预测为第i类关系的实例数，正确预测数目，准确率，召回率，F1值，i]。
    """
    # if method == "MACRO":
    static_res = []
    # 统计计算每一分类的结果
    for i in range(rel_num+1):
        if i != 9:
            wrong_direction = confusion_matrix[1][2 * i, 2 * i + 1] + confusion_matrix[1][2 * i + 1, 2 * i]
        else:
            wrong_direction = 0
        tp_fp = sum(confusion_matrix[0][:, i])
        tp_fn = sum(confusion_matrix[0][i, :])
        tp = confusion_matrix[0][i, i] - wrong_direction
        p = float(tp)/float(tp_fp) if tp_fp else 0
        r = float(tp)/float(tp_fn) if tp_fn else 0
        f1 = (2 * p * r) / (p + r) if p and r else 0
        static_res.append([tp_fn, tp_fp, tp, p, r, f1, i])
    # 计算总体准确率召回率、F1值，不包含“other”关系
    p_r_f1 = [0, 0, 0]
    # static_res中是否有"other"关系
    if len(static_res) == 1 and static_res[0][6] == 9:
        return static_res, p_r_f1
    else:
        for res in static_res:
            # 如果不是other关系，则将关系加入到统计序列中
            if res[6] != 9:
                for x in (0, 1, 2):
                    p_r_f1[x] += res[x+3]
        for x in (0, 1, 2):
            p_r_f1[x] /= 9

        return static_res, p_r_f1


def get_rel_list(predicted, answer, start_index, rel_dict):
    """
    将预测的结果和实际结果转换为perl程序规定的形式。
    """
    assert len(predicted) == len(answer)
    sample_num = len(predicted)
    reversed_rel_dict = dict()
    for k, v in rel_dict.iteritems():
        reversed_rel_dict.setdefault(v, k)

    ans_list = list()
    pro_list = list()
    for i in range(sample_num):
        ans_list.append(str(start_index) + "\t" + reversed_rel_dict[predicted[i]] + "\n")
        pro_list.append(str(start_index) + "\t" + reversed_rel_dict[answer[i]] + "\n")
        start_index += 1

    return ans_list, pro_list


def get_f1_from_file(file_name):
    """
    从文件中读取f1值，这个f1值是视为9类且考虑方向的宏平均F1值
    :param file_name:
    :return:
    """
    with open(file_name) as f:
        info = f.readlines()
    begin_str = "<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:\n"
    end_str_t = "<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1"
    end_str = filter(lambda x: end_str_t in x, info)[0]
    f1_index_str = "MACRO-averaged result (excluding Other):\n"

    seg = info[info.index(begin_str):info.index(end_str)]
    f1_line_index = seg.index(f1_index_str) + 1
    f1_line = seg[f1_line_index]

    pf1 = re.compile('F1.*%')
    f1 = re.findall(pf1, f1_line)[0]
    f1 = f1.replace("F1 =  ", "")
    f1 = float(f1.replace("%", ""))

    return f1


def as_floatX(variable):
    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)
    return T.cast(variable, theano.config.floatX)


def load_data(data_file):
    """
    读取数据，生成shared变量。
    """
    print '... loading data'
    with open(data_file) as f:
        dataset = cPickle.load(f)
    # train_set = dataset.ix[0:6999]
    train_set = dataset.ix[0:7999]
    # valid_set = dataset.ix[7000:7999]
    test_set = dataset.ix[8000:9999]

    def shared_dataset(data_set, borrow=True):
        data_x, data_y = list(data_set["Sentences"]), list(data_set["Relations"])
        data_pos = list(data_set["EntitiesPos"])
        data_pos1 = map(lambda s: map(lambda p: p[0], s), data_pos)
        data_pos2 = map(lambda s: map(lambda p: p[1], s), data_pos)
        data_l1 = list(data_set["L1"])
        data_l2 = list(data_set["L2"])
        data_l3l, data_l3r = map(lambda x: x[0], list(data_set["L3"])), map(lambda x: x[1], list(data_set["L3"]))
        data_l4l, data_l4r = map(lambda x: x[0], list(data_set["L4"])), map(lambda x: x[1], list(data_set["L4"]))
        data_l51, data_l52 = map(lambda x: x[0], list(data_set["L5"])), map(lambda x: x[1], list(data_set["L5"]))

        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        shared_pos1 = theano.shared(numpy.asarray(data_pos1, dtype=theano.config.floatX), borrow=borrow)
        shared_pos2 = theano.shared(numpy.asarray(data_pos2, dtype=theano.config.floatX), borrow=borrow)
        shared_l1 = theano.shared(numpy.asarray(data_l1, dtype=theano.config.floatX), borrow=borrow)
        shared_l2 = theano.shared(numpy.asarray(data_l2, dtype=theano.config.floatX), borrow=borrow)
        shared_l3l = theano.shared(numpy.asarray(data_l3l, dtype=theano.config.floatX), borrow=borrow)
        shared_l3r = theano.shared(numpy.asarray(data_l3r, dtype=theano.config.floatX), borrow=borrow)
        shared_l4l = theano.shared(numpy.asarray(data_l4l, dtype=theano.config.floatX), borrow=borrow)
        shared_l4r = theano.shared(numpy.asarray(data_l4r, dtype=theano.config.floatX), borrow=borrow)
        shared_l51 = theano.shared(numpy.asarray(data_l51, dtype=theano.config.floatX), borrow=borrow)
        shared_l52 = theano.shared(numpy.asarray(data_l52, dtype=theano.config.floatX), borrow=borrow)

        return (shared_x, T.cast(shared_y, 'int32'), shared_pos1, shared_pos2,
                shared_l1, shared_l2,
                shared_l3l, shared_l3r,
                shared_l4l, shared_l4r,
                shared_l51, shared_l52
                )

    train_set_x, train_set_y, train_set_pos1, train_set_pos2, train_set_l1, train_set_l2, train_set_l3l, train_set_l3r, train_set_l4l, train_set_l4r, train_set_l51, train_set_l52 = shared_dataset(train_set)
    # valid_set_x, valid_set_y, valid_set_pos1, valid_set_pos2, valid_set_l1, valid_set_l2, valid_set_l3l, valid_set_l3r, valid_set_l4l, valid_set_l4r, valid_set_l51, valid_set_l52 = shared_dataset(valid_set)
    test_set_x, test_set_y, test_set_pos1, test_set_pos2, test_set_l1, test_set_l2, test_set_l3l, test_set_l3r, test_set_l4l, test_set_l4r, test_set_l51, test_set_l52 = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y, train_set_pos1, train_set_pos2, train_set_l1, train_set_l2, train_set_l3l, train_set_l3r, train_set_l4l, train_set_l4r, train_set_l51, train_set_l52),
            # (valid_set_x, valid_set_y, valid_set_pos1, valid_set_pos2, valid_set_l1, valid_set_l2, valid_set_l3l, valid_set_l3r, valid_set_l4l, valid_set_l4r, valid_set_l51, valid_set_l52),
            (test_set_x, test_set_y, test_set_pos1, test_set_pos2, test_set_l1, test_set_l2, test_set_l3l, test_set_l3r, test_set_l4l, test_set_l4r, test_set_l51, test_set_l52)]

    return rval


def evaluate_recnn(learning_rate=0.1,
                   window_size=3,
                   n_epochs=200,
                   dataset="data_processed\\data_aligned.pkl",
                   batch_size=200,
                   n=(200, 100),
                   wordidx_matrix="tmp_files\\matrix_wordidx.pkl",
                   posidx_matrix="tmp_files\\matrix_posidx.pkl",
                   relation_dict="tmp_files\\dict_relation.pkl",
                   word_dim=50,
                   ):
    rng = numpy.random.RandomState(7841)
    datasets= load_data(dataset)

    # 各类特征，share变量形式
    train_set_x, train_set_y, train_set_pos1, train_set_pos2, train_set_l1, train_set_l2, train_set_l3l, train_set_l3r, train_set_l4l, train_set_l4r, train_set_l51, train_set_l52 = datasets[0]
    # valid_set_x, valid_set_y, valid_set_pos1, valid_set_pos2, valid_set_l1, valid_set_l2, valid_set_l3l, valid_set_l3r, valid_set_l4l, valid_set_l4r, valid_set_l51, valid_set_l52 = datasets[1]
    # test_set_x, test_set_y, test_set_pos1, test_set_pos2, test_set_l1, test_set_l2, test_set_l3l, test_set_l3r, test_set_l4l, test_set_l4r, test_set_l51, test_set_l52 = datasets[2]
    test_set_x, test_set_y, test_set_pos1, test_set_pos2, test_set_l1, test_set_l2, test_set_l3l, test_set_l3r, test_set_l4l, test_set_l4r, test_set_l51, test_set_l52 = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    # n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    # n_valid_batches /= batch_size
    n_test_batches /= batch_size

    matrix_wordidx = numpy.asarray(cPickle.load(open(wordidx_matrix)), dtype=theano.config.floatX)
    matrix_posidx = numpy.asarray(cPickle.load(open(posidx_matrix)), dtype=theano.config.floatX)
    dict_rel = cPickle.load(open(relation_dict))

    sent_length = train_set_x.get_value().shape[1]

    index = T.lscalar()

    # input, output and features
    x = T.matrix('x')
    y = T.ivector('y')

    pos1 = T.matrix('pos1')
    pos2 = T.matrix('pos2')

    l1 = T.vector('pos2')
    l2 = T.vector('pos2')
    l3l = T.vector('pos2')
    l3r = T.vector('pos2')
    l4l = T.vector('pos2')
    l4r = T.vector('pos2')
    l51 = T.vector('pos2')
    l52 = T.vector('pos2')
    # use for set zero vector after train
    zero_vect_word_tensor = T.vector()
    zero_vect_pos_tensor = T.vector()

    zero_vec_word = numpy.zeros(word_dim, dtype=theano.config.floatX)
    zero_vec_pos = numpy.zeros(matrix_posidx.shape[1], dtype=theano.config.floatX)


    print '...building the model'

    initlayer = TransLayer(
        sent=x.reshape((batch_size, 1, sent_length, 1)),
        pos1=pos1.reshape((batch_size, 1, sent_length, 1)),
        pos2=pos2.reshape((batch_size, 1, sent_length, 1)),
        wordidx_matrix=matrix_wordidx,
        posidx_matrix=matrix_posidx,
    )
    # get_init = theano.function(
    #     [index],
    #     initlayer.output,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         pos1: train_set_pos1[index * batch_size: (index + 1) * batch_size],
    #         pos2: train_set_pos2[index * batch_size: (index + 1) * batch_size],
    #     }
    # )
    # res_init = get_init(1)

    layer0 = ConvPoolLayer(
        rng,
        input=initlayer.output,
        image_shape=(batch_size, 1, sent_length, (matrix_wordidx.shape[1] + 2 * matrix_posidx.shape[1])),
        filter_shape=(n[0], 1, window_size, (matrix_wordidx.shape[1] + 2 * matrix_posidx.shape[1])),
        poolsize=(int(sent_length - window_size + 1), 1)
    )

    # 转换为(batch_size, nkern * 200 * 1)
    layer1_input = layer0.output.flatten(2)

    layer1 = HiddenLayer(
        rng,
        input=layer1_input,
        n_in=n[0]*1*1,
        n_out=n[1],
        activation=T.tanh
    )
    # get_layer1 = theano.function(
    #     [index],
    #     layer1.output,
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         pos1: train_set_pos1[index * batch_size: (index + 1) * batch_size],
    #         pos2: train_set_pos2[index * batch_size: (index + 1) * batch_size],
    #     }
    # )
    # res_l1 = get_layer1(1)

    # Lexical Features
    jointlayer = JointLayer(
        sent_feature=layer1.output.reshape((batch_size, 1, 1, n[1])),
        l1=l1.reshape((batch_size, 1, 1, 1)),
        l2=l2.reshape((batch_size, 1, 1, 1)),
        l3l=l3l.reshape((batch_size, 1, 1, 1)),
        l3r=l3r.reshape((batch_size, 1, 1, 1)),
        l4l=l4l.reshape((batch_size, 1, 1, 1)),
        l4r=l4r.reshape((batch_size, 1, 1, 1)),
        l51=l51.reshape((batch_size, 1, 1, 1)),
        l52=l52.reshape((batch_size, 1, 1, 1)),
        wordidx_matrix=initlayer.wordidx_matrix,
        dim_word=word_dim
    )

    # get_jointlayer = theano.function(
    #     [index],
    #     [jointlayer.sent_featue, jointlayer.l1, jointlayer.l2, jointlayer.l3l, jointlayer.l3r, jointlayer.l4l,jointlayer.l4r, jointlayer.l51, jointlayer.l52, jointlayer.output],
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         pos1: train_set_pos1[index * batch_size: (index + 1) * batch_size],
    #         pos2: train_set_pos2[index * batch_size: (index + 1) * batch_size],
    #         l1: train_set_l1[index * batch_size: (index + 1) * batch_size],
    #         l2: train_set_l2[index * batch_size: (index + 1) * batch_size],
    #         l3l: train_set_l3l[index * batch_size: (index + 1) * batch_size],
    #         l3r: train_set_l3r[index * batch_size: (index + 1) * batch_size],
    #         l4l: train_set_l4l[index * batch_size: (index + 1) * batch_size],
    #         l4r: train_set_l4r[index * batch_size: (index + 1) * batch_size],
    #         l51: train_set_l51[index * batch_size: (index + 1) * batch_size],
    #         l52: train_set_l52[index * batch_size: (index + 1) * batch_size],
    #     }
    # )
    # res_jl = get_jointlayer(1)

    # logisticRegression的输入input为matrix(batch_size, layer.output.size)每一行代表一个句子特征。
    # layer2 = LogisticRegression(input=layer1.output, n_in=n[1], n_out=19)
    layer2_input_dim = word_dim * 8 + n[1]
    layer2 = LogisticRegression(input=jointlayer.output, n_in=layer2_input_dim, n_out=19)

    # get_layer2 = theano.function(
    #     [index],
    #     [layer2.W, layer2.b, layer2.p_y_given_x],
    #     givens={
    #         x: train_set_x[index * batch_size: (index + 1) * batch_size],
    #         pos1: train_set_pos1[index * batch_size: (index + 1) * batch_size],
    #         pos2: train_set_pos2[index * batch_size: (index + 1) * batch_size],
    #         l1: train_set_l1[index * batch_size: (index + 1) * batch_size],
    #         l2: train_set_l2[index * batch_size: (index + 1) * batch_size],
    #         l3l: train_set_l3l[index * batch_size: (index + 1) * batch_size],
    #         l3r: train_set_l3r[index * batch_size: (index + 1) * batch_size],
    #         l4l: train_set_l4l[index * batch_size: (index + 1) * batch_size],
    #         l4r: train_set_l4r[index * batch_size: (index + 1) * batch_size],
    #         l51: train_set_l51[index * batch_size: (index + 1) * batch_size],
    #         l52: train_set_l52[index * batch_size: (index + 1) * batch_size],
    #     }
    # )
    # res_l2 = get_layer2(1)

    # 求似然
    cost = layer2.negative_log_likelihood(y)

    params = layer2.params + layer1.params + layer0.params + initlayer.params
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    set_zero_word = theano.function([zero_vect_word_tensor], updates=[(initlayer.wordidx_matrix, T.set_subtensor(initlayer.wordidx_matrix[0, :], zero_vect_word_tensor))])
    set_zero_pos = theano.function([zero_vect_pos_tensor], updates=[(initlayer.posidx_matrix, T.set_subtensor(initlayer.posidx_matrix[199, :], zero_vect_pos_tensor))])

    train_model = theano.function(
        [index],
        [cost, layer2.errors(y)],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            pos1: train_set_pos1[index * batch_size: (index + 1) * batch_size],
            pos2: train_set_pos2[index * batch_size: (index + 1) * batch_size],
            l1: train_set_l1[index * batch_size: (index + 1) * batch_size],
            l2: train_set_l2[index * batch_size: (index + 1) * batch_size],
            l3l: train_set_l3l[index * batch_size: (index + 1) * batch_size],
            l3r: train_set_l3r[index * batch_size: (index + 1) * batch_size],
            l4l: train_set_l4l[index * batch_size: (index + 1) * batch_size],
            l4r: train_set_l4r[index * batch_size: (index + 1) * batch_size],
            l51: train_set_l51[index * batch_size: (index + 1) * batch_size],
            l52: train_set_l52[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size],
        }
    )

    test_model = theano.function(
        [index],
        [layer2.errors(y), layer2.y_pred, y],
        givens={
            y: test_set_y[index * batch_size: (index + 1) * batch_size],
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            pos1: test_set_pos1[index * batch_size: (index + 1) * batch_size],
            pos2: test_set_pos2[index * batch_size: (index + 1) * batch_size],
            l1: test_set_l1[index * batch_size: (index + 1) * batch_size],
            l2: test_set_l2[index * batch_size: (index + 1) * batch_size],
            l3l: test_set_l3l[index * batch_size: (index + 1) * batch_size],
            l3r: test_set_l3r[index * batch_size: (index + 1) * batch_size],
            l4l: test_set_l4l[index * batch_size: (index + 1) * batch_size],
            l4r: test_set_l4r[index * batch_size: (index + 1) * batch_size],
            l51: test_set_l51[index * batch_size: (index + 1) * batch_size],
            l52: test_set_l52[index * batch_size: (index + 1) * batch_size],
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    best_test_f1 = 0
    best_test_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        train_error_list = []
        train_cost_list = []
        # totally n mini batches
        for minibatch_index in xrange(n_train_batches):
            # calculate one mini batch in one iter
            iter_n = (epoch - 1) * n_train_batches + minibatch_index
            # cost of this mini batch
            cost_ij, train_error = train_model(minibatch_index)
            set_zero_word(zero_vec_word)
            set_zero_pos(zero_vec_pos)

            train_error_list.append(train_error)
            train_cost_list.append(cost_ij)

            if (iter_n + 1) % n_train_batches == 0:
                if n_test_batches:
                    # compute zero-one loss on validation set
                    test_res = [
                            test_model(i)
                            for i in xrange(n_test_batches)
                        ]
                else:
                    test_res = [test_model(0)]

                this_train_losses = numpy.mean(train_error_list)
                this_train_costes = numpy.mean(train_cost_list)
                this_test_losses = numpy.mean(map(lambda l: l[0], test_res))

                # predict relations of test set
                predict_rel = list()
                for predict in map(lambda x: list(x[1]), test_res):
                    predict_rel += predict

                # proposed relations of test set
                anser_rel = list()
                for propose in map(lambda x: list(x[2]), test_res):
                    anser_rel += propose

                # 自己写的macro_avg_f1
                confusion_matrix = confusion_matrix_generator(anser_rel, predict_rel)
                static_res, p_r_f1 = static_cm(confusion_matrix)
                macro_avg_f1 = p_r_f1[2]

                if macro_avg_f1 > best_test_f1:
                    best_test_iter = iter_n
                    best_test_f1 = macro_avg_f1

                    pred_res, answer_res = get_rel_list(predict_rel, anser_rel, 1, dict_rel)

                    pred_file = "res_analysis\\pred_file.txt"
                    answer_file = "res_analysis\\answer_file.txt"
                    score_file = "res_analysis\\score_file.txt"
                    static_res_file = "res_analysis\\static_res_file.pkl"
                    p_r_f1_file = "res_analysis\\p_r_f1_file.pkl"
                    confusion_matrix_file = "res_analysis\\confusion_matrix_file.pkl"

                    with open(pred_file, "w") as pred_write:
                        pred_write.writelines(pred_res)
                    with open(answer_file, "w") as answer_write:
                        answer_write.writelines(answer_res)
                    with open(static_res_file, "w") as static_res_write:
                        cPickle.dump(static_res, static_res_write)
                    with open(p_r_f1_file, "w") as p_r_f1_write:
                        cPickle.dump(p_r_f1, p_r_f1_write)
                    with open(confusion_matrix_file, "w") as confusion_matrix_write:
                        cPickle.dump(confusion_matrix, confusion_matrix_write)

                    # 调用perl程序计算F1值
                    command = "dataset\\SemEval2010_task8_all_data\\SemEval2010_task8_scorer-v1.2\\semeval2010_task8_scorer-v1.2.pl "\
                              + pred_file + " " + answer_file + " > " + score_file
                    os.system(command.encode('gbk'))

                    print('epoch %i, iter %i, minibatch %i/%i, train cost %f, train error %f %%, test error %f %%, F1(best) %f %%' %
                          (epoch, iter_n + 1, minibatch_index + 1, n_train_batches, this_train_costes, this_train_losses * 100,
                           this_test_losses * 100., macro_avg_f1 * 100) + '\t\t' + time.ctime())

                else:
                    print('epoch %i, iter %i, minibatch %i/%i, train cost %f, train error %f %%, test error %f %%, F1 %f %%' %
                          (epoch, iter_n + 1, minibatch_index + 1, n_train_batches, this_train_costes, this_train_losses * 100,
                           this_test_losses * 100., macro_avg_f1 * 100) + '\t\t' + time.ctime())

    end_time = time.clock()

    print('Optimization complete.')

    print('Best F1 score of %f %% obtained at iteration %i, ' %
          (best_test_f1 * 100., best_test_iter + 1))

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    print 'Parameters:\nwindow_size: %d\nlearning_rate: %f\nn_epochs: %d\nbatch_size: %d\nn1,n2: %d,%d' %\
          (window_size, learning_rate, n_epochs, batch_size, n[0], n[1])


if __name__ == "__main__":
    evaluate_recnn(learning_rate=0.01,
                   window_size=3,
                   n_epochs=10000,
                   dataset="E:\\Work Space\\Code\\Python_My\\RE\\data_processed\\data_aligned.pkl",
                   batch_size=1000,
                   n=(200, 100),
                   word_dim=50)
