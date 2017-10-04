# -*- coding: utf-8
# TODO
# eos tag

import tensorflow as tf
from nltk.tokenize import word_tokenize
import numpy as np
import functools
from itertools import chain
from sklearn import model_selection, metrics
import pprint

from process_data import load_entities, load_task, vectorize, vectorize_kv_pairs, save_pickle, load_pickle, load_kv_pairs
# from net.memn2n import MemN2N
from net.memn2n_kv import MemN2N_KV, add_gradient_noise

pp = pprint.PrettyPrinter()

tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.1, "Lambda for l2 loss.")
tf.flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
tf.flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
tf.flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
tf.flags.DEFINE_integer("lindim", 75, "linear part of the state [75]") # TODO ?
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout")
tf.flags.DEFINE_integer("evaluation_interval", 50, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 8, "Batch size for training.")
tf.flags.DEFINE_integer("feature_size", 40, "Feature size")
tf.flags.DEFINE_integer("n_hop", 1, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("n_epoch", 30, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embd_size", 100, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("mem_size", 20, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en-10k/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model (bow, simple_gru)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "automatically choose an existing and supported device to run the operations in case the specified one doesn't exist")
tf.flags.DEFINE_boolean("log_device_placement", False, "To find out which devices your operations and tensors are assigned to")
tf.flags.DEFINE_string("output_file", "single_scores.csv", "Name of output file for final bAbI accuracy scores.")
tf.flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
tf.flags.DEFINE_boolean("show", False, "print progress [False]")
tf.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")

FLAGS = tf.flags.FLAGS

def main(_):
    is_babi = False
    is_load_pickle = True

    train_data, test_data = None, None
    entities = None # only for movie dialog

    if is_load_pickle:
        train_data = load_pickle('mov_task1_qa_pipe_train.pickle')[:1000]
        # test_data = load_pickle('mov_task1_qa_pipe_test.pickle')
        kv_pairs = load_pickle('mov_kv_pairs.pickle')[:10000]
    else:
        if is_babi:
            train_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt')
            test_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt')
        else: # movie qa
            train_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_train.txt')
            test_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_test.txt')
            entities = load_entities('./data/movie_dialog_dataset/entities.txt')
            # save_pickle(train_data, 'mov_task1_qa_pipe_train.pickle')
            # save_pickle(test_data, 'mov_task1_qa_pipe_test.pickle')
            # save_pickle(entities, 'mov_entities.pickle')

    # data = train_data + test_data
    data = train_data
    if test_data:
        data += test_data

    if is_load_pickle:
        vocab = load_pickle('mov_vocab.pickle')
        w2i = load_pickle('mov_w2i.pickle')
        i2w = load_pickle('mov_i2w.pickle')
        entities = load_pickle('mov_entities.pickle')
    else:
        vocab = functools.reduce(lambda x, y: x | y, (set(chain(chain.from_iterable(s), q, a)) for s, q, a in data))
        w2i = dict((c, i) for i, c in enumerate(vocab, 1))
        i2w = dict((i, c) for i, c in enumerate(vocab, 1))
        # save_pickle(vocab, 'vocab.pickle')
        # save_pickle(w2i, 'w2i.pickle')
        # save_pickle(i2w, 'i2w.pickle')
    if is_babi:
        FLAGS.n_entity = None
    else:
        FLAGS.n_entity = len(entities)

    max_story_size = max(map(len, (s for s, _, _ in data)))
    len_list = list(map(len, chain.from_iterable(s for s, _, _ in data)))
    if len(len_list) != 0:
        sentence_size = max(len_list)
    else:
        sentence_size = 0
    question_size = max(map(len, chain.from_iterable(q for _, q, _ in data)))
    max_sentence_size = max(sentence_size, question_size)
    vocab_size = len(w2i) + 1 # +1 for nil word

    FLAGS.story_size = max_story_size
    FLAGS.mem_size = min(FLAGS.mem_size, max_story_size + 1) # avoid memory_size == 0
    if not is_babi:
        FLAGS.mem_size = len(kv_pairs)
    FLAGS.query_size = max_sentence_size
    FLAGS.sentence_size = max_sentence_size
    FLAGS.memory_key_size = FLAGS.mem_size
    FLAGS.memory_value_size = FLAGS.mem_size
    FLAGS.vocab_size = vocab_size
    print('max_story_size={}\nmax_sentence_size={}\nvocab_size={}'.format(max_story_size, max_sentence_size, vocab_size))


    S, Q, A = vectorize(data, w2i, max_sentence_size, FLAGS.mem_size, entities)
    trainS, valS, trainQ, valQ, trainA, valA = model_selection.train_test_split(S, Q, A, test_size=0.1)
    # testS, testQ, testA = vectorize(test_data, w2i, max_sentence_size, FLAGS.mem_size, entities)

    n_train = trainS.shape[0]
    # n_test = testS.shape[0]
    n_valid = valS.shape[0]
    print('Train size:', n_train)
    # print('Test size:', n_test)
    print('Valid size:', n_valid)

    train_labels = np.argmax(trainA, axis=1)
    # test_labels = np.argmax(testA, axis=1)
    valid_labels = np.argmax(valA, axis=1)

    batch_size = FLAGS.batch_size
    batch_indices = list(zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size)))
    if not is_babi:
        # kv_pairs = vectorize_kv_pairs(kv_pairs, max_sentence_size, FLAGS.mem_size, entities)
        FLAGS.max_memory_kv_size = 2
        kv_pairs = vectorize_kv_pairs(kv_pairs, FLAGS.max_memory_kv_size, FLAGS.mem_size, entities)
        kv_pairs_batch = np.resize(kv_pairs, (batch_size, kv_pairs.shape[0], kv_pairs.shape[1]))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )

        global_step = tf.Variable(0, name='global_step', trainable=False)
        # decay leaning rate
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        lr = tf.train.exponential_decay(FLAGS.init_lr, global_step, 20000, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=FLAGS.epsilon)

        with tf.Session() as sess:
            model = MemN2N_KV(FLAGS, is_babi)#, sess)
            # model.build_model()

            # TODO
            grads_and_vars = optimizer.compute_gradients(model.loss_op)
            grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                                for g, v in grads_and_vars if g is not None]
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
            nil_grads_and_vars = []
            for g, v in grads_and_vars:
                if v.name in model._nil_vars:
                    nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    nil_grads_and_vars.append((g, v))

            train_op = optimizer.apply_gradients(nil_grads_and_vars, name="train_op", global_step=global_step)
            sess.run(tf.global_variables_initializer())

            def train_step(s, q, a, k=None):
                if is_babi:
                    feed_dict = {
                        model.query: q,
                        model.memory_key: s,
                        model.memory_value: s,
                        model.labels: a,
                        model.keep_prob: FLAGS.keep_prob
                    }
                else:
                    feed_dict = {
                        model.query: q,
                        model.memory_key: k,
                        model.memory_value: k, # k==v in Sentence Level
                        model.labels: a,
                        model.keep_prob: FLAGS.keep_prob
                    }

                _, step, predict_op = sess.run([train_op, global_step, model.predict_op], feed_dict=feed_dict)
                return predict_op

            def test_step(s, q):
                feed_dict = {
                    model.query: q,
                    model.memory_key: s,
                    model.memory_value: s,
                    model.keep_prob: 1
                }
                predicts = sess.run(model.predict_op, feed_dict=feed_dict)
                return predicts

            for epoch in range(1, FLAGS.n_epoch+1):
                print('Epoch', epoch)
                np.random.shuffle(batch_indices)
                train_preds = []
                for start in range(0, n_train, batch_size):
                    # if start + batch_size >= n_train: break # TODO 切り捨て
                    end = start + batch_size
                    s = trainS[start:end] # (bs, story_size, sentence_size) = (32, 10, 6)
                    q = trainQ[start:end] # (bs, sentence_size) = (32, 6)
                    a = trainA[start:end] # (bs, vocab_size) = (32, 20)
                    # predict_op = train_step(s, q, a)
                    if is_babi:
                        predict_op = train_step(s, q, a)
                    else:
                        # predict_op = train_step(s, q, a, np.repeat(kv_pairs, batch_size, axis=0))
                        if start + batch_size < n_train:
                            predict_op = train_step(s, q, a, kv_pairs_batch)
                        else:
                            predict_op = train_step(s, q, a, kv_pairs_batch[:(n_train - start)])

                    train_preds += list(predict_op)
                
                train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
                print('Training Acc: {0:.6f}'.format(train_acc))

                if epoch % FLAGS.evaluation_interval == 0:
                    val_preds = test_step(valS, valQ)
                    val_acc = metrics.accuracy_score(np.array(val_preds), valid_labels)
                    print(val_preds)
                    print('Epoch', epoch)
                    print('Validation Acc: {0:.6f}'.format(val_acc))

            # test on train dataset
            # train_preds = test_step(trainS, trainQ)
            # train_acc = metrics.accuracy_score(train_labels, train_preds)
            # train_acc = '{0:.6f}'.format(train_acc)
            # eval dataset
            # val_preds = test_step(valS, valQ)
            # val_acc = metrics.accuracy_score(valid_labels, val_preds)
            # val_acc = '{0:.2f}'.format(val_acc)
            # tesing dataset
            # test_preds = test_step(testS, testQ)
            # test_acc = metrics.accuracy_score(test_labels, test_preds)
            # test_acc = '{0:.2f}'.format(test_acc)

            # print('===================================')
            # print('Training Acc:', train_acc)
            # print('Validating Acc:', val_acc)
            # print('Testing Acc:', test_acc)
            # print('===================================')
            # with open(FLAGS.open_file, 'a') as f:
                # f.write('{}, {}, {}, {}\n'.format(FLAGS.task_id, test_acc, train_acc, val_acc))

            # TODO: Should like this
            # if FLAGS.is_test:
            #     model.run(valid_data, test_data)
            # else:
            #     model.run(train_data, valid_data)


if __name__ == '__main__':
    tf.app.run()
