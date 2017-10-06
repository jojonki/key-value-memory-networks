# see original code: https://github.com/carpedm20/MemN2N-tensorflow
"""Key Value Memory Networks with GRU reader.
The implementation is based on https://arxiv.org/abs/1606.03126
The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from six.moves import range
import numpy as np
# from attention_reader import Attention_Reader

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.
    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

class MemN2N_KV(object):
    """Key Value Memory Network."""
    def __init__(self, config, is_babi, feature_size=30, reader='bow', l2_lambda=0.2, name='KeyValueM2mN2N'): # TODO: feature_size
        # batch_size, vocab_size,
        #              query_size, story_size, memory_key_size,
        #              memory_value_size, embedding_size,
        #              feature_size=30,
        #              hops=3,
        #              reader='bow',
        #              l2_lambda=0.2,
        #              name='KeyValueMemN2N'):
        """Creates an Key Value Memory Network

        Args:
        batch_size: The size of the batch.

        vocab_size: The size of the vocabulary (should include the nil word). The nil word one-hot encoding should be 0.

        query_size: largest number of words in question

        story_size: largest number of words in story

        embedding_size: The size of the word embedding.

        memory_key_size: the size of memory slots for keys
        memory_value_size: the size of memory slots for values
        
        feature_size: dimension of feature extraced from word embedding

        hops: The number of hops. A hop consists of reading and addressing a memory slot.

        debug_mode: If true, print some debug info about tensors
        name: Name of the End-To-End Memory Network.\
        Defaults to `KeyValueMemN2N`.
        """
        self.story_size = config.story_size
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.query_size = config.query_size
        self.sentence_size = config.sentence_size
        #self._wiki_sentence_size = doc_size
        self.memory_key_size = config.memory_key_size
        self.memory_value_size = config.memory_value_size
        # self.max_memory_kv_size = config.max_memory_kv_size
        if not is_babi:
            self.max_memory_sentence_size = config.max_memory_sentence_size
        self.embd_size = config.embd_size
        self.n_hop = config.n_hop
        self.name = name
        self.encoding = tf.constant(position_encoding(self.sentence_size, self.embd_size), name="encoding")
        self.reader = 'bow'
        self.is_babi = is_babi
        self.n_entity = config.n_entity
        self._build_inputs()

        d = feature_size
        self.feature_size = feature_size
        # self.n_hidden = feature_size
        self.reader_feature_size = 0

        # trainable variables
        if reader == 'bow':
            self.reader_feature_size = self.embd_size
        # elif reader == 'simple_gru':
        #     self.reader_feature_size = self.n_hidden

        self.A = tf.get_variable('A', shape=[self.feature_size, self.reader_feature_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        #self.A_mvalue = tf.get_variable('A_mvalue', shape=[self._feature_size, self.reader_feature_size],
        #                                initializer=tf.contrib.layers.xavier_initializer())
        self.TK = tf.get_variable('TK', shape=[self.memory_value_size, self.reader_feature_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.TV = tf.get_variable('TV', shape=[self.memory_value_size, self.reader_feature_size],
                                  initializer=tf.contrib.layers.xavier_initializer())

        # Embedding layer
        # with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            nil_word_slot = tf.zeros([1, self.embd_size])
            self.W = tf.concat([nil_word_slot, tf.get_variable('W', shape=[self.vocab_size-1, self.embd_size],
                                                                  initializer=tf.contrib.layers.xavier_initializer())], 0)
            if self.is_babi:
                self.W_memory = tf.concat([nil_word_slot, tf.get_variable('W_memory', shape=[self.vocab_size-1, self.embd_size],
                                                                         initializer=tf.contrib.layers.xavier_initializer())], 0)
            else:
                self.W_memory = tf.concat([nil_word_slot, tf.get_variable('W_memory', shape=[self.n_entity-1, self.embd_size],
                                                                         initializer=tf.contrib.layers.xavier_initializer())], 0)
            # self.W_memory = self.W
            self._nil_vars = set([self.W.name, self.W_memory.name])
            # shape: [batch_size, query_size, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.query)
            # shape: [batch_size, memory_size, story_size, embedding_size]
            self.mkeys_embedded_chars = tf.nn.embedding_lookup(self.W_memory, self.memory_key)
            # shape: [batch_size, memory_size, story_size, embedding_size]
            self.mvalues_embedded_chars = tf.nn.embedding_lookup(self.W_memory, self.memory_value)

        if reader == 'bow':
            if is_babi:
                q_r = tf.reduce_sum(self.embedded_chars*self.encoding, 1) 
                doc_r = tf.reduce_sum(self.mkeys_embedded_chars*self.encoding, 2)
                value_r = tf.reduce_sum(self.mvalues_embedded_chars*self.encoding, 2)
            else:
                q_r = tf.reduce_sum(self.embedded_chars, 1) 
                doc_r = tf.reduce_sum(self.mkeys_embedded_chars, 2)
                value_r = tf.reduce_sum(self.mvalues_embedded_chars, 2)

        r_list = []
        for _ in range(self.n_hop):
            # define R for variables
            R = tf.get_variable('R{}'.format(_), shape=[self.feature_size, self.feature_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            r_list.append(R)

        o = self._key_addressing(doc_r, value_r, q_r, r_list)
        o = tf.transpose(o)
        if reader == 'bow':
            self.B = self.A
            #self.B = tf.get_variable('B', shape=[self._feature_size, self.reader_feature_size],
            #                         initializer=tf.contrib.layers.xavier_initializer())
        
        #logits_bias = tf.get_variable('logits_bias', [self.vocab_size])
        y_tmp = tf.matmul(self.B, self.W_memory, transpose_b=True)
        with tf.name_scope("prediction"):
            logits = tf.matmul(o, y_tmp)# + logits_bias
            #logits = tf.nn.dropout(tf.matmul(o, self.B) + logits_bias, self.keep_prob)
            probs = tf.nn.softmax(tf.cast(logits, tf.float32))
            
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self.labels, tf.float32), name='cross_entropy')
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

            # loss op
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])
            loss_op = cross_entropy_sum + l2_lambda*lossL2
            # predict ops
            predict_op = tf.argmax(probs, 1, name="predict_op")

            # assign ops
            self.loss_op = loss_op
            self.predict_op = predict_op
            self.probs = probs

    def _build_inputs(self):
        with tf.name_scope("input"):
            self.query = tf.placeholder(tf.int32, [None, self.query_size], name='question')

            if self.is_babi:
                self.memory_key = tf.placeholder(tf.int32, [None, self.memory_key_size, self.sentence_size], name='memory_key')
                self.memory_value = tf.placeholder(tf.int32, [None, self.memory_value_size, self.sentence_size], name='memory_value')
                self.labels = tf.placeholder(tf.float32, [None, self.vocab_size], name='answer')
            else:
                self.memory_key = tf.placeholder(tf.int32, [None, self.memory_key_size, self.max_memory_sentence_size], name='memory_key')
                self.memory_value = tf.placeholder(tf.int32, [None, self.memory_value_size, self.max_memory_sentence_size], name='memory_value')
                self.labels = tf.placeholder(tf.float32, [None, self.n_entity], name='answer')

            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    '''
    mkeys: the vector representation for keys in memory
    -- shape of each mkeys: [1, embedding_size]
    mvalues: the vector representation for values in memory
    -- shape of each mvalues: [1, embedding_size]
    questions: the vector representation for the question
    -- shape of questions: [1, embedding_size]
    -- shape of R: [feature_size, feature_size]
    -- shape of self.A: [feature_size, embedding_size]
    -- shape of self.B: [feature_size, embedding_size]
    self.A, self.B and R are the parameters to learn
    '''
    def _key_addressing(self, mkeys, mvalues, questions, r_list): # doc_r, value_r, q_r, r_list)
        
        with tf.variable_scope(self.name):
            # [feature_size, batch_size]
            u = tf.matmul(self.A, questions, transpose_b=True)
            u = [u]
            for _ in range(self.n_hop):
                R = r_list[_]
                u_temp = u[-1]
                mk_temp = mkeys + self.TK
                # [reader_size, batch_size x memory_size]
                k_temp = tf.reshape(tf.transpose(mk_temp, [2, 0, 1]), [self.reader_feature_size, -1])
                # [feature_size, batch_size x memory_size]
                a_k_temp = tf.matmul(self.A, k_temp)
                # [batch_size, memory_size, feature_size]
                a_k = tf.reshape(tf.transpose(a_k_temp), [-1, self.memory_key_size, self.feature_size])
                # [batch_size, 1, feature_size]
                u_expanded = tf.expand_dims(tf.transpose(u_temp), [1])
                # [batch_size, memory_size]
                dotted = tf.reduce_sum(a_k*u_expanded, 2)

                # Calculate probabilities
                # [batch_size, memory_size]
                probs = tf.nn.softmax(dotted)
                # [batch_size, memory_size, 1]
                probs_expand = tf.expand_dims(probs, -1)
                mv_temp = mvalues + self.TV
                # [reader_size, batch_size x memory_size]
                v_temp = tf.reshape(tf.transpose(mv_temp, [2, 0, 1]), [self.reader_feature_size, -1])
                # [feature_size, batch_size x memory_size]
                a_v_temp = tf.matmul(self.A, v_temp)
                # [batch_size, memory_size, feature_size]
                a_v = tf.reshape(tf.transpose(a_v_temp), [-1, self.memory_key_size, self.feature_size])
                # [batch_size, feature_size]
                o_k = tf.reduce_sum(probs_expand*a_v, 1)
                # [feature_size, batch_size]
                o_k = tf.transpose(o_k)
                # [feature_size, batch_size]
                u_k = tf.matmul(R, u[-1]+o_k)

                u.append(u_k)

            return u[-1]
