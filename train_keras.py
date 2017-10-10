from __future__ import print_function

# from keras import backend as K
# from keras.models import Sequential, Model
# from keras.layers.embeddings import Embedding
# from keras.layers import Input, Activation, Dense, Lambda, Permute, Dropout, add, dot, concatenate
# from keras.layers import LSTM
# from keras.utils.data_utils import get_file
# from keras.preprocessing.sequence import pad_sequences
from functools import reduce
from itertools import chain
# # import tarfile
import numpy as np
# import re

import tensorflow as tf


from process_data import load_entities, save_pickle, load_pickle, load_kv_pairs, lower_list, vectorize, vectorize_kv_pairs
from net.memn2n_kv import MemNNKV

# def tokenize(sent):
#     '''Return the tokens of a sentence including punctuation.
#     >>> tokenize('Bob dropped the apple. Where is the apple?')
#     ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
#     '''
#     return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


is_babi = False
if is_babi:
    train_stories = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt', is_babi)
    test_stories = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt', is_babi)
else:
    N = 10000
    train_stories = load_pickle('mov_task1_qa_pipe_train.pickle')[:N]
    test_stories = load_pickle('mov_task1_qa_pipe_test.pickle')[:N]
    kv_pairs = load_pickle('mov_kv_pairs.pickle')
    train_kv_indices = load_pickle('mov_train_kv_indices.pickle')[:N]
    test_kv_indices = load_pickle('mov_test_kv_indices.pickle')[:N]
    train_kv = [ [kv_pairs[ind] for ind in indices] for indices in train_kv_indices ]
    test_kv = [ [kv_pairs[ind] for ind in indices] for indices in test_kv_indices ]
    train_kv = np.array([list(chain(*x)) for x in train_kv])
    test_kv = np.array([list(chain(*x)) for x in test_kv])
    print(len(train_kv), train_kv[0])
    
    entities = load_pickle('mov_entities.pickle')
    entity_size = len(entities)

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + answer)
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

print('len(entities)', len(entities))
w2i = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize(train_stories,
                                                               w2i,
                                                               story_maxlen,
                                                               query_maxlen, entities)
inputs_test, queries_test, answers_test = vectorize(test_stories,
                                                            w2i,
                                                            story_maxlen,
                                                            query_maxlen, entities)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)


print('train_kv[0]:', train_kv[0], ', mem_size:', len(train_kv[0]))
# mem_maxlen = max(map(len, (x for x in train_kv+test_kv)))
train_mem_maxlen = max(map(len, (x for x in train_kv)))
test_mem_maxlen = max(map(len, (x for x in test_kv)))
mem_maxlen = max(train_mem_maxlen, test_mem_maxlen)

print('mem_maxlen:', mem_maxlen)
vec_train_kv = vectorize_kv_pairs(train_kv, mem_maxlen, vocab)
vec_test_kv = vectorize_kv_pairs(test_kv, mem_maxlen, vocab)

embd_size = 64
memnn_kv = MemNNKV(mem_maxlen, query_maxlen, vocab_size, entity_size, embd_size)
print(memnn_kv.summary())
# train_cands = 
memnn_kv.fit([vec_train_kv, vec_train_kv, queries_train, answers_train], answers_train,
          batch_size=32,
          epochs=10)#,
          # validation_data=([vec_test_kv, vec_test_kv, queries_test, answers_test], answers_test))
