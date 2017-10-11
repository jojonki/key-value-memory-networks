from __future__ import print_function

from functools import reduce
from itertools import chain
import numpy as np

from process_data import load_entities, save_pickle, load_pickle, load_kv_pairs, lower_list, vectorize, vectorize_kv, get_relative_kv
from net.memnn_kv import MemNNKV

is_babi = False
if is_babi:
    train_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt', is_babi)
    test_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt', is_babi)
else:
    # N = 49900
    N = 50000000
    mem_maxlen = 100 # 1つのエピソードに関連しているKVの数に対する制限
    train_data = load_pickle('mov_task1_qa_pipe_train.pickle')[:N]
    test_data = load_pickle('mov_task1_qa_pipe_test.pickle')[:N]
    kv_pairs = load_pickle('mov_kv_pairs.pickle')
    train_kv_indices = load_pickle('mov_train_kv_indices.pickle')[:N]
    test_kv_indices = load_pickle('mov_test_kv_indices.pickle')[:N]
    train_k, train_v = get_relative_kv(train_kv_indices, kv_pairs)
    test_k, test_v = get_relative_kv(test_kv_indices, kv_pairs)
    train_k = np.array([list(chain(*x))[:mem_maxlen] for x in train_k])
    train_v = np.array([list(chain(*x))[:mem_maxlen] for x in train_v])
    test_k = np.array([list(chain(*x))[:mem_maxlen] for x in test_k])
    test_v = np.array([list(chain(*x))[:mem_maxlen] for x in test_v])
    entities = load_pickle('mov_entities.pickle')
    entity_size = len(entities)

# TODO
vocab = set(entities +  ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags', 'has_plot'] )
for story, q, answer in train_data + test_data:
    vocab |= set(story + q + answer)
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab)
story_maxlen = max(map(len, (x for x, _, _ in train_data + test_data)))
query_maxlen = max(map(len, (x for _, x, _ in train_data + test_data)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training data:', len(train_data))
print('Number of test data:', len(test_data))
print('-')
print('Here\'s what a "data" tuple looks like (input, query, answer):')
print(train_data[0])
print('-')
print('Vectorizing the word sequences...')

print('Number of entities', len(entities))
w2i = dict((c, i) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize(train_data,
                                                               w2i,
                                                               story_maxlen,
                                                               query_maxlen)
inputs_test, queries_test, answers_test = vectorize(test_data,
                                                            w2i,
                                                            story_maxlen,
                                                            query_maxlen)

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


# mem_maxlen = max(map(len, (x for x in train_kv+test_kv)))
# train_mem_maxlen = max(map(len, (x for x in train_kv)))
# test_mem_maxlen = max(map(len, (x for x in test_kv)))
# mem_maxlen = max(train_mem_maxlen, test_mem_maxlen)
#
# print('mem_maxlen:', mem_maxlen)
# vec_train_kv = vectorize_kv_pairs(train_kv, mem_maxlen, vocab)
# vec_test_kv = vectorize_kv_pairs(test_kv, mem_maxlen, vocab)

# e2i = dict((e, i) for i, e in enumerate(entities))
max_memory_num = 200
vec_train_k = vectorize_kv(train_k, mem_maxlen, w2i)
vec_train_v = vectorize_kv(train_v, mem_maxlen, w2i)
vec_test_k = vectorize_kv(test_k, mem_maxlen, w2i)
vec_test_v = vectorize_kv(test_v, mem_maxlen, w2i)
print('vec_k', vec_train_k.shape)
print('vec_v', vec_train_v.shape)

embd_size = 256
memnn_kv = MemNNKV(mem_maxlen, query_maxlen, vocab_size, embd_size)
print(memnn_kv.summary())
memnn_kv.fit([vec_train_k, vec_train_v, queries_train], answers_train,
          batch_size=32,
          epochs=30,
          validation_data=([vec_test_k, vec_test_v, queries_test], answers_test))

print('save model')
memnn_kv.save('model_memnn_kv.h5')
