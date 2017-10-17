from __future__ import print_function

from functools import reduce
from itertools import chain
import numpy as np
import datetime

from keras.callbacks import ModelCheckpoint

from process_data import load_entities, save_pickle, load_pickle, load_kv_pairs, lower_list, vectorize, vectorize_kv, load_kv_dataset
from net.memnn_kv import MemNNKV

is_babi = False
if is_babi:
    train_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt', is_babi)
    test_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt', is_babi)
else:
    # mem_maxlen         = 100 # 1つのエピソードに関連しているKVの数に対する制限
    train_data         = load_pickle('pickle/mov_task1_qa_pipe_train.pickle')
    test_data          = load_pickle('pickle/mov_task1_qa_pipe_test.pickle')
    kv_pairs           = load_pickle('pickle/mov_kv_pairs.pickle')
    train_k            = np.array(load_pickle('pickle/mov_train_k.pickle'))
    train_v            = np.array(load_pickle('pickle/mov_train_v.pickle'))
    test_k             = np.array(load_pickle('pickle/mov_test_k.pickle'))
    test_v             = np.array(load_pickle('pickle/mov_test_v.pickle'))
    entities           = load_pickle('pickle/mov_entities.pickle')
    entity_size        = len(entities)

# TODO
# vocab = set(entities 
#             + ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags', 'has_plot'] 
#             + ['!directed_by', '!written_by', '!starred_actors', '!release_year', '!has_genre', '!has_tags', '!has_plot'] )
# for _, q, answer in train_data + test_data:
#         vocab |= set(q + answer)
#         vocab = sorted(vocab)
vocab = load_pickle('mov_vocab.pickle')
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
stopwords = load_pickle('mov_stopwords.pickle')
# w2i = dict((c, i) for i, c in enumerate(vocab))
# i2w = dict((i, c) for i, c in enumerate(vocab))
# save_pickle(w2i, 'mov_w2i.pickle')
# save_pickle(i2w, 'mov_i2w.pickle')
w2i = load_pickle('pickle/mov_w2i.pickle')
i2w = load_pickle('pickle/mov_i2w.pickle')
queries_train, answers_train = vectorize(train_data, w2i, query_maxlen)
queries_test, answers_test = vectorize(test_data, w2i, query_maxlen)

print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)


max_mem_len = 4
max_mem_size = 15
vec_train_k = vectorize_kv(train_k, max_mem_len, max_mem_size, w2i)
vec_train_v = vectorize_kv(train_v, max_mem_len, max_mem_size, w2i)
vec_test_k = vectorize_kv(test_k, max_mem_len, max_mem_size, w2i)
vec_test_v = vectorize_kv(test_v, max_mem_len, max_mem_size, w2i)
print('vec_k', vec_train_k.shape)
print('vec_v', vec_train_v.shape)

embd_size = 200
memnn_kv =MemNNKV(max_mem_len, max_mem_size, query_maxlen, vocab_size, embd_size, None)
print(memnn_kv.summary())
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
model_path = 'saved_models/' + now + '_kvnn-weights-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
memnn_kv.fit([vec_train_k, vec_train_v, queries_train], answers_train,
          batch_size=32,
          epochs=30,
          callbacks=callbacks_list,
          validation_data=([vec_test_k, vec_test_v, queries_test], answers_test))

# print('save model')
# memnn_kv.save('model_memnn_kv.h5')
