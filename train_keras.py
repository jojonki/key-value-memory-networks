from __future__ import print_function

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Lambda, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re

import tensorflow as tf


from itertools import chain
from process_data import load_entities, save_pickle, load_pickle, load_kv_pairs, lower_list


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def load_task(fpath, is_babi, max_length=None):
    with open (fpath, encoding='utf-8') as f:
        lines = f.readlines()
        data, story = [], []
        for l in lines:
            l = l.rstrip()
            turn, left = l.split(' ', 1)
            
            if turn == '1': # new story
                story = []

            if '\t' in left: # question
                q, a = left.split('\t', 1)
                q = tokenize(q)
                q = lower_list(q)
                if q[-1] == '?':
                    q = q[:-1]
                if '\t' in a:
                    a = a.split('\t')[0] # discard reward
                a = a.split('|') # may contain several labels
                a = lower_list(a)

                substory = [x for x in story if x]

                data.append((substory, q, a))
                story.append('')
            else: # normal sentence
                s = tokenize(left)
                if s[-1] == '.':
                    s = s[:-1]
                s = lower_list(s)
                story.append(s)

    if is_babi:
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
        
    return data

def vectorize(data, w2i, story_maxlen, query_maxlen, entities=None):
    if entities:
        e2i = dict((e, i) for i, e in enumerate(entities))

    S, Q, A = [], [], []
    for story, question, answer in data:
        # Vectroize story
        s_pad_len = max(0, story_maxlen - len(story))
        s = [w2i[w] for w in story] + [0] * s_pad_len

        # Vectroize question
        q_pad_len = max(0, query_maxlen - len(question))
        q = [w2i[w] for w in question] + [0] * q_pad_len
        q = q[:query_maxlen]

        # Vectroize answer
        if entities:
            y = np.zeros(len(entities), dtype='byte')
            for a in answer:
                y[e2i[a]] = 1
        else:
            y = np.zeros(len(w2i) + 1) # +1 for nil word
            for a in answer:
                y[w2i[a]] = 1

        S.append(s)
        Q.append(q)
        A.append(y)
    
    S = np.array(S, dtype=np.uint16)
    Q = np.array(Q, dtype=np.uint16)
    A = np.array(A, dtype='byte')

    return S, Q, A

def vectorize_kv_pairs(kv_pairs, memory_size, entities):
    vec_kv_pairs = []
    w2i = dict((e, i) for i, e in enumerate(entities))
    w2i['directed_by'] = len(w2i)
    w2i['written_by'] = len(w2i)
    w2i['starred_actors'] = len(w2i)
    w2i['release_year'] = len(w2i)
    w2i['has_genre'] = len(w2i)
    w2i['has_tags'] = len(w2i)
    w2i['has_plot'] = len(w2i)
    for ent_list in kv_pairs:
#         print('----ent_list', ent_list)
#         print(len(ent_list))
        kv = [w2i[e] for e in ent_list if e in w2i]
        mem_pad_len = max(0, memory_size - len(kv))
        vec_kv_pairs.append(kv + [0] * mem_pad_len)


    return np.array(vec_kv_pairs, dtype=np.uint16)

is_babi = False
if is_babi:
    train_stories = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt', is_babi)
    test_stories = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt', is_babi)
else:
    N = 50000
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

def MemNNKV(mem_size, query_maxlen, vocab_size, entity_size, embd_size):
    print('mem_size:', mem_size)
    print('q_max', query_maxlen)
    print('embd_size', embd_size)
    print('vocab_size', vocab_size)
    print('entity_size', entity_size)
    # placeholders
    key = Input((mem_size,), name='Key_Input')
    val = Input((mem_size,), name='Val_Input')
    question = Input((query_maxlen,), name='Question_Input')

    # encoders
    # memory encoders
    key_encoder = Sequential(name='Key_Encoder')
    key_encoder.add(Embedding(input_dim=entity_size, output_dim=embd_size, input_length=mem_size))
    val_encoder = Sequential(name='Val_Encoder')
    val_encoder.add(Embedding(input_dim=entity_size, output_dim=embd_size, input_length=mem_size))
    # output: (samples, mem_size, embd_size)

    # embed the question into a sequence of vectors
    question_encoder = Sequential(name='Question_Encoder')
    question_encoder.add(Embedding(input_dim=vocab_size, output_dim=embd_size, input_length=query_maxlen))
#     question_encoder.add(Dropout(0.3))
    # output: (samples, query_maxlen, embd_size)

    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    key_encoded = key_encoder(key) # (None, mem_size, embd_size)
    val_encoded = val_encoder(val) # (None, mem_size, embd_size)
    question_encoded = question_encoder(question) # (None, query_max_len, embd_size)

    ph = dot([question_encoded, key_encoded], axes=(2, 2)) 
    ph = Permute((2, 1))(ph) # (None, mem_size, query_max_len)
    o = dot([ph, val_encoded], axes=(1, 1)) # (None, query_max_len, embd_size)
    R = Dense(embd_size, input_shape=(embd_size,), name='R_Dense')     
    q2 = R(add([question_encoded,  o])) # (None, query_max_len, embd_size)
    
    cand_encoder = Sequential(name='cand_encoder')
    cand_encoder.add(Embedding(input_dim=entity_size, output_dim=embd_size, input_length=1))
#     cand_encoder.add(Dropout(0.3))
    
    cand = Input((entity_size,), name='Cand_Input')
    y_encoded = cand_encoder(cand) # (None, entity_size, embd_size)
#     print('y_encoded', y_encoded.shape)
    
    answer = dot([q2, y_encoded], axes=(2, 2)) # (None, query_max_len, entity_size)
    answer = Lambda(lambda x: K.sum(x, axis=1), output_shape=(entity_size, )) (answer)
    preds = Activation('softmax')(answer)
    print('--answer', answer.shape)
    
    # build the final model
    model = Model([key, val, question, cand], answer)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

embd_size = 64
memnn_kv = MemNNKV(mem_maxlen, query_maxlen, vocab_size, entity_size, embd_size)
print(memnn_kv.summary())
# train_cands = 
memnn_kv.fit([vec_train_kv, vec_train_kv, queries_train, answers_train], answers_train,
          batch_size=32,
          epochs=10)#,
          # validation_data=([vec_test_kv, vec_test_kv, queries_test, answers_test], answers_test))
