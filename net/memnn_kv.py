from __future__ import print_function

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Lambda, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
from keras import metrics
import numpy as np

def MemNNKV(mem_size, query_maxlen, vocab_size, embd_size):
    print('mem_size:', mem_size)
    print('q_max', query_maxlen)
    print('embd_size', embd_size)
    print('vocab_size', vocab_size)
#     print('entity_size', entity_size)
    print('-----------')

    # placeholders
    key = Input((mem_size,), name='Key_Input')
    val = Input((mem_size,), name='Val_Input')
    question = Input((query_maxlen,), name='Question_Input')

    # encoders
    # memory encoders
    # output: (None, mem_size, embd_size)
    # shared_embd_A = Embedding(input_dim=vocab_size, output_dim=embd_size, input_length=mem_size)
    shared_embd_A = Embedding(input_dim=vocab_size, output_dim=embd_size)
    key_encoder = Sequential(name='Key_Encoder')
    key_encoder.add(shared_embd_A)
#     key_encoder.add(Dropout(0.3))
    val_encoder = Sequential(name='Val_Encoder')
    val_encoder.add(shared_embd_A)
#     val_encoder.add(Dropout(0.3))

    # embed the question into a sequence of vectors
    # output: (None, query_maxlen, embd_size)
    question_encoder = Sequential(name='Question_Encoder')
#     question_encoder.add(Embedding(input_dim=vocab_size, output_dim=embd_size, input_length=query_maxlen))
    question_encoder.add(shared_embd_A)
#     question_encoder.add(Dropout(0.3))

    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    key_encoded = key_encoder(key) # (None, mem_size, embd_size)
    val_encoded = val_encoder(val) # (None, mem_size, embd_size)
    question_encoded = question_encoder(question) # (None, query_max_len, embd_size)
    
    ph = dot([question_encoded, key_encoded], axes=(2, 2)) # (None, query_max_len, mem_size)
    ph = Permute((2, 1))(ph) # (None, mem_size, query_max_len)
    o = dot([ph, val_encoded], axes=(1, 1)) # (None, query_max_len, embd_size)
    R = Dense(embd_size, input_shape=(embd_size,), name='R_Dense')     

    q2 = R(add([question_encoded,  o])) # (None, query_max_len, embd_size)
    
    answer = Dense(vocab_size, name='last_Dense')(q2)
    answer = Lambda(lambda x: K.sum(x, axis=1), output_shape=(vocab_size, )) (answer)
    print('answer.shape', answer.shape)
    preds = Activation('softmax')(answer)
    # preds = Activation('sigmoid')(answer)
    
    # build the final model
    model = Model([key, val, question], preds)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy', metrics.categorical_accuracy, metrics.sparse_categorical_accuracy])
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

