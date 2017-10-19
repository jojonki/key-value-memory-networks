from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Lambda, Permute, Dropout, add, multiply, dot
from keras.layers.normalization import BatchNormalization

def MemNNKV(mem_len, mem_size, query_maxlen, vocab_size, embd_size, answer_size):
    print('mem_size:', mem_size)
    print('q_max', query_maxlen)
    print('embd_size', embd_size)
    print('vocab_size', vocab_size)
#     print('entity_size', entity_size)
    print('-----------')

    # placeholders
    key = Input((mem_size, mem_len,), name='Key_Input')
    val = Input((mem_size, mem_len,), name='Val_Input')
    question = Input((query_maxlen,), name='Question_Input')
    # print('key:', key.shape)

    shared_embd_A = Embedding(input_dim=vocab_size, output_dim=embd_size)

    key_encoded = shared_embd_A(key) # (None, mem_size, mem_len, embd_size)
#     key_encoded = Dropout(.3)(key_encoded)
    # print('key_encoded', key_encoded.shape)
    key_encoded = Lambda(lambda x: K.sum(x, axis=2)) (key_encoded) #(None, mem_size, embd_size)
    key_encoded = BatchNormalization()(key_encoded)
    # print('key_encoded', key_encoded.shape)
    val_encoded = shared_embd_A(val) # (None, mem_size, embd_size)
#     val_encoded = Dropout(.3)(val_encoded)
    val_encoded = Lambda(lambda x: K.sum(x, axis=2)) (val_encoded)
    val_encoded = BatchNormalization()(val_encoded)
    
    question_encoded = shared_embd_A(question) # (None, query_max_len, embd_size)
#     question_encoded = Dropout(.3)(question_encoded)
#     question_encoded = Lambda(lambda x: K.sum(x, axis=1)) (val_encoded)
    question_encoded = Lambda(lambda x: K.sum(x, axis=1)) (question_encoded) #(None, embd_size)
    question_encoded = BatchNormalization()(question_encoded)
    # print('q_encoded', question_encoded.shape)
    q= question_encoded
    for h in range(2):
        # print('---hop', h)
        ph = dot([q, key_encoded], axes=(1, 2))  # (None, mem_size)
        # print('ph', ph.shape)
        ph = Activation('softmax')(ph)
        o = multiply([ph, Permute((2, 1))(val_encoded)]) # (None, embd_size, mem_size)
        # print('o', o.shape)
        o = Lambda(lambda x: K.sum(x, axis=2))(o) # (None, embd_size)
        # print('o', o.shape)
#         R = Dense(embd_size, input_shape=(embd_size,), name='R_Dense_h' + str(h+1), kernel_regularizer=regularizers.l2(0.01))
        R = Dense(embd_size, input_shape=(embd_size,), name='R_Dense_h' + str(h+1))
        q = R(add([q,  o])) # (None, embd_size)
        q = BatchNormalization()(q)
        # print('q', q.shape)

#     answer = Dense(answer_size, name='last_Dense')(q) #(None, answer_size)
#     answer = Dense(vocab_size, name='last_Dense', kernel_regularizer=regularizers.l2(0.01))(q) #(None, vocab_size)
    answer = Dense(vocab_size, name='last_Dense')(q) #(None, vocab_size)
    answer = BatchNormalization()(answer)
    # print('answer.shape', answer.shape)
    preds = Activation('softmax')(answer)
    
    # build the final model
    model = Model([key, val, question], preds)
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model

