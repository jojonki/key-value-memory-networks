# from keras import backend as K
import numpy as np
from keras.models import load_model
from nltk.tokenize import word_tokenize
from process_data import find_ngrams, load_pickle, lower_list

mem_maxlen = 100
query_maxlen = 21

print('load data...')
model = load_model('demo_model_memnn_kv.h5')

vocab = load_pickle('mov_vocab.pickle')
w2i = load_pickle('mov_w2i.pickle')
i2w = load_pickle('mov_i2w.pickle')
kv_pairs = load_pickle('mov_kv_pairs.pickle')
stopwords = load_pickle('mov_stopwords.pickle')

def predict(q):
    # tokenize a question
    q_tokens = word_tokenize(q)
    q_tokens = lower_list(q_tokens)
    q_tokens = find_ngrams(vocab, q_tokens, 100000)
    print('q_tokens:', q_tokens)

    # vectorize a question
    vec_q = [w2i[w] for w in q_tokens if w in w2i]
    q_pad_len = max(0, query_maxlen - len(vec_q))
    vec_q += [0] * q_pad_len
    vec_q = np.array(vec_q)
    vec_q = np.reshape(vec_q, (1, len(vec_q)))
    # print('vec_q:', vec_q)

    # get related kv
    k_list, v_list = [], []
    for w in q_tokens:
        if w not in stopwords:
            for kv_ind, (k, v) in enumerate(kv_pairs):
                if w in (k+v):
                    k_list += k
                    v_list += v
        else:
            pass
            # print(w, 'in stopwords')

    def _vec_kv(data, w2i, mem_maxlen):
        vec = [w2i[e] for e in data if e in w2i]
        vec += [0] * max(0, mem_maxlen - len(vec))
        vec = vec[:mem_maxlen]
        vec = np.array(vec)
        vec = np.reshape(vec, (1, 100))
        
        return vec
    # vectroize kv
    vec_k, vec_v = None, None
    vec_k = _vec_kv(k_list, w2i, mem_maxlen)
    vec_v = _vec_kv(v_list, w2i, mem_maxlen)

    int_predict = model.predict([vec_k, vec_v, vec_q], batch_size=1, verbose=0)     
    # print('q:',q)
    print('A:', i2w[np.argmax(int_predict[0])])

while True:
    q = input("Question: ")
    if q != '':
        predict(q)
