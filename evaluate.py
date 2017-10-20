from keras.models import load_model
import numpy as np
import argparse
from process_data import load_pickle, vectorize, vectorize_kv

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model',
                    required=True,
                    help='saved keras model')
parser.add_argument('--max_mem_len',
                    default=3,
                    help='max the number of words in one memory')
parser.add_argument('--max_mem_size',
                    default=100,
                    help='max the number of memories related one episode')
parser.add_argument('--max_query_len',
                    default=16,
                    help='max the number of words in question')
args = parser.parse_args()
print(args)
max_mem_len = args.max_mem_len
max_mem_size = args.max_mem_size
max_query_len = args.max_query_len
model_name = args.model

test_data = load_pickle('pickle/mov_task1_qa_pipe_test.pickle')
kv_pairs  = load_pickle('pickle/mov_kv_pairs.pickle')
test_k    = np.array(load_pickle('pickle/mov_test_k.pickle'))
test_v    = np.array(load_pickle('pickle/mov_test_v.pickle'))

vocab = load_pickle('pickle/mov_vocab.pickle')
vocab_size = len(vocab)
w2i = load_pickle('pickle/mov_w2i.pickle')
i2w = load_pickle('pickle/mov_i2w.pickle')
w2i_label = load_pickle('pickle/mov_w2i_label.pickle')
i2w_label = load_pickle('pickle/mov_i2w_label.pickle')


queries_test, answers_test = vectorize(test_data, w2i, max_query_len, w2i_label)
vec_test_k = vectorize_kv(test_k, max_mem_len, max_mem_size, w2i)
vec_test_v = vectorize_kv(test_v, max_mem_len, max_mem_size, w2i)

model = load_model(model_name)
ret = model.evaluate([vec_test_k, vec_test_v, queries_test], answers_test, verbose=1)
print('=====result=====')
print('loss: {:.5f}, acc: {:.5f}'.format(ret[0], ret[1]))

print('=====wrong examples=====')
pred = model.predict([vec_test_k, vec_test_v, queries_test], batch_size=32, verbose=1)
wrong_ct = 0
for i, (p, a) in enumerate(zip(pred, answers_test)):
    p_id = np.argmax(p)
#     a_ids = [i for i, aid in enumerate(a) if aid == 1 ]
#     if p_id not in a_ids:
    if a[p_id] == 0:
        wrong_ct += 1
print('')
n_data = len(answers_test)
print('acc:', (n_data-wrong_ct)/n_data, '=', wrong_ct, '/', n_data)