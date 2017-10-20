import numpy as np
import datetime
import json
import argparse

from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model

from process_data import save_pickle, load_pickle, load_kv_pairs, lower_list, vectorize, vectorize_kv, load_kv_dataset
from net.memnn_kv import MemNNKV


parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model', help='begin training from saved keras model')
args = parser.parse_args()

train_data  = load_pickle('pickle/mov_task1_qa_pipe_train.pickle')
test_data   = load_pickle('pickle/mov_task1_qa_pipe_test.pickle')
kv_pairs    = load_pickle('pickle/mov_kv_pairs.pickle')
train_k     = np.array(load_pickle('pickle/mov_train_k.pickle'))
train_v     = np.array(load_pickle('pickle/mov_train_v.pickle'))
test_k      = np.array(load_pickle('pickle/mov_test_k.pickle'))
test_v      = np.array(load_pickle('pickle/mov_test_v.pickle'))
entities    = load_pickle('pickle/mov_entities.pickle')
entity_size = len(entities)

# TODO
# vocab = set(entities 
#             + ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags', 'in_language'] 
#             + ['!directed_by', '!written_by', '!starred_actors', '!release_year', '!has_genre', '!has_tags', '!in_language'] )
# for _, q, answer in train_data + test_data:
#         vocab |= set(q + answer)
#         vocab = sorted(vocab)
vocab = load_pickle('pickle/mov_vocab.pickle')
vocab_size = len(vocab)
# label_list = []
# for _,_,a in (train_data+test_data):
#     for aa in a:
#         if aa not in label_list: label_list.append(aa)
# label_list = sorted(label_list)
# w2i_label = dict((c, i) for i, c in enumerate(label_list))
# i2w_label = dict((i, c) for i, c in enumerate(label_list))
# save_pickle(label_list, 'pickle/mov_label_list.pickle')
# save_pickle(w2i_label, 'pickle/mov_w2i_label.pickle')
# save_pickle(i2w_label, 'pickle/mov_i2w_label.pickle')

train_indices, test_indices = [], []
for i, k in enumerate(train_k):
    if len(k) != 0 and len(k) < 100:
        train_indices.append(i)
for i, k in enumerate(test_k):
    if len(k) != 0 and len(k) < 100:
        test_indices.append(i)
print('before filter:', len(train_data), len(test_data))
train_data = [train_data[i] for i in train_indices]
train_k = [train_k[i] for i in train_indices]
train_v = [train_v[i] for i in train_indices]
test_data = [test_data[i] for i in test_indices]
test_k = [test_k[i] for i in test_indices]
test_v = [test_v[i] for i in test_indices]
print('after filter:', len(train_data), len(test_data))

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
stopwords = load_pickle('pickle/mov_stopwords.pickle')
# w2i = dict((c, i) for i, c in enumerate(vocab))
# i2w = dict((i, c) for i, c in enumerate(vocab))
# save_pickle(w2i, 'mov_w2i.pickle')
# save_pickle(i2w, 'mov_i2w.pickle')
w2i = load_pickle('pickle/mov_w2i.pickle')
i2w = load_pickle('pickle/mov_i2w.pickle')
w2i_label = load_pickle('pickle/mov_w2i_label.pickle')
i2w_label = load_pickle('pickle/mov_i2w_label.pickle')
queries_train, answers_train = vectorize(train_data, w2i, query_maxlen, w2i_label)
queries_test, answers_test = vectorize(test_data, w2i, query_maxlen, w2i_label)

print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)


# max_mem_len = 3
mem_key_len = 2
mem_val_len = 1
max_mem_size = 100
vec_train_k = vectorize_kv(train_k, mem_key_len, max_mem_size, w2i)
vec_train_v = vectorize_kv(train_v, mem_val_len, max_mem_size, w2i)
vec_test_k = vectorize_kv(test_k, mem_key_len, max_mem_size, w2i)
vec_test_v = vectorize_kv(test_v, mem_val_len, max_mem_size, w2i)
print('vec_k', vec_train_k.shape)
print('vec_v', vec_train_v.shape)

embd_size = 500
if args.model:
    print('load saved model')
    memnn_kv = load_model(args.model)
else:
    memnn_kv =MemNNKV(mem_key_len, mem_val_len, max_mem_size, query_maxlen, vocab_size, embd_size, len(w2i_label))
print(memnn_kv.summary())

now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
model_path = '/media/jonki/H/models/' + now + '_kvnn-weights-{epoch:02d}-{val_acc:.4f}.hdf5'
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
log_path = 'result/' + now + '_emb{}-memsize{}'.format(embd_size, max_mem_size) + '.json'
class History(Callback):
    def on_train_begin(self, logs={}):
        self.result = []

    def on_epoch_end(self, epoch, logs={}):
        global log_path
        logs['epoch'] = epoch
        self.result.append(logs)
        with open(log_path, 'wt') as f:
            f.write(json.dumps(self.result, indent=4, sort_keys=True))

history = History()
callbacks_list = [checkpoint, history]
memnn_kv.fit([vec_train_k, vec_train_v, queries_train], answers_train,
          batch_size=64,
          epochs=30,
          callbacks=callbacks_list,
          validation_data=([vec_test_k, vec_test_v, queries_test], answers_test))
# print(json.dumps(history.result))
