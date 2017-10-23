# -*- coding: utf-8 -*-

import pickle
from nltk.tokenize import word_tokenize
import numpy as np
import functools
from itertools import chain
import copy

def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)

def load_pickle(path):
    print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)

def lower_list(word_list):
    return [w.lower() for w in word_list]

def load_entities(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        entities = [e.lower().rstrip() for e in lines]
        return list(set(entities))

def find_ngrams(token_dict, text, n):
    """ See: https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/dict.py#L31
        token_dict:  {'hello world', 'ol boy'}
        text: ['hello', 'world', 'buddy', 'ol', 'boy']
        n: max n of n-gram
        ret: ['hello world', 'buddy', 'ol boy']
    """
    """Breaks text into ngrams that appear in ``token_dict``."""
    # base case
    if n <= 1:
        return text
    # tokens committed to output
    saved_tokens = []
    # tokens remaining to be searched in sentence
    search_tokens = text[:]
    # tokens stored until next ngram found
    next_search = []
    while len(search_tokens) >= n:
        ngram = ' '.join(search_tokens[:n])
        if ngram in token_dict:
            # first, search previous unmatched words for smaller ngrams
            sub_n = min(len(next_search), n - 1)
            saved_tokens.extend(find_ngrams(token_dict, next_search, sub_n))
            next_search.clear()
            # then add this ngram
            saved_tokens.append(ngram)
            # then pop this ngram from the remaining words to search
            search_tokens = search_tokens[n:]
        else:
            next_search.append(search_tokens.pop(0))
    remainder = next_search + search_tokens
    sub_n = min(len(remainder), n - 1)
    saved_tokens.extend(find_ngrams(token_dict, remainder, sub_n))
    return saved_tokens

def load_task(fpath):
    print('load', fpath)
    with open (fpath, encoding='utf-8') as f:
        lines = f.readlines()
        data = []
        for i, l in enumerate(lines):
            l = l.rstrip()
            turn, left = l.split(' ', 1)
            
            if '\t' in left: # question
                q, a = left.split('\t', 1)
                q = q.split('?', 1)[1] # use split by n-gram
                q = q.split(' 1:')[1:]
                q = lower_list(q)
            
                a = a.split(', ') # may contain several labels
                a = lower_list(a)
                data.append((q, a))
    return data
    # data_q = load_questions(q_fpath)
    # data_a = load_answers(a_fpath)
    # return [(q, a) for q, a in zip(data_q, data_a)]

# def load_task(fpath, token_dict=None, max_token_length=None):
#     # print('load', fpath)
#     # with open (fpath, encoding='utf-8') as f:
#     #     lines = f.readlines()
#     #     data, story = [], []
#     #     for i, l in enumerate(lines):
#     #         if i % 2000 == 0: print(i, '/', len(lines))
#     #         l = l.rstrip()
#     #         turn, left = l.split(' ', 1)
            
#     #         if turn == '1': # new story
#     #             story = []

#     #         if '\t' in left: # question
#     #             q, a = left.split('\t', 1)
#     #             if q[-1] == '?':
#     #                 q = q[:-1]
#     #             # q = word_tokenize(q)
#     #             q = q.split(' ')
#     #             q = lower_list(q)
#     #             if token_dict and max_token_length:
#     #                 q = find_ngrams(token_dict, q, max_token_length)

#     #             if '\t' in a:
#     #                 a = a.split('\t')[0] # discard reward
#     #             a = a.split('|') # may contain several labels
#     #             a = lower_list(a)

#     #             substory = [x for x in story if x]

#     #             data.append((substory, q, a))
#     #             story.append('')
#     #         else: # normal sentence
#     #             # s = word_tokenize(left)
#     #             s = left.split(' ')
#     #             if s[-1] == '.':
#     #                 s = s[:-1]
#     #             s = lower_list(s)
#     #             story.append(s)

#     # return data

def vectorize(data, w2i, query_maxlen, w2i_label, use_multi_label=False):
    Q, A = [], []
    for question, answer in data:
        # Vectroize question
        q = [w2i[w] for w in question if w in w2i]
        q = q[:query_maxlen]
        q_pad_len = max(0, query_maxlen - len(q))
        q += [0] * q_pad_len

#         y = np.zeros(len(w2i_label))
#         y[w2i_label[answer[0]]] = 1
        y = np.zeros(len(w2i_label))
        if use_multi_label:
            for a in answer:
                y[w2i_label[a]] = 1
        else:
            y[w2i_label[answer[0]]] = 1

        Q.append(q)
        A.append(y)
    
    Q = np.array(Q, dtype=np.uint32)
    A = np.array(A, dtype='byte')

    return Q, A

def load_kv_pairs(path, token_dict, max_token_length, is_save_pickle=False):
    """load key-value paris from KB"""
    # rel = ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags', 'has_plot', 'in_language'] # TODO hard code
    rel = ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags', 'in_language'] # TODO hard code, not use has_plot tag
    kv_pairs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if i % 5000 == 0: print('load_kv_pairs:', i, '/', len(lines))
            if l == '\n': continue
            turn, left = l.rstrip().split(' ', 1)
            for r in rel:
                if r in left:
                    k, v = [], []
                    tmp = left.split(r)
                    lhs = tmp[0].rstrip().lower()
                    k.append(lhs)
                    k.append(r)
                    rhs = tmp[1].strip().lower()
                    vals = rhs.split(', ')
                    for v in vals:
                        kv_pairs.append((k, [v]))

                        # double KB by considering reversed relation. see 3.2
                        k_r = [v, '!'+r]
                        v_r = [lhs]
                        kv_pairs.append((k_r, v_r))

                    break

    if is_save_pickle:
        save_pickle(kv_pairs, 'pickle/mov_kv_pairs.pickle')

    return kv_pairs

def vectorize_kv(data, max_mem_len, max_mem_size, w2i):
    all_vec_list = []
    for i, kv_list in enumerate(data):
        if i % 5000 == 0: print('vectorize_kv:', i, '/', len(data))
        vec_list = []
        for kv in kv_list[:max_mem_len+100]: #TODO: +100 for unknown entity in w2i
            vec = [w2i[e] for e in kv if e in w2i]
            # vec = [w2i[e] for e in kv]
            vec = vec[:max_mem_len]
            mem_pad_len = max(0, max_mem_len - len(vec))
            vec = vec + [0] * mem_pad_len
            vec_list.append(vec)
        vec_list = vec_list[:max_mem_size]
        mem_pad_size = max(0, max_mem_size - len(vec_list))
        for _ in range(mem_pad_size):
            vec_list.append([0] * max_mem_len)
        all_vec_list.append(vec_list)

    return np.array(all_vec_list, dtype=np.uint32)

def load_kv_dataset(data, kv_pairs, stopwords):
    print('---',len(data), len(kv_pairs))
    data_k, data_v = [], []
    for i, (q, _) in enumerate(data):
        if i%100 == 0: print('load_kv_dataset:', i, '/', len(data))
        k_list, v_list = [], []
        for w in q:
            if w not in stopwords:
                for kv_ind, (k, v) in enumerate(kv_pairs):
                    if w in (k): # the key shares at least one word with question with F<1000
                        k_list.append(k)
                        v_list.append(v)
        if len(k_list) == 0:
            print('==================no kv!')
            print(q)
        if len(k_list) > 100:
            print('==================too many kv! > 100')
            print(q)
            print(len(k_list))
        data_k.append(k_list)
        data_v.append(v_list)
        
    return data_k, data_v

def get_stop_words(data, freq, token_dict, max_token_length, is_save_pickle):
    # train_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_train.txt')
    # test_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_test.txt')
    # dev_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_dev.txt')
    # data = train_data + test_data + dev_data
    bow = {}
    for i, (q, _) in enumerate(data):
        if i % 2000 == 0: print(i, '/', len(data))
        for qq in q:
            q_tokens = find_ngrams(token_dict, qq.split(' '), max_token_length)
            for w in q_tokens:
                if w not in bow:
                    bow[w] = 0
                else:
                    bow[w] += 1

    stopwords = [k for k, v in bow.items() if v >= freq]
    if is_save_pickle:
        save_pickle(stopwords, 'pickle/mov_stopwords.pickle')

    return stopwords

def filter_data(data, data_k, data_v, kv_min, kv_max):
    indices = []
    for i, k in enumerate(data_k):
        if len(k) > kv_min and len(k) <= kv_max:
            indices.append(i)
    data = [data[i] for i in indices]
    data_k = [data_k[i] for i in indices]
    data_v = [data_v[i] for i in indices]
    return data, data_k, data_v

if __name__ == '__main__':
    # --- entities
    entities = load_pickle('pickle/mov_entities.pickle')
    # entities = load_entities('./data/movieqa/knowledge_source/entities.txt')
    # save_pickle(entities, 'mov_entities.pickle')
    # max_entity_length = max(map(len, (e.split(' ') for e in entities))) # for searching n-gram
    # vocab = load_pickle('pickle/mov_vocab.pickle')

    # --- movie-qa train/test dataset
    train_data = load_task('./data/WikiMovies/train_1.txt')
    test_data = load_task('./data/WikiMovies/test_1.txt')
    dev_data = load_task('./data/WikiMovies/dev_1.txt')
    save_pickle(train_data, 'pickle/mov_task1_qa_pipe_train.pickle')
    save_pickle(test_data, 'pickle/mov_task1_qa_pipe_test.pickle')
    save_pickle(dev_data, 'pickle/mov_task1_qa_pipe_dev.pickle')
    # train_data = load_pickle('pickle/mov_task1_qa_pipe_train.pickle')
    # test_data = load_pickle('pickle/mov_task1_qa_pipe_test.pickle')
    # dev_data = load_pickle('pickle/mov_task1_qa_pipe_dev.pickle')
    print(len(train_data))

    # -- update vocab and w2i/i2w
    vocab = set(entities 
                + ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags', 'in_language'] 
                + ['!directed_by', '!written_by', '!starred_actors', '!release_year', '!has_genre', '!has_tags', '!in_language'] )
    for q, answer in train_data + test_data + dev_data:
        vocab |= set(q + answer)
    vocab = sorted(list(vocab))
    save_pickle(vocab, 'pickle/mov_vocab.pickle')
    w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    i2w = dict((i, c) for i, c in enumerate(vocab, 1))
    save_pickle(w2i, 'pickle/mov_w2i.pickle')
    save_pickle(i2w, 'pickle/mov_i2w.pickle')
    # vocab = load_pickle('pickle/mov_vocab.pickle')
    
    # generate kv_pairs
    # kv_pairs = load_kv_pairs('./data/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt', entities,  100, True)
    # kv_pairs = load_pickle('pickle/mov_kv_pairs.pickle')
    # vec_kv_pairs = vectorize_kv_pairs(kv_pairs, 10, 30, entities)
    
    # generate stopwords
    # stopwords = get_stop_words(train_data+test_data+dev_data, 1000, vocab, 100, True)
    # stopwords = load_pickle('pickle/mov_stopwords.pickle')

    # train_k, train_v = load_kv_dataset(train_data, kv_pairs, stopwords)
    # save_pickle(train_k, 'pickle/mov_train_k.pickle')
    # save_pickle(train_v, 'pickle/mov_train_v.pickle')
    # test_k, test_v = load_kv_dataset(test_data, kv_pairs, stopwords)
    # save_pickle(test_k, 'pickle/mov_test_k.pickle')
    # save_pickle(test_v, 'pickle/mov_test_v.pickle')
    # dev_k, dev_v = load_kv_dataset(dev_data, kv_pairs, stopwords)
    # save_pickle(test_k, 'pickle/mov_dev_k.pickle')
    # save_pickle(test_v, 'pickle/mov_dev_v.pickle')
    
