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

def load_task(fpath, token_dict=None, max_token_length=None):
    print('load', fpath)
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
                q = word_tokenize(q)
                q = lower_list(q)
                if q[-1] == '?':
                    q = q[:-1]
                if token_dict and max_token_length:
                    q = find_ngrams(token_dict, q, max_token_length)

                if '\t' in a:
                    a = a.split('\t')[0] # discard reward
                a = a.split('|') # may contain several labels
                a = lower_list(a)

                substory = [x for x in story if x]

                data.append((substory, q, a))
                story.append('')
            else: # normal sentence
                s = word_tokenize(left)
                if s[-1] == '.':
                    s = s[:-1]
                s = lower_list(s)
                story.append(s)

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

def load_kv_pairs(path, token_dict, max_token_length, is_save_pickle=False):
    """load key-value paris from KB"""
    rel = ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags', 'has_plot'] # TODO hard code
    kv_pairs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if i % 500 == 0: print(i, '/', len(lines))
            if l == '\n': continue
            turn, left = l.rstrip().split(' ', 1)
            for r in rel:
                if r in left:
                    k, v = [], []
                    tmp = left.split(r)
                    lhs = tmp[0].rstrip().lower()
                    k.append(lhs)
                    k.append(r)
                    vals = word_tokenize(tmp[1].strip().lower())#.split(' ')
                    vals = find_ngrams(token_dict, vals, max_token_length)
                    while ',' in vals:
                        vals.remove(',') #@TODO
                    if r == 'has_plot':
                        kv_pairs.append((k, vals))
                    else:
                        for v in vals:
                            kv_pairs.append((k, [v]))

                    # double KB by considering reversed relation. see 3.2
                    if r == 'has_plot':
                        kv_pairs.append((vals + ['!'+r], [lhs]))
                    else:
                        for v in vals:
                            k_r = [v, '!'+r]
                            v_r = [lhs]
                            kv_pairs.append((k_r, v_r))
                        
                    break

    if is_save_pickle:
        save_pickle(kv_pairs, 'mov_kv_pairs.pickle')

    return kv_pairs

def get_relative_kv(kv_indices, kv_pairs):
    print('get relative key-values...')
    data_k, data_v = [], []
    for i, indices in enumerate(kv_indices):
        # if i % 10000 == 0: print(i, '/', len(kv_indices))
        k_list, v_list = [], []
        for kv_ind in indices:
            k_list.append(kv_pairs[kv_ind][0])
            v_list.append(kv_pairs[kv_ind][1])
        data_k.append(k_list)
        data_v.append(v_list)
    return data_k, data_v
        
def vectorize_kv(kv, memory_size, w2i):
    vec_list = []
    w2i['directed_by'] = len(w2i)
    w2i['written_by'] = len(w2i)
    w2i['starred_actors'] = len(w2i)
    w2i['release_year'] = len(w2i)
    w2i['has_genre'] = len(w2i)
    w2i['has_tags'] = len(w2i)
    w2i['has_plot'] = len(w2i)
    def _vectroize(data, w2i, memory_size):
        vec = [w2i[e] for e in data if e in w2i]
        mem_pad_len = max(0, memory_size - len(vec))
        return (vec_k + [0] * mem_pad_len)

    for data in kv:
        vec = [w2i[e] for e in data if e in w2i]
        mem_pad_len = max(0, memory_size - len(vec))
        vec_list.append(vec + [0] * mem_pad_len)

    return np.array(vec_list, dtype=np.uint16)

def get_kv_indices(data, kv_pairs):
    kv_indices = []
    for i, (_, q, _) in enumerate(data):
        if i%100 == 0: print(i, '/', len(data))
        indices = []
        for w in q:
            if w not in stopwords:
                for kv_ind, (k, v) in enumerate(kv_pairs):
                    if w in (k+v):
                        indices.append(kv_ind)
        kv_indices.append(indices)
    return kv_indices

def get_stop_words(freq, is_save_pickle):
    train_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_train.txt')
    test_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_test.txt')
    dev_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_dev.txt')
    data = train_data + test_data + dev_data
    bow = {}
    for _, q, _ in data:
        for qq in q:
            for w in qq.split(' '):
                if w not in bow:
                    bow[w] = 0
                else:
                    bow[w] += 1

    stopwords = [k for k, v in bow.items() if v >= freq]
    if is_save_pickle:
        save_pickle(stopwords, 'mov_stopwords.pickle')
    


if __name__ == '__main__':
    # --- entities
    entities = load_pickle('mov_entities.pickle')
    # entities = load_entities('./data/movieqa/knowledge_source/entities.txt')
    # save_pickle(entities, 'mov_entities.pickle')
    max_entity_length = max(map(len, (e.split(' ') for e in entities)))

    # --- movie-qa train/test dataset
    # train_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_train.txt', entities, max_entity_length)
    # test_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_test.txt', entities, max_entity_length)
    # save_pickle(train_data, 'mov_task1_qa_pipe_train.pickle')
    # save_pickle(test_data, 'mov_task1_qa_pipe_test.pickle')
    train_data = load_pickle('mov_task1_qa_pipe_train.pickle')
    test_data = load_pickle('mov_task1_qa_pipe_test.pickle')

    # -- update vocab and w2i/i2w
    # vocab = functools.reduce(lambda x, y: x | y, (set(chain(chain.from_iterable(s), q, a)) for s, q, a in train_data+test_data))
    # w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    # i2w = dict((i, c) for i, c in enumerate(vocab, 1))
    # save_pickle(vocab, 'vocab.pickle')
    # save_pickle(w2i, 'w2i.pickle')
    # save_pickle(i2w, 'i2w.pickle')
    
    # generate kv_pairs
    # kv_pairs = load_kv_pairs('./data/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt', entities,  max_entity_length, True)
    kv_pairs = load_pickle('mov_kv_pairs.pickle')
    # vec_kv_pairs = vectorize_kv_pairs(kv_pairs, 10, 30, entities)
    

    # generate stopwords
    # get_stop_words(1000, True)
    stopwords = load_pickle('mov_stopwords.pickle')

    train_kv_indices = get_kv_indices(train_data, kv_pairs)
    save_pickle(train_kv_indices, 'mov_train_kv_indices.pickle')
    test_kv_indices = get_kv_indices(test_data, kv_pairs)
    save_pickle(test_kv_indices, 'mov_test_kv_indices.pickle')
    
    # data = load_task('./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')
    # data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt')

    # vocab = functools.reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data))
    # w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    # i2w = dict((i, c) for i, c in enumerate(vocab, 1))
    # print(len(vocab))    print("HOGE")
    # S, Q, A = vectorize(data,)
