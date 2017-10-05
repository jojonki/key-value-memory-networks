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
        return [e.lower().rstrip() for e in lines]

def load_task(fpath):
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

def vectorize(data, w2i, max_sentence_size, memory_size, entities=None):
    if entities:
        e2i = dict((e, i) for i, e in enumerate(entities))

    S, Q, A = [], [], []
    for story, question, answer in data:
        # Vectroize story
        ss = []
        for sentence in story:
            s_pad_len = max(0, max_sentence_size - len(sentence))
            ss.append([w2i[w] for w in sentence] + [0] * s_pad_len)

        ss = ss[::-1][:memory_size] # discard old memory lager than memory_size(max_story_size)
        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * max_sentence_size)

        # Vectroize question
        q_pad_len = max(0, max_sentence_size - len(question))
        q = [w2i[w] for w in question] + [0] * q_pad_len

        # Vectroize answer
        if entities:
            y = np.zeros(len(entities), dtype='byte')
            for a in answer:
                y[e2i[a]] = 1
        else:
            y = np.zeros(len(w2i) + 1) # +1 for nil word
            for a in answer:
                y[w2i[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    
    S = np.array(S, dtype=np.uint16)
    Q = np.array(Q, dtype=np.uint16)
    A = np.array(A, dtype='byte')

    return S, Q, A

def load_kv_pairs(path, entities, is_save_pickle=False):
    """load key-value paris from KB"""
    rel = ['directed_by', 'written_by', 'starred_actors', 'release_year', 'has_genre', 'has_tags']#, 'has_plot'] # TODO hard code
    kv_pairs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if l == '\n': continue
            k = []
            turn, left = l.rstrip().split(' ', 1)

            for r in rel:
                if r in left:
                    tmp = left.split(r)
                    # TODO
                    k.append(tmp[0].rstrip().lower())
                    k.append(tmp[1].split(',')[0].lstrip().lower()) # TODO may not valid entity
                    k.append(r)
                    kv_pairs.append(k)
                    break
        # kv_pairs = [l.rstrip().split(' ', 1)[1].split() for l in lines if l != '\n'] # key==value in Sentence Level

    if is_save_pickle:
        save_pickle(kv_pairs, 'mov_kv_pairs.pickle')

    return kv_pairs
        
def vectorize_kv_pairs(kv_pairs, max_sentence_size, memory_size, entities):
    vec_kv_pairs = []
    w2i = dict((e, i) for i, e in enumerate(entities))
    w2i['directed_by'] = len(w2i)
    w2i['written_by'] = len(w2i)
    w2i['starred_actors'] = len(w2i)
    w2i['release_year'] = len(w2i)
    w2i['has_genre'] = len(w2i)
    w2i['has_tags'] = len(w2i)
    # w2i['has_plot'] = len(w2i)
    for sentence in kv_pairs:
        kv = [w2i[w] for w in sentence if w in w2i]
        pad_len = max(0, max_sentence_size - len(kv))
        vec_kv_pairs.append(kv + [0] * pad_len)
    
    # pad to memory_size
    lm = max(0, memory_size - len(vec_kv_pairs))
    for _ in range(lm):
        vec_kv_pairs.append([0] * max_sentence_size)

    return np.array(vec_kv_pairs, dtype=np.uint16)

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
    # entities = load_pickle('mov_entities.pickle')
    # kv_pairs = load_kv_pairs('./data/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt', entities, True)
    # vec_kv_pairs = vectorize_kv_pairs(kv_pairs, 10, 30, entities)

    get_stop_words(1000, True)
    
    # data = load_task('./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')
    # data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt')

    # vocab = functools.reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data))
    # w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    # i2w = dict((i, c) for i, c in enumerate(vocab, 1))
    # print(len(vocab))    print("HOGE")
    # S, Q, A = vectorize(data,)
