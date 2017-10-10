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
        # q = q[:max_sentence_size]

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

def load_kv_pairs(path, is_save_pickle=False):
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
                    k.append(r)
                    # k += word_tokenize(tmp[1].rstrip().lower())
                    k += tmp[1].strip().lower().split(', ')
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
    for kv_list in kv_pairs:
        vec_kv = []
        for sentence in kv_list:
#             print(sentence)
            kv = [w2i[w] for w in sentence if w in w2i]
            pad_len = max(0, max_sentence_size - len(kv))
            vec_kv.append(kv + [0] * pad_len)
    
        # pad to memory_size
        lm = max(0, memory_size - len(vec_kv))
        for _ in range(lm):
            vec_kv.append([0] * max_sentence_size)

        vec_kv_pairs.append(vec_kv)

    return np.array(vec_kv_pairs, dtype=np.uint16)

def get_kv_indices(data, kv_pairs):
    kv_indices = []
    for i, (_, q, _) in enumerate(train_data):
        if i%100 == 0: print(i, '/', len(train_data))
        ind = []
        question = ' '.join(q)
        for j, kv in enumerate(kv_pairs):
            for ent in kv:
                if ent in question:
                    ind.append(j)
                    break
        kv_indices.append(ind)
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

    # --- movie-qa train/test dataset
    # max_entity_length = max(map(len, (e.split(' ') for e in entities)))
    # train_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_train.txt', entities, max_entity_length)
    # test_data = load_task('./data/movie_dialog_dataset/task1_qa/task1_qa_pipe_test.txt', entities, max_entity_length)
    # save_pickle(train_data, 'mov_task1_qa_pipe_train.pickle')
    # save_pickle(test_data, 'mov_task1_qa_pipe_test.pickle')

    # vocab = functools.reduce(lambda x, y: x | y, (set(chain(chain.from_iterable(s), q, a)) for s, q, a in train_data+test_data))
    # w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    # i2w = dict((i, c) for i, c in enumerate(vocab, 1))
    # save_pickle(vocab, 'vocab.pickle')
    # save_pickle(w2i, 'w2i.pickle')
    # save_pickle(i2w, 'i2w.pickle')
    
    # generate kv_pairs
    # kv_pairs = load_kv_pairs('./data/movieqa/knowledge_source/wiki_entities/wiki_entities_kb.txt', True)
    # vec_kv_pairs = vectorize_kv_pairs(kv_pairs, 10, 30, entities)

    # generate stopwords
    # get_stop_words(1000, True)
    
    # data = load_task('./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')
    # data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt')

    # vocab = functools.reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data))
    # w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    # i2w = dict((i, c) for i, c in enumerate(vocab, 1))
    # print(len(vocab))    print("HOGE")
    # S, Q, A = vectorize(data,)
