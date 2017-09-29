import pickle
from nltk.tokenize import word_tokenize
import numpy as np
import functools
from itertools import chain


def save_pickle(d, path):
    with open(path, mode='wb') as f:
        pickle.dump(d, f)

def load_pickle(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)

def lower_list(word_list):
    return [w.lower() for w in word_list]

def load_task(fpath):
    with open (fpath) as f:
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

def vectorize(data, w2i, max_sentence_size, memory_size):
    S, Q, A = [], [], []
    for story, question, answer in data:
        # Vectroize story
        ss = []
        for sentence in story:
            s_pad_len = max(0, max_sentence_size - len(sentence))
            ss.append([w2i[w] for w in sentence] + [0] * s_pad_len)
        ss = ss[::-1][:memory_size] # discard old memory lager than memory_size

        # Vectroize question
        q_pad_len = max(0, max_sentence_size - len(question))
        q = [w2i[w] for w in question] + [0] * q_pad_len

        # Vectroize answer
        y = np.zeros(len(w2i) + 1) # +1 for nil word
        for a in answer:
            y[w2i[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)

    return np.array(S), np.arary(A), np.array(A)

if __name__ == '__main__':
    # data = load_task('./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')
    data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt')

    vocab = functools.reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data))
    w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    i2w = dict((i, c) for i, c in enumerate(vocab, 1))
    # print(len(vocab))    print("HOGE")
    # S, Q, A = vectorize(data,)