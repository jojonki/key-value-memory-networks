
import pickle
from nltk.tokenize import word_tokenize
import numpy as np


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


if __name__ == '__main__':
    # data = load_task('./data/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')
    data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt')