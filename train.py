from nltk.tokenize import word_tokenize
import numpy as np
import functools
from itertools import chain
from sklearn.model_selection import train_test_split

from process_data import load_task, vectorize

if __name__ == '__main__':
    # train_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_train.txt')
    test_data = load_task('./data/tasks_1-20_v1-2/en/qa5_three-arg-relations_test.txt')
    # data = train_data + test_data
    data = test_data

    vocab = functools.reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data))
    w2i = dict((c, i) for i, c in enumerate(vocab, 1))
    i2w = dict((i, c) for i, c in enumerate(vocab, 1))

    memory_size = 10 # TBD
    max_story_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    question_size = max(map(len, chain.from_iterable(q for _, q, _ in data)))
    max_sentence_size = max(sentence_size, question_size)
    vocab_size = len(w2i) + 1 # +1 for nil word

    print('max_story_size={}\nmax_sentence_size={}\nvocab_size={}'.format(max_story_size, max_sentence_size, vocab_size))

    S, Q, A = vectorize(data, w2i, max_sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = train_test_split(S, Q, A, test_size=0.1)

    print("end")
