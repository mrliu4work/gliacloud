import numpy as np
from sklearn.decomposition import PCA
from os import path

def q21_cooccur_matrix(filename='./data/raw_sentences.txt', window=2):
    filename = path.join(path.dirname(__file__), filename)
    content = open(filename).read()
    tokens = content.replace('\n', '').split(' ')

    index = 0
    vocab = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = index
            index += 1
    inv_vocab = dict(zip(vocab.values(), vocab.keys()))

    cooccur_matrix = np.zeros((index + 1, index + 1))
    for position, token in enumerate(tokens[window:-window]):
        index = vocab[token]
        for j in range(1, window + 1):
            jndex = vocab[tokens[position - j]]
            cooccur_matrix[index][jndex] += 1
        for j in range(1, window + 1):
            jndex = vocab[tokens[position + j]]
            cooccur_matrix[index][jndex] += 1

    return vocab, inv_vocab, cooccur_matrix

def q22_word_vectors(cooccur_matrix, dim=10):
    pca = PCA(n_components=dim)
    pca.fit(cooccur_matrix)
    return pca.transform(cooccur_matrix)

    '''
    cov = np.cov(cooccur_matrix - np.mean(cooccur_matrix, axis=0))
    eig_val, eig_vec = np.linalg.eig(cov)
    return eig_vec[:10].T.shape
    '''

if __name__ == '__main__':
    vocab, inv_vocab, cooccur_matrix = q21_cooccur_matrix()

    for key in vocab:
        assert key == inv_vocab[vocab[key]]

    for el in cooccur_matrix.reshape(-1):
        assert el >= 0

    word_vectors = q22_word_vectors(cooccur_matrix)
