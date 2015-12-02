import numpy as np
def pad_sequences(X,maxlen=-1):
    """
    assumes matrix of w2v dim x sentence_length
    pads sentence length
    """
    dim = np.shape(X[0][0])
    padder = np.zeros(dim)
    lengths = [len(q) for q in X if type(q) != np.ndarray] #make sure to guard against 1 word
                                                              #questions(should probably clean them)
    padding = max(max(lengths),maxlen)

    X1 = []
    for q_a in X:
        if len(q_a) < padding:
            q_a = np.array([padder for i in range(padding-len(q_a))] + q_a)
        X1.append(q_a)
    print len(X1[0])
    return X1
def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            # hdf5 dataset only support list object as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]


if __name__ == '__main__':
    X = [[np.array([4,5,6]), np.array([1,2,3])],[np.array([1,2,3]), np.array([4,5,6]),np.array([1,2,3])]]
    X = pad_sequences(X)
