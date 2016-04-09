import scipy.io
import numpy as np
import pudb
from brian2 import ms

class data:

    def __init__(self, data_set, shuffle=False):
        self.shuffle = shuffle
        self.data_set=data_set
        if data_set == 'mnist':
            self.load_mnist()
        elif data_set == 'xor':
            self.load_xor()
        else:
            self.load_random()

    def rflatten(self, A):
        if A.dtype == 'O':
            dim = np.shape(A)
            n = len(dim)
            ad = np.zeros(n)
            i = 0
            tmp = []
            for a in A:
                tmp.append(self.rflatten(a))
            return_val = np.concatenate(tmp)
        else:
            return_val = A.flatten()

        return return_val

    def load_mnist(self):
        c1_train = scipy.io.loadmat('../data/train-1.mat')['c1a'][0]
        c1_test = scipy.io.loadmat('../data/test-1.mat')['c1b'][0]

        N_train, N_test = len(c1_train), len(c1_test)
        train_features = np.empty(N_train, dtype=object)
        test_features = np.empty(N_test, dtype=object)

        for i in xrange(N_train):
            train_features[i] = self.floats_to_times(self.rflatten(c1_train[i]))*ms
        for i in xrange(N_test):
            test_features[i] = self.floats_to_times(self.rflatten(c1_test[i]))*ms

        self.X, self.Y = {}, {}
        train_labels = scipy.io.loadmat('../data/train-label.mat')['train_labels_body']
        test_labels = scipy.io.loadmat('../data/test-label.mat')['test_labels_body']

        if self.shuffle == True:
            indices_train = np.random.shuffle(np.arange(len(N_train)))
            indices_test = np.random.shuffle(np.arange(len(N_train)))

            self.X['train'] = train_features[indices_train]
            self.X['test'] = test_features[indices_test]
            self.Y['train'] = self.rflatten(train_labels)[indices_train]
            self.Y['test'] = self.rflatten(test_labels)[indices_test]

        self.X['train'] = train_features
        self.X['test'] = test_features
        self.Y['train'] = self.rflatten(train_labels)
        self.Y['test'] = self.rflatten(test_labels)

    def load_xor(self):
        """
            0: 00 -> 6 6 0 -> ONE
            1: 11 -> 1 1 0 -> ONE
            2: 01 -> 1 6 0 -> ZERO
            3: 10 -> 6 1 0 -> ZERO
        """
        self.X, self.Y = {}, {}
        self.X['train'] = np.empty(4, dtype=np.ndarray)
        self.X['train'][0] = np.array([6, 6, 0])*ms
        self.X['train'][1] = np.array([0, 0, 0])*ms
        self.X['train'][2] = np.array([0, 6, 0])*ms
        self.X['train'][3] = np.array([6, 0, 0])*ms
        #self.Y['train'] = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        self.Y['train'] = np.array([[0], [0], [1], [1]])
        #label = self.set_xor_times(index)

    def floats_to_times(self, x, tc=1.7, n_out=10):
        times = tc / x
        m = np.max(times)
        times -= np.min(times)
        times *= 0.7
        return times

    """
    def set_xor_times(self, y):
        indices = np.asarray([0, 1, 2])
        if y == 0:
            times = np.asarray([6, 6, 0])
            label = 0
        elif y == 1:
            times = np.asarray([0, 0, 0])
            label = 0
        elif y == 2:
            times = np.asarray([0, 6, 0])
            label = 1
        elif y == 3:
            times = np.asarray([6, 0, 0])
            label = 1
        self.xl = 31*0.001
        self.xe = 25*0.001
        times = times*0.001
        desired = np.asarray([self.xl])
        desired -= label*(self.xl - self.xe)

        self.set_train_spikes(indices=indices, times=times, desired=desired)
        #self.net.store()

        return desired
    """

    def desired_times(self, y, n_out=10, binary=False, data_set='mnist'):
        """
            Sets all desired times
        """
        desired = np.zeros(n_out)
        if binary == False:
            if data_set == 'mnist':
                desired[y] = 1
                desired *= 0.001*(31)
                desired[y] *= 0.7
            elif data_set == 'xor':
                pass
