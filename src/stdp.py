import brian2 as br
import cPickle as pickle
import numpy as np
import math as ma
import scipy
import pudb
import lif

class stdp_encoder():

    def __init__(self, N_hidden, seed=5, data='mnist'):
        #pudb.set_trace()
        self.changes = []
        self.trained = False
        self.r = 4.0
        self.dta = 0.2*br.ms
        self.N_hidden = N_hidden
        self.tauLP = 5.0
        self.seed = seed
        np.random.seed(self.seed)
        self.a, self.d = None, None
        self.a_post, self.d_post = [], []
        self.a_pre, self.d_pre = [], []
        self.data, self.labels = None, None
        if data == 'mnist':
            self.load()
            self.N_inputs = len(self.data['train'][0])
            self.N_output = 10
        else:
            self.N_inputs = N_inputs
        self.__groups()

    def __groups(self):
        inputs = br.SpikeGeneratorGroup(self.N_inputs, 
                                        indices=np.asarray([]), 
                                        times=np.asarray([])*br.ms, 
                                        name='input')
        hidden = br.NeuronGroup(self.N_hidden, \
                               model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1
                                        gL = 30                                     : 1
                                        El = -70                                    : 1
                                        vt = 20                                     : 1
                                        Cm = 3.0                                    : 1
                                        D                                           : 1''',
                                        method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                                        reset='v=El', name='hidden', dt=self.dta)
        Sh = br.Synapses(inputs, hidden,
                    model='''
                            # STDP variables
                            gmax = 10000                                        : 1
                            taupre = 0.020                                      : second (shared)
                            taupost = 0.020                                     : second (shared)
                            dApre = 1                                           : 1
                            dApost = -dApre * 1.05                              : 1
                            dApre/dt = -Apre /  (0.020*second)                  : 1 (event-driven)
                            dApost/dt = -Apost / (0.020*second)                 : 1 (event-driven)

                            tl                                                  : second
                            tp                                                  : second
                            tau1 = 0.0025                                       : second (shared)
                            tau2 = 0.000625                                     : second (shared)
                            tauL = 0.010                                        : second (shared)
                            tauLp = 0.1*tauL                                    : second (shared)

                            w                                                   : 1

                            up = (sign(t - tp) + 1.0) / 2                       : 1
                            ul = (sign(t - tl - 3*ms) + 1.0) / 2                : 1
                            u = (sign(t) + 1.0) / 2                             : 1

                            c = 100*exp((tp - t)/tau1) - exp((tp - t)/tau2)     : 1
                            f = w*c                                             : 1
                            D_post = w*c*ul                                     : 1 (summed) ''',
                    pre=''' tp=t
                            Apre += dApre
                            w = clip(w + Apre, 0, gmax)''',
                    post='''tl=t
                            Apost += dApost
                            w = clip(w + Apre, 0, gmax)''', 
                    name='synapses', dt=self.dta)
        Sh.connect('True')
        Sh.tl[:, :] = '-1*second'
        Sh.tp[:, :] = '-1*second'
        Sh.w[:, :] = '(1000*rand()+3750)'
        hidden.v[:] = -70
        T = br.SpikeMonitor(hidden, name='crossings')
        M = br.StateMonitor(hidden, variables='v', record=True, name='monitor_v')
        N = br.StateMonitor(Sh, variables='Apre', record=True, name='monitor_Apre')
        self.net = br.Network(inputs, hidden, Sh, T, M, N)
        self.net.store()

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

    def load(self):
        c1_train = scipy.io.loadmat('../data/train-1.mat')['c1a'][0]
        c1_test = scipy.io.loadmat('../data/test-1.mat')['c1b'][0]

        N_train, N_test = len(c1_train), len(c1_test)
        train_features = np.empty(N_train, dtype=object)
        test_features = np.empty(N_test, dtype=object)

        for i in xrange(N_train):
            train_features[i] = self.rflatten(c1_train[i])
        for i in xrange(N_test):
            test_features[i] = self.rflatten(c1_test[i])

        self.data, self.labels = {}, {}
        train_labels = scipy.io.loadmat('../data/train-label.mat')['train_labels_body']
        test_labels = scipy.io.loadmat('../data/test-label.mat')['test_labels_body']

        self.data['train'] = train_features
        self.data['test'] = test_features
        self.labels['train'] = self.rflatten(train_labels)
        self.labels['test'] = self.rflatten(test_labels)

        del train_features
        del test_features
        del train_labels
        del test_labels

    def set_train_spikes(self, indices=[], times=[], desired=[]):
        self.net.restore()
        self.indices, self.times, self.desired = indices, times*br.ms, desired*br.ms
        self.net['input'].set_spikes(indices=self.indices, times=self.times)
        self.net.store()

    def read_image(self, index, kind='train'):
        array = self.data[kind][index]
        label = self.labels[kind][index]
        times = self.tauLP / array
        indices = np.arange(len(array))
        desired = np.zeros(self.N_output)
        self.T = int(ma.ceil(max(np.max(desired), np.max(times)) + self.tauLP))
        desired[label] = int(ma.ceil(self.T))
        self.T += 25
        self.set_train_spikes(indices=indices, times=times, desired=desired)
        self.net
        self.net.store()

    def pretrain(self, a, b):
        print "Unsupervised pre-training with STDP"
        i, norm = 0, float("inf")
        threshold = 0.05 * self.N_hidden
        if a == 0:
            numbers = np.asarray(range(b))
        else:
            numbers = np.asarray(range(a, b))
        while norm > threshold * 2:
            np.random.shuffle(numbers)
            print i, " norm: ", norm
            i, norm = i+1, 0
            for j in numbers:
                print "\tj = ", j,
                self.net.restore()
                self.read_image(j)
                self.net.run(self.T*br.ms)
                w = self.net['synapses'].w[:]
                self.net.restore()
                norm_tmp = np.linalg.norm(self.net['synapses'].w[:] - w[:])
                print "\tnorm_tmp: ", norm_tmp
                norm += norm_tmp
                self.net['synapses'].w[:] = w[:]
                self.net.store()

        print "Saving to mnist_encoded.bin"

        self.save_outputs(a, b)

    def save_outputs(self, a, b):
        if a == 0:
            iterator = xrange(b)
        else:
            iterator = xrange(a, b)
        
        #pudb.set_trace()
        n = len(iterator)
        fname = "mnist_encoded.bin"
        f = open(fname, "w+")
        p = pickle.Pickler(f)
        p.dump(n)
        for i in iterator:
            print "Dumping encoded image ", i
            self.net.restore()
            self.read_image(i)
            self.net.run(self.T*br.ms)
            q = self.net['crossings']
            #pudb.set_trace()
            p.dump(i)
            p.dump(q.it[0]['i'])
            p.dump(q.it[1]/br.ms)
            r = self.net['monitor_v']
        f.close()

    def plot(self, name='monitor_v'):
        for j in range(self.N_hidden):
            br.plot(self.net[name][j].t, self.net[name][j].v, label=name[-1] + ' ' + str(j))
        br.show()
