import scipy.io as sio
import os.path as op
import brian2 as br
import numpy as np
import math as ma
import scipy
import pudb

import display
import spike_correlation
import train
br.prefs.codegen.target = 'weave'  # use the Python fallback

class net:

    ###################
    ### MODEL SETUP ###
    ###################

    def __init__(self, N_hidden=5, N_output=10, N_input=4, data='mnist', seed=5):
        #pudb.set_trace()
        self.changes = []
        self.trained = False
        self.rb = 1.0
        self.r = 1.0
        self.dta = 0.2*br.ms
        self.N_hidden = N_hidden
        self.N_output = N_output
        self.tauLP = 5.0
        #self.tauIN = 5.0
        self.seed = seed
        np.random.seed(self.seed)
        self.a, self.d = None, None
        self.a_post, self.d_post = [], []
        self.a_pre, self.d_pre = [], []
        self.data, self.labels = None, None
        self.T = 40
        if data == 'mnist':
            self.load()
            #pudb.set_trace()
            self.N_inputs = len(self.data['train'][0])
            #self.N_output = 10
        else:
            self.N_inputs = N_inputs
        self.__groups()

    def rand_weights(self, test=False):
        #Sh.w[:, :] = '(1000*rand()+750)'
        #So.w[:, :] = '(1000*rand()+750)'
        #So.w[1, 0] = '200'
        Sh = self.net['synapses_hidden']
        So = self.net['synapses_output']
        #if test==False:
        #    Sh.connect('True')
        #    So.connect('True')
        Sh.w[:, :] = '(40*rand()+40)'
        So.w[:, :] = '(23*rand()+10)'
        Sh.tl[:, :] = '-1*second'
        Sh.tp[:, :] = '-1*second'
        So.tl[:, :] = '-1*second'
        So.tp[:, :] = '-1*second'
        self.net.store()

    def __groups(self):
        inputs = br.SpikeGeneratorGroup(self.N_inputs, 
                                        indices=np.asarray([]), 
                                        times=np.asarray([])*br.ms, 
                                        name='input')
        hidden = br.NeuronGroup(self.N_hidden, \
                   model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1 (unless refractory)
                            gL = 30                                     : 1 (shared)
                            El = -70                                    : 1 (shared)
                            vt = 20                                     : 1 (shared)
                            Cm = 12.0                                   : 1 (shared)
                            D                                           : 1''',
                            method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                            reset='v=El', name='hidden', dt=self.dta)
        output = br.NeuronGroup(self.N_output, \
                            model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1 (unless refractory)
                                        gL = 30          : 1 (shared)
                                        El = -70         : 1 (shared)
                                        vt = 20          : 1 (shared)
                                        Cm = 3.0         : 1 (shared)
                                        D                : 1''',
                                        method='rk2', refractory=0*br.ms, 
                                        threshold='v>=vt', reset='v=El', 
                                        name='output', dt=self.dta)

        Sh = br.Synapses(inputs, hidden,
                    model='''
                            tl                                    : second
                            tp                                    : second
                            tauC = 5                              : 1      (shared)
                            tau1 = 0.0050                         : second (shared)
                            tau2 = 0.001250                       : second (shared)
                            tauL = 0.010                          : second (shared)
                            tauLp = 0.1*tauL                      : second (shared)

                            w                                     : 1

                            up = (sign(t - tp) + 1.0) / 2         : 1
                            ul = (sign(t - tl - 3*ms) + 1.0) / 2  : 1
                            u = (sign(t) + 1.0) / 2               : 1

                      c = 200*exp((tp - t)/(tau1*tauC)) - exp((tp - t)/(tau2*tauC)): 1
                            f = w*c                               : 1
                            D_post = f*ul                         : 1 (summed) ''',
                    pre='tp=t', post='tl=t', name='synapses_hidden', dt=self.dta)
        So = br.Synapses(hidden, output, 
                   model='''tl                                   : second
                            tp                                   : second
                            tauC = 5                             : 1 (shared)
                            tau1 = 0.0050                        : second (shared)
                            tau2 = 0.001250                      : second (shared)
                            tauL = 0.010                         : second (shared)
                            tauLp = 0.1*tauL                     : second (shared)

                            w                                    : 1

                            up = (sign(t - tp) + 1.0) / 2        : 1
                            ul = (sign(t - tl - 3*ms) + 1.0) / 2 : 1
                            u = (sign(t) + 1.0) / 2              : 1

                     c = 200*exp((tp - t)/(tau1*tauC)) - exp((tp - t)/(tau2*tauC)): 1
                            f = w*c                              : 1
                            D_post = f*ul                        : 1 (summed) ''',
                   pre='tp=t', post='tl=t', name='synapses_output', dt=self.dta)

        hidden.v[:] = -70
        output.v[:] = -70
        M = br.StateMonitor(output, 'v', record=True, name='monitor_v')
        N = br.StateMonitor(So, 'c', record=True, name='monitor_o_c')
        Th = br.SpikeMonitor(hidden, variables='v', name='crossings_h')
        To = br.SpikeMonitor(output, variables='v', name='crossings_o')
        Sh.connect('True')
        So.connect('True')
        self.net = br.Network(inputs, hidden, Sh, Th, output, So, M, N, To)
        self.rand_weights()
        self.actual = self.net['crossings_o'].all_values()['t']
        self.net.store()
        #self.read_weights()

    ##################
    ### DATA SETUP ### 
    ##################

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

    def save_weights(self):
        folder = '../files/'
        hidden, output = 'synapses_hidden', 'synapses_output'
        Fh = open(folder + hidden + '.txt', 'w')
        Fo = open(folder + output + '.txt', 'w')
        Wh = self.net[hidden]
        Wo = self.net[output]
        m = len(Wh.w[:])
        n = len(Wo.w[:])
        for i in range(m):
            Fh.write(str(Wh.w[i]))
            Fh.write('\n')
        for i in range(n):
            Fo.write(str(Wo.w[i]))
            Fo.write('\n')
        Fh.close()
        Fo.close()

    def read_weights(self):
        #pudb.set_trace()
        self.net.restore()
        folder = '../files/'
        hidden, output = 'synapses_hidden', 'synapses_output'
        Fh = open(folder + hidden + '.txt', 'r')
        Fo = open(folder + output + '.txt', 'r')
        string_h, string_o = Fh.readlines(), Fo.readlines()
        m, n = len(string_h), len(string_o)
        weights_h = np.empty(m, dtype=float)
        weights_o = np.empty(n, dtype=float)
        for i in xrange(m):
            weights_h[i] = float(string_h[i][:-1])
        for i in xrange(n):
            weights_o[i] = float(string_o[i][:-1])

        h = self.net[hidden]
        o = self.net[output]
        if len(h.w) == 0 or len(o.w) == 0:
            h.connect('True')
            o.connect('True')
        h.w[:] = weights_h[:]
        o.w[:] = weights_o[:]
        h.tl[:, :] = '-1*second'
        h.tp[:, :] = '-1*second'
        o.tl[:, :] = '-1*second'
        o.tp[:, :] = '-1*second'
        self.net.store()

    ##########################
    ### SET INPUT / OUTPUT ###
    ##########################

    def get_spikes(self, name='hidden', t='dict'):
        if name != 'hidden':
            name = 'output'
        if name == 'hidden':
            S = self.net['crossings_h']
        elif name == 'output':
            S = self.net['crossings_o']

        if t=='dict':
            return S.i, S.t
        return S.all_values()['t']

    def set_train_spikes(self, indices=[], times=[], desired=[]):
        self.net.restore()
        self.indices, self.times, self.desired = indices, times*br.ms, desired
        #pudb.set_trace()
        self.net['input'].set_spikes(indices=self.indices, times=self.times)
        s = self.net['input']
        self.net.store()

    def read_image(self, index, kind='train'):
        array = self.data[kind][index]
        label = self.labels[kind][index]
        times = self.tauLP / array
        #pudb.set_trace()
        m = np.max(times)
        times -= np.min(times)
        times *= m / np.max(times)
        indices = np.arange(len(array))
        #self.T = int(ma.ceil(np.max(times)) + self.tauLP)
        desired = np.ones(self.N_output) 
        desired *= 0.001*(35)
        desired[label] *= 0.7
        self.set_train_spikes(indices=indices, times=times, desired=desired)
        self.net.store()

        return label

    def indices(self, N, numbers):
        n = len(numbers)
        indices = []
        count = [0]*n

        index = 1
        while True:
            #print "index: ", index
            label = self.labels['train'][index]
            #if label == 0 or label == 8:
            #    pudb.set_trace()
            if label in numbers:
                index_put = [i for i in range(n) if numbers[i] == label][0]
                if count[index_put] < N:
                    count[index_put] += 1
                    indices.append(index)
            #b = [count[i] == N for i in range(n)]
            if count == [N]*n:
                break
            index += 1
        return indices

    def uniform_input(self):
        self.net.restore()
        times = np.zeros(self.N_inputs)
        indices = np.arange(self.N_inputs)
        self.set_train_spikes(indices=indices, times=times, desired=np.array([]))
        self.T = 20

    def reset(self):
        self.net.restore()
        self.net['synapses'].w[:, :] = '0'
        self._input_output()

    ##################
    ### EVAULATION ###
    ##################

    def accuracy(self, a=0, b=50000, data='train'):
        false_p, false_n = np.zeros(self.N_output), np.zeros(self.N_output)
        true_p, true_n = np.zeros(self.N_output), np.zeros(self.N_output)
        for i in range(a, b):
            self.read_image(i, kind=data)
            self.run(self.T)
            if self.neuron_right_outputs():
                pass

    def output_class(self):
        actual, desired = self.actual, self.desired
        for i in range(len(actual)):
            if len(actual[i]) >= 1:
                return i
        return -1

    def performance(self):
        #pudb.set_trace()
        actual, desired = self.actual, self.desired
        on = np.min(desired)
        off = np.max(desired)
        p = 0
        for i in range(len(desired)):
            if len(actual[i]) == 0:
                p += 20
            else:
                for j in range(len(actual[i])):
                    #pudb.set_trace()
                    p += ((j+1)**2)*(actual[i][j]/br.msecond - 1000*desired[i])**2
        return (p / float(len(desired)))**0.5

    def neuron_right_outputs(self, label):
        #pudb.set_trace()
        actual, desired = self.actual, self.desired
        if len(actual[label]) == 0:
            return False
        #pudb.set_trace()
        indices = range(len(desired))
        indices.pop(label)
        if len(actual[label]) == 0:
            return False
        actual[label].sort()
        for i in indices:
            #pudb.set_trace()
            actual[i].sort()
            if len(actual[i]) > 0 and actual[i][0] <= actual[label][0]:
                return False
        return True

    def tdiff_rms(self):
        """ OUTDATED FUNCTION """
        if not self.neuron_right_outputs():
            return -1
        actual, desired = np.sort(self.actual), np.sort(self.desired)
        n = len(actual)
        if n > 0:
            r, m = 0, 0
            for i in range(n):
                r += (actual[i][0] - desired[i])**2
            return (r / float(n))**0.5

    ##########################
    ### TRAINING / RUNNING ###
    ##########################

    def single_neuron_update(self, dw, m, n, c, neuron_index, time_index):
        """ Computes weight updates for single neuron """
        # m neurons in network
        # n inputs
        dw_t = np.empty(n)
        dw_t[:] = c[neuron_index:m:m*n, time_index]

        dw_n = np.linalg.norm(dw_t)
        if dw_n > 0:
            return dw_t
        return 0

    def test(self, N=100, K=250, T=80):
        """
        N: range of number of input synapses to test
        K: number of runs for each parameter set
        T: time (ms) for each run
        """

        self.T = T
        self.n_inputs(2*self.N, 6*self.N)
        self.n_outputs(1, int(self.N / 10))
        for i in range(K):
            self.__groups()
            self._input_output()
            self.train()

    def run(self, T):
        self.net.restore()
        if T != None and T >= self.T:
            self.net.run(2*T*br.ms)
            #Sh = self.net['crossings_h'].all_values()['t']
        else:
            self.net.run(2*self.T*br.ms)

    def preset_weights(self, images):
        if op.isfile("../files/synapses_hidden.txt") and op.isfile("../files/synapses_output.txt"):
            #pudb.set_trace()
            self.read_weights()
        else:
            #self.rand_weights(test=True)
            #self.save_weights()
            mod = True
            k = 0
            np.random.shuffle(images)
            n = min(len(images), 1)
            self.run(None)
            self.net.restore()
            while mod:
                #k += 1
                mod = False
                #pudb.set_trace()
                print k,
                for i in images[:n]:
                    print "!",
                    self.read_image(i)
                    self.run(None)
                    #print "\t run_try", i,
                    #train.print_times(self)
                    #pudb.set_trace()
                    if train.synaptic_scaling(self):
                        mod = True
                print
            #pudb.set_trace()
            self.save_weights()

    def train(self, iteration, images, method='resume', threshold=0.7):
        print "PRESETTING WEIGHTS"
        self.preset_weights(images)
        #pudb.set_trace()
        i, j, k = 0, 0, 0
        pmin = 10000
        p = pmin
        #ch = False
        print "TRAINING"
        while True:
            i += 1
            j += 1
            #print "Iter-Epoch ", iteration, ", ", i
            print i, " - ",
            pold = p
            N, correct, p = train.train_epoch(self, i, images, method=method)
            #if i > 1 and p - pold == 0:
            #    hidden = True
            if p < pmin:
                pmin = p
                j = 0
            self.r = self.rb*(min(p, 4)**2) / 4
            print "p, pmin: ", p, ", ", pmin, ", ",
            print float(correct) / N
            if float(correct) / N > 0.85:
                self.net.restore()
                return i, pmin
