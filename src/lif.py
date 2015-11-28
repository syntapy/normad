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

    def __init__(self, N_hidden=20, N_output=10, N_input=4, data='mnist', seed=5):
        self.changes = []
        self.trained = False
        self.r = 4.0
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
        if data == 'mnist':
            self.load()
            self.N_inputs = len(self.data['train'][0])
            self.N_output = 10
        else:
            self.N_inputs = N_inputs
        #pudb.set_trace()
        self.__groups()

    def __groups(self):
        inputs_hidden = br.SpikeGeneratorGroup(self.N_inputs, 
                                        indices=np.asarray([]), 
                                        times=np.asarray([])*br.ms, 
                                        name='input_hidden')
        hidden = br.NeuronGroup(self.N_hidden, \
                               model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1
                                        gL = 30                                     : 1 (shared)
                                        El = -70                                    : 1 (shared)
                                        vt = 20                                     : 1 (shared)
                                        Cm = 3.0                                    : 1 (shared)
                                        D                                           : 1''',
                                        method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                                        reset='v=El', name='hidden', dt=self.dta)
        inputs_out = br.SpikeGeneratorGroup(self.N_hidden, 
                                        indices=np.asarray([]), 
                                        times=np.asarray([])*br.ms, 
                                        name='input_out')
        output = br.NeuronGroup(self.N_output, \
                               model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1
                                        gL = 30                                     : 1 (shared)
                                        El = -70                                    : 1 (shared)
                                        vt = 20                                     : 1 (shared)
                                        Cm = 3.0                                    : 1 (shared)
                                        D                                           : 1''',
                                        method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                                        reset='v=El', name='output', dt=self.dta)
        #hidden = br.Subgroup(neurons, 0, self.N_hidden, name='hidden')
        #output = br.Subgroup(neurons, self.N_hidden, self.N_hidden+self.N_output, name='output')

        Sh = br.Synapses(inputs_hidden, hidden,
                    model='''
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
                    pre='tp=t', post='tl=t', name='synapses_hidden', dt=self.dta)
        So = br.Synapses(inputs_out, output, 
                   model='''tl                                                  : second
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
                   pre='tp=t', post='tl=t', name='synapses_output', dt=self.dta)
        #Rh = br.Synapses(hidden, hidden, pre='c_post=0', name='winner_take_all_a', dt=self.dta)
        #Ro = br.Synapses(output, output, pre='c_post=0', name='winner_take_all_b', dt=self.dta)

        #Rh.connect('i!=j')
        #Ro.connect('i!=j')
        Sh.connect('True')
        So.connect('True')

        Sh.w[:, :] = '(1000*rand()+750)'
        So.w[:, :] = '0*(1000*rand()+750)'
        #So.w[0, 1] = '700'
        Sh.tl[:, :] = '-1*second'
        Sh.tp[:, :] = '-1*second'
        So.tl[:, :] = '-1*second'
        So.tp[:, :] = '-1*second'
        hidden.v[:] = -70
        output.v[:] = -70
        M = br.StateMonitor(output, 'v', record=True, name='monitor_v')
        N = br.StateMonitor(So, 'c', record=True, name='monitor_o_c')
        #O = br.StateMonitor(S, 'tp', record=True, name='monitor_o')
        #F = br.StateMonitor(S, 'f', record=True, name='monitor_f')
        Th = br.SpikeMonitor(hidden, variables='v', name='crossings_h')
        To = br.SpikeMonitor(output, variables='v', name='crossings_o')
        self.net_hidden = br.Network(inputs_hidden, hidden, Sh, Th)
        self.net_out = br.Network(inputs_out, output, So, M, N, To)
        self.actual = self.net_out['crossings_o'].all_values()['t']
        self.net_hidden.store()
        self.net_out.store()

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

    def save_weights(self, fname='weights.txt'):
        folder = '../files/'
        F = open(folder + fname, 'w')
        s = self.net['synapses']
        n = len(s.w[:])
        for i in range(n):
            F.write(str(s.w[i]))
            F.write('\n')
        F.close()

    def read_weights(self, fname='weights.txt'):
        folder = '../files/'
        F = open(folder + fname, 'r')
        string = F.readlines()
        n = len(string)
        weights = np.empty(n, dtype=float)
        for i in xrange(n):
            weights[i] = float(string[i][:-1])

        self.net['synapses'].w[:] = weights[:]

    ##########################
    ### SET INPUT / OUTPUT ###
    ##########################

    def get_spikes(self, name='hidden', t='dict'):
        if name != 'hidden':
            name = 'output'
        if name == 'hidden':
            S = self.net_hidden['crossings_h']
        elif name == 'output':
            S = self.net_out['crossings_o']

        if t=='dict':
            return S.i, S.t
        return S.all_values()['t']

    def set_train_spikes(self, indices=[], times=[], desired=[]):
        self.net_hidden.restore()
        self.indices, self.times, self.desired = indices, times*br.ms, desired*br.ms
        self.net_hidden['input_hidden'].set_spikes(indices=self.indices, times=self.times)
        self.net_hidden.store()

    def read_image(self, index, kind='train'):
        array = self.data[kind][index]
        label = self.labels[kind][index]
        times = self.tauLP / array
        indices = np.arange(len(array))
        desired = np.zeros(self.N_output)
        self.T = int(ma.ceil(max(np.max(desired), np.max(times)) + self.tauLP))
        desired[label] = int(ma.ceil(self.T))
        #self.T += 5
        self.set_train_spikes(indices=indices, times=times, desired=desired)
        self.net_hidden.store()

    def transfer_spikes(self):
        i, t = self.get_spikes()
        #pudb.set_trace()
        if len(t) > 0:
            t = t / br.second
            t -= np.min(t)
            t = t*br.second
        self.net_out.restore()
        self.net_out['input_out'].set_spikes(indices=i, times=t)
        self.net_out.store()

    def uniform_input(self):
        self.net.restore()
        times = np.zeros(self.N_inputs)
        indices = np.arange(self.N_inputs)
        self.set_train_spikes(indices=indices, times=times, desired=np.array([]))
        self.T = 20

    def reset(self):
        self.net_hidden.restore()
        self.net_out.restore()
        self.net_hidden['synapses'].w[:, :] = '0'
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

    def neuron_right_outputs(self):
        actual, desired = self.actual, self.desired
        for i in range(len(desired)):
            if desired[i] == 0:
                if len(actual[i]) > 0:
                    return False
            else:
                if len(actual[i]) != 1:
                    return False
        return True

    def tdiff_rms(self):
        """ OUTDATED FUNCTION """
        return 1
        pudb.set_trace()
        actual, desired = np.sort(self.actual), np.sort(self.desired)
        if len(actual) != len(desired):
            return 0

        n = len(actual)
        if n > 0:
            r, m = 0, 0
            for i in range(n):
                r += (actual[i] - desired[i])**2
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
        self.net_hidden.restore()
        self.net_out.restore()
        if T != None and T >= self.T:
            self.net_hidden.run(T*br.ms)
            Sh = self.net['crossings_h'].all_values()['t']
        else:
            self.net_hidden.run(self.T*br.ms)
            self.transfer_spikes()
            self.net_out.run(self.T*br.ms)

    def train(self, a, b, method='resume', threshold=0.7):
        i = 0
        self.r = 1
        while True:
            i += 1
            print "Epoch ", i
            correct = train.train_epoch(self, a, b, method=method)
            p_correct = float(correct) / (b - a)
            print  ": %", p_correct, " correct"
            if p_correct > threshold:
                break
            self.r = 32 / (1 + 1024*p_correct)
        return p_correct
