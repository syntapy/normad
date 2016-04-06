import scipy.io as sio
import os.path as op
import brian2 as br
import numpy as np
import math as ma
import scipy
import pudb
import sys

import display
import spike_correlation
import train
br.prefs.codegen.target = 'weave'  # use the Python fallback

class activity:

    def __init__(self, **keywds):
        """
            S: brian2 SpikeMonitor
            V: brian2 StateMonitor
        """

        if keywds.has_key('S'):
            self.S = keywds['S']
        if keywds.has_key('M'):
            self.M = keywds['M']
        else: self.M = None

    def reread(self):
        self.i, self.t = self.S.it_
        #pudb.set_trace()
        if self.M != None:
            self.v = self.M.v

class net_info:

    def __init__(self, **keywds):
        """
            a: no. inputs
            b: no. second layer (output or hidden)
            c: no. last layer
            p: no. subconnections
            Wh: input - hidden weights
            Wo: input or hidden - output weights
            Dh: input - hidden delays
            Do: input or hidden - output delays
            hidden: hidden activity object
            output: output activity object
        """
        multilayer = False
        self.a = keywds['a']
        self.c = keywds['c']
        if keywds.has_key('b'):
            self.b = keywds['b']
            multilayer = True
        self.p = keywds['p']
        if keywds.has_key('Wh'):
            self.Wh = keywds['Wh']
            self.d_Wh = np.zeros(np.shape(self.Wh[:]), dtype=np.float64)
            multilayer = True
        else: self.Wh = None
        if keywds.has_key('Dh'):
            self.Dh = keywds['Dh']
            self.d_Dh = np.zeros(np.shape(self.Dh[:]), dtype=np.float64)
            multilayer = True
        else: self.Dh = None
        #pudb.set_trace()
        self.Wo = keywds['Wo']
        self.d_Wo = np.zeros(np.shape(self.Wo[:]), dtype=np.float64)
        self.Do = keywds['Do']
        self.d_Do = np.zeros(np.shape(self.Do[:]), dtype=np.float64)
        if keywds.has_key('hidden'):
            self.H = keywds['hidden']
            multilayer = True
        else: self.H = None
        self.O = keywds['output']

    def set_inputs(self, indices, times):
        self.ii, self.ta = indices, times

    def get_inputs(self):
        return self.ii, self.ta

    def reread(self):
        if self.H != None:
            self.H.reread()
        self.O.reread()
        self.direction()

    def direction(self):
        self.d = [0] * len(self.y)
        y = self.y
        S = self.O.S.all_values()['t']
        for i in range(len(y)):
            #pudb.set_trace()
            if y[i] == 1:
                if len(S[i]) == 0:
                    self.d[i] = 1
            elif y[i] == 0:
                if len(S[i]) > 0:
                    self.d[i] = -1

    def set_y(self, y):
        self.y = y
    def reset(self):
        if self.Wh != None:
            self.d_Wh *= 0
        if self.Dh != None:
            self.d_Dh *= 0
        self.Wo *= 0
        self.Do *= 0

class net:

    ###################
    ### MODEL SETUP ###
    ###################

    def __init__(self, hidden=5, output=2, inputs=3, subc=3, seed=5):
        #pudb.set_trace()
        self.changes = []
        self.trained = False
        self.rb = 1.0
        self.r = 10.0
        self.dta = 0.2*br.ms
        self.N_inputs = inputs
        self.N_hidden = hidden
        self.N_output = output
        self.N_subc = subc
        self.tauLP = 5.0
        #self.tauIN = 5.0
        #self.seed = seed
        #np.random.seed(self.seed)
        #self.a, self.d = None, None
        #self.a_post, self.d_post = [], []
        #self.a_pre, self.d_pre = [], []
        #self.data, self.labels = None, None
        self.T = 40
        self.__groups()

    def rand_weights_singlelayer(self, test=False):
        So = self.net['synapses_output']
        p = self.N_subc
        So.w[:, :, :] = '80'
        So.w[:, :, :int(np.ceil(p/3))] *= -1

        So.delay[:, :, :] = '11*rand()*ms'

        So.tl[:, :, :] = '-1*second'
        So.tp[:, :, :] = '-1*second'

        self.net.store()

    def rand_weights_multilayer(self, test=False):
        Sh = self.net['synapses_hidden']
        So = self.net['synapses_output']
        p = self.N_subc
        Sh.w[:, :, :] = '120'
        So.w[:, :, :] = '80'
        Sh.w[:, :, :int(np.ceil(p/3))] *= -1
        So.w[:, :, :int(np.ceil(p/3))] *= -1

        Sh.delay[:, :, :] = '11*rand()*ms'
        So.delay[:, :, :] = '11*rand()*ms'

        Sh.tl[:, :, :] = '-1*second'
        Sh.tp[:, :, :] = '-1*second'
        So.tl[:, :, :] = '-1*second'
        So.tp[:, :, :] = '-1*second'

        self.net.store()

    def __gen_neuron_group(self, N_neurons, name):
        neurons = br.NeuronGroup(N_neurons, \
                   model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1 (unless refractory)
                            gL = 30                                     : 1 (shared)
                            El = -70                                    : 1 (shared)
                            vt = 20                                     : 1 (shared)
                            Cm = 06.0                                   : 1 (shared)
                            D                                           : 1''',
                            method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                            reset='v=El', name=name, dt=self.dta)
        return neurons

    def __gen_synapse_group(self, neurons_a, neurons_b, name):
        S = br.Synapses(neurons_a, neurons_b,
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
                    pre='tp=t', post='tl=t', name=name, dt=self.dta)

        return S
    
    def __gen_singlelayer_nn(self, inputs):
        #pudb.set_trace()
        output = self.__gen_neuron_group(self.N_output,'output')
        So = self.__gen_synapse_group(inputs, output, 'synapses_output')

        output.v[:] = -70
        N = br.StateMonitor(So, 'c', record=True, name='monitor_o_c')
        M = br.StateMonitor(output, 'v', record=True, name='monitor_v')
        To = br.SpikeMonitor(output, variables='v', name='crossings_o')
        So.connect('True', n=self.N_subc)
        self.net = br.Network(inputs, output, So, M, N, To)
        self.rand_weights_singlelayer()
        self.actual = self.net['crossings_o'].all_values()['t']
        self.net.store()
        self.output_a = activity(S=To, M=M)
        Wo = self.net['synapses_output'].w
        Do = self.net['synapses_output'].delay
        self.info = net_info(a=self.N_inputs, c=self.N_output, p=self.N_subc, \
                output=self.output_a, Wo=Wo, Do=Do)

    def __gen_multilayer_nn(self, inputs):
        hidden = self.__gen_neuron_group(self.N_hidden, 'hidden')
        output = self.__gen_neuron_group(self.N_output,'output')

        Sh = self.__gen_synapse_group(inputs, hidden, 'synapses_hidden')
        So = self.__gen_synapse_group(hidden, output, name='synapses_output')

        hidden.v[:] = -70
        output.v[:] = -70
        N = br.StateMonitor(So, 'c', record=True, name='monitor_o_c')
        Vo = br.StateMonitor(output, 'v', record=True, name='values_vo')
        Vh = br.StateMonitor(hidden, 'v', record=True, name='values_vh')
        Th = br.SpikeMonitor(hidden, variables='v', name='crossings_h')
        To = br.SpikeMonitor(output, variables='v', name='crossings_o')
        Sh.connect('True', n=self.N_subc)
        So.connect('True', n=self.N_subc)
        self.net = br.Network(inputs, hidden, Sh, Th, output, So, Vo, Vh, N, To)
        self.rand_weights_multilayer()
        self.net.store()
        output_a = activity(S=To, M=Vo)
        hidden_a = activity(S=Th, M=Vh)
        Wo = self.net['synapses_output'].w
        Wh = self.net['synapses_hidden'].w
        Do = self.net['synapses_output'].delay
        Dh = self.net['synapses_hidden'].delay
        self.info = net_info(a=self.N_inputs, b=self.N_hidden, c=self.N_output, p=self.N_subc, \
                hidden=hidden_a, output=output_a, Wh=Wh, Wo=Wo, Dh=Dh, Do=Do)

    def __groups(self):
        inputs = br.SpikeGeneratorGroup(self.N_inputs, 
                                        indices=np.asarray([]), 
                                        times=np.asarray([])*br.ms, 
                                        name='input')
        if self.N_hidden > 0:
            self.__gen_multilayer_nn(inputs)
            self.hidden = True
        else:
            self.hidden = False
            self.__gen_singlelayer_nn(inputs)
        #self.read_weights()

    def save_weights_singlelayer(self, file_o=None):
        if file_o == None:
            folder = "../weights/"
            name_o = "synapses_output-"
            param, ext = str(self.N_hidden) + "_" + str(self.N_output), ".txt"
            file_o = folder + name_o + param + ext
        output = 'synapses_output'
        Fo = open(file_o, 'w')
        Wo = self.net[output]
        n = len(Wo.w[:])
        for i in range(n):
            Fo.write(str(Wo.w[i]))
            Fo.write('\n')
        Fo.close()

    def read_weights_singlelayer(self, file_o=None):
        if file_o == None:
            folder = "../weights/"
            name_o = "synapses_output-"
            param, ext = "0_" + str(self.N_output), ".txt"
            file_o = folder + name_o + param + ext
        self.net.restore()
        output = 'synapses_output'
        Fo = open(file_o, 'r')
        string_o = Fh.readlines(), Fo.readlines()
        n = len(string_o)
        weights_o = np.empty(n, dtype=float)
        for i in xrange(n):
            weights_o[i] = float(string_o[i][:-1])

        o = self.net[output]
        if len(o.w) == 0:
            o.connect('True')
        o.w[:] = weights_o[:]
        o.tl[:, :] = '-1*second'
        o.tp[:, :] = '-1*second'
        self.net.store()

    def save_weights_multilayer(self, file_h=None, file_o=None):
        if file_h == None or file_o == None:
            folder = "../weights/"
            name_h, name_o = "synapses_hidden-", "synapses_output-"
            param, ext = str(self.N_hidden) + "_" + str(self.N_output), ".txt"
            file_h, file_o = folder + name_h + param + ext, folder + name_o + param + ext
        hidden, output = 'synapses_hidden', 'synapses_output'
        Fh = open(file_h, 'w')
        Fo = open(file_o, 'w')
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

    def read_weights_multilayer(self, file_h=None, file_o=None):
        if file_h == None or file_o == None:
            folder = "../weights/"
            name_h, name_o = "synapses_hidden-", "synapses_output-"
            param, ext = str(self.N_hidden) + "_" + str(self.N_output), ".txt"
            file_h, file_o = folder + name_h + param + ext, folder + name_o + param + ext
        self.net.restore()
        hidden, output = 'synapses_hidden', 'synapses_output'
        Fh = open(file_h, 'r')
        Fo = open(file_o, 'r')
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

    def spikes(self, name='hidden', t='dict'):
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
        #s = self.net['input']
        self.net.store()

    def set_mnist_times(self, index, kind='train'):
        array = self.data[kind][index]
        label = self.labels[kind][index]
        times = self.tauLP / array
        #pudb.set_trace()
        m = np.max(times)
        times -= np.min(times)
        #times *= m / np.max(times)
        times *= 0.7
        indices = np.arange(len(array))
        #self.T = int(ma.ceil(np.max(times)) + self.tauLP)
        desired = np.ones(self.N_output) 

        return label

    def indices(self, N, numbers):
        n = len(numbers)
        indices = []
        count = [0]*n

        index = 1
        while True:
            label = self.labels['train'][index]
            if label in numbers:
                index_put = [i for i in range(n) if numbers[i] == label][0]
                if count[index_put] < N:
                    count[index_put] += 1
                    indices.append(index)
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
        pudb.set_trace()
        self.net['synapses_output'].w[:, :] = '0'
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
        if self.data == 'mnist':
            actual, desired = self.actual, self.desired
            on = np.min(desired)
            off = np.max(desired)
            p = 0
            for i in range(len(desired)):
                if len(actual[i]) == 0:
                    p += 20
                else:
                    for j in range(len(actual[i])):
                        p += ((j+1)**2)*(actual[i][j]/br.msecond - 1000*desired[i])**2
            return (p / float(len(desired)))**0.5
        elif self.data == 'xor':
            #pudb.set_trace()
            actual, desired = self.actual[0] / br.ms, self.desired
            if len(actual) != 1:
                train.synaptic_scalling_wrap(self, 1)
                #pudb.set_trace()
                #return "nan"
            #pudb.set_trace()
            return abs(actual[0] - 1000*self.desired[0])
            #if self.label == 0:
            #    return abs(actual[0] - 1000*self.xl) / abs(actual[0] - 1000*self.xe)
            #else:
            #    return abs(actual[0] - 1000*self.xe) / abs(actual[0] - 1000*self.xl)

    def mnist_right_outputs(self, label):
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

    #def xor_right_outputs(self, label):
    #    actual, desired = self.actual / br.ms, self.desired
    #    if len(actual[0]) != 0
    #        return False
    #
    #    if actual[0]:
        
    def neuron_right_outputs(self, label):
        if self.data == 'mnist':
            return self.mnist_right_outputs(label)
        elif self.data == 'xor':
            print pudb.set_trace()
            #    return self.xor_right_outputs(label)

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
    ### PRESET --- WEIGHTS ###
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

    def run(self, T=None):
        if T != None and T >= self.T:
            self.net.run(T*br.ms)
        else:
            self.net.run(self.T*br.ms)
        #self.actual = self.net['crossings_o'].all_values()['t']

    def preset_weights_singlelayer(self, images):
        folder = "../weights/"
        name_o = "synapses_output-"
        param, ext = "0_" + str(self.N_output), ".txt"
        file_o = folder + name_o + param + ext
        if op.isfile(file_o):
            self.read_weights_singlelayer(file_o=file_o)
        mod = True
        k = 0
        np.random.shuffle(images)
        n = min(len(images), 10)
        iteration = 0
        while mod:
            mod = False
            iteration += 1
            for i in images[:n]:
                self.read_image(i)
                self.run(None)
                if train.synaptic_scaling(self, 2, iteration=iteration):
                    mod = True
                self.net.restore()
        self.save_weights_singlelayer(file_o=file_o)

    def preset_weights_multilayer(self, images):
        folder = "../weights/"
        name_h, name_o = "synapses_hidden-", "synapses_output-"
        param, ext = str(self.N_hidden) + "_" + str(self.N_output), ".txt"
        file_h, file_o = folder + name_h + param + ext, folder + name_o + param + ext
        if op.isfile(file_h) and op.isfile(file_o):
            self.read_weights_multilayer(file_h=file_h, file_o=file_o)
        mod = True
        k = 0
        np.random.shuffle(images)
        n = min(len(images), 10)
        iteration = 0
        while mod:
            mod = False
            iteration += 1
            for i in images[:n]:
                self.read_image(i)
                self.run(None)
                if train.synaptic_scaling(self, 2, iteration=iteration):
                    mod = True
                self.net.restore()
        self.save_weights_multilayer(file_h=file_h, file_o=file_o)

    def preset_weights(self, images):
        if self.N_hidden > 0:
            self.preset_weights_multilayer(images)
        else:
            self.preset_weights_singlelayer(images)

    ############################
    ### TRAINING --- TESTING ###
    ############################

    def set_inputs(self, x):
        indices = np.arange(self.N_inputs)
        times = x
        self.indices, self.times = indices, times
        self.net['input'].set_spikes(indices=indices, times=times)
        self.net.store()
        self.info.set_inputs(indices, times)

    def fit(self, X, Y, method='tempotron', threshold=0.7):
        def print_zeros(i, max_order=4):
            for j in range(1, 4):
                if i < 10**j:
                    print ' ',
        #print "PRESETTING WEIGHTS"
        #self.preset_weights(images)
        i, j, k = 0, 0, 0
        pmin = 10000
        p = pmin
        #print "TRAINING - ",
        #print "N_input, N_output, N_hidden: ", self.N_inputs, self.N_output, self.N_hidden
        while True:
            i += 1
            j += 1
            pold = p
            p = train.train_epoch(self, X, Y, method=method)
            if p < pmin:
                pmin = p
                j = 0
            print "i, p, pmin: ", i, p, pmin
            self.r = self.rb*(min(p, 4)**2) / 4
            if p < 1:
                break
        self.save_weights()

    def compute(self, images):
        test_result = []
        times = []
        print "======= R e s u l t s ========"
        print "times \t\t actual \t\t desired"
        for i in images:
            label = self.read_image(i)
            self.run(None)
            p = self.performance()
            actual, desired = self.actual, self.desired
            print self.times, "\t\t", self.actual, "\t\t", self.desired
            self.net.restore()
