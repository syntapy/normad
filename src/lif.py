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

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FC
from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
from subprocess import call

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

    def get_times(self):
        return self.S.all_values()['t']

    def print_times(self, S, layer_name, spike_type, tabs, tabs_1):
        if tabs > 0:
            print "\t"*tabs,
        print spike_type, "times", 
        if layer_name != None:
            print layer_name,
        print ":", "\t"*tabs_1,
        if S != None:
            for i in range(len(S)):
                if type(S[i]) == tuple:
                    print S[i][0],
                else: print S[i],
        print

    def get_spike_times(self):
        ii, it = self.S.it_
        return it

    def print_spike_times(self, layer_name=None, tabs=0, tabs_1=1):
        S = self.S.all_values()['t']
        self.print_times(S, layer_name, layer_name, tabs, tabs_1)

    def print_desired_times(self, layer_name=None, tabs=0, tabs_1=1):
        S = self.d_times
        self.print_times(S, layer_name, "desired", tabs, tabs_1)

    def print_sd_times(self, tabs=0):
        self.print_spike_times(layer_name="output", tabs=tabs, tabs_1=2)
        self.print_desired_times(tabs=tabs, tabs_1=2)

class net_info:

    def __init__(self, **keywds):
        """
            a: no. inputs
            b: no. second layer (output or hidden)
            c: no. last layer
            p: no. subconnections
            net: network object
            Wh: input - hidden weights
            Wo: input or hidden - output weights
            Dh: input - hidden delays
            Do: input or hidden - output delays
            hidden: hidden activity object
            output: output activity object
        """
        self.multilayer = False
        self.a = keywds['a']
        self.c = keywds['c']
        if keywds.has_key('b'):
            self.b = keywds['b']
            self.multilayer = True
        else:
            self.b = None
        self.p = keywds['p']
        if keywds.has_key('hidden'):
            self.H = keywds['hidden']
            self.multilayer = True
        else: self.H = None
        self.O = keywds['output']
        self.y = None

        self.net = keywds['net']
        self.read_weights()

    def read_weights(self):
        self.Wh = None
        self.Dh = None
        self.d_Wh = None
        self.d_Dh = None
        if self.multilayer == True:
            self.Wh = self.net['synapses_hidden'].w[:]
            self.Dh = self.net['synapses_hidden'].delay[:]
            self.d_Wh = np.zeros(np.shape(self.Wh[:]), dtype=np.float64)
            self.d_Dh = np.zeros(np.shape(self.Dh[:]), dtype=np.float64)
        self.Wo = self.net['synapses_output'].w[:]
        self.Do = self.net['synapses_output'].delay[:]
        self.d_Wo = np.zeros(np.shape(self.Wo[:]), dtype=np.float64)
        self.d_Do = np.zeros(np.shape(self.Do[:]), dtype=np.float64)

        #if keywds.has_key('Wh'):
        #    self.Wh = keywds['Wh']
        #    multilayer = True
        #else: self.Wh = None
        #if keywds.has_key('Dh'):
        #    self.Dh = keywds['Dh']
        #    multilayer = True
        #else: self.Dh = None
        ##pudb.set_trace()
        #self.Wo = keywds['Wo']
        #self.Do = keywds['Do']

    def set_inputs(self, indices, times, y=None):
        #pudb.set_trace()
        self.ii, self.ta = indices, times / br.second
        self.y = y
        if y != None:
            self.bin_to_times()
        else: self.bin_to_none()

    def get_inputs(self):
        ii = np.argsort(self.ii)

        return self.ii[ii], self.ta[ii]

    def bin_to_times(self):
        self.d_times = np.zeros(len(self.y))
        for i in range(len(self.y)):
            if self.y[i] == 1:
                self.d_times[i] = 26.0
            else: self.d_times[i] = 33.0
        self.d_times *= 0.001
        self.O.d_times = self.d_times

    def bin_to_none(self):
        self.d_times = None
        self.O.d_times = None

    def set_y_times(self, desired):
        self.d_times = np.zeros(1)
        self.d_times[0] = desired

    def set_y(self, y):
        self.y = y
        self.bin_to_times()

    def reread(self):
        if self.H != None:
            self.H.reread()
        self.O.reread()
        if self.y != None:
            self.direction()

    def direction(self):
        self.d = [0] * len(self.y)
        y = self.y
        S = self.O.S.all_values()['t']
        for i in range(len(y)):
            if y[i] == 1:
                if len(S[i]) == 0:
                    self.d[i] = 1
            elif y[i] == 0:
                if len(S[i]) > 0:
                    self.d[i] = -1

    def reset(self):
        if self.multilayer == True:
            self.d_Wh *= 0
            self.d_Dh *= 0
        self.d_Wo *= 0
        self.d_Do *= 0

    def set_d_delays(self, d_Do, d_Dh=None):
        if d_Dh != None:
            self.d_Dh = d_Dh
        self.d_Do = d_Do

    def update_d_weights(self, d_Wo, d_Wh=None):
        if d_Wh != None:
            self.d_Wh += d_Wh
        self.d_Wo += d_Wo

    def reset_d_weights(self):
        if self.multilayer == True:
            self.d_Wh *= 0
        self.d_Wo *= 0

    def weights(self):
        return self.Wh, self.Wo

    def d_weights(self):
        return self.d_Wh, self.d_Wo

    def delays(self):
        if self.Dh == None:
            return self.Dh, self.Do / br.ms
        return self.Dh / br.ms, self.Do / br.ms

    def update_weights(self, r):
        self.net.restore()
        if self.d_Wh != None:
            self.net['synapses_hidden'].w += r*self.d_Wh
            self.net['synapses_hidden'].delay += r*self.d_Dh
        self.net['synapses_output'].w += r*self.d_Wo
        self.net['synapses_output'].delay += r*self.d_Do
        self.net.store()

    def performance(self):
        S = self.O.S.all_values()['t']
        D = self.d_times
        p = 0
        for i in range(len(S)):
            p += len(S[i])*30
        p -= 30*len(D)
        p = abs(p)

        if p < 30:
            #pudb.set_trace()
            for i in range(len(D)):
                p += np.abs(1000*D[i] - S[i][0]/br.ms)**2
        return p

class net:

    ###################
    ### MODEL SETUP ###
    ###################

    def __init__(self, hidden=5, output=2, inputs=3, subc=3, delay=11, seed=5):
        #pudb.set_trace()
        self.changes = []
        self.trained = False
        self.rb = 1.0
        self.r = 10.0
        self.dta = 0.2*br.ms
        self.delay = delay
        self.N_inputs = inputs
        self.N_hidden = hidden
        self.N_output = output
        self.N_subc = subc
        self.tauLP = 5.0
        #self.tauIN = 5.0
        self.seed = seed
        np.random.seed(self.seed)
        #self.a, self.d = None, None
        #self.a_post, self.d_post = [], []
        #self.a_pre, self.d_pre = [], []
        #self.data, self.labels = None, None
        self.T = 50
        self.__groups()

    def rand_weights_singlelayer(self, test=False):
        """
            with m inputs, n outputs, p subconnections:
            So.w[i, j, k] = w[k + p*j + n_p*i]
        """

        So = self.net['synapses_output']
        p = self.N_subc
        So.w[:, :, :] = '64000*(0.8*rand()-0.2)*2'
        #So.w[:, :, :int(np.ceil(p/5))] *= -1
        So.w[:, :, :] /= self.N_inputs*p

        So.delay[:, :, :] = str(self.delay) + '*rand()*ms'

        So.tl[:, :, :] = '-1*second'
        So.tp[:, :, :] = '-1*second'

        self.net.store()

    def rand_weights_multilayer(self, test=False):
        """
            with m inputs, n outputs, p subconnections:
            So.w[i, j, k] = w[k + p*j + n_p*i]
        """

        p = self.N_subc
        Sh = self.net['synapses_hidden']
        So = self.net['synapses_output']
        Sh.w[:, :, :] = '800*(0.8*rand() - 0.2)*2'
        So.w[:, :, :] = '800*(0.8*rand() - 0.2)*2'
        #Sh.w[:, :, :int(np.ceil(p/5))] *= -1
        #So.w[:, :, :int(np.ceil(p/5))] *= -1
        Sh.w[:, :, :] /= self.N_inputs*p
        So.w[:, :, :] /= self.N_hidden*p
        #Sh.w[2, 0, :] = 0
        #pudb.set_trace()
        #So.w[:, 0, :] = 0

        Sh.delay[:, :, :] = str(self.delay) +'*rand()*ms'
        So.delay[:, :, :] = str(self.delay) +'*rand()*ms'

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
                            method='rk2', refractory=3*br.ms, threshold='v>=vt', 
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

                      c = 100*(exp((tp - t)/tau1) - exp((tp - t)/tau2)): 1
                            f = w*c                               : 1
                            D_post = f*ul                         : 1 (summed) ''',
                    pre='tp=t', post='tl=t', name=name, dt=self.dta)
        return S

    def __gen_singlelayer_nn(self, inputs):
        #pudb.set_trace()
        output = self.__gen_neuron_group(self.N_output,'output')
        So = self.__gen_synapse_group(inputs, output, 'synapses_output')

        output.v[:] = -70
        So.connect('True', n=self.N_subc)
        N = br.StateMonitor(So, 'c', record=True, name='monitor_o_c')
        M = br.StateMonitor(output, 'v', record=True, name='monitor_v')
        To = br.SpikeMonitor(output, variables='v', name='crossings_o')
        self.net = br.Network(inputs, output, So, M, N, To)
        self.rand_weights_singlelayer()
        self.actual = self.net['crossings_o'].all_values()['t']
        self.net.store()
        self.output_a = activity(S=To, M=M)
        #Wo = self.net['synapses_output'].w
        #Do = self.net['synapses_output'].delay
        self.info = net_info(a=self.N_inputs, c=self.N_output, p=self.N_subc, \
                output=self.output_a, net=self.net)

    def __gen_multilayer_nn(self, inputs):
        hidden = self.__gen_neuron_group(self.N_hidden, 'hidden')
        output = self.__gen_neuron_group(self.N_output,'output')

        Sh = self.__gen_synapse_group(inputs, hidden, 'synapses_hidden')
        So = self.__gen_synapse_group(hidden, output, name='synapses_output')

        hidden.v[:] = -70
        output.v[:] = -70
        #pudb.set_trace()
        Sh.connect('True', n=self.N_subc)
        So.connect('True', n=self.N_subc)
        N = br.StateMonitor(So, 'f', record=True, name='monitor_o_c')
        Vo = br.StateMonitor(output, 'v', record=True, name='values_vo')
        Vh = br.StateMonitor(hidden, 'v', record=True, name='values_vh')
        Th = br.SpikeMonitor(hidden, variables='v', name='crossings_h')
        To = br.SpikeMonitor(output, variables='v', name='crossings_o')
        self.net = br.Network(inputs, hidden, Sh, Th, output, So, Vo, Vh, N, To)
        self.rand_weights_multilayer()
        self.net.store()
        output_a = activity(S=To, M=Vo)
        hidden_a = activity(S=Th, M=Vh)
        #Wo = self.net['synapses_output'].w
        #Wh = self.net['synapses_hidden'].w
        #Do = self.net['synapses_output'].delay
        #Dh = self.net['synapses_hidden'].delay
        #pudb.set_trace()
        self.info = net_info(a=self.N_inputs, b=self.N_hidden, c=self.N_output, p=self.N_subc, \
                hidden=hidden_a, output=output_a, net=self.net)

    def __groups(self):
        inputs = br.SpikeGeneratorGroup(self.N_inputs, 
                                        indices=np.asarray([]), 
                                        times=np.asarray([])*br.ms, 
                                        name='input')
        if self.N_hidden > 0:
            self.__gen_multilayer_nn(inputs)
            #self.hidden = True
        else:
            #self.hidden = False
            self.__gen_singlelayer_nn(inputs)
        #self.read_weights()

    def save_weights(self):
        pudb.set_trace()
        if self.info.multilayer == True:
            self.save_weights_multilayer()
        else: 
            self.save_weights_singlelayer()

    def read_weights(self):
        if self.info.multilayer == True:
            self.read_weights_multilayer()
        else:
            self.read_weights_singlelayer()

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
        if op.exists(file_o):
            Fo = open(file_o, 'r')
            string_o = Fo.readlines()
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
            Fo.close()

    def save_weights_multilayer(self, file_h=None, file_o=None):
        if file_h == None or file_o == None:
            folder = "../weights/"
            name_h, name_o = "synapses_hidden-", "synapses_output-"
            dname_h, dname_o = "delays_hidden-", "delays_output-"
            param, ext = str(self.N_hidden) + "_" + str(self.N_output), ".txt"
            file_h, file_o = folder + name_h + param + ext, folder + name_o + param + ext
            dfile_h, dfile_o = folder + dname_h + param + ext, folder + dname_o + param + ext
        hidden, output = 'synapses_hidden', 'synapses_output'
        #pudb.set_trace()
        Fh = open(file_h, 'w')
        Fo = open(file_o, 'w')
        dFh = open(dfile_h, 'w')
        dFo = open(dfile_o, 'w')
        Wh = self.net[hidden]
        Wo = self.net[output]
        m = len(Wh.w[:])
        n = len(Wo.w[:])
        for i in range(m):
            Fh.write(str(Wh.w[i]))
            Fh.write('\n')
            dFh.write(str(Wh.delay[i]))
            dFh.write('\n')
        for i in range(n):
            Fo.write(str(Wo.w[i]))
            Fo.write('\n')
            dFo.write(str(Wo.delay[i]))
            dFo.write('\n')
        Fh.close()
        Fo.close()
        dFh.close()
        dFo.close()

    def read_weights_multilayer(self, file_h=None, file_o=None, dfile_h=None, dfile_o=None):
        if file_h == None or file_o == None:
            folder = "../weights/"
            name_h, name_o = "synapses_hidden-", "synapses_output-"
            dname_h, dname_o = "delays_hidden-", "delays_output-"
            param, ext = str(self.N_hidden) + "_" + str(self.N_output), ".txt"
            file_h, file_o = folder + name_h + param + ext, folder + name_o + param + ext
            dfile_h, dfile_o = folder + dname_h + param + ext, folder + dname_o + param + ext
        self.net.restore()
        hidden, output = 'synapses_hidden', 'synapses_output'
        if op.exists(file_h) and op.exists(file_o) and op.exists(dfile_h) and op.exists(dfile_o):
            Fh = open(file_h, 'r')
            Fo = open(file_o, 'r')
            dFh = open(dfile_h, 'r')
            dFo = open(dfile_o, 'r')
            string_h, string_o = Fh.readlines(), Fo.readlines()
            dstring_h, dstring_o = dFh.readlines(), dFo.readlines()
            m, n = len(string_h), len(string_o)
            weights_h = np.empty(m, dtype=float)
            weights_o = np.empty(n, dtype=float)
            delays_h = np.empty(m, dtype=float)
            delays_o = np.empty(n, dtype=float)
            for i in xrange(m):
                weights_h[i] = float(string_h[i][:-1])
                delays_h[i] = float(dstring_h[i][:-1])
            for i in xrange(n):
                weights_o[i] = float(string_o[i][:-1])
                delays_o[i] = float(dstring_o[i][:-1])

            h = self.net[hidden]
            o = self.net[output]
            if len(h.w) == 0 or len(o.w) == 0:
                h.connect('True')
                o.connect('True')
            h.w[:] = weights_h[:]
            o.w[:] = weights_o[:]
            h.delay[:] = delays_h[:]*br.second
            o.delay[:] = delays_o[:]*br.second
            h.tl[:, :, :] = '-1*second'
            h.tp[:, :, :] = '-1*second'
            o.tl[:, :, :] = '-1*second'
            o.tp[:, :, :] = '-1*second'
            self.net.store()
            self.info.read_weights()

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

    def set_inputs_inner(self, indices, times):
        #pudb.set_trace()
        self.indices, self.times = indices, times
        self.net['input'].set_spikes(indices=indices, times=times)
        self.net.store()
        self.info.set_inputs(indices, times)
        
    def set_inputs(self, x):
        indices = np.arange(self.N_inputs)
        times = x
        self.set_inputs_inner(indices, times)

    def get_indices(self, plist):
        pass

    def fit(self, X, Y, method_o='tempotron', method_h=None, threshold=0.7):
        if self.N_hidden == 0:
            method_h = None
        elif self.N_hidden > 0:
            if method_h == None:
                method_h = method_o
        def print_zeros(i, max_order=4):
            for j in range(1, 4):
                if i < 10**j:
                    print ' ',
        #print "PRESETTING WEIGHTS"
        #self.preset_weights(images)
        #pudb.set_trace()
        self.read_weights()
        #train.synaptic_scalling_wrap(self, 1, 1)
        #self.save_weights()
        i, j, k = 0, 0, 0
        pmin = 10000
        p = pmin
        #print "TRAINING - ",
        #print "N_input, N_output, N_hidden: ", self.N_inputs, self.N_output, self.N_hidden
        scaling = True
        min_spikes, max_spikes = 1, 1
        indices = [0, 1, 2, 3]
        #indices = [3, 0, 1, 2]
        #indices = [0, 1]
        r = 10
        plist = None
        p_graph = -1
        while True:
            i += 1
            j += 1
            pold = p
            #if i > 15:
            #    pudb.set_trace()

            plist = train.train_epoch(self, r, \
                i, indices, pmin, X, Y, min_spikes, max_spikes, \
                method_o=method_o, method_h=method_h, scaling=scaling)
            p = sum(plist)
            index_worst = np.argmax(plist)
            #indices = range(len(X))
            #if pmin < 0.85*len(indices):
            #    break
            #    #pudb.set_trace()
            #    factor = float(np.sum(plist) - plist[index_worst]) / (len(plist) - 1)
            #    indices += [index_worst]*int(factor)
            #    indices = np.sort(indices)
            #self.r = self.rb*(min(p, 4)**2) / 4
            #if i > 2:
            #    pudb.set_trace()
            #if p < 1:
            #    break
            if p < pmin:
                pmin = p
                #if i % 10 == 0:
                #self.save_weights()
            if pmin < 50:
                #r = min((np.float(pmin) / 250)**2, 1) * 5.5
                min_spikes = 1
                max_spikes = 1
            p_graph = p
            print "i, r, p, pmin: ", i, r, p, pmin
        #self.save_weights()

    def predict(self, xi, xt, plot=False):
        self.net.restore()
        self.set_inputs_inner(xi, xt)
        self.run()
        self.info.reread()

        return self.info.O.get_spike_times()

    def topology_chart(self, spikes):
        if len(spikes) == 0:
            return -1
        elif len(spikes) > 1:
            return len(spikes)*1000
        else: return spikes[0]*100000

    def topology(self, it_min=0, it_max=10, num=4):
        self.net.restore()
        inputs = np.array([0, 0, 0])*br.ms
        self.set_inputs(inputs)
        indices = np.arange(3)
        times = np.zeros(3)
        t_array = np.linspace(it_min, it_max, num=num)
        o_array = np.empty((len(t_array), len(t_array)), dtype=np.float64)
        if True:
            for i in range(num):
                for j in range(num):
                    times[0] = t_array[i]
                    times[1] = t_array[j]
                    a = self.predict(indices, times*br.ms)
                    print "Predicted",
                    print a
                    o_array[i, j] = self.topology_chart(a)
            return t_array, o_array

    def test_topology(self, n=15, it_min=0, it_max=10, num=20):
        indices = np.array([0])
        indices_net = np.arange(3)
        #i_times = np.zeros(3)
        min_spikes, max_spikes = 1, 1
        i_times = np.array([8, 2, 0])
        desired = 23
        self.read_weights()
        inputs = np.array([0, 0, 0])*br.ms
        self.net.restore()
        self.set_inputs(inputs)
        #pudb.set_trace()
        self.info.set_y_times(desired)
        self.info.O.print_sd_times(tabs=2)
        #self.net.store()
        train.synaptic_scalling_wrap(self, min_spikes, max_spikes)
        self.save_weights()
        for count in range(n):
            times, grid = self.topology(it_min=it_min, it_max=it_max, num=num)
            self.plot_2d(grid, times, count, i_times, desired)
            #pudb.set_trace()
            self.net.restore()
            self.set_inputs_inner(indices=indices_net, times=i_times*br.ms)
            self.info.set_y_times(desired*0.001)
            self.info.reread()
            train.train_step(self, count, 0, 10, method_o="resume", method_h="resume")
            self.info.update_weights(1.0)
            self.info.reset_d_weights()
            #plist = train.train_epoch(self, \
            #    count, , pmin, X, Y, min_spikes, max_spikes, \
            #    method_o=method_o, method_h=method_h, scaling=scaling)

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

    def plot_2d(self, p, grid, axes_times, index_a, index_b, i_time, desired):
        """
        1st index of i_time = vertical axiss
        2nd index = horizontal axis
        """
        methods = ['none']
        #pudb.set_trace()

        #grid = np.random.rand(4, 4)
        fig = plt.figure()
        axes = fig.add_subplot(111)
        plot = axes.pcolor(grid)
        fig.colorbar(plot)
        #pudb.set_trace()
        #interp_method = 'none'
        #axes.imshow(grid, interpolation=interp_method)
        #plt.colorbar(fig)
        #print axes
        fig.suptitle("xor: 3in, 7hid, 1out, 10subc: input " + str(i_time[0]) + ", " + str(i_time[1]) + " ms to " + str(desired) + ": p = " + str(p))

        #plt.grid()
        #plt.figlegend()
        #plt.show()
        img_name = 'xor-test-0/'
        #pudb.set_trace()
        #'h2b/foo-'
        if index_a < 10:
            img_name += '0'
        if index_a < 100:
            img_name += '0'
        img_name+=str(index_a)+'-'+str(index_b)+'.png'
        fig.savefig(img_name)
        fig.clf()
        #fig.cla()
        #fig.close()
        #call(["feh", img_name])

    def plot_desired(self):
        desired = self.desired
        for i in range(len(desired)):
            x = desired[i]
            br.plot((x, x), (0, 100), 'r--')

    def plot_actual(self):
        actual = self.info.O.get_times()
        for i in range(len(actual[0])):
            x = actual[i]
            br.plot((x, x), (0, 100), 'r-')

    def plot(self, figname='test.png', save=False, sh=True, i=None):
        """
            http://stackoverflow.com/questions/14088687/how-to-change-plot-background-color
        """

        S = self.net['synapses_output']
        fig = Figure()
        cv = FC(fig)
        axes = fig.add_subplot(1, 1, 1, axisbg='black')
        #fig = br.figure(figsize=(8, 5))
        #self.plot_desired()
        #self.plot_actual()
        axes.plot((0, self.T)*br.ms, (90, 90), 'b--')
        #a, b, c = S.w[0, 0], S.w[1, 0], S.w[2, 0]
        #a /= 150.0
        #b /= 150.0
        #c /= 150.0
        #pudb.set_trace()
        x = self.net['monitor_o_c']
        n = self.N_hidden*self.N_subc
        for i in range(n):
            #pass
            axes.plot(self.net['monitor_o_c'][i].t, self.net['monitor_o_c'][i].f / 32, 'r-', lw=3, label='C' + str(i))
        #axes.plot(self.net['monitor_o_c'][1].t, self.net['monitor_o_c'][1].c, 'r-', lw=3, label='C1')
        #axes.plot(self.net['monitor_o_c'][2].t, self.net['monitor_o_c'][2].c, 'r-', lw=3, label='C1')
        #axes.plot(self.net['monitor_o_c'].t, b*self.net['monitor_o_c'].c, 'g-', lw=3, label='C2')
        #axes.plot((10, 10)*br.ms, (0, 90), 'b-', lw=2, label="Spike Time")
        #axes.plot(self.net['monitor_o_c'][2].t, c*self.net['monitor_o_c'][2].c, 'y-', lw=3, label='Spike Kernel')
        axes.plot(self.net['values_vo'][0].t, (self.net['values_vo'][0].v+70), 'b-', lw=3, label='V')
        #axes.legend()
        axes.set_xlim([0, 50]*br.ms)
        axes.set_ylim([0, 180])
        #pudb.set_trace()
        if i != None and save == True:
            file_name = '../figs/'
            file_name = './'
            for j in range(4):
                if i < 10**(j+1):
                    file_name += '0'
            file_name += str(i) + '.png'
            self.fig.savefig(file_name)
        if sh==True:
            cv.print_figure(figname)
