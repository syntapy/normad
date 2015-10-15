import os.path as op
import brian2 as br
import math as ma
import numpy as np
import pudb

class neuron:

    def __init__(self, N=2, T=50):
        self.changes = []
        self.r = 1.0
        r = self.r
        self.dta = 0.05*br.ms
        self.dtb = 0.05*br.ms
        self.N = N
        self.T = T
        self.tauLP = 1.0
        np.random.seed(10)
        self.a, self.d = None, None
        self.a_post, self.d_post = [], []

        Ni = br.SpikeGeneratorGroup(self.N, indices=np.asarray([]), times=np.asarray([])*br.ms, name='input')
        Nh = br.NeuronGroup(1, model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1
                                        D                                           : 1 (shared)
                                        gL = 6.0                                    : 1
                                        El = -70                                    : 1
                                        vt = 20                                     : 1
                                        Cm = 3.0                                    : 1 ''',
                                        method='rk2', refractory=3*br.ms, threshold='v>=vt', 
                                        reset='v=El', name='hidden', dt=self.dta)
        S = br.Synapses(Ni, Nh,
                   model='''tl                                              : second
                            tp                                              : second
                            tau1 = 0.005                                    : second
                            tau2 = 0.00125                                  : second
                            tauL = 0.010                                    : second
                            tauLp = 0.1*tauL                                : second

                            w                                               : 1

                            up = (sign(t - tp) + 1.0) / 2                   : 1
                            ul = (sign(t - tl) + 1.0) / 2                   : 1
                            u = (sign(t) + 1.0) / 2                         : 1

                            c = exp((tp - t)/tau1) - exp((tp - t)/tau2)     : 1
                            D_post = w*c*100*ul                             : 1 (summed) ''',
                   post='tl=t+3*ms', pre='tp=t', name='synapses', dt=self.dta)
        S.connect('True')
        S.w[:, :] = '(800*rand()+320)'
        S.tl[:, :] = '-1*second'
        S.tp[:, :] = '-1*second'
        Nh.v[:] = -70
        M = br.StateMonitor(Nh, 'v', record=True, name='monitor')
        N = br.StateMonitor(Nh, 'D', record=True, name='monitor1')
        T = br.SpikeMonitor(Nh, variables='v', name='crossings', when='after_thresholds', order=0)
        self.network_op = br.NetworkOperation(self.supervised_update_setup, dt=self.dtb, when='after_thresholds', order=1)
        self.net = br.Network(Ni, S, Nh, M, N, T, self.network_op)
        self.actual = self.net['crossings'].all_values()['t'][0]
        self.w_shape = S.w[:, :].shape

    def input_output(self):
        n = np.random.randint(1, 3) # of spikes per input neuron
        o = np.random.randint(1, 5)
        self.times = np.unique(np.random.random_integers(0, self.T, n*self.N))*br.ms
        self.indices = np.random.random_integers(0, self.N, len(self.times))
        min_time = 3 + self.times.min() / br.ms
        self.desired = np.unique(np.random.random_integers(int(min_time), self.T, o))*br.ms

        self.net['input'].set_spikes(indices=self.indices, times=self.times)
        if op.isfile('weights.txt') == True:
            self.read_weights('weights.txt')
        self.net.store()

    def spiking(self, t):
        """ Desides whether t is within self.dtb of one of the spikes 
            issued by self.net['hidden'] """
        spikes = self.net['crossings'].all_values()['t'][0]
        t_tmp = t
        if len(spikes) > 0:
            for i in range(len(spikes)):
                if t_tmp - spikes[i] > 0*br.ms and t_tmp - spikes[i] < self.dtb:
                    return True
        return False

    def supervised_update_setup(self):
        """ Sets the arrays to ultimately define total weight changes
            to be applied after each run / epoch """
        if self.d == 1:
            self.d_post.append([self.net['synapses'].t, self.net['synapses'].c])
            self.d = None
        if self.a == 1:
            self.a_post.append([self.net['synapses'].t, self.net['synapses'].c])
            self.a = None
        if self.net['synapses'].t in self.desired:
            #pudb.set_trace()
            dw_d = self.net['synapses'].c
            self.changes.append([dw_d, self.net['synapses'].t, "Desired"])
            self.d = 1
            dw_dS = np.sum(dw_d)
            if dw_dS > 0:
                if self.dw_d == None:
                    self.dw_d =  dw_d / dw_dS
                else:
                    self.dw_d += dw_d / dw_dS
        t = self.net['hidden'].t
        if self.spiking(t):
            #pudb.set_trace()
            dw_a = self.net['synapses'].c
            self.changes.append([dw_a, self.net['synapses'].t, "Actual"])
            self.a = 1
            dw_aS = np.sum(dw_a)
            if self.dw_a == None:
                if dw_aS > 0:
                    self.dw_a = dw_a / dw_aS
            else:
                self.dw_a += dw_a / dw_aS

    def supervised_update_apply(self):
        """ Applies all weight change vectors that have been summed and normalized
            over entire time period.
            Uses basic adaptive learning rate. """
        dw = np.zeros(self.w_shape)
        if self.dw_d != None:
            dw += self.dw_d
        if self.dw_a != None:
            dw -= self.dw_a

        # Adaptive learning rate
        if self.dw_d != None or self.dw_a != None:
            dw = dw / np.linalg.norm(dw)
            p = abs(len(self.net['crossings'].all_values()['t'][0]) - len(self.desired))
            c = 2
            self.net['synapses'].w[:, :] += self.r*dw*(c**(p))

            #print "\tdw:", dw,
            #if abs(dw[0] + dw[1]) < 0.0001:
            #    self.save_weights()
            #    pudb.set_trace()
            #print "\tw: ", self.net['synapses'].w[:, :],
            #print "\tr:", self.r*(c**(p)),

    def save_weights(self, fname='weights.txt'):
        F = open(fname, 'w')
        s = self.net['synapses']
        n = len(s.w[:])
        for i in range(n):
            F.write(str(s.w[i]))
            F.write('\n')
        F.close()

    def read_weights(self, fname='weights.txt'):
        F = open(fname, 'r')
        string = F.readlines()
        n = len(string)
        weights = np.empty(n, dtype=float)
        for i in xrange(n):
            weights[i] = float(string[i][:-1])

        self.net['synapses'].w[:] = weights[:]

    def LNonOverlap(self, Smax, Smin):
        """ Finds i in which Smax[i] > Smin[0] """
        nmax, nmin, i = len(Smax), len(Smin), 0
        if Smin[0] > Smax[0]:
            while Smin[0] > Smax[i]:
                i += 1
        return i

    def LOverlapPrevious(self, Smax, Smin, index):
        i, SC_overlap = 0, 0
        while i < len(Smin) and Smin[i] <= Smax[index]:
            SC_overlap += ma.exp((Smin[i]-Smax[index])/self.tauLP)
            i += 1
        return SC_overlap

    def L(self, t1, t2):
        if t2 > t1:
            return ma.exp((t1 - t2) / self.tauLP)
        return ma.exp((t2 - t1) / self.tauLP)

    def SCorrelation(self, S1, S2):
        """ S1 and S2 are lists of spike times. """
        S1, S2 = np.sort(S1), np.sort(S2)
        total, integral1, integral2 = 0, 0, 0
        i, j, = 0, 0
        n1, n2 = len(S1), len(S2)
        i_index, j_index = min(i, n1-1), min(j, n2-1)
        t1_last, t2_last = -10, -10 
        if n1 > 0 and n2 > 0:
            while i < n1 or j < n2:
                if S1[i_index] < S2[j_index]:
                    integral1 /= np.exp((S1[i_index] - t1_last) / self.tauLP)
                    integral1 += 1
                    total += integral1 / np.exp((S2[j_index] - S1[i_index]) / self.tauLP)
                    i += 1
                    i_index = min(i, n1-1)
                elif S1[i_index] > S2[j_index]:
                    integral2 /= np.exp((S2[j_index] - t2_last) / self.tauLP)
                    integral2 += 1
                    total += integral2 / np.exp((S1[i_index] - S2[j_index]) / self.tauLP)
                    j += 1
                    j_index = min(j, n2-1)
                elif S1[i_index] == S2[j_index]:
                    integral1 /= np.exp((S1[i_index] - t1_last) / self.tauLP)
                    integral2 /= np.exp((S2[j_index] - t2_last) / self.tauLP)
                    integral1, integral2 = integral1 + 1, integral2 + 1
                    total += integral1*integral2
                    i, j = i + 1, j + 1
                    i_index, j_index = min(i, n1 - 1), min(j, n2 - 1)
            return total / float(n1 * n2)
        elif n1 > 0 or n2 > 0:
            return 0
        else:
            return 1

    def matches(self, S1, S2):
        if len(S1) > 0 and len(S2) > 0:
            i_min, j_min, min_diff = -1, -1, float("inf")
            for i in range(len(S1)):
                for j in range(len(S2)):
                    diff = abs(S1[i] - S2[j])
                    if diff < min_diff:
                        i_min, j_min, min_diff = i, j, diff
            return i_min, j_min

    def SCorrelationSIMPLE(self, S1, S2, dt):
        S1, S2 = np.sort(S1), np.sort(S2)
        i, j = 0, 0
        n1, n2 = len(S1), len(S2)
        total = 0
        if n1 != n2:
            total += np.exp(abs(n1-n2))
        matches = self.match(S1, S2)
        for i in range(len(matches)):
            total += abs(S1[matches[i][0]] - S2[matches[i][1]])
        return total

    def restore(self):
        self.w = self.net['synapses'].w[:, :]
        self.net.restore()
        self.net['synapses'].w = self.w
        self.net.store()

    def untrained(self):
        d = self.desired
        a = self.net['crossings'].all_values()['t'][0]

        if len(d) != len(a):
            return True
        
        if len(d) == len(a) and len(d) > 0:
            for i in range(len(d)):
                if abs((d[i] - a[i]) / br.ms) > 1:
                    return True
        return False

    def train(self):
        self.dw_d, self.dw_a = None, None
        r = self.r
        self.run(self.T)
        self.actual = self.net['crossings'].all_values()['t'][0]
        #print "\tself.actual: ", self.actual,
        #print "\tself.desired: ", self.desired,
        #print "\t",
        #if self.dw_d == None:
        #    print "dw_d == None: ",
        #else:
        #    print "dw_d == OBJECT",
        #if self.dw_a == None:
        #    print "dw_a == None: ",
        #else:
        #    print "dw_a == OBJECT",
        self.supervised_update_apply()

    def run(self, T=None):
        if T != None:
            T = T*br.ms
            self.net.run(T)
        else:
            self.net.run(self.T*br.ms)

    def plot(self, i=None, save=False, show=True):
        br.plot(self.net['monitor'][0].t, 100*self.net['monitor'][0].v)
        br.plot(self.net['monitor1'][0].t, self.net['monitor1'][0].D / 10.0)
        if i != None and save == True:
            file_name = '../figs/'
            for j in range(4):
                if i < 10**(j+1):
                    file_name += '0'
            file_name += str(i) + '.png'
            br.savefig(file_name)
        if show==True:
            br.show()
        br.clf()
