import os.path as op
import brian2 as br
import math as ma
import numpy as np
import pudb

class neuron:

    def __init__(self, N=2, T=30):
        self.changes = []
        self.r = 4.0
        r = self.r
        self.dta = 0.2*br.ms
        self.N = N
        self.T = T
        self.tauLP = 1.0
        np.random.seed(12)
        self.a, self.d = None, None
        self.a_post, self.d_post = [], []
        self.a_pre, self.d_pre = [], []

        Ni = br.SpikeGeneratorGroup(self.N, indices=np.asarray([]), times=np.asarray([])*br.ms, name='input')
        Nh = br.NeuronGroup(1, model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1
                                        gL = 96                                     : 1
                                        El = -70                                    : 1
                                        vt = 20                                     : 1
                                        Cm = 3.0                                    : 1                                         
                                        D                                           : 1 (shared)''',
                                        method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                                        reset='v=El', name='hidden', dt=self.dta)
        S = br.Synapses(Ni, Nh,
                   model='''tl                                                      : second
                            tp                                                      : second
                            tau1 = 0.010                                            : second
                            tau2 = 0.00250                                          : second
                            tauL = 0.010                                            : second
                            tauLp = 0.1*tauL                                        : second

                            w                                                       : 1

                            up = (sign(t - tp) + 1.0) / 2                           : 1
                            ul = (sign(t - tl - 3*ms) + 1.0) / 2                    : 1
                            u = (sign(t) + 1.0) / 2                                 : 1

                            c = 100*exp((tp - t)/tau1) - exp((tp - t)/tau2)         : 1
                            D_post = w*c*ul                                         : 1 (summed) ''',
                   post='tl=t+0*ms', pre='tp=t', name='synapses', dt=self.dta)
        S.connect('True')
        S.w[:, :] = '(0*rand()+0)'
        S.tl[:, :] = '-1*second'
        S.tp[:, :] = '-1*second'
        Nh.v[:] = -70
        M = br.StateMonitor(Nh, 'v', record=True, name='monitor0')
        N = br.StateMonitor(S, 'c', record=True, name='monitor1')
        T = br.SpikeMonitor(Nh, variables='v', name='crossings')
        #self.network_op = br.NetworkOperation(self.supervised_update_setup, dt=self.dtb)
        self.net = br.Network(Ni, S, Nh, M, N, T)
        self.actual = self.net['crossings'].all_values()['t'][0]
        self.w_shape = S.w[:, :].shape
        self.net.store()

    def input_output(self):
        n = np.random.randint(1, 3) # of spikes per input neuron
        o = np.random.randint(1, 5)
        self.times = np.array([11])*br.ms
        self.indices = np.array([0])
        max_time = 3 + self.times.max() / br.ms
        self.desired = np.array([15])*br.ms

        #self.times = np.unique(np.random.random_integers(0, int(0.5*self.T), n*self.N))*br.ms
        #self.indices = np.random.random_integers(0, self.N, len(self.times))
        #max_time = 3 + self.times.max() / br.ms
        #self.desired = np.unique(np.random.random_integers(int(max_time), self.T, o))*br.ms

        self.net['input'].set_spikes(indices=self.indices, times=self.times)
        if op.isfile('weights.txt') == True:
            self.read_weights('weights.txt')
        self.net.store()

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

    ### SPIKE CORRELATION FUNCTIONS (IN PROGRESS) ###

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

    ### TRAINING / RUNNING ###

    def data(self):
        self.actual = self.net['crossings'].all_values()['t'][0]

    def supervised_update_setup(self):
        dt = self.dta
        n = len(self.net['synapses'].w)
        v = self.net['monitor0'].v
        c = self.net['monitor1'].c
        actual, desired = self.actual, self.desired
        dw, dw_t = np.zeros(n), np.zeros(n)
        dw_aS, dw_dS = 0, 0
        self.dw_a, self.dw_d = None, None
        for i in range(len(actual)):
            self.dw_a = 0
            index = int(actual[i] / dt)
            for j in range(len(c)):
                dw_t[j] = c[j][index]
            dw -= dw_t / np.linalg.norm(dw_t)
        for i in range(len(desired)):
            self.dw_d = 0
            index = int(desired[i] / dt)
            for j in range(len(c)):
                dw_t[j] = c[j][index]
            dw += dw_t / np.linalg.norm(dw_t)

        return dw

    def supervised_update(self, display=False):
        self.data()
        dw = self.supervised_update_setup()
        self.net.restore()
        self.net['synapses'].w[:, :] += self.r*dw
        self.net.store()
        if display:
            self.print_dw_vec(dw, self.r)
            self.print_dws()

    def train(self, T=None):
        self.run(T)
        self.supervised_update(True)

    def run(self, T=None):
        self.net.restore()
        if T != None:
            self.net.run(T*br.ms)
        else:
            self.net.run(self.T*br.ms)

    ### PLOTTING / DISPLAY ###

    def print_dws(self):
        print "\tself.actual: ", self.actual,
        print "\tself.desired: ", self.desired,
        print "\t",
        if self.dw_d == None:
            print "dw_d == None: ",
        else:
            print "dw_d == OBJECT",
        if self.dw_a == None:
            print "dw_a == None: ",
        else:
            print "dw_a == OBJECT",

    def print_dw_vec(self, dw, r):
        print "\tdw:", dw,
        if abs(dw[0] + dw[1]) < 0.0001:
            self.save_weights()
            #pudb.set_trace()
        print "\tw: ", self.net['synapses'].w[:, :],
        print "\tr:", r,

    def plot_desired(self):
        desired = self.desired
        for i in range(len(desired)):
            x = desired[i]
            br.plot((x, x), (0, 100), 'r--')

    def plot_actual(self):
        actual = self.actual
        for i in range(len(actual)):
            x = actual[i]
            br.plot((x, x), (0, 100), 'r-')

    def plot(self, save=False, show=True, i=None):
        self.fig = br.figure(figsize=(8, 5))
        self.plot_desired()
        self.plot_actual()
        br.plot((0, self.T)*br.ms, (90, 90), 'b--')
        br.plot(self.net['monitor0'][0].t, self.net['monitor0'][0].v+70, 'k-')
        br.plot(self.net['monitor1'][0].t, self.net['monitor1'][0].c, 'g-')
        br.plot(self.net['monitor1'][1].t, self.net['monitor1'][1].c, 'g-')
        if i != None and save == True:
            file_name = '../figs/'
            for j in range(4):
                if i < 10**(j+1):
                    file_name += '0'
            file_name += str(i) + '.png'
            self.fig.savefig(file_name)
        if show==True:
            br.show()
        self.fig.close()
