import os.path as op
import brian2 as br
import math as ma
import numpy as np
import scipy
import pudb

class neuron:

    ###################
    ### MODEL SETUP ###
    ###################

    def __init__(self, N=80, T=250):
        self.changes = []
        self.trained = False
        self.r = 4.0
        self.dta = 0.2*br.ms
        self.N = N
        self.T = T
        self.tauLP = 5.0
        self.seed = 12
        np.random.seed(self.seed)
        self.a, self.d = None, None
        self.a_post, self.d_post = [], []
        self.a_pre, self.d_pre = [], []
        #self._input_output()
        self._groups()

    def _groups(self):
        Ni = br.SpikeGeneratorGroup(self.N, indices=np.asarray([]), times=np.asarray([])*br.ms, name='input')
        Nh = br.NeuronGroup(1, model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1
                                        gL = 30                                     : 1
                                        El = -70                                    : 1
                                        vt = 20                                     : 1
                                        Cm = 3.0                                    : 1
                                        D                                           : 1 (shared)''',
                                        method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                                        reset='v=El', name='hidden', dt=self.dta)
        S = br.Synapses(Ni, Nh,
                   model='''tl                                                      : second
                            tp                                                      : second
                            tau1 = 0.0025                                           : second
                            tau2 = 0.000625                                         : second
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
        S.w[:, :] = '(200*rand()+0)'
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

    def set_train_spikes(self, indices=[], times=[], desired=[]):
        self.net.restore()
        self.indices, self.times, self.desired = indices, times*br.ms, desired*br.ms
        self.net['input'].set_spikes(indices=self.indices, times=self.times)
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

    def _rflatten(A):
        if A.dtype == 'O':
            dim = np.shape(A)
            n = len(dim)
            ad = np.zeros(n)
            i = 0
            tmp = []
            for a in A:
                tmp.append(rflatten(a))
            return_val = np.concatenate(tmp)
        else:
            return_val = A.flatten()

        return return_val

    def read_images(self):
        c1_train = scipy.io.loadmat('../data/train-1.mat')['c1a'][0]
        c1_test = scipy.io.loadmat('../data/test-1.mat')['c1b'][0]

        N_train, N_test = len(c1_train), len(c1_test)
        self.train_features = np.empty(N_train, dtype=object)
        self.test_features = np.empty(N_test, dtype=object)

        for i in xrange(N_train):
            train_features[i] = self._rflatten(c1_train[i])
        for i in xrange(N_train):
            test_features[i] = self._rflatten(c1_test[i])

        train_labels = scipy.io.loadmat('../data/train-label.mat')['train_labels_body']
        test_labels = scipy.io.loadmat('../data/test-label.mat')['test_labels_body']
        self.train_mnist = [train_features, train_labels]
        self.test_mnist = [test_features, test_labels]

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
    ### TRAINING / RUNNING ###
    ##########################

    def supervised_update_setup(self):
        self.actual = self.net['crossings'].all_values()['t'][0]
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
                print "\ti, j = ", i, ", ", j
                dw_t[j] = c[j][index]
            dw += dw_t / np.linalg.norm(dw_t)

        return dw / np.linalg.norm(dw)

    def supervised_update(self, display=True):
        dw = self.supervised_update_setup()
        self.net.restore()
        self.net['synapses'].w[:, :] += self.r*dw
        self.net.store()
        if display:
            #self.print_dw_vec(dw, self.r)
            self.print_dws(dw)

    def reset(self):
        self.net.restore()
        self.net['synapses'].w[:, :] = '0'
        self._input_output()

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
            self._groups()
            self._input_output()
            self.train()

    def test_if_trained(self, Dt=1.0):
        self.trained = True
        if len(self.actual) != len(self.desired):
            self.trained = False
        else:
            for i in range(len(self.actual)):
                if abs((self.actual[i] - self.desired[i])/br.ms) > Dt:
                    self.trained = False

    def train_step(self, T=None):
        self.run(T)
        self.supervised_update()

    def train(self, T=None, dsp=True):
        if dsp == True:
            i = 0
            while self.not_trained():
                print "i = ", i,
                i += 1
                self.train_step(T)

    def run(self, T=None):
        self.net.restore()
        if T != None:
            self.net.run(T*br.ms)
        else:
            self.net.run(self.T*br.ms)

    ##########################
    ### PLOTTING / DISPLAY ###
    ##########################

    def times_format(self):
        inputs = []
        for i in range(len(self.times)):
            inputs.append([self.indices[i], self.times[i]])

        return inputs

    def print_dws(self, dw):
        #print "\tinput: ", self.times_format(), "\n"
        #print "\tdw: ", dw,
        #print "\tw: ", self.net['synapses'].w[:, :],
        print "\tactual: ", self.actual,
        print "\tdesired: ", self.desired
        #if self.dw_d == None:
        #    print "dw_d == None: ",
        #else:
        #    print "dw_d == OBJECT",
        #if self.dw_a == None:
        #    print "dw_a == None: ",
        #else:
        #    print "dw_a == OBJECT",

    def print_dw_vec(self, dw, r):
        #if abs(dw[0] + dw[1]) < 0.0001:
        #    self.save_weights()
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

    ##################################################
     ### SPIKE CORRELATION FOUTINES (IN PROGRESS) ###
    ##################################################

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
        """ Atomic low-pass filter function """
        if t2 > t1:
            return ma.exp((t1 - t2) / self.tauLP)
        return ma.exp((t2 - t1) / self.tauLP)

    def F(self, S, t):
        """ Returns summ of low-passed spikes at time t """
        return_val = 0
        if len(S) > 0:
            i = 0
            while i < len(S) and S[i] <= t:
                return_val += self.L(S[i], t)
                i += 1
        return return_val

    def SCorrelationSlow(self, S1, S2, dt=0.05):
        """
            Trapezoid integration rule to compute inner product:
            
                <L(S1), L(S2)> / (||L(S1)|| * ||L(S2)||)
        """
        na = self.F(S1, 0) + self.F(S1, self.T-dt)
        nb = self.F(S2, 0) + self.F(S2, self.T-dt)
        return_val = self.F(S1, 0) * self.F(S2, 0) + \
                    self.F(S1, self.T-dt) * self.F(S2, self.T-dt)
        for t in range(1, self.T - 1):
            return_val += 2*self.F(S1, t) * self.F(S2, t) * dt
            na += self.F(S1, t) * dt
            nb += self.F(S2, t) * dt
        return return_val / (na * nb)

    def _prel_SC(self, Smax, Smin):
        i, t_last = 0, 0

        while Smax[i] < Smin[0]:
            tot *= np.exp((t_last - Smax[i]) / self.tauLP)
            tot += 1
            t_last = Smax[i]
            i += 1
        tot *= np.exp((t_last - Smin[0]) / self.tauLP)
        if Smax[i] == Smin[0]:
            tot += 1
        return tot*self.tauLP, i

    def _equal_len_SC(self, Smax, Smin):
        return_val = 0
        integral = 0
        t_last = 0
        if Smax[0] < Smin[0]:
            return_val, i = self._prel_SC(Smax, Smin)
            j = 0
            while i < len(Smax):
                while Smin[j] <= Smax[i]:
                    integral += 1
                    integral *= np.exp((Smin[j] - Smax[i]) / self.tauLP)
                    j += 1
                i += 1

    def SC_step(self, total, t1, t2, int1, int2, i, j, S1, S2):
        if S1[i] < S2[j]:
            
        elif S1[i] > S2[j]:
            pass
        elif S1[i] == S2[j]:
            pass

    def _equal_len_SC(self, S1, S2):
        total, int1, int2 = 0, 0, 0
        i, j = 0, 0
        t1, t2 = -10, -10

        while i < len(S1) and j < len(S2):
            total, int1, int2, i, j = self.SC_step(total, t1, t2, int1, int2, i, j, S1, S2)

    def SCorrelation(self, S1, S2):
        S1, S2 = np.sort(S1), np.sort(S2)
        total, integral1, integral2 = 0, 0, 0
        i, j = 0, 0

        if len(S1) == len(S2):
            if len(S1) > 0:
                if S1[-1] > S2[-1]:
                    return self._equal_len_SC(S1, S2)
                return self._equal_len_SC(S2, S1)
            return 1
        return 0
                    

    def SCorrelation(self, S1, S2):
        """ 
            Analytical approach to compute the inner product:
            
                <L(S1), L(S2)> / (||L(S1)|| * ||L(S2)||)
        """

        pudb.set_trace()
        S1, S2 = np.sort(S1), np.sort(S2)
        total, integral1, integral2 = 0, 0, 0
        i, j, = 0, 0
        n1, n2 = len(S1), len(S2)
        i_index, j_index = 0, 0
        t1_last, t2_last = -10, -10
        if n1 > 0 and n2 > 0:
            while i < n1 or j < n2:
                if S1[i_index] < S2[j_index]:
                    integral1 /= np.exp((S1[i_index] - t1_last) / self.tauLP)
                    integral1 += 1
                    total += integral1 * np.exp((S2[i_index] - S1[j_index]) / self.tauLP)
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
