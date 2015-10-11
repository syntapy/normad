import brian2 as br
import math as ma
import numpy as np
import pudb

class neuron:

    def __init__(self, N=500, T=250):
        self.r = 1.0
        r = self.r
        self.dta = 0.05*br.ms
        self.dtb = 1*br.ms
        self.N = N
        self.T = T*br.ms
        self.tauLP = 100.0

        Ni = br.SpikeGeneratorGroup(self.N, indices=np.asarray([]), times=np.asarray([])*br.ms, name='input')
        Nh = br.NeuronGroup(1, model='''V = El + D                          : 1
                                        D                                   : 1 
                                        Dtilda                              : 1
                                        El                                  : 1
                                        ws                                  : 1
                                        Vt = 20                             : 1''', 
                                        refractory='V>Vt', threshold='V>Vt', name='hidden', dt=self.dta)
        S = br.Synapses(Ni, Nh,
                   model='''
                            tl                                              : second
                            tp                                              : second
                            tau1 = 0.005                                    : second
                            tau2 = 0.00125                                  : second
                            tauL = 0.010                                    : second
                            tauLp = 0.1*tauL                                : second
                            El_post = -70                                   : 1
                            Cm = 300.0                                      : 1

                            w                                               : 1

                            up = (sign(t - tp) + 1.0) / 2                   : 1
                            ul = (sign(t - tl - 3*ms) + 1.0) / 2            : 1
                            u = (sign(t) + 1.0) / 2                         : 1

                            h = exp((tp - t)/tauL)*u / Cm                   : 1
                            hhat = exp((tp - t)/tauLp)*up / Cm              : 1

                            c = (exp((tp - t)/tau1) - exp((tp - t)/tau2))   : 1
                            d = c*ul*up*h                                   : 1
                            dtilda = c*up*hhat                              : 1
                            D_post = w*d                                    : 1 (summed)
                            Dtilda_post = w*dtilda                          : 1 (summed)''',
                   post='''
                            tl=t
                            tp=-10000*second
                        ''', pre='tp=t', name='synapses', dt=self.dta)
        S.connect('True')
        S.w[:, :] = '(0*8000*rand()+15800)'
        S.tl[:, :] = '-10000*second'
        S.tp[:, :] = '-10000*second'
        #pudb.set_trace()
        M = br.StateMonitor(Nh, 'V', record=True, name='monitor')
        N = br.StateMonitor(S, 'dtilda', record=True, name='monitor2')
        #O = br.StateMonitor(S, 'c', record=True, name='monitor3')
        T = br.SpikeMonitor(Nh, variables='V', name='crossings')
        self.network_op = br.NetworkOperation(self.supervised_update_setup, dt=self.dtb)
        self.net = br.Network(Ni, S, Nh, M, N, T, self.network_op)

    def input_output(self):
        n = np.randint(1, 10)
        o = np.randint(1, 10)
        self.indices = np.random.random_integers(0, self.N, n*self.N)
        self.times = np.random.random_integers(0, self.T, n*self.N)*br.ms
        self.desired = np.random.random_integers(0, self.T, o)*br.ms

        self.net['input'].set_spikes(indices=indices, times=times)
        self.net.store()

    def spiking(self, t):
        """
            Desides whether t is within self.dtb of one of the spikes 
            issued by self.net['hidden']
        """
        spikes = self.net['crossings'].all_values()['t'][0]
        t_tmp = t

        if len(spikes) > 0:
            for i in range(len(spikes)):
                if t_tmp - spikes[i] > 0*br.ms and t_tmp - spikes[i] < self.dtb:
                    return True
        return False

    def LNonOverlap(self, Smax, Smin):

        nmax, nmin = len(Smax), len(Smin)
        if Smin[0] > Smax[0]:
            i = 0
            while Smin[i] > Smax[0]:
                i += 1

            return (i+1)
        return 0

    def LOverlapPrevious(self, Smax, Sprev, index):

        i, SC_overlap = 0, 0
        while i < len(Sprev) and Sprev[i] <= Smax[index]:
            SC_overlap += ma.exp((Sprev[i]-Smax[index])/self.tauLP)
            i += 1

        return SC_overlap

    def SCorrelation(self, S1, S2):
        """
            S1 and S2 are lists of spike times.
        """

        if len(S1) > 0 and len(S2) > 0:
            #pudb.set_trace()
            S1, S2 = np.sort(S1), np.sort(S2)
            if S1[-1] > S2[-1]:
                Smax, Smin = S1, S2
            else:
                Smax, Smin = S2, S1

            nmax = len(Smax)
            SC = self.LNonOverlap(Smax, Smin)
            for index in range(nmax):
                SC += self.LOverlapPrevious(Smax, Smin, index)
            return SC / float(len(S1) * len(S2))
        elif len(S1) > 0 or len(S2) > 0:
            return 0
        else:
            return 1

    def supervised_update_setup(self):
        """
            Sets the arrays to ultimately define total weight changes
            to be applied after each run / epoch
        """
        if self.train == True:
            if self.net['synapses'].t in self.desired:
                print "\tDESIRED",
                if self.dw_d == None:
                    self.dw_d = self.net['synapses'].dtilda / np.sum(self.net['synapses'].dtilda)
                else:
                    self.dw_d += self.net['synapses'].dtilda / np.sum(self.net['synapses'].dtilda)

            t = self.net['hidden'].t
            if self.spiking(t):
                print "\tACTUAL",
                if self.dw_a == None:
                    self.dw_a = self.net['synapses'].dtilda / np.sum(self.net['synapses'].dtilda)
                else:
                    self.dw_a += self.net['synapses'].dtilda / np.sum(self.net['synapses'].dtilda)

    def supervised_update_apply(self):
        """
            Applies all weight change vectors that have been summed and normalized
            over entire time period.

            Uses basic adaptive learning rate
        """
        dw = None
        if self.dw_d != None and self.dw_a != None:
            if False not in (self.dw_d == self.dw_a):
                ra = np.random.rand(self.dw_d.shape)
                if np.random.rand() > 0.5:
                    ra *= 2*np.min(self.dw_d)
                    self.dw_d += ra
                else:
                    ra *= 2*np.min(self.dw_a)
                    self.dw_a += ra
            dw = self.dw_d - self.dw_a
        elif self.dw_d != None:
            dw = self.dw_d
        elif self.dw_a != None:
            dw = -self.dw_a

        # Adaptive learning rate
        if dw != None:
            dw = dw / np.linalg.norm(dw)
            p = abs(len(self.net['crossings'].all_values()['t'][0]) - len(self.desired))
            c = 7
            self.net['synapses'].w[:, :] += self.r*dw*(c**(2*p))

            print "\n\tA: ", self.dw_a,
            print "\n\tD: ", self.dw_d,
            print "\n\tdw:", dw,
            print "\n\tW:", self.net['synapses'].w[:, :],
            print "\n\tr:", self.r*(c**(2*p)),
            print "\n",

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

    def train(self, desired):
        self.train = True
        self.dw_d, self.dw_a = None, None
        self.net.run(30*br.ms)
        r = self.r
        self.net.run(30*br.ms)
        if self.dw_d == None:
            print "dw_d == None: ",
        else:
            print "dw_d == OBJECT",
        if self.dw_a == None:
            print "dw_a == None: ",
        else:
            print "dw_a == OBJECT",
        self.supervised_update_apply()

    def run(self):
        self.train = False
        self.net.run(30*br.ms)

    def plot(self, i=None, save=False, show=True):
        a = self.net['monitor2']
        br.plot(self.net['monitor'][0].t, self.net['monitor'][0].V)

        for j in range(3):
            br.plot(self.net['monitor2'][0].t, a[j].dtilda)

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
