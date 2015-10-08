import brian2 as br
import numpy as np
import pudb

class neuron:

    def __init__(self):
        self.r = 10.0
        r = self.r
        self.dta = 0.05*br.ms
        self.dtb = 1*br.ms

        Ni = br.SpikeGeneratorGroup(3, indices=np.asarray([]), times=np.asarray([])*br.ms, name='input')
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
                            D_post = w*d                                    : 1 (summed)
                            Dtilda_post = w*c*hhat                          : 1 (summed)''',
                   post='tl=t', pre='tp=t', name='synapses', dt=self.dta)
        S.connect('True')
        S.w[:, :] = '(14000*rand()+1600)'
        S.tl[:, :] = '-10000*second'
        S.tp[:, :] = '-10000*second'
        M = br.StateMonitor(Nh, 'V', record=True, name='monitor')
        N = br.StateMonitor(S, 'd', record=True, name='monitor2')
        O = br.StateMonitor(S, 'c', record=True, name='monitor3')
        T = br.SpikeMonitor(Nh, variables='V', name='crossings')
        self.network_op = br.NetworkOperation(self.supervised_update, dt=self.dtb)
        self.net = br.Network(Ni, S, Nh, M, N, O, T, self.network_op)

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

    def supervised_update(self):
        dw_d, dw_a = None, None
        if self.net['synapses'].t in self.desired:
            print "\tDESIRED",
            r = self.r
            dw_d = r*self.net['synapses'].Dtilda / np.sum(self.net['synapses'].Dtilda)

        t = self.net['hidden'].t
        if self.spiking(t):
            print "\tACTUAL",
            r = self.r
            dw_a = r*self.net['synapses'].Dtilda / np.sum(self.net['synapses'].Dtilda)

        pudb.set_trace()
        if dw_d != None and dw_a != None:
            if False not in dw_d == dw_a:
                ra = np.rand(dw_d.shape)
                if np.random.rand() > 0.5:
                    ra *= 2*np.min(dw_d)
                    dw_d += ra
                else:
                    ra *= 2*np.min(dw_a)
                    dw_a += ra
            self.net['synapses'].w[:, :] += dw_d - dw_a
        elif dw_d != None:
            self.net['synapses'].w[:, :] += dw_d
        elif dw_a != None:
            self.net['synapses'].w[:, :] -= dw_a


    def input_output(self, inputs=[[0, 1], [6, 9]], desired=[18]):
        indices = np.asarray(inputs[0])
        times = np.asarray(inputs[1]) * br.ms
        self.desired = np.asarray(desired) * br.ms

        self.net['input'].set_spikes(indices=indices, times=times)
        self.net.store()

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

    def run(self):
        desired = self.desired[0]
        r = self.r
        self.net.run(30*br.ms)

    def plot(self, i=None, show=True):
        a = self.net['monitor2']
        b = self.net['monitor3']
        br.plot(self.net['monitor'][0].t, self.net['monitor'][0].V)
        md = a[0].d + a[1].d + a[2].d
        mc = b[0].c + b[1].c + b[2].c

        br.plot(self.net['monitor2'][0].t, md)
        br.plot(self.net['monitor3'][0].t, 1.1*mc)
        if i != None:
            file_name = '../figs/'
            if i < 10:
                file_name += '0'
            if i < 100:
                file_name += '0'
            file_name += str(i) + '.png'
            br.savefig(file_name)
            br.clf()
        if show==True:
            br.show()
            br.clf()
