import brian2 as br
import numpy as np
import pudb

class neuron:

    def __init__(self):
        self.tau1 = 5.00*br.ms
        self.tau2 = 1.25*br.ms
        self.tauL = 10.0*br.ms
        self.tauLp = 0.7*self.tauL
        self.El = -70.0
        self.Cm = 300.0
        self.Vt = 20.0

    def setup(self):
        Ni = br.SpikeGeneratorGroup(3, indices=np.asarray([]), times=np.asarray([])*br.ms, name='input')
        Nh = br.NeuronGroup(1, model='''V = El + D : 1
                                        D : 1''', 
                                        threshold='V>Vt', name='hidden')
        S = br.Synapses(Ni, Nh,
                   model='''tl                                              : second
                            tp                                              : second
                            w                                               : 1
                            c = exp((tl - t)/tau1) - exp((tl - t)/tau2)     : 1
                            u = (sign(t - tp) + 1.0) / 2                    : 1
                            h = exp((tl - t)/tauL)*u / Cm                   : 1
                            hhat = exp(-t/tauLp)*u / Cm                     : 1
                            d = c*u*h                                       : 1
                            dtilda = c*h                                    : 1
                            D_post = w*d                                    : 1 (summed)
                         ''',
                   pre='tl=t',
                   post='tp=t', name='synapses')
        S.connect('True')
        S.w[:, :] = '(5+3*rand())'
        S.tl[:, :] = '-10000*second'
        S.tp[:, :] = '-10000*second'
        M = br.StateMonitor(Nh, 'V', record=True, name='monitor')
        self.net = br.Network(Ni, S, Nh, M)

    def input_output(self, inputs=[[0, 1], [6, 2]], desired=[12]):
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

    def run(self):
        #self.net['input'].t = 0*br.ms
        #self.net['synapses'].t = 0*br.ms
        tau1 = self.tau1
        tau2 = self.tau2
        tauL = self.tauL
        El = self.El
        Cm = self.Cm
        Vt = self.Vt
        desired = self.desired[0]
        #self.net.restore()
        self.net.run(20*br.ms)

    def plot(self):
        br.plot(self.net['monitor'][0].t, self.net['monitor'][0].V)
        br.show()
