import brian2 as br
import numpy as np
import pudb

#pudb.set_trace()

tau1 = 5.00*br.ms
tau2 = 1.25*br.ms
tauL = 10.0*br.ms
El = -70.0
Cm = 300.0

indices = np.asarray([0, 1])
times = np.asarray([6, 2]) * br.ms

Ni = br.SpikeGeneratorGroup(3, indices=indices, times=times)
Nh = br.NeuronGroup(1, model="""V = El + D : 1
                                D : 1""")
S = br.Synapses(Ni, Nh,
           model='''
                    tl                                              : second
                    w                                               : 1
                    a = exp((tl - t)/tau1) - exp((tl - t)/tau2)     : 1
                    u = (sign(t - tl) + 1.0) / 2                    : 1
                    h = exp((tl - t)/tauL)*u / Cm                   : 1
                    d = a*u*h                                       : 1
                    D_post = w*d                                    : 1 (summed)
                 ''',
           pre='tl+=t - tl')
S.connect('True')
S.w[:, :] = '(3+5*rand())'
S.tl[:, :] = '-10000*second'
M = br.StateMonitor(Nh, 'V', record=True)
br.run(20*br.ms)
br.plot(M[0].t, M[0].V)
br.show()
