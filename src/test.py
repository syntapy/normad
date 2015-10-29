import brian2 as br
import numpy as np
import pudb

indices=np.asarray([0])
times=np.asarray([0]) * br.ms

Ni = br.SpikeGeneratorGroup(1, indices=indices, times=times, name='input')
Nh = br.NeuronGroup(2, model='''dv/dt = ((-gL*(v - El)) + D) / (Cm*second)  : 1
                                gL = 30                                     : 1
                                El = -70                                    : 1
                                vt = 20                                     : 1
                                Cm = 3.0                                    : 1
                                D                                           : 1 (shared)''',
                                method='rk2', refractory=0*br.ms, threshold='v>=vt', 
                                reset='v=El', name='hidden')
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
           post='tl=t+0*ms', pre='tp=t', name='synapses')
S.connect('True')
S.w[:, :] = '400*(2*i + 4*j + 1)'
S.tl[:, :] = '-1*second'
S.tp[:, :] = '-1*second'
Nh.v[:] = -70

M = br.StateMonitor(Nh, 'D', record=True, name='monitor')
N = br.StateMonitor(S, 'c', record=True, name='monitor_c')

br.run(30*br.ms)
pudb.set_trace()
