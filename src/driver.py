import lif
import pudb

#pudb.set_trace()
nn = lif.neuron()
nn.setup()
nn.input_output()
for i in range(5):
    nn.run()
    nn.plot()
    nn.restore()

#How to set arbitrary events in NeurongGroup to be used by Synapse object?
#
#Hi
#
#So, when generating Synapse object we have the pre and post variables that can be used to define what happens when a pre/post synaptic spike occurs. Now, in the same way I figured that you can also define what happens at an aribtrary set of times or conditions that are not necessarily spike events, for example when defining desired spike times in a supervised learning fashion. I'm trying to use the events option in the NeuronGroup class, and the on_events option in the Synapse class to do this, but I can't figure out how or what kinds of variables I should pass to them.
#
#I've tried the following, but I get an error sayinng "ValueError: Source group does not define an event 'a'."
#
#        Nh = br.NeuronGroup(1, model='''V = El + D : 1
#                                        D : 1''',
#                                        threshold='V>Vt', events={'a':'t==desired'}, name='hidden')
#        S = br.Synapses(Ni, Nh,
#                   model='''tl                                              : second
#                            tp                                              : second
#                            w                                               : 1
#                            r                                               : 1
#                            a = exp((tl - t)/tau1) - exp((tl - t)/tau2)     : 1
#                            u = (sign(t - tp) + 1.0) / 2                    : 1
#                            h = exp((tl - t)/tauL)*u / Cm                   : 1
#                            d = a*u*h                                       : 1
#                            D_post = w*d                                    : 1 (summed)
#                         ''',
#                   pre='tl=t', post='tp=t', on_event=''a", name='synapses')
