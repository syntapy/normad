import numpy as np
import pudb
import brian2 as br
import weight_updates_numba as weight_updates
import weight_updates_py

br.prefs.codegen.target = 'weave'  # use the Python fallback
def resume_supervised_update_setup(self, hidden=True):
    #pudb.set_trace()
    a = self.net['input']
    ii, ti = self.indices, self.times
    ih, th = self.net['crossings_h'].it_
    ia, ta = self.net['crossings_o'].it_
    d = self.desired

    #print "\t", ta
    #print "\t", ia

    #pudb.set_trace()
    m, n, o= self.N_inputs, self.N_hidden, self.N_output
    #Si = self.times
    #Sa, Sd = self.actual, self.desired
    #Sa, Sh = weight_updates.sort(Sa), weight_updates.sort(Sh)
    w_ho = self.net['synapses_output'].w[:]
    w_ih = self.net['synapses_hidden'].w[:]
    dw_ho = np.zeros(np.shape(w_ho), dtype=np.float64)
    dw_ih = np.zeros(np.shape(w_ih), dtype=np.float64)
    tau = self.net['synapses_hidden'].tau1 / (1000*br.msecond)
    dw_o = weight_updates.resume_update_output_weights(\
                dw_ho, m, n, o, ih[:], th[:], ia[:], ta[:], d, tau)
    if hidden == True:
        dw_h = weight_updates.resume_update_hidden_weights(\
                    dw_ih, w_ho, m, n, o, ii, ti/br.second, \
                    ih[:], th[:], ia[:], ta[:], d, tau)
        return dw_o, dw_h
    return dw_o
    #dw_o_py = weight_updates_py.resume_update_output_weights(self)
    #dw_h_py = weight_updates_py.resume_update_hidden_weights(self)

    #ocmp = np.allclose(dw_h, dw_h_py, rtol=4*(1e-2), atol=3e-1)
    #hcmp = np.allclose(dw_o, dw_o_py, rtol=4*(1e-2), atol=3e-1)
    #if not ocmp or not hcmp:
    #    pudb.set_trace()


    #print "\tDIFFS:"
    #print "\t\tout:\t", dw_o - dw_o_py
    #print "\t\thidden:\t", dw_h - dw_h_py
    #pudb.set_trace()

def supervised_update(self, display=False, method='resume', hidden=True):
    if hidden == True:
        dw_o, dw_h = resume_supervised_update_setup(self, hidden=hidden)
        self.net.restore()
        self.net['synapses_output'].w += self.r*dw_o[:]
        self.net['synapses_hidden'].w += self.r*dw_h[:]
        w_o = self.net['synapses_output'].w
        w_h = self.net['synapses_output'].w
        self.net.store()
    else:
        dw_o = resume_supervised_update_setup(self, hidden=hidden)
        self.net.restore()
        self.net['synapses_output'].w += self.r*dw_o[:]
        w_o = self.net['synapses_output'].w
        w_h = self.net['synapses_output'].w
        self.net.store()

def synaptic_scaling_step(w, m, n, spikes):
    f = 0.02
    ### m neuron layer to n neuron layer
    ### w[n*i + j] acceses the synapse from neuron i to neuron j

    mod = False

    for j in spikes:
        if len(spikes[j]) > 1:
            w[j:m*n:n] *= 1 - f
            mod = True
            print " -",
        elif len(spikes[j]) < 1:
            w[j:m*n:n] *= 1 + f
            mod = True
            print " +",
        else:
            print " =",
    return mod

def synaptic_scaling(self):
    a = self.net['crossings_o']
    b = self.net['crossings_h']

    actual = a.all_values()['t']
    hidden = b.all_values()['t']

    #pudb.set_trace()
    w_ih = self.net['synapses_hidden'].w
    w_ho = self.net['synapses_output'].w

    self.net.restore()
    print "[", hidden,
    moda = synaptic_scaling_step(w_ih, self.N_inputs, self.N_hidden, hidden)
    print "] [",
    modb = synaptic_scaling_step(w_ho, self.N_hidden, self.N_output, actual)
    print "]", actual,
    self.net.store()
    print "\n",

    return moda or modb

def train_step(self, T=None, method='resume', hidden=True):
    mod = True
    i = 1
    #w_ih = self.net['synapses_hidden'].w
    #w_ho = self.net['synapses_output'].w
    #pudb.set_trace()
    while mod:
        self.run(T)
        print "\t\t\t run_try", i,
        mod = synaptic_scaling(self)
        i += 1

    supervised_update(self, method=method, hidden=hidden)

def train_epoch(self, images, method='resume', dsp=True, ch=False, hidden=True):
    correct = 0
    #i, j = a, 0
    p = 0
    for i in images:
        #for i in range(a, b):
        label = self.read_image(i, ch=ch)
        #if label == 0:
        #j += 1
        train_step(self, method=method, hidden=hidden)
        p += self.performance()
        print "\tImage ", i, " trained"
        if self.neuron_right_outputs():
            correct += 1
        #i += 1
    return correct, p
