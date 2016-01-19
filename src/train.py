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
    a = self.net['crossings_o']
    b = self.net['crossings_h']

    actual_s = a.all_values()['t']
    hidden_s = b.all_values()['t']

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
    #if hidden == True:
    dw_h = weight_updates.resume_update_hidden_weights(\
                dw_ih, w_ho, m, n, o, ii, ti/br.second, \
                ih[:], th[:], ia[:], ta[:], d, tau)
    return dw_o, dw_h
    #return dw_o
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

def supervised_update(self, iteration, display=False, method='resume'):
    #if iteration == 3:
    #    pudb.set_trace()
    #if hidden == True:
    #pudb.set_trace()
    dw_o, dw_h = resume_supervised_update_setup(self)
    a, b = np.abs(dw_o).max(), np.abs(dw_h).max()
    #print "MAX:", max(a, b),
    self.actual = self.net['crossings_o'].all_values()['t']
    #print self.self.label
    #print self.actual[0] / br.ms
    #print self.desired
    self.net.restore()
    self.net['synapses_output'].w += self.r*dw_o[:]
    self.net['synapses_hidden'].w += self.r*dw_h[:]
    w_o = self.net['synapses_output'].w
    w_h = self.net['synapses_output'].w
    self.net.store()
    #else:
    #    dw_o = resume_supervised_update_setup(self, hidden=hidden)
    #    print "MAX2:", np.max(np.max(dw_o)),
    #    self.actual = self.net['crossings_o'].all_values()['t']
    #    self.net.restore()
    #    self.net['synapses_output'].w += self.r*dw_o[:]
    #    w_o = self.net['synapses_output'].w
    #    w_h = self.net['synapses_output'].w
    #    self.net.store()

def synaptic_scaling_step(w, m, n, tomod, spikes, max_spikes):
    f = 0.20
    ### m neuron layer to n neuron layer
    ### w[n*i + j] acceses the synapse from neuron i to neuron j

    mod = False
    #pudb.set_trace()
    for j in tomod:
        if len(spikes[j]) > max_spikes:
            w[j:m*n:n] *= 1 - f
        if len(spikes[j]) == 0:
            w[j:m*n:n] *= 1 + f

def print_times(self):
    a = self.net['crossings_o']
    #b = self.net['crossings_h']
    #sa = self.net['synapses_hidden']
    #sb = self.net['synapses_output']

    actual = a.all_values()['t']
    #hidden = b.all_values()['t']
    #desired = self.desired
    #w_ho = sb.w
    #w_ih = sa.w
    
    #print "\n\t\t[(", np.std(w_ih), ", ", np.mean(w_ih), ") (", np.std(w_ho), ",", np.mean(w_ho), ")]"#\n\t\t[", actual, "]\n"
    #print "HIDDEN: ", hidden
    print "ACTUAL: ", actual
    #print "DESIRED: ", desired

def synaptic_scaling(self, max_spikes):
    w_ih = self.net['synapses_hidden'].w
    w_ho = self.net['synapses_output'].w

    #a = self.net['synapses_hidden']
    #b = self.net['synapses_output']

    if False: #np.min(w_ih) < 1:
        np.clip(w_ih, 1, 10000, out=w_ih)
    elif False: #np.min(w_ho) < 1:
        np.clip(w_ho, 1, 10000, out=w_ho)
    else:
        a = self.net['crossings_o']
        b = self.net['crossings_h']

        actual = a.all_values()['t']
        hidden = b.all_values()['t']
        desired = self.desired

        #print "\n\t\t[", hidden, "]\n\t\t[", actual, "]\n"
        #print w_ho
        #print actual, "\t", desired, "\n"
        #print w_ih
        #print hidden
        #pudb.set_trace()
        tomod_a = [i for i in actual if len(actual[i]) == 0 or len(actual[i]) > max_spikes]
        tomod_h = [i for i in hidden if len(hidden[i]) == 0 or len(hidden[i]) > max_spikes]
        if tomod_a != [] or tomod_h != []:
            self.net.restore()
            synaptic_scaling_step(w_ih, self.N_inputs, self.N_hidden, tomod_h, hidden, max_spikes)
            synaptic_scaling_step(w_ho, self.N_hidden, self.N_output, tomod_a, actual, max_spikes)
            self.net.store()

            w_ih_diff = w_ih - self.net['synapses_hidden'].w
            w_ho_diff = w_ho - self.net['synapses_output'].w
            #print 
            return True
    #self.net.restore()
    return False

def train_step(self, iteration, T=None, method='resume', hidden=True):
    mod = True
    i = 1
    while mod:
        self.run(T)
        #print "!",
        mod = synaptic_scaling(self, 5)
        #if i == 100:
        #pudb.set_trace()
        i += 1

    supervised_update(self, iteration, method=method)

def train_epoch(self, iteration, images, method='resume', hidden=True):
    correct = 0
    p, p_total = 0, 0
    for i in images:
        k = 0
        label = self.read_image(i)
        train_step(self, iteration, method=method)
        p_init = self.performance()
        while True:
            train_step(self, iteration, method=method)
            #print "%0.2f" % self.performance(),
            #print self.actual
            p = self.performance()
            print "\t ", k, i, p
            #if p < 2 and (p < 0.5*p_init or k > 100 or p < 0.3):
            #    break
            break
            k += 1
        #print "\t============="
        p_total += p
        #print_times(self)
    print " ",

    return p_total
