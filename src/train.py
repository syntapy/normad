import numpy as np
import pudb
import brian2 as br
import weight_updates_numba as weight_updates
import weight_updates_py

br.prefs.codegen.target = 'weave'  # use the Python fallback
def resume_supervised_update_setup(self, hidden=True):
    #pudb.set_trace()

    #STDP Parameters
    Ap, Am, a_nh = 1.2, 0.5, 0.05
    tau = 0.005

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
    p = self.N_subc
    #Si = self.times
    #Sa, Sd = self.actual, self.desired
    #Sa, Sh = weight_updates.sort(Sa), weight_updates.sort(Sh)
    w_ho = self.net['synapses_output'].w[:]
    w_ih = self.net['synapses_hidden'].w[:]
    dw_ho = np.zeros(np.shape(w_ho), dtype=np.float64)
    dw_ih = np.zeros(np.shape(w_ih), dtype=np.float64)
    delay_ho = self.net['synapses_output'].delay
    delay_ih = self.net['synapses_hidden'].delay
    dw_o = weight_updates.resume_update_output_weights(\
                dw_ho, m, n, o, p, ih[:], th[:], ia[:], ta[:], d, tau, \
                Ap, Am, a_nh)
    #if hidden == True:
    dw_h = weight_updates.resume_update_hidden_weights(\
                dw_ih, w_ho, delay_ih, delay_ho, \
                m, n, o, p, ii, ti/br.second, \
                ih[:], th[:], ia[:], ta[:], d, tau, \
                Ap, Am, a_nh)
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
    dw_o, dw_h = resume_supervised_update_setup(self)
    a, b = np.abs(dw_o).max(), np.abs(dw_h).max()
    self.actual = self.net['crossings_o'].all_values()['t']
    self.net.restore()
    self.net['synapses_output'].w += self.r*dw_o[:]
    self.net['synapses_hidden'].w += self.r*dw_h[:]
    w_o = self.net['synapses_output'].w
    w_h = self.net['synapses_output'].w
    self.net.store()

def synaptic_scaling_step(w, m, n, p, tomod, spikes, max_spikes):
    f = 0.05
    ### m neuron layer to n neuron layer
    ### w[n*i + j] acceses the synapse from neuron i to neuron j

    mod = False
    for j in tomod:
        if len(spikes[j]) > max_spikes:
            w[j:m*n*p:n*p] *= np.float(1 - f)**np.sign(w[j:m*n*p:n*p])
        if len(spikes[j]) == 0:
            w[j:m*n*p:n*p] *= np.float(1 + f)**np.sign(w[j:m*n*p:n*p])

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


def synaptic_scaling_singlelayer(self, max_spikes, iteration=0):
    w_io = self.net['synapses_output'].w

    # KEEP FOLOWING COMMENTS !!!
    #if False: #np.min(w_ih) < 1:
    #    np.clip(w_ih, 1, 10000, out=w_ih)
    #elif False: #np.min(w_ho) < 1:
    #    np.clip(w_ho, 1, 10000, out=w_ho)
    #else:
    #if iteration == 100:
    #    pudb.set_trace()
    # KEEP PREVIOUS COMMENTS !!!
    a = self.net['crossings_o']

    actual = a.all_values()['t']
    desired = self.desired

    tomod_a = [i for i in actual if len(actual[i]) == 0 or len(actual[i]) > max_spikes]
    if tomod_a != []:
        self.net.restore()
        synaptic_scaling_step(w_io, self.N_inputs, self.N_output, self.N_subc, tomod_a, actual, max_spikes)
        self.net.store()

        #w_io_diff = w_io - self.net['synapses'].w
        return True
    return False

def synaptic_scaling_multilayer(self, max_spikes, iteration=0):
    w_ih = self.net['synapses_hidden'].w
    w_ho = self.net['synapses_output'].w

    # KEEP FOLOWING COMMENTS !!!
    #if False: #np.min(w_ih) < 1:
    #    np.clip(w_ih, 1, 10000, out=w_ih)
    #elif False: #np.min(w_ho) < 1:
    #    np.clip(w_ho, 1, 10000, out=w_ho)
    #else:
    #if iteration == 100:
    #    pudb.set_trace()
    # KEEP PREVIOUS COMMENTS !!!
    a = self.net['crossings_o']
    b = self.net['crossings_h']

    actual = a.all_values()['t']
    hidden = b.all_values()['t']
    desired = self.desired

    tomod_a = [i for i in actual if len(actual[i]) == 0 or len(actual[i]) > max_spikes]
    tomod_h = [i for i in hidden if len(hidden[i]) == 0 or len(hidden[i]) > max_spikes]
    if tomod_a != [] or tomod_h != []:
        self.net.restore()
        synaptic_scaling_step(w_ih, self.N_inputs, self.N_hidden, self.N_subc, tomod_h, hidden, max_spikes)
        synaptic_scaling_step(w_ho, self.N_hidden, self.N_output, self.N_subc, tomod_a, actual, max_spikes)
        self.net.store()

        w_ih_diff = w_ih - self.net['synapses_hidden'].w
        w_ho_diff = w_ho - self.net['synapses_output'].w
        return True
    return False

def synaptic_scaling(self, max_spikes, iteration=0):
    if self.N_hidden > 0:
        return synaptic_scaling_multilayer(self, max_spikes, iteration=iteration)
    else:
        return synaptic_scaling_singlelayer(self, max_spikes, iteration=iteration)

def synaptic_scalling_wrap(self, max_spikes):
    mod = True
    i = 1
    while mod:
        self.run(None)
        #print "!",
        mod = synaptic_scaling(self, max_spikes)
        #if i == 100:
        #pudb.set_trace()
        i += 1

def train_step(self, iteration, T=None, method='resume', hidden=True):
    synaptic_scalling_wrap(self, 1)
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
            if type(p) == str:
                pudb.set_trace()
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
