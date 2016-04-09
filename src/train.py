import numpy as np
import pudb
import brian2 as br
import weight_updates_numba as weight_updates
import weight_updates_py

class resume_params:

    def __init__(self,Ap=1.2, Am=0.5, a_nh=0.05, tau=0.005):
        self.Ap = Ap
        self.Am = Am
        self.a_nh = a_nh
        self.tau = tau

    def get_params(self):
        return self.Ap, self.Am, self.a_nh, self.tau

br.prefs.codegen.target = 'weave'  # use the Python fallback
def supervised_update_setup(self, method_o='tempotron', method_h=None):
    #pudb.set_trace()
    if method_o == 'resume':
        update_function_o = weight_updates.resume_update_output_weights
    elif method_o == 'tempotron':
        update_function_o = weight_updates.tempotron_update_output_weights

    if method_h == 'resume':
        update_function_h = weight_updates.resume_update_hidden_weights
    elif method_h == 'tempotron':
        update_function_h = weight_updates.tempotron_update_hidden_weights

    #pudb.set_trace()
    self.info.params = resume_params()
    #print "s",
    dw_o = update_function_o(self.info)
    #print "!!",
    dw_h = update_function_h(self.info)
    self.info.set_d_weights(dw_o, d_Wh=dw_h)
    #print "e"
    #pudb.set_trace()

def supervised_update(self, method_o='tempotron', method_h=None):
    supervised_update_setup(self, method_o=method_o, method_h=method_h)
    #self.net.restore()
    #if self.hidden == True:
    #    dw_h = self.info.d_Wh
    #    m = np.abs(dw_h).max()
    #    self.net['synapses_hidden'].w += self.r*dw_h
    #dw_o = self.info.d_Wo
    #n = np.abs(dw_o).max()
    #self.r = 1.0
    #self.net['synapses_output'].w += self.r*dw_o
    #self.net.store()
    #p = 0
    #for i in range(len(self.info.d)):
    #    if self.info.d[i] == 0:
    #        p += 1.0
    #return p / len(self.info.d)

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
    #np.clip(w_ih, -80, 10000, out=w_ih)
    #elif False: #np.min(w_ho) < 1:
    #np.clip(w_ho, -80, 10000, out=w_ho)
    #else:
    #if iteration == 100:
    #    pudb.set_trace()
    # KEEP PREVIOUS COMMENTS !!!
    a = self.net['crossings_o']
    b = self.net['crossings_h']

    actual = a.all_values()['t']
    hidden = b.all_values()['t']
    #desired = self.desired

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
        mod = synaptic_scaling(self, max_spikes)
        i += 1
        if i > 15:
            pudb.set_trace()

def train_step(self, method_o='tempotron', method_h=None):
    if method_o != 'tempotron' or method_h != 'tempotron':
        pass
        #synaptic_scalling_wrap(self, 5)
    supervised_update(self, method_o=method_o, method_h=method_h)

def train_epoch(self, index, X, Y, method_o='tempotron', method_h=None):
    correct = 0
    p = 0
    for i in range(len(X)):
        self.net.restore()
        self.set_inputs(X[i])
        #pudb.set_trace()
        self.info.set_y(Y[i])
        self.run()
        #self.info.H.print_spike_times(tabs=1)
        self.info.O.print_sd_times(tabs=1)
        self.info.reread()
        #if index == 6:
        #    pudb.set_trace()
        p_tmp = self.info.performance()
        #if p_tmp < 4.0:
        #    pudb.set_trace()
        train_step(self, method_o=method_o, method_h=method_h)
        #print "\t", p_tmp
        p += p_tmp
        self.info.update_weights(0.2)
        #print self.info.d_Wh[:]
        #pudb.set_trace()
        #self.info.update_weights(self.net)

    return p
