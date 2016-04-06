import numpy as np
import pudb
import brian2 as br
import weight_updates_numba as weight_updates
import weight_updates_py

br.prefs.codegen.target = 'weave'  # use the Python fallback
def supervised_update_setup(self, hidden=True, method='tempotron'):
    #pudb.set_trace()

    #STDP Parameters
    #pudb.set_trace()
    tau = 0.005
    
    if method == 'resume':
        Ap, Am, a_nh = 1.2, 0.5, 0.05
        update_function_o = weight_updates.resume_update_output_weights
    elif method == 'tempotron':
        update_function_o = weight_updates.tempotron_update_output_weights

    if hidden == True:
        if method == 'resume':
            update_function_h = weight_updates.resume_update_hidden_weights
            dw_h = update_function_h(self.info)
        elif method == 'tempotron':
            update_function_h = weight_updates.tempotron_update_hidden_weights
            update_function_h(self.info)

    update_function_o(self.info)

def supervised_update(self, method='tempotron'):
    supervised_update_setup(self)
    self.net.restore()
    if self.hidden == True:
        dw_h = self.info.d_Wh
        m = np.abs(dw_h).max()
        self.net['synapses_hidden'].w += self.r*dw_h
    dw_o = self.info.d_Wo
    n = np.abs(dw_o).max()
    self.r = 1.0
    self.net['synapses_output'].w += self.r*dw_o
    self.net.store()
    p = 0
    for i in range(len(self.info.d)):
        if self.info.d[i] == 0:
            p += 1.0
    return p / len(self.info.d)

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
        mod = synaptic_scaling(self, max_spikes)
        i += 1

def train_step(self, method='tempotron', hidden=True):
    if method != 'tempotron':
        synaptic_scalling_wrap(self, 1)
    return supervised_update(self, method=method)

def train_epoch(self, X, Y, method='tempotron', hidden=True):
    correct = 0
    p = 0
    for i in range(len(X)):
        self.net.restore()
        self.set_inputs(X[i])
        self.info.set_y(Y[i])
        self.run()
        self.info.reread()
        p += train_step(self, method=method)

    return p
