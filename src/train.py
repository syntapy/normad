import numpy as np
import pudb
import brian2 as br
import weight_updates_numba as weight_updates
import weight_updates_py
from aux import spike_count

class resume_params:

    def __init__(self,Ap=1.2, Am=0.5, a_nh=0.05, tau=0.005):
        self.Ap = Ap
        self.Am = Am
        self.a_nh = a_nh
        self.tau = tau

    def get_params(self):
        return self.Ap, self.Am, self.a_nh, self.tau

br.prefs.codegen.target = 'weave'  # use the Python fallback
def supervised_update(self, method_o='tempotron', method_h=None):
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
    if self.info.multilayer == True:
        if method_h != None:
            dw_h = update_function_h(self.info)
    else: dw_h = None
    if method_o != None:
        dw_o = update_function_o(self.info)
    #self.info.update_d_weights(dw_o, d_Wh=dw_h)


def synaptic_scaling_step(w, m, n, p, spikes, min_spikes, max_spikes):
    f = 0.05
    ### m neuron layer to n neuron layer
    ### w[n*i + j] acceses the synapse from neuron i to neuron j

    mod = False
    for j in range(len(spikes)):
        if spikes[j] > max_spikes:
            for i in range(m):
                w[n*p*i+j*p:n*p*i+(j+1)*p] *= np.float(1 - f)**np.sign(w[n*p*i+j*p:n*p*i+(j+1)*p])
        if spikes[j] < min_spikes:
            for i in range(m):
                w[n*p*i+j*p:n*p*i+(j+1)*p] *= np.float(1 + f)**np.sign(w[n*p*i+j*p:n*p*i+(j+1)*p])

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

def synaptic_scaling_singlelayer(self, min_spikes, max_spikes, iteration=0):
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
    self.info.O.print_spike_times()
    #desired = self.desired

    tomod_a = [i for i in actual if len(actual[i]) < min_spikes or len(actual[i]) > max_spikes]
    if tomod_a != []:
        self.net.restore()
        synaptic_scaling_step(w_io, self.N_inputs, self.N_output, self.N_subc, tomod_a, actual, min_spikes, max_spikes)
        self.net.store()

        #w_io_diff = w_io - self.net['synapses'].w
        return True
    return False

def synaptic_scaling_multilayer(self, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h, iteration=0):
    w_ih = self.net['synapses_hidden'].w
    w_ho = self.net['synapses_output'].w

    a, b, c, p = self.info.a, self.info.b, self.info.c, self.info.p

    # KEEP FOLOWING COMMENTS !!!
    #if False: #np.min(w_ih) < 1:
    #np.clip(w_ih, -80, 10000, out=w_ih)
    #elif False: #np.min(w_ho) < 1:
    #np.clip(w_ho, -80, 10000, out=w_ho)
    #else:
    #if iteration == 100:
    #    pudb.set_trace()
    # KEEP PREVIOUS COMMENTS !!!
    o_mon = self.net['crossings_o']
    h_mon = self.net['crossings_h']

    actual = o_mon.all_values()['t']
    hidden = h_mon.all_values()['t']

    ia, _ = o_mon.it_
    ih, _ = h_mon.it_

    a_count, h_count = spike_count(ia, self.info.c), spike_count(ih, self.info.b)
    #desired = self.desired

    #tomod_a = [i for i in actual if len(actual[i]) < min_spikes_o or len(actual[i]) > max_spikes_o]
    #tomod_h = [i for i in hidden if len(hidden[i]) < min_spikes_h or len(hidden[i]) > max_spikes_h]

    tomod_a = np.any(a_count > max_spikes_o) or np.any(a_count < min_spikes_o)
    tomod_h = np.any(h_count > max_spikes_h) or np.any(h_count < min_spikes_h)
    #pudb.set_trace()
    self.net.restore()
    if tomod_a or tomod_h:
        #pudb.set_trace()
        synaptic_scaling_step(w_ih, a, b, p, h_count, min_spikes_h, max_spikes_h)
        synaptic_scaling_step(w_ho, b, c, p, a_count, min_spikes_o, max_spikes_o)
        #print "W_HO: ", w_ho
        #print "W_IH: ", w_ih
        self.net.store()

        #w_ih_diff = w_ih - self.net['synapses_hidden'].w
        #w_ho_diff = w_ho - self.net['synapses_output'].w
        #pudb.set_trace()

        return True
    return False

def synaptic_scaling(self, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h, iteration=0):
    if self.N_hidden > 0:
        return synaptic_scaling_multilayer(self, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h, iteration=iteration)
    else:
        return synaptic_scaling_singlelayer(self, min_spikes, max_spikes, iteration=iteration)

def synaptic_scalling_wrap(self, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h):
    i = 1
    mod = synaptic_scaling(self, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h)
    while mod:
        self.run()
        self.info.reread()
        self.info.H.print_spike_times(layer_name="hidden", tabs=2)
        #pudb.set_trace()
        self.info.O.print_sd_times(tabs=2)
        mod = synaptic_scaling(self, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h)
        i += 1
        #if i > 5:
        #    self.save_weights()

def train_step(self, index, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h, method_o='tempotron', method_h=None, scaling=True):
    if (method_o != 'tempotron' or method_h != 'tempotron') and scaling == True:
        pass
        #pudb.set_trace()
        #synaptic_scalling_wrap(self, 1, 1)
    supervised_update(self, method_o=method_o, method_h=method_h)

def train_epoch(self, r, index, indices, pmin, X, Y, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h, method_o='tempotron', method_h=None, scaling=True):
    correct = 0
    p = 0
    #indices = np.arange(len(X))
    #np.random.shuffle(indices)

    indices_unique = np.unique(indices)
    plist = np.zeros(len(indices_unique))
    #print 
    #print " $$$$ --- E P O C H --- $$$$"
    #print 
    for i in indices:
        #times, grid = self.topology(num=80)
        #print "w_ih:\t", self.net['synapses_hidden'].w[:]
        #print "d_ih:\t", self.net['synapses_hidden'].delay[:]
        #print
        #print "w_ho:\t", self.net['synapses_output'].w[:]
        #print "d_ho:\t", self.net['synapses_output'].delay[:]
        #print "- - - - - - - "*2
        #pudb.set_trace()
        self.net.restore()
        self.set_inputs(X[i])
        self.info.set_y(Y[i])
        #desired = self.info.d_times[0]*1000
        #_ , i_times = self.info.get_inputs()
        #self.plot_2d(p_old, grid, times, index, i, i_times*1000, desired)
        #self.set_inputs(X[i])
        #self.info.set_y(Y[i])
        self.run()
        #pudb.set_trace()
        #self.info.H.print_spike_times(layer_name="hidden", tabs=1)
        self.info.O.print_sd_times(tabs=1)
        #print "=============="*2
        self.info.reread()
        #if index == 6:
        #    pudb.set_trace()
        plist[i] = self.info.performance(continuous=False)
        #if p_tmp < 4.0:
        #    pudb.set_trace()
        train_step(self, index, min_spikes_o, max_spikes_o, min_spikes_h, max_spikes_h, method_o=method_o, method_h=method_h, scaling=scaling)
        self.info.update_weights(r)
        self.info.reset_d_weights()
        #print "\t", p_tmp
        
        #print self.info.d_Wh[:]
        #pudb.set_trace()
        #self.info.update_weights(self.net)

    return plist
