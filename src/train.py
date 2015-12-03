import numpy as np
import pudb
import brian2 as br
import weight_updates
br.prefs.codegen.target = 'weave'  # use the Python fallback

def resume_supervised_update_setup(self):
    dw = np.zeros(2, dtype=object)
    Sh = self.net_hidden['crossings_h'].all_values()['t']

    m, n, o= self.N_inputs, self.N_hidden, self.N_output
    Si = self.times
    Sa, Sd = self.actual, self.desired
    Sa, Sh = weight_updates.sort(Sa), weight_updates.sort(Sh)
    w_ho = self.net_out['synapses_output'].w[:]
    w_ih = self.net_hidden['synapses_hidden'].w[:]
    dw_ho = np.zeros(np.shape(w_ho))
    dw_ih = np.zeros(np.shape(w_ih))
    tau=self.net_hidden['synapses_hidden'].tau1
    dw[1] = weight_updates.resume_update_output_weights(dw_ho, n, o, 1, Sh, Sa, Sd, tau)
    dw[0] = weight_updates.resume_update_hidden_weights(dw_ih, w_ho, m, n, o, Si, Sh, Sa, Sd, tau)

    return dw

def supervised_update(self, display=False, method='resume'):
    if method == 'resume':
        dw = resume_supervised_update_setup(self)
        self.net_out.restore()
        self.net_hidden.restore()
        self.net_out['synapses_output'].w += self.r*dw[1]
        self.net_hidden['synapses_hidden'].w += self.r*dw[0]
    else:
        dw = normad_supervised_update_setup(self)
        self.net_out.restore()
        self.net_hidden.restore()
        self.net['synapses_output'].w += self.r*dw
    self.net_out.store()
    self.net_hidden.store()
    if display:
        #self.print_dw_vec(dw, self.r)
        self.print_dws(dw)

def train_step(self, T=None, method='resume'):
    self.run(T)
    #pudb.set_trace()
    self.actual = self.net_out['crossings_o'].all_values()['t']
    #a = self.net['crossings'].all_values()['t']
    #tdf = self.tdiff_rms()
    supervised_update(self, method=method)
    #return tdf

def train_epoch(self, a, b, method='resume', dsp=True):
    correct = 0
    for i in range(a, b):
        self.read_image(i)
        train_step(self, method=method)
        print "\tImage ", i, " trained"
        if self.neuron_right_outputs():
            correct += 1
    return correct

