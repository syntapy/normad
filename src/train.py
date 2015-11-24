import numpy as np
import pudb

def smaller_indices(self, a, B):
    indices = []
    for i in range(len(B)):
        if B[i] <= a:
            indices.append(i)
    return np.asarray(indices)

def resume_kernel(self, s):
    A = 3.0
    tau=self.net['synapses_hidden'].tau1
    return A*np.exp(-s/tau)

def resume_update_output_weights(self):
    pudb.set_trace()
    w = self.net['synapses_output'].w[:]
    dw = np.zeros(np.shape(w))
    #Sh = self.net['crossings_h'].all_values()['t'][0]
    Sh = self.net['crossings_h'].all_values()['t']
    Sa, Sd = self.actual, self.desired
    nh = self.N_hidden
    for i in range(len(Sh)):
        s_d = smaller_indices(self, Sh[i], Sd)
        s_a = smaller_indices(self, Sh[i], Sa)
        for j in range(len(s_d)):
            pass

def resume_supervised_update_setup(self):
    dw = np.empty(2, dtype=object)
    dw[1] = resume_update_output_weights(self)
    dw[0] = resume_update_hidden_weights(self)

    return dw

def normad_supervised_update_setup(self):
    """ Normad training step """
    #self.actual = self.net['crossings_o'].all_values()['t']
    actual, desired = self.actual, self.desired
    dt = self.dta
    v = self.net['monitor_v'].v
    c = self.net['monitor_o_c'].c
    w = self.net['synapses_output'].w
    #t = self.net['monitor_o'].tp
    #f = self.net['monitor_f'].f
    #a = [max(f[i]) for i in range(len(f))]

    # m neurons, n inputs
    #pudb.set_trace()
    m, n = len(v), len(self.net['synapses_output'].w[:, 0])
    m_n = m*n
    dW, dw = np.zeros(m_n), np.zeros(n)
    for i in range(m):
        if len(actual[i]) > 0:
            index_a = int(actual[i] / dt)
            dw_tmp = c[i:m_n:m, index_a]
            dw_tmp_norm = np.linalg.norm(dw_tmp)
            if dw_tmp_norm > 0:
                dw[:] -= dw_tmp / dw_tmp_norm
        if desired[i] > 0:
            index_d = int(desired[i] / dt)
            dw_tmp = c[i:m_n:m, index_d]
            dw_tmp_norm = np.linalg.norm(dw_tmp)
            if dw_tmp_norm > 0:
                dw[:] += dw_tmp / dw_tmp_norm
        dwn = np.linalg.norm(dw)
        if dwn > 0:
            dW[i:m_n:m] = dw / dwn
        dw *= 0
    return dW

def supervised_update(self, display=False, method='resume'):
    if method == 'resume':
        dw = resume_supervised_update_setup(self)
        self.net.restore()
        self.net['synapses_output'].w += self.r*dw[1]
        self.net['synapses_hidden'].w += self.r*dw[0]
    else:
        dw = self.normad_supervised_update_setup()
        self.net.restore()
        self.net['synapses_output'].w += self.r*dw
    self.net.store()
    if display:
        #self.print_dw_vec(dw, self.r)
        self.print_dws(dw)

def train_step(self, T=None, method='resume'):
    self.run(T)
    #pudb.set_trace()
    self.actual = self.net['crossings_o'].all_values()['t']
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
