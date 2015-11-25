import numpy as np
import pudb

def sort(S):
    if len(S) > 0:
        for i in range(len(S)):
            S[i] = np.sort(S[i])
    return S

def smaller_indices(a, B):
    indices = []
    for i in range(len(B)):
        if B[i] <= a:
            indices.append(i)
    return np.asarray(indices)

def larger_indices(a, B):
    indices = []
    for i in range(len(B)):
        if B[i] > a:
            indices.append(i)
    return np.asarray(indices)

def resume_kernel(s):
    A = 3.0
    tau=self.net['synapses_hidden'].tau1
    return A*np.exp(-s/tau)

def resume_update_hidden_weights(self):
    a = 1.0     # non-hebbian weight term
    m, n, o= self.N_input, self.N_hidden, self.N_output
    n_o, m_n_o = n*o, m*n*o
    w_ih = self.net['synapses_hidden'].w[:]
    w_ho = self.net['synapses_output'].w[:]
    Si = self.times
    Sh = self.net['crossings_h'].all_values()['t']
    Sa, Sd = self.actual, self.desired

    ### m input neurons, n hidden neurons, o output neurons
    ### (i, j) denotes the synapse from input i to hidden j
    ### w[n_o*i+o*j+k] --------------> (i, j, k)
    ### w[o*j+k:m_n_o:n_o] ----------> (:, j, k)
    ### w[n_o*i+k:n_o*(i+1)+k:o] ----> (i, :, k)
    Sa, Sh = sort(Sa), sort(Sh)
    dw = np.zeros(np.shape(w_ih))
    for j in range(n):
        for i in range(m):
            dw_tmp = 0
            for g = in range(o):
                if Sd[g] <= Si[i]:
                    s = Si[i] - Sd[g]
                    dw_tmp += resume_kernel(s)
                s_ia = smaller_indices(Si[i], Sa[g])
                for h in range(len(s_ia)):
                    s = Si[i] - Sa[s_ia[g]]
                    dw_tmp -= resume_kernel(s)
                dw_tmp *= w_ho[n_o*i+o*j]
                if Si[i] < Sd[g]:
                    s = Sd[g] - Si[i]
                    dw_tmp += a + resume_kernel(s)
                h = 0
                if len(Sa[g][h]) > 0:
                    while Si[i] <= Sa[g][h]:
                        s = Sa[g][h] - Si[i]
                        dw_tmp -= a + resume_kernel(s)
                        h += 1
                dw_tmp *= w_ho[n_o*i+o*j] / float(m*n)
            dw[n_o*i+o*j] = dw_tmp
    return dw

def resume_update_output_weights(self):
    a = 1.0     # non-hebbian weight term
    pudb.set_trace()
    Sh = self.net['crossings_h'].all_values()['t']
    Sa, Sd = self.actual, self.desired
    m, n, o= self.N_hidden, self.N_output, 1
    n_o, m_n_o = n*o, m*n*o

    Sa, Sh = sort(Sa), sort(Sh)
    ### m hidden neurons, n output neurons, o synapses per neuron/input pair
    ### (i, j, k) denotes kth synapse from hidden i to output j
    ### w[n_o*i+o*j+k] --------------> (i, j, k)
    ### w[o*j+k:m_n_o:n_o] ----------> (:, j, k)
    ### w[n_o*i+k:n_o*(i+1)+k:o] ----> (i, :, k)
    dw = np.zeros(np.shape(w))
    for j in range(n): # output neurons
        for i in range(m): # hidden neurons
            dw_tmp = 0
            s_dh = smaller_indices(Sd[j], Sh)
            s_hd = larger_indices(Sd[j], Sh)
            for g in range(len(s_dh)):
                s = Sd[j] - Sh[s_dh[g]]
                dw_tmp += a - resume_kernel(s)
            for g in range(len(s_hd)):
                s = Sh[s_hd[g]] - Sd[j]
                dw_tmp += a + resume_kernel(s)
            for g in range(len(Sh[i])):
                s_ha = smaller_indices(Sh[i][g], Sa[j])
                for h in range(len(s_ha)):
                    s = Sh[i][g] - Sa[j][s_ha[h]]
                    dw_tmp -= a - resume_kernel(s)
            for g in range(len(Sa[i])):
                s_ah = smaller_indices(Sh[i])
                for h in range(len(s_ah)):
                    s = Sa[i][g] - Sh[i][s_ah[h]]
                    dw_tmp -= a + resume_kernel(s)
            dw[n_o*i+o*j] = dw_tmp
    return dw

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
