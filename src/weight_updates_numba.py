import numpy as np
import numba

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

@numba.jit(nopython=True)
def resume_kernel(s, tau):
    A = 3.0
    return A*np.exp(-s/tau)

@numba.jit(nopython=True)
def resume_update_hidden_weights(dw_ih, w_ho, m, n, o, ii, ti, ih, th, ia, ta, d, tau):
    a = 1.0     # non-hebbian weight term
    n_o, m_n_o = n*o, m*n*o

    ### m input neurons, n hidden neurons, o output neurons
    ### (i, j) denotes the synapse from input i to hidden j
    ### w[n*i+j] ---------------------> (i, j, k)
    ### w[o*j:m_n:n] -----------------> (:, j, k)
    ### w[n*i:n*(i+1)] ---------------> (i, :, k)

    # loop over input neurons
    for I in range(len(ii)):
        i = ii[I]
        # loop over hidden neurons
        for k in range(n):
            # loop over output neurons
            for J in range(len(ta)):
                j = ia[J]
                s = ta[j] - ti[i]
                if s < 0:
                    dw_ih[n*i+k] += resume_kernel(-s, tau)*w_ho[o*k+j]
                else:
                    dw_ih[n*i+k] -= (a + resume_kernel(s, tau))*w_ho[o*k+j]
            for j in range(len(d)):
                if d[j] != 0:
                    s = d[j] - ti[i]
                    if s < 0:
                        dw_ih[n*i+k] -= resume_kernel(-s, tau)*w_ho[o*k+j]
                    else:
                        dw_ih[n*i+k] += a + resume_kernel(s, tau)*w_ho[o*k+j]
    dw_ih /= float(m*n)
    return dw_ih

@numba.jit(nopython=True)
def resume_update_output_weights(dw_ho, m, n, o, ih, th, ia, ta, d, tau):
    a = 1.0     # non-hebbian weight term
    #pudb.set_trace()
    n_o, m_n_o = n*o, m*n*o

    ### m hidden neurons, n output neurons, o synapses per neuron/input pair
    ### (i, j) denotes synapse from hidden i ta output j
    ### w[n*i+o*j] ------------------> (i, j)
    ### w[o*j:m_n:n] ----------------> (:, j)
    ### w[n*i:n*(i+1)] --------------> (i, :)

    # loop over hidden spikes
    for I in range(len(ih)):
        i = ih[I]
        # loop over output spikes
        for J in range(len(ia)):
            j = ia[J]
            s = ta[J] - th[I]
            if s < 0:
                dw_ho[o*i+j] += resume_kernel(-s, tau)
            else:
                dw_ho[o*i+j] -= a + resume_kernel(s, tau)
        # loop over desired spikes
        for j in range(len(d)):
            if d[j] != 0:
                s = d[j] - th[I]
                if s < 0:
                    dw_ho[o*i+j] -= resume_kernel(s, tau)
                else:
                    dw_ho[o*i+j] += a + resume_kernel(-s, tau)
    dw_ho /= float(n)
    return dw_ho

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
