import numpy as np

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

def resume_kernel(s, tau):
    A = 3.0
    return A*np.exp(-s/tau)

def resume_update_hidden_weights(dw_ih, w_ho, m, n, o, Si, Sh, Sa, Sd, tau):
    a = 1.0     # non-hebbian weight term
    n_o, m_n_o = n*o, m*n*o

    ### m input neurons, n hidden neurons, o output neurons
    ### (i, j) denotes the synapse from input i to hidden j
    ### w[n*i+j] ---------------------> (i, j, k)
    ### w[o*j:m_n:n] -----------------> (:, j, k)
    ### w[n*i:n*(i+1)] ---------------> (i, :, k)
    for j in range(n):
        for i in range(m):
            dw_tmp = 0
            for g in range(o):
                if Sd[g] <= Si[i]:
                    s = Si[i] - Sd[g]
                    dw_tmp += resume_kernel(s, tau)
                s_ia = smaller_indices(Si[i], Sa[g])
                for h in range(len(s_ia)):
                    s = Si[i] - Sa[g][s_ia[h]]
                    dw_tmp -= resume_kernel(s, tau)
                dw_tmp *= w_ho[n*i+j]
                if Si[i] < Sd[g]:
                    s = Sd[g] - Si[i]
                    dw_tmp += a + resume_kernel(s, tau)
                h = 0
                while h < len(Sa[g]) and Si[i] <= Sa[g][h] :
                    s = Sa[g][h] - Si[i]
                    dw_tmp -= a + resume_kernel(s, tau)
                    h += 1
                dw_tmp *= w_ho[n*i+j] / float(m*n)
            dw_ih[n*i+j] += dw_tmp
    return dw_ih

def resume_update_output_weights(dw_ho, m, n, o, Sh, Sa, Sd, tau):
    a = 1.0     # non-hebbian weight term
    #pudb.set_trace()
    n_o, m_n_o = n*o, m*n*o

    ### m hidden neurons, n output neurons, o synapses per neuron/input pair
    ### (i, j) denotes synapse from hidden i to output j
    ### w[n*i+o*j] ------------------> (i, j)
    ### w[o*j:m_n:n] ----------------> (:, j)
    ### w[n*i:n*(i+1)] --------------> (i, :)
    for j in range(n): # output neurons
        for i in range(m): # hidden neurons
            dw_tmp = 0
            s_dh = smaller_indices(Sd[j], Sh[i])
            s_hd = larger_indices(Sd[j], Sh[i])
            for g in range(len(s_dh)):
                s = Sd[j] - Sh[i][s_dh[g]]
                dw_tmp += a - resume_kernel(s, tau)
            for g in range(len(s_hd)):
                s = Sh[i][s_hd[g]] - Sd[j]
                dw_tmp += a + resume_kernel(s, tau)
            for g in range(len(Sh[i])):
                s_ha = smaller_indices(Sh[i][g], Sa[j])
                for h in range(len(s_ha)):
                    s = Sh[i][g] - Sa[j][s_ha[h]]
                    dw_tmp -= a - resume_kernel(s, tau)
            for g in range(len(Sh[i])):
                s_ah = smaller_indices(Sh[i][g], Sa[j])
                for h in range(len(s_ah)):
                    #pudb.set_trace()
                    s = Sh[i][g] -  Sa[j][s_ah[h]] 
                    dw_tmp -= a + resume_kernel(s, tau)
            dw_ho[n*i+j] = dw_tmp
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
