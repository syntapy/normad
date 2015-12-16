import numpy as np
import pudb
import brian2 as br
br.prefs.codegen.target = 'weave'  # use the Python fallback

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

def resume_kernel(self, s):
    A = 3.0
    tau=self.net_hidden['synapses_hidden'].tau1
    return A*np.exp(-s/tau)

def resume_update_hidden_weights(self):
    a = 1.0     # non-hebbian weight term
    m, n, o= self.N_inputs, self.N_hidden, self.N_output
    n_o, m_n_o = n*o, m*n*o
    w_ih = self.net_hidden['synapses_hidden'].w[:]
    w_ho = self.net_out['synapses_output'].w[:]
    Si = self.times
    Sh = self.net_hidden['crossings_h'].all_values()['t']
    Sa, Sd = self.actual, self.desired

    ### m input neurons, n hidden neurons, o output neurons
    ### (i, j) denotes the synapse from input i to hidden j
    ### w[n*i+j] ---------------------> (i, j, k)
    ### w[o*j:m_n:n] -----------------> (:, j, k)
    ### w[n*i:n*(i+1)] ---------------> (i, :, k)
    Sa, Sh = sort(Sa), sort(Sh)
    dw = np.zeros(np.shape(w_ih))
    for j in range(n):
        for i in range(m):
            dw_tmp = 0
            for g in range(o):
                if Sd[g] <= Si[i]:
                    s = Si[i] - Sd[g]
                    dw_tmp += resume_kernel(self, s)
                s_ia = smaller_indices(Si[i], Sa[g])
                for h in range(len(s_ia)):
                    s = Si[i] - Sa[g][s_ia[h]]
                    dw_tmp -= resume_kernel(self, s)
                dw_tmp *= w_ho[n*i+j]
                if Si[i] < Sd[g]:
                    s = Sd[g] - Si[i]
                    dw_tmp += a + resume_kernel(self, s)
                h = 0
                while h < len(Sa[g]) and Si[i] <= Sa[g][h] :
                    s = Sa[g][h] - Si[i]
                    dw_tmp -= a + resume_kernel(self, s)
                    h += 1
                dw_tmp *= w_ho[n*i+j] / float(m*n)
            dw[n*i+j] = dw_tmp
    return dw

def resume_update_output_weights(self):
    a = 1.0     # non-hebbian weight term
    #pudb.set_trace()
    Sh = self.net_hidden['crossings_h'].all_values()['t']
    Sa, Sd = self.actual, self.desired
    m, n, o= self.N_hidden, self.N_output, 1
    n_o, m_n_o = n*o, m*n*o
    w = self.net_out['synapses_output'].w[:]

    Sa, Sh = sort(Sa), sort(Sh)
    ### m hidden neurons, n output neurons, o synapses per neuron/input pair
    ### (i, j) denotes synapse from hidden i to output j
    ### w[n*i+o*j] ------------------> (i, j)
    ### w[o*j:m_n:n] ----------------> (:, j)
    ### w[n*i:n*(i+1)] --------------> (i, :)
    dw = np.zeros(np.shape(w))
    for j in range(n): # output neurons
        for i in range(m): # hidden neurons
            dw_tmp = 0
            s_dh = smaller_indices(Sd[j], Sh[i])
            s_hd = larger_indices(Sd[j], Sh[i])
            for g in range(len(s_dh)):
                s = Sd[j] - Sh[i][s_dh[g]]
                dw_tmp += a - resume_kernel(self, s)
            for g in range(len(s_hd)):
                s = Sh[i][s_hd[g]] - Sd[j]
                dw_tmp += a + resume_kernel(self, s)
            for g in range(len(Sh[i])):
                s_ha = smaller_indices(Sh[i][g], Sa[j])
                for h in range(len(s_ha)):
                    s = Sh[i][g] - Sa[j][s_ha[h]]
                    dw_tmp -= a - resume_kernel(self, s)
            for g in range(len(Sh[i])):
                s_ah = smaller_indices(Sh[i][g], Sa[j])
                for h in range(len(s_ah)):
                    #pudb.set_trace()
                    s = Sh[i][g] -  Sa[j][s_ah[h]] 
                    dw_tmp -= a + resume_kernel(self, s)
            dw[n*i+j] = dw_tmp
    return dw
