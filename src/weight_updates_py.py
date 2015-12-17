import numpy as np
import pudb
import brian2 as br
br.prefs.codegen.target = 'weave'  # use the Python fallback

def sort(S, trim):
    if len(S) > 0:
        if trim == True:
            for i in range(len(S)):
                S[i] = np.sort(S[i])/br.second
        else:
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
    A = 1.0
    tau=self.net['synapses_hidden'].tau1
    return A*np.exp(s/tau)

def resume_update_hidden_weights(self):
    a = 0.1     # non-hebbian weight term
    m, n, o= self.N_inputs, self.N_hidden, self.N_output
    w_ih = self.net['synapses_hidden'].w[:]
    w_ho = self.net['synapses_output'].w[:]
    Si = self.times / br.second
    Sh = self.net['crossings_h'].all_values()['t']
    Sa, Sd = self.actual, self.desired

    ### m input neurons, n hidden neurons, o output neurons
    ### (i, j) denotes the synapse from input i to hidden j
    ### w[n*i+j] ---------------------> (i, j, k)
    ### w[o*j:m_n:n] -----------------> (:, j, k)
    ### w[n*i:n*(i+1)] ---------------> (i, :, k)
    Sa, Sh = sort(Sa, trim=False), sort(Sh, trim=True)
    dw = np.zeros(np.shape(w_ih))
    #pudb.set_trace()
    for j in range(n):
        for i in range(m):
            for g in range(o):
                dw_tmp = 0
                if Sd[g] <= Si[i]:
                    s = Sd[g] - Si[i] 
                    dw_tmp -= resume_kernel(self, s)
                else:
                    s = Si[i] - Sd[g] 
                    dw_tmp += a + resume_kernel(self, s)
                s_ia = smaller_indices(Si[i], Sa[g])
                for h in range(len(s_ia)):
                    s = Sa[g][s_ia[h]] - Si[i]
                    dw_tmp += resume_kernel(self, s)
                s_ai = larger_indices(Si[i], Sa[g])
                for h in range(len(s_ai)):
                    s = Si[i] - Sa[g][s_ai[h]]
                    dw_tmp -= a + resume_kernel(self, s)
                dw[n*i+j] += dw_tmp * w_ho[o*j+g]
    return dw / float(m*n)

def resume_update_output_weights(self):
    a = 0.1     # non-hebbian weight term
    #pudb.set_trace()
    Sh = self.net['crossings_h'].all_values()['t']
    Sa, Sd = self.actual, self.desired
    m, n, o= self.N_hidden, self.N_output, 1
    n_o, m_n_o = n*o, m*n*o
    w = self.net['synapses_output'].w[:]

    Sa, Sh = sort(Sa, trim=True), sort(Sh, trim=True)
    ### m hidden neurons, n output neurons, o synapses per neuron/input pair
    ### (i, j) denotes synapse from hidden i to output j
    ### w[n*i+o*j] ------------------> (i, j)
    ### w[o*j:m_n:n] ----------------> (:, j)
    ### w[n*i:n*(i+1)] --------------> (i, :)
    #pudb.set_trace()
    dw = np.zeros(np.shape(w))
    for j in range(n): # output neurons
        for i in range(m): # hidden neurons
            dw_tmp = 0
            s_dh = smaller_indices(Sd[j], Sh[i])
            s_hd = larger_indices(Sd[j], Sh[i])
            for g in range(len(s_dh)):
                s = Sh[i][s_dh[g]] - Sd[j]
                dw_tmp += a + resume_kernel(self, s)
            for g in range(len(s_hd)):
                s = Sd[j] - Sh[i][s_hd[g]]
                dw_tmp -= resume_kernel(self, s)
            for g in range(len(Sh[i])):
                s_ha = smaller_indices(Sh[i][g], Sa[j])
                for h in range(len(s_ha)):
                    s = Sa[j][s_ha[h]] - Sh[i][g]
                    dw_tmp += resume_kernel(self, s)
            for g in range(len(Sh[i])):
                s_ah = larger_indices(Sh[i][g], Sa[j])
                for h in range(len(s_ah)):
                    #pudb.set_trace()
                    s = Sh[i][g] -  Sa[j][s_ah[h]] 
                    dw_tmp -= a + resume_kernel(self, s)
            dw[n*i+j] = dw_tmp
    return dw / np.float(m)
