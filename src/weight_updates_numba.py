import pudb
import numpy as np
#import numba

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

#@numba.jit(nopython=True)
def resume_kernel(s, tau):
    return np.exp(s/tau)

#@numba.jit(nopython=True)
def resume_update_output_weights(info):
    #pudb.set_trace()
    m, n, o, p = info.a, info.b, info.c, info.p
    n_o, m_n_o = n*o, m*n*o
    n_p, o_p = n*p, o*p

    params = info.params
    #ii, ti = info.get_inputs()
    ia, ta = info.O.S.it_
    ih, th = info.H.S.it_
    Ap, Am, a_nh, tau = params.get_params()
    d = info.d_times

    dw_ih, dw_ho = info.d_weights()
    w_ih, w_ho = info.weights()
    delay_ih, delay_ho = info.delays()
    ### n hidden neurons, o output neurons, p synapses per neuron/input pair
    ### w_ho[o_p*i + p*j + k] acceses the kth synapses from hidden i to output j

    #pudb.set_trace()
    # loop over output spikes
    for J in range(len(ia)):
        j = ia[J]
        # loop over hidden spikes
        for I in range(len(ih)):
            i = ih[I]
            #for k in range(p):
            index_ho = o_p*i + p*j
            delay = delay_ho[index_ho:index_ho+p]
            S = ta[J] - th[I] - delay
            for l in range(len(S)):
                s = S[l]
                if s <= 0:
                    dw_ho[index_ho+l] += Am*resume_kernel(s, tau)
                if s >= 0:
                    dw_ho[index_ho+l] -= Ap*resume_kernel(-s, tau)
        for i in range(n):
            index_ho = o_p*i + p*j
            dw_ho[index_ho:index_ho+p] -= a_nh

    # loop over desired spikes
    for j in range(len(d)):
        # loop over hidden spikes
        for I in range(len(ih)):
            i = ih[I]
            for k in range(p):
                index_ho = o_p*i+p*j
                delay = delay_ho[index_ho:index_ho+p]
                #pudb.set_trace()
                S = d[j] - th[I] - delay
                for l in range(len(S)):
                    s = S[l]
                    if s <= 0:
                        dw_ho[index_ho+l] -= Am*resume_kernel(s, tau)
                    if s >= 0:
                        dw_ho[index_ho+l] += Ap*resume_kernel(-s, tau)
        for i in range(n):
            index_ho = o_p*i + p*j
            dw_ho[index_ho:index_ho+p] += a_nh

    dw_ho /= float(n_p)

#@numba.jit(nopython=True)
def resume_update_hidden_weights(info):
    m, n, o, p = info.a, info.b, info.c, info.p
    m_n, n_o, m_n_o = m*n, n*o, m*n*o
    n_p, o_p = n*p, o*p

    params = info.params
    ii, ti = info.get_inputs()
    Wo, d_Wo = info.Wo, info.d_Wo
    ia, ta = info.O.S.it_
    Ap, Am, a_nh, tau = params.get_params()
    dw_ih, dw_ho = info.weights()
    w_ih, w_ho = info.weights()
    delay_ih, delay_ho = info.delays()
    d = info.d_times
    #v = info.O.v

    ### m input neurons, n hidden neurons, o output neurons, p subconnections
    ### w_ih[n_p*i + p*j + k] acceses the kth synapse from input i to hidden j
    ### w_ho[o_p*i + p*j + k] acceses the kth synapse from hidden i to output j

    #ii, ta = info.get_inputs()

    # loop over hidden neurons
    for k in range(n):
        # loop over output neurons
        for J in range(len(ta)):
            j = ia[J]
            index_ho = o_p*k+p*j
            # loop over input neurons
            for I in range(len(ii)):
                i = ii[I]
                index_ih = n_p*i+p*k
                delay = delay_ih[index_ih:index_ih+p]
                #pudb.set_trace()
                S = ta[J] - ti[I] - delay
                for l in range(len(S)):
                    s = S[l]
                    if s <= 0:
                        dw_ih[index_ih+l] += Am*resume_kernel(s, tau)*np.abs(w_ho[index_ho+l])
                    if s >= 0 :
                        dw_ih[index_ih+l] -= Ap*resume_kernel(-s, tau)*np.abs(w_ho[index_ho+l])
            for i in range(m):
                index_ih = n_p*i+p*k
                dw_ih[index_ih:index_ih+p] -= a_nh*np.abs(w_ho[index_ho:index_ho+p])
        for j in range(len(d)):
            index_ho = o_p*k+p*j
            # loop over input neurons
            for I in range(len(ii)):
                i = ii[I]
                index_ih = n_p*i+p*k
                delay = delay_ih[index_ih:index_ih+p]
                #pudb.set_trace()
                S = d[j] - ti[I] - delay
                for l in range(len(S)):
                    s = S[l]
                    if s <= 0:
                        #pudb.set_trace()
                        dw_ih[index_ih+l] -= Am*resume_kernel(s, tau)*np.abs(w_ho[index_ho+l])
                    if s >= 0:
                        dw_ih[index_ih+l] += Ap*resume_kernel(-s, tau)*np.abs(w_ho[index_ho+l])
            for i in range(m):
                index_ih = n_p*i+p*k
                dw_ih[index_ih:index_ih+p] += a_nh*np.abs(w_ho[index_ho:index_ho+p])
    dw_ih /= float(m_n*p*p)

def normad_update_output_weights(self):
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
    #return dW

def tempotron_update_hidden_weights(info):
    pass

def tempotron_update_output_weights(info):
    #pudb.set_trace()
    ii, it = info.get_inputs()
    Wo, d_Wo = info.Wo, info.d_Wo
    S, v = info.O.S.all_values()['t'], info.O.v

    A, C, P = info.a, info.c, info.p
    d = info.d

    lam = 5.0
    for c in range(C):
        i_max = np.argmax(v[c])
        #pudb.set_trace()
        if d[c] != 0:
            for a in range(A):
                for p in range(P):
                    d_Wo[a*C*P + c*P + p] += d[c]*lam*v[c][i_max]
