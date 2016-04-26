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
    m, n, o, p = info.a, info.b, info.c, info.p
    if n == None:
        n = m
    n_p, o_p = n*p, o*p

    params = info.params
    #ii, ti = info.get_inputs()
    ia, ta = info.O.S.it_
    #pudb.set_trace()
    if info.H != None:
        ih, th = info.H.S.it_
    else:
        ih, th = info.get_inputs()
    Ap, Am, a_nh, tau = params.get_params()
    d = info.d_times

    dw_ih, dw_ho = info.d_weights()
    w_ih, w_ho = info.weights()
    delay_ih, delay_ho = info.delays()
    ### n hidden neurons, o output neurons, p synapses per neuron/input pair
    ### w_ho[o_p*i + p*j + k] acceses the kth synapses from hidden i to output j

    # loop over hidden spikes
    for H in range(len(ih)):
        h = ih[H]
        # loop over output spikes
        for J in range(len(ia)):
            j = ia[J]
            index_ho = o_p*h + p*j
            delay = delay_ho[index_ho:index_ho+p]*0.001
            S = ta[J] - th[H] - delay
            for l in range(len(S)):
                s = S[l]
                if s <= 0:
                    # th + d - s = ta
                    # s = -ta + d + th
                    dw_ho[index_ho+l] += Am*resume_kernel(s, tau)
                if s >= 0:
                    # ta - d - s = th
                    # s = ta - d - th
                    dw_ho[index_ho+l] -= Ap*resume_kernel(-s, tau)
        # loop over desired spikes
        for j in range(len(d)):
            index_ho = o_p*h+p*j
            delay = delay_ho[index_ho:index_ho+p]*0.001
            S = d[j] - th[H] - delay
            for l in range(len(S)):
                s = S[l]
                if s <= 0:
                    # th + d - s = td
                    # s = -td + d + th
                    dw_ho[index_ho+l] -= Am*resume_kernel(s, tau)
                if s >= 0:
                    # td - d - s = th
                    # s = td - d - th
                    dw_ho[index_ho+l] += Ap*resume_kernel(-s, tau)
        # loop over output neurons
        #for j in range(o):
        #    index_ho = o_p*h + p*j
        #    dw_ho[index_ho:index_ho+p] += a_nh
    # Non-hebbian term
    # loop over hidden neurons
    for h in range(n):
        # loop over output spikes
        for J in range(len(ia)):
            j = ia[J]
            index_ho = o_p*h + p*j
            dw_ho[index_ho:index_ho+p] -= a_nh
        # loop over desired spikes
        for j in range(len(d)):
            index_ho = o_p*h + p*j
            dw_ho[index_ho:index_ho+p] += a_nh

    return dw_ho / float(n_p)

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
    dw_ih, dw_ho = info.d_weights()
    w_ih, w_ho = info.weights()
    delay_ih, delay_ho = info.delays()
    d = info.d_times
    #v = info.O.v

    ### m input neurons, n hidden neurons, o output neurons, p subconnections
    ### w_ih[n_p*i + p*j + k] acceses the kth synapse from input i to hidden j
    ### w_ho[o_p*i + p*j + k] acceses the kth synapse from hidden i to output j

    #pudb.set_trace()
    # loop over hidden neurons
    for h in range(n):
        # loop over input spikes
        for I in range(len(ii)):
            i = ii[I]
            index_ih = n_p*i+p*h
            delay = delay_ih[index_ih:index_ih+p]*0.001
            # loop over output spikes
            for J in range(len(ta)):
                j = ia[J]
                index_ho = o_p*h+p*j
                S = ta[J] - ti[I] - delay
                for l in range(len(S)):
                    s = S[l]
                    if s <= 0:
                        # ti + d - s = ta
                        # s = -ta + d + ti
                        dw_ih[index_ih+l] += Am*resume_kernel(s, tau)*np.abs(w_ho[index_ho+l])
                    if s >= 0 :
                        # ta - d - s = ti
                        # s = ta - d - ti
                        dw_ih[index_ih+l] -= Ap*resume_kernel(-s, tau)*np.abs(w_ho[index_ho+l])
            # loop over desired spikes
            for j in range(len(d)):
                index_ho = o_p*h+p*j
                delay = delay_ih[index_ih:index_ih+p]*0.001
                S = d[j] - ti[I] - delay
                for l in range(len(S)):
                    s = S[l]
                    if s <= 0:
                        # ti + d - s = td
                        # s = -td + d + ti
                        dw_ih[index_ih+l] -= Am*resume_kernel(s, tau)*np.abs(w_ho[index_ho+l])
                    if s >= 0:
                        # td - d - s = ti
                        # s = td - d - ti
                        dw_ih[index_ih+l] += Ap*resume_kernel(-s, tau)*np.abs(w_ho[index_ho+l])
            # loop over output neurons
            #for j in range(o):
            #    index_ho = o_p*h+p*j
            #    dw_ih[index_ih:index_ih+p] += a_nh*np.abs(w_ho[index_ho:index_ho+p])

    # loop over hidden neurons
    for h in range(n):
        # Non-hebbian term
        # loop over input neurons
        for i in range(m):
            index_ih = n_p*i+p*h
            # loop over output spikes
            for J in range(len(ta)):
                index_ho = o_p*h+p*j
                dw_ih[index_ih:index_ih+p] -= a_nh*np.abs(w_ho[index_ho:index_ho+p])
            # loop over desired spikes
            for j in range(len(d)):
                index_ho = o_p*h+p*j
                dw_ih[index_ih:index_ih+p] += a_nh*np.abs(w_ho[index_ho:index_ho+p])

    dw_ih /= float(m_n*p*p)
    #pudb.set_trace()
    return dw_ih

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
    m, n, o, p = info.a, info.b, info.c, info.p
    if n == None:
        n = m
    n_p, o_p = n*p, o*p

    ia, ta = info.O.S.it_
    cout = np.zeros(o)
    spike_out = np.bincount(ia)
    cout[:len(spike_out)] += spike_out
    if info.H != None:
        ih, th = info.H.S.it_
    else:
        ih, th = info.get_inputs()
    d = info.y - np.clip(cout, 0, 1)

    dw_ih, dw_ho = info.d_weights()
    w_ih, w_ho = info.weights()
    delay_ih, delay_ho = info.delays()

    #pudb.set_trace()
    Wo, d_Wo = info.Wo, info.d_Wo
    S, v = info.O.S.all_values()['t'], info.O.v

    pudb.set_trace()
    lam = 1.0
    for j in range(o):
        j_max = np.argmax(v[j])
        if d[j] != 0:
            for i in range(n):
                for k in range(p):
                    d_Wo[i*o_p + j*p + k] += d[j]*lam*(v[j][j_max]  + 70)/ 90.0

    return d_Wo
