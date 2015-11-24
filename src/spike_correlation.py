##################################################
#### SPIKE CORRELATION FOUTINES (IN PROGRESS) ####
##################################################

def LNonOverlap(self, Smax, Smin):
    """ Finds i in which Smax[i] > Smin[0] """
    nmax, nmin, i = len(Smax), len(Smin), 0
    if Smin[0] > Smax[0]:
        while Smin[0] > Smax[i]:
            i += 1
    return i

def LOverlapPrevious(self, Smax, Smin, index):
    i, SC_overlap = 0, 0
    while i < len(Smin) and Smin[i] <= Smax[index]:
        SC_overlap += ma.exp((Smin[i]-Smax[index])/self.tauLP)
        i += 1
    return SC_overlap

def L(self, t1, t2):
    """ Atomic low-pass filter function """
    if t2 > t1:
        return ma.exp((t1 - t2) / self.tauLP)
    return ma.exp((t2 - t1) / self.tauLP)

def F(self, S, t):
    """ Returns summ of low-passed spikes at time t """
    return_val = 0
    if len(S) > 0:
        i = 0
        while i < len(S) and S[i] <= t:
            return_val += self.L(S[i], t)
            i += 1
    return return_val

def SCorrelationSlow(self, S1, S2, dt=0.05):
    """
        Trapezoid integration rule to compute inner product:
        
            <L(S1), L(S2)> / (||L(S1)|| * ||L(S2)||)
    """
    na = self.F(S1, 0) + self.F(S1, self.T-dt)
    nb = self.F(S2, 0) + self.F(S2, self.T-dt)
    return_val = self.F(S1, 0) * self.F(S2, 0) + \
                self.F(S1, self.T-dt) * self.F(S2, self.T-dt)
    for t in range(1, self.T - 1):
        return_val += 2*self.F(S1, t) * self.F(S2, t) * dt
        na += self.F(S1, t) * dt
        nb += self.F(S2, t) * dt
    return return_val / (na * nb)

def _prel_SC(self, Smax, Smin):
    i, t_last = 0, 0

    while Smax[i] < Smin[0]:
        tot *= np.exp((t_last - Smax[i]) / self.tauLP)
        tot += 1
        t_last = Smax[i]
        i += 1
    tot *= np.exp((t_last - Smin[0]) / self.tauLP)
    if Smax[i] == Smin[0]:
        tot += 1
    return tot*self.tauLP, i

def _equal_len_SC(self, Smax, Smin):
    return_val = 0
    integral = 0
    t_last = 0
    if Smax[0] < Smin[0]:
        return_val, i = self._prel_SC(Smax, Smin)
        j = 0
        while i < len(Smax):
            while Smin[j] <= Smax[i]:
                integral += 1
                integral *= np.exp((Smin[j] - Smax[i]) / self.tauLP)
                j += 1
            i += 1

def SC_step(self, total, t1, t2, int1, int2, i, j, S1, S2):
    if S1[i] < S2[j]:
        pass
    elif S1[i] > S2[j]:
        pass
    elif S1[i] == S2[j]:
        pass

def _equal_len_SC(self, S1, S2):
    total, int1, int2 = 0, 0, 0
    i, j = 0, 0
    t1, t2 = -10, -10

    while i < len(S1) and j < len(S2):
        total, int1, int2, i, j = self.SC_step(total, t1, t2, int1, int2, i, j, S1, S2)

def SCorrelation(self, S1, S2):
    S1, S2 = np.sort(S1), np.sort(S2)
    total, integral1, integral2 = 0, 0, 0
    i, j = 0, 0

    if len(S1) == len(S2):
        if len(S1) > 0:
            if S1[-1] > S2[-1]:
                return self._equal_len_SC(S1, S2)
            return self._equal_len_SC(S2, S1)
        return 1
    return 0
                

def SCorrelation(self, S1, S2):
    """ 
        Analytical approach to compute the inner product:
        
            <L(S1), L(S2)> / (||L(S1)|| * ||L(S2)||)
    """

    pudb.set_trace()
    S1, S2 = np.sort(S1), np.sort(S2)
    total, integral1, integral2 = 0, 0, 0
    i, j, = 0, 0
    n1, n2 = len(S1), len(S2)
    i_index, j_index = 0, 0
    t1_last, t2_last = -10, -10
    if n1 > 0 and n2 > 0:
        while i < n1 or j < n2:
            if S1[i_index] < S2[j_index]:
                integral1 /= np.exp((S1[i_index] - t1_last) / self.tauLP)
                integral1 += 1
                total += integral1 * np.exp((S2[i_index] - S1[j_index]) / self.tauLP)
                i += 1
                i_index = min(i, n1-1)
            elif S1[i_index] > S2[j_index]:
                integral2 /= np.exp((S2[j_index] - t2_last) / self.tauLP)
                integral2 += 1
                total += integral2 / np.exp((S1[i_index] - S2[j_index]) / self.tauLP)
                j += 1
                j_index = min(j, n2-1)
            elif S1[i_index] == S2[j_index]:
                integral1 /= np.exp((S1[i_index] - t1_last) / self.tauLP)
                integral2 /= np.exp((S2[j_index] - t2_last) / self.tauLP)
                integral1, integral2 = integral1 + 1, integral2 + 1
                total += integral1*integral2
                i, j = i + 1, j + 1
                i_index, j_index = min(i, n1 - 1), min(j, n2 - 1)
        return total / float(n1 * n2)
    elif n1 > 0 or n2 > 0:
        return 0
    else:
        return 1

def matches(self, S1, S2):
    if len(S1) > 0 and len(S2) > 0:
        i_min, j_min, min_diff = -1, -1, float("inf")
        for i in range(len(S1)):
            for j in range(len(S2)):
                diff = abs(S1[i] - S2[j])
                if diff < min_diff:
                    i_min, j_min, min_diff = i, j, diff
        return i_min, j_min

def SCorrelationSIMPLE(self, S1, S2, dt):
    S1, S2 = np.sort(S1), np.sort(S2)
    i, j = 0, 0
    n1, n2 = len(S1), len(S2)
    total = 0
    if n1 != n2:
        total += np.exp(abs(n1-n2))
    matches = self.match(S1, S2)
    for i in range(len(matches)):
        total += abs(S1[matches[i][0]] - S2[matches[i][1]])
    return total

def untrained(self):
    d = self.desired
    a = self.net['crossings'].all_values()['t'][0]

    if len(d) != len(a):
        return True
    
    if len(d) == len(a) and len(d) > 0:
        for i in range(len(d)):
            if abs((d[i] - a[i]) / br.ms) > 1:
                return True
    return False
