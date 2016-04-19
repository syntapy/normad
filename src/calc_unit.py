import numpy as np
import pudb

#th = [[7.8*0.001, 25.2*0.001], [ 9.6*0.001]]
#ta = 24.8*0.001

ti = [0.006, 0, 0]
th = [[0.0078, 0.0252], [ 0.0096]]

ta = 0.0248
td = 0.028

m, n, o, p = 3, 2, 1, 1
Ap, Am, a, tau = 1.2, 0.5, 0.05, 0.005

def Apre(s):
    return -Am*np.exp(s / tau)

def Apost(s):
    return Ap*np.exp(-s / tau)

w_ih = np.array([1192.38587144,  1213.54141314,  1215.89900524,  1211.53072387,  1213.30483321, 1208.16402457])
dw_ih = np.zeros(np.shape(w_ih))
d_ih = np.array([8.48817697,  1.78895925,  0.54363214,  3.61538446,  2.75400929,  5.30000225])*0.001
dd_ih = np.zeros(np.shape(d_ih))

w_ho = np.array([831.6878502, 831.46842741])
dw_ho = np.zeros(np.shape(w_ho))
d_ho = np.array([ 3.05918916, 3.04474359])*0.001
dd_ho = np.zeros(np.shape(d_ho))

print d_ho

w_ih_new = np.array([ 1218.59773157, 1253.21098736, 1257.26227549, 1250.27975471, 1254.67675206, 1247.49591808])
d_ih_new = np.array([ 8.48817697, 1.78895925, 0.54363214, 3.61538446, 2.75400929, 5.30000225])*0.001

w_ho_new = np.array([831.29887157,  831.3935979])
d_ho_new = np.array([3.05918916, 3.04474359])*0.001


"""
    IH Routine
"""

pudb.set_trace()
i, h = 0, 0
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * Apost(ta - tid)
dw_ih[i*2 + h] += abs(w_ho[h]) * Apost(td - tid)

i, h = 1, 0
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * Apost(ta - tid)
dw_ih[i*2 + h] += abs(w_ho[h]) * Apost(td - tid)

i, h = 2, 0
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * Apost(ta - tid)
dw_ih[i*2 + h] += abs(w_ho[h]) * Apost(td - tid)

i, h = 0, 1
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * Apost(ta - tid)
dw_ih[i*2 + h] += abs(w_ho[h]) * Apost(td - tid)

i, h = 1, 1
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * Apost(ta - tid)
dw_ih[i*2 + h] += abs(w_ho[h]) * Apost(td - tid)

i, h = 2, 1
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * Apost(ta - tid)
dw_ih[i*2 + h] += abs(w_ho[h]) * Apost(td - tid)


# Non-Hebbian terms
i, h = 0, 0
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * a
dw_ih[i*2 + h] += abs(w_ho[h]) * a

i, h = 1, 0
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * a
dw_ih[i*2 + h] += abs(w_ho[h]) * a

i, h = 2, 0
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * a
dw_ih[i*2 + h] += abs(w_ho[h]) * a

i, h = 0, 1
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * a
dw_ih[i*2 + h] += abs(w_ho[h]) * a


i, h = 1, 1
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * a
dw_ih[i*2 + h] += abs(w_ho[h]) * a

i, h = 2, 1
tid = ti[i] + d_ih[i*2 + h]
dw_ih[i*2 + h] -= abs(w_ho[h]) * a
dw_ih[i*2 + h] += abs(w_ho[h]) * a

dw_ih /= (6.0)


"""
    HO Routine
"""
# h = 0, o = 0
thd = th[0][0] + d_ho[0]
# ta - d - s = th
# s = ta - th - d = POSITIVE
dw_ho[0] -= Apost(ta - thd)
dw_ho[0] += Apost(td - thd)

# h = 1, 0 = 0
thd = th[1][0] + d_ho[1]
dw_ho[1] -= Apost(ta - thd)
dw_ho[1] += Apost(td - thd)
#pudb.set_trace()

# h = 0, o = 0
thd = th[0][1] + d_ho[0]
# th + d - s = ta
# s = th + d - ta
dw_ho[0] -= Apre(ta - thd)
dw_ho[0] += Apre(td - thd)

# Nonhebbiann terms
dw_ho[0] -= a
dw_ho[0] += a

# Factors
dw_ho /= 2.0
#pudb.set_trace()

print w_ho + dw_ho - w_ho_new
print w_ih + dw_ih - w_ih_new

#pudb.set_trace()
