import numpy as np
import pudb
import brian2 as br
import weight_updates_numba as weight_updates
import weight_updates_py

#br.prefs.codegen.target = 'weave'  # use the Python fallback
def resume_supervised_update_setup(self):
    #pudb.set_trace()
    a = self.net['input']
    ii, ti = self.indices, self.times
    ih, th = self.net['crossings_h'].it_
    ia, ta = self.net['crossings_o'].it_
    d = self.desired

    #print "\t", ta
    #print "\t", ia

    #pudb.set_trace()
    m, n, o= self.N_inputs, self.N_hidden, self.N_output
    #Si = self.times
    #Sa, Sd = self.actual, self.desired
    #Sa, Sh = weight_updates.sort(Sa), weight_updates.sort(Sh)
    w_ho = self.net['synapses_output'].w[:]
    w_ih = self.net['synapses_hidden'].w[:]
    dw_ho = np.zeros(np.shape(w_ho), dtype=np.float64)
    dw_ih = np.zeros(np.shape(w_ih), dtype=np.float64)
    tau=self.net['synapses_hidden'].tau1 / (1000*br.msecond)
    dw_o = weight_updates.resume_update_output_weights(\
                dw_ho, m, n, o, ih[:], th[:], ia[:], ta[:], d, tau)
    dw_h = weight_updates.resume_update_hidden_weights(\
                dw_ih, w_ho, m, n, o, ii, ti/br.second, ih[:], th[:], ia[:], ta[:], d, tau)
    #dw_o_py = weight_updates_py.resume_update_output_weights(self)
    #dw_h_py = weight_updates_py.resume_update_hidden_weights(self)

    #ocmp = np.allclose(dw_h, dw_h_py, rtol=4*(1e-2), atol=3e-1)
    #hcmp = np.allclose(dw_o, dw_o_py, rtol=4*(1e-2), atol=3e-1)
    #if not ocmp or not hcmp:
    #    pudb.set_trace()


    #print "\tDIFFS:"
    #print "\t\tout:\t", dw_o - dw_o_py
    #print "\t\thidden:\t", dw_h - dw_h_py
    #pudb.set_trace()

    return dw_o, dw_h

def supervised_update(self, display=False, method='resume'):
    #pudb.set_trace()
    #pudb.set_trace()
    dw_o, dw_h = resume_supervised_update_setup(self)
    #print '\t', np.mean(dw[1]), np.mean(dw[0])
    #w_o = self.net['synapses_output'].w
    #w_h = self.net['synapses_hidden'].w
    #w_o_OLD = w_o
    #w_h_OLD = w_h
    #print '\t', dw_o
    #print "- "*10
    #print '\t', dw_h
    #print "="*10
    #print np.mean(w_o), np.mean(w_h)
    self.net.restore()
    #pudb.set_trace()
    self.net['synapses_output'].w += self.r*dw_o[:]
    self.net['synapses_hidden'].w += self.r*dw_h[:]
    w_o = self.net['synapses_output'].w
    w_h = self.net['synapses_output'].w
    self.net.store()

    #print '\t', np.max(np.abs(w_o_OLD - w_o))
    #print '\t', np.max(np.abs(w_h_OLD - w_h))

def train_step(self, T=None, method='resume'):
    self.run(T)
    #pudb.set_trace()
    a = self.net['crossings_o']
    self.actual = a.all_values()['t']
    supervised_update(self, method=method)

def train_epoch(self, a, b, method='resume', dsp=True):
    correct = 0
    for i in range(a, b):
        self.read_image(i)
        train_step(self, method=method)
        print "\tImage ", i, " trained"
        if self.neuron_right_outputs():
            correct += 1
    p = self.performance()
    return correct, p
