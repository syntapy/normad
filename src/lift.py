import lif
import pudb
import numpy as np

class lif_tester:

    def __init__(self, neuron=None, Na=None, Nb=None):
        if neuron == None:
            self.neuron = lif.neuron()
        else:
            self.neuron = neuron

        self.N = self.neuron.N
        self.T = self.neuron.T
        if Na == None:
            self.na, self.nb = 2*self.N, 6*self.N
        else:
            self.na, self.nb = Na[0], Na[1]

        if Nb == None:
            self.ma, self.mb = self.na/2, self.nb/2
        else:
            self.ma, self.mb = Nb[0], Nb[1]

    def test_SC(self):
        S1, S2 = [2.5, 15.5], [2.5, 15.5]
        #a = self.neuron.SCorrelationSlow(S1, S2)
        b = self.neuron.SCorrelation(S1, S2)

        #print a
        print b

    def _spaced_next_number(self, spacing, a):
        return a + int(np.ma.round(abs(np.random.poisson(lam=10))))

    def _get_random_spikes(self, spacing, a, b, n):
        """
        Generate array of n numbers between a and b 
        with minimum spacing of spacing
        """
        return_val = np.empty(n)
        while True:
            return_val[0] = self._spaced_next_number(spacing, a)
            for i in range(1, n):
                return_val[i] = self._spaced_next_number(spacing, return_val[i-1])
            if return_val[-1] < b:
                break
        return return_val
        #a, b = max(int(np.ceil(a/spacing)), 1), int(np.floor(b/spacing))
        #ints = np.sort(np.random.random_integers(a, b, n))*spacing
        #add = np.arange(start=0, stop=spacing*(len(ints) + 0), step=spacing)
        #return ints + add

    def _input_output(self, classes=1):
        """
        Sets the input and desired output spike times if no arguments are given
        """
        pudb.set_trace()
        self.times, self.indices, self.desired = [], [], []
        self.classes = classes
        for i in range(classes):
            a, b = int(0.8*self.T / 10.0), int(1.2*self.T / 10.0)
            n = np.random.randint(a, b) # of spikes per input synapse
            input_cmpnt = self._get_random_spikes(5, 0, self.T - 10, (n+2)*self.N)
            indices_cmpnt = np.random.random_integers(0, self.N, len(input_cmpnt))
            min_time = int(np.ceil((5 + (input_cmpnt[i].min())) / 20.0) * 20)
            m = np.random.randint(0.4*n, 0.6*n) # of spikes per output synapse
            desired_cmpnt = self._get_random_spikes(20, min_time, self.T-1, m)

            self.indices.append(indices_cmpnt)
            self.times.append(input_cmpnt)
            self.desired.append(desired_cmpnt)
            print "i = ", i

    def test_spike_consistency(self):
        m, n = len(self.times), len(self.desired)
        if m != n:
            return False
        for i in range(m):
            if self.times[0] >= self.desired[0] - 5:
                return False
            if self.desired[-1] >= self.neuron.T:
                return False
            if self.times[-1] > self.neuron.T:
                return False

    def setup(self, classes=1):
        self._input_output(classes)

    def test(self, classes=1):
        self._input_output(classes)
        while self.neuron.trained == False:
            self.neuron.trained = True
            for i in range(self.classes):
                self.neuron.set_train_spikes(self.indices[i], self.times[i], self.desired[i])
                rms = self.neuron.train_step()
                self.neuron.test_if_trained()
                print "\trms: ", rms
