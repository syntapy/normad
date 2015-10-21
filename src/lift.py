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

    def _get_random_spikes(self, s, a, b, n):
        """
        Generate array of n numbers between a and b 
        with minimum spacing of s
        """
        a, b = max(int(a/s), 1), int(b/s)
        ints = np.sort(np.random.random_integers(a, b, n))
        add = np.arange(start=0, stop=s*len(ints), step=s)
        return ints + add

    def _input_output(self, classes=1):
        """
        Sets the input and desired output spike times if no arguments are given
        """
        self.times, self.indices, self.desired = [], [], []
        self.classes = classes
        for i in range(classes):
            a, b = int(0.8*self.T / 10.0), int(1.2*self.T / 10.0)
            n = np.random.randint(a, b) # of spikes per input synapse
            input_cmpnt = self._get_random_spikes(5, 0, self.T - 10, (n+2)*self.N)
            indices_cmpnt = np.random.random_integers(0, self.N, len(input_cmpnt))
            min_time = 5 + (input_cmpnt[i].min())
            m = np.random.randint(0.4*n, 0.6*n) # of spikes per output synapse
            desired_cmpnt = self._get_random_spikes(20, min_time, self.T-1, m)

            self.indices.append(indices_cmpnt)
            self.times.append(input_cmpnt)
            self.desired.append(desired_cmpnt)

    def test(self, classes=2):
        self._input_output(classes)
        for i in range(self.classes):
            self.neuron.set_train_spikes(self.indices[i], self.times[i], self.desired[i])
            self.neuron.train_step()
