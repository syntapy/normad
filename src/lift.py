import lif
import pudb
import numpy as np

class lif_tester:

    def __init__(self, neuron=None, seed=5, Na=None, Nb=None):
        if neuron == None:
            self.neuron = lif.neuron(seed=seed)
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

    def _poisson(self, r, spacing, Ti, Tf):
        """
        Generates a sequnce of spikes with mean arrival rate r,
        min seperation of spacing, on the range of [Ti, Tf).

        Returns numpy array and size of spike train.
        """

        spikes = [0]
        while True:
            spikes[0] = np.random.poisson(lam=r)
            if spikes[0] >= Ti:
                break
        a = np.random.poisson(lam=r)
        while spikes[-1] + a < Tf and a >= spacing:
            spikes.append(spikes[-1] + a)
            a = np.random.poisson(lam=r)

        return spikes

    def rflatten(self, A):
        if A.dtype == 'O':
            dim = np.shape(A)
            n = len(dim)
            ad = np.zeros(n)
            i = 0
            tmp = []
            for a in A:
                tmp.append(self.rflatten(a))
            return_val = np.concatenate(tmp)
        else:
            return_val = A.flatten()

        return return_val

    def _get_random_spikes(self, N, r, Ti, Tf, spacing, g_indices=True):
        """
        Generate array of spikes from poisson distribution with mean rate of r 
        and subject to minimum spacing of spacing. Spike will be no earlier than
        T0 and no later than T1

        Produces spikes for each of the N inputs, and records the earliest spike
        for each value of N in min_times array

        If g_indices == True, routine also returns, in addition to array of 
        times, both an array of indices along with min_times
        """
        spikes_list, indices_list = [], []
        min_times = np.empty(N)
        for i in range(N):
            if type(Ti) == np.ndarray or type(Ti) == list:
                spikes = self._poisson(r, spacing, Ti[i], Tf)
            else:
                spikes = self._poisson(r, spacing, Ti, Tf)
            min_times[i] = spikes[0] + spacing
            indices = [i]*len(spikes)
            spikes_list.append(np.asarray(spikes))
            indices_list.append(np.asarray(indices))
        times = self.rflatten(np.asarray(spikes_list))
        indices = self.rflatten(np.asarray(indices_list))
        if g_indices == True:
            return times, min_times, indices
        return times

    def _input_output(self, classes=1):
        """
        Sets the input and desired output spike times if no arguments are given
        """
        #pudb.set_trace()
        N, ri, ro, spacing = self.N, 10, 20, 5
        self.times, self.indices, self.desired = [], [], []
        self.classes = classes
        self.times_list, self.indices_list, self.desired_list = [], [], []
        for i in range(classes):
            times, min_times, indices = self._get_random_spikes(N, ri, 0, self.T, spacing)
            desired = self._get_random_spikes(20, ro, min_times, self.T, spacing, g_indices=False)

            self.times_list.append(times)
            self.indices_list.append(indices)
            self.desired_list.append(desired)

    def test_spike_consistency(self, classes=1, spacing=5):
        self._input_output(classes)
        l, m, n = len(self.times_list), len(self.indices_list), len(self.desired_list)
        if l != m or m != n:
            return False
        for i in range(l):
            if self.times_list[i][0] > self.desired_list[i][0] - spacing:
                return False
            if self.desired_list[i][-1] >= self.neuron.T:
                return False
            if self.times_list[i][-1] > self.neuron.T:
                return False

        return True

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
