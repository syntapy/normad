import numpy as np
import lift
import pudb
import train
import lif
import brian2 as br

N = 60
pmins = np.zeros(N)
iters = np.zeros(N)
nn = lif.net(N_hidden=6, N_input=4, seed=(35495)%20)
for i in range(N):
    nn.rand_weights(test=True)
    iters[i], pmins[i] = nn.train(i, [0])

print "P:"
print "Mean\t Stdev\t min\t max\t"
print np.mean(pmins), "\t", np.std(pmins), "\t", np.min(pmins), "\t", np.max(pmins)
print
print
print "Iters:"
print np.mean(iters), "\t", np.std(iters), "\t", np.min(iters), "\t", np.max(iters)

a = [1, 21, 34, 37, 51, 56, 63, 68, 69, 75]
