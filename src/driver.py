import numpy as np
import lift
import pudb
import train
import lif
import brian2 as br

pmins = np.zeros(60)
nn = lif.net(N_hidden=6, N_input=4, seed=(37485)%20)
for i in range(60):
    pmins[i] = nn.train(i, [0])
    nn.rand_weights()

print "Mean\t Stdev\t min\t max\t"
print np.mean(pmins), "\t", np.std(pmins), "\t", np.min(pmins), "\t", np.max(pmins)
a = [1, 21, 34, 37, 51, 56, 63, 68, 69, 75]
