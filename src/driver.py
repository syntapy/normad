import numpy as np
#import lift
import pudb
#import train
import lif
import brian2 as br

#N = 60
#pmins = np.zeros(N)
#iters = np.zeros(N)
#numbers = [0, 1, 2]
#pudb.set_trace()
nn = lif.net(N_hidden=5, seed=(35495)%20)
#indices = nn.indices(1, numbers)
#print indices
#print [nn.labels['train'][i] for i in indices]

#for i in range(N):
#iters[i], pmins[i] = 
nn.train(0, range(4))
#pudb.set_trace()
nn.test(range(4))

#nn.rand_weights(test=True)
#print "P:"
#print "Mean\t Stdev\t min\t max\t"
#print np.mean(pmins), "\t", np.std(pmins), "\t", np.min(pmins), "\t", np.max(pmins)
#print
#print
#print "Iters:"
#print np.mean(iters), "\t", np.std(iters), "\t", np.min(iters), "\t", np.max(iters)
#
#a = [1, 21, 34, 37, 51, 56, 63, 68, 69, 75]
