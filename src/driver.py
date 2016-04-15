import numpy as np
import data as dt
#import lift
import pudb
#import train
import lif
import brian2 as br

xor = dt.data('xor')
X, Y = xor.X['train'], xor.Y['train']
nn = lif.net(inputs=3, hidden=0, output=1, subc=2, delay=10, seed=(45495)%20)
#nn.plot_2d()
nn.test_topology()

#nn.topology()
#nn.fit(X, Y, method_o='resume', method_h='resume')
#nn.predict(X[0], 3, plot=True)
#pudb.set_trace()
#nn.fit(X, Y, method_o='resume', method_h='resume')

#mt = max([np.max(mnist.X['train'][i]) for i in range(len(mnist.X['train']))])
#print mt

#N = 60
#pmins = np.zeros(N)
#iters = np.zeros(N)
#numbers = [0, 1, 2]
#pudb.set_trace()
#pudb.set_trace()
###nn.load('mnist')
###label = nn.read_data(0)

#nn.run(None)
#indices = nn.indices(1, numbers)
#print indices
#print [nn.labels['train'][i] for i in indices]

#for i in range(N):
#iters[i], pmins[i] = 
#pudb.set_trace()
###nn.fit(range(4))
#nn.test(range(4))

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
