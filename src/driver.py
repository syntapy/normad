import lif
import numpy as np
import pudb

nn = lif.neuron()
#S1, S2 = np.asarray([0, 0.1, 0.173, 1]), np.asarray([0, 0.1, 0.123, 1])
S1, S2 = np.asarray([7.7]), np.asarray([0.123])
#print nn.LNonOverlap(S1, S2)
#print nn.LOverlapPrevious(S1, S2, 0)
pudb.set_trace()
print nn.SCorrelation(S1, S2)
#nn.input_output()
#nn.run()
#nn.plot(i=1, show=False)
"""
i = 0
while True:
    print "i=", i, "",
    nn.run()
    nn.plot(save=False, show=True)
    #print "\n",
    if nn.untrained() == False or True:
        nn.restore()
        break
    nn.restore()
    i += 1
"""
