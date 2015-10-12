import lif
import numpy as np
import pudb

nn = lif.neuron()
nn.input_output()
i = 0
while True:
    print "i=", i, "",
    nn.train()
    nn.plot(i=i, save=True, show=False)
    nn.restore()
    print "\n",
    i += 1
