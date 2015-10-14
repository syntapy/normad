import lif
import numpy as np
import pudb

nn = lif.neuron()
nn.input_output()
nn.train()
nn.plot()
i = 0
while False:
    print "i=", i, "",
    nn.train()
    nn.plot(i=i, save=True, show=False)
    nn.restore()
    print "\n",
    i += 1
