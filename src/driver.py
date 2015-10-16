import lif
import numpy as np
import pudb

#pudb.set_trace()
nn = lif.neuron()
nn.input_output()
#nn.run()
#nn.plot()
i = 0
while True:
    print "i=", i, "",
    nn.train()
    nn.plot(save=True, show=False, i=i)
    nn.restore()
    print "\n",
    i += 1

"""
changes = nn.changes
for i in range(len(changes)):
    print changes[i]

print "\n\na_pre:" 
for i in range(len(nn.a_pre)):
    print "\t", nn.a_pre[i]

print "\n\nd_pre:"
for i in range(len(nn.d_pre)):
    print "\t", nn.d_pre[i]

print "\n\na_post:" 
for i in range(len(nn.a_post)):
    print "\t", nn.a_post[i]

print "\n\nd_post:"
for i in range(len(nn.a_post)):
    print "\t", nn.d_post[i]
"""
