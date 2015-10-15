import lif
import numpy as np
import pudb

nn = lif.neuron()
nn.input_output()
#nn.train()
#nn.plot()
i = 0
while i < 10:
    #print "i=", i, "",
    nn.train()
    nn.plot(i=i, save=True, show=False)
    nn.restore()
    #print "\n",
    i += 1

changes = nn.changes
for i in range(len(changes)):
    print changes[i]

print "\n\na_post:"

for i in range(len(nn.a_post)):
    print "\t", nn.a_post[i]

print "\n\nd_post: "
for i in range(len(nn.a_post)):
    print "\t", nn.d_post[i]
