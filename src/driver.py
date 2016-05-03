import numpy as np
import data as dt
#import lift
import pudb
#import train
from aux import spike_count
import lif
import brian2 as br


# Uncomment any of the following lines to chose which benchmark to solve

# needs 3 inputs and at least 2 outputs
linsep = dt.data('linsep')
X, Y = linsep.X['train'], linsep.Y['train']

# needs 5 inputs and at least 3 outputs
#iris = dt.data('iris')
#X, Y = iris.X['data'], iris.Y['data']

# needs 3 inputs and at least 2 outputs
xor = dt.data('xor')
X, Y = xor.X['train'], xor.Y['train']

nn = lif.net(inputs=3, hidden=40, output=3, subc=10, delay=10)
p = nn.fit(X, Y, method_o='tempotron', method_h='tempotron', goal_accuracy=1.0)

print "Test Performance:", p
