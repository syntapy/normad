import lif
import lift
import pudb
import brian2 as br
import train

nn = lif.net(N_hidden=6, N_input=4)
a = [1, 21, 34, 37, 51, 56, 63, 68, 69, 75]
nn.train([0])
