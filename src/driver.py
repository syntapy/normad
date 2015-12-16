import lif
import lift
import pudb
import brian2 as br
import train

nn = lif.net(N_hidden=16, N_input=4)
nn.train(0, 1)
