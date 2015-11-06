import lif
import lift
import pudb
import brian2 as br

nn = lif.net(N_hidden=10, N_input=4)
nn.train(0, 2)
