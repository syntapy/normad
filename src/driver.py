import lif
import pudb
import brian2 as br

network = lif.net()
#network.read_image(0)
network.train(0, 2)
