import lif
import pudb
import brian2 as br

network = lif.net()
#network.read_image(0)
while True:
    network.train(1, 2)
    network.train(0, 1)
