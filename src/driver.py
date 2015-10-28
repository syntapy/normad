import lif
import pudb
import brian2 as br

network = lif.net()
pudb.set_trace()
network.read_image(0)
network.train_step()
