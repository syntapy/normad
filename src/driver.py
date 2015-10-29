import lif
import pudb
import brian2 as br

pudb.set_trace()
network = lif.net()
network.read_image(0)
network.train_step()
