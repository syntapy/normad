import lif
import pudb
import brian2 as br

network = lif.net()
network.set_train_spikes(indices=[0, 1, 2, 3], times=[0, 0, 0, 0])
network.run(35)
network.plot()
