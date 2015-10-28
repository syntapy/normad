import lif
import pudb
import brian2 as br

network = lif.net()
#pudb.set_trace()
network.uniform_input()
network.test_weight_order()
