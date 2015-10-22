import lift
import pudb

nt = lift.lif_tester()
nt.setup(200)
nt.test_spike_consistency()
