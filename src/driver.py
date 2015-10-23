import lift
import pudb

nt = lift.lif_tester(seed=9)
print nt.test_spike_consistency(80)
