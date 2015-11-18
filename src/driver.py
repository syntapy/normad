import stdp
import pudb

encoder = stdp.stdp_encoder(10)
#encoder.save_outputs(0, 10)
encoder.pretrain(18, 20)
