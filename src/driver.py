import lif
import lift
import pudb
import brian2 as br
import train

#nn.read_image(0)
#nn.run(50)
#train.resume_update_output_weights(nn)
#train.resume_update_hidden_weights(nn)
#nn.run(40)
nn = lif.net(N_hidden=16, N_input=4)
nn.train(0, 1)
