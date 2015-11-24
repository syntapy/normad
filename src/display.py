##########################
### PLOTTING / DISPLAY ###
##########################

def times_format(self):
    inputs = []
    for i in range(len(self.times)):
        inputs.append([self.indices[i], self.times[i]])

    return inputs

def print_dws(self, dw):
    #print "\tinput: ", self.times_format(), "\n"
    #print "\tdw: ", np.sum(dw),
    #print "\tw: ", np.sum(self.net['synapses'].w[:, :]),
    #print "\tactual: ", self.actual,
    #print "\tdesired: ", self.desired,
    print "\tlen_dif: ", len(self.desired) - len(self.actual),
    #if self.dw_d == None:
    #    print "dw_d == None: ",
    #else:
    #    print "dw_d == OBJECT",
    #if self.dw_a == None:
    #    print "dw_a == None: ",
    #else:
    #    print "dw_a == OBJECT",

def print_dw_vec(self, dw, r):
    #if abs(dw[0] + dw[1]) < 0.0001:
    #    self.save_weights()
    print "\tr:", r,

def plot_desired(self):
    desired = self.desired
    for i in range(len(desired)):
        x = desired[i]
        br.plot((x, x), (0, 100), 'r--')

def plot_actual(self):
    actual = self.actual
    for i in range(len(actual)):
        x = actual[i]
        br.plot((x, x), (0, 100), 'r-')

def plot(self, save=False, show=True, i=None):
    self.fig = br.figure(figsize=(8, 5))
    #self.plot_desired()
    self.plot_actual()
    br.plot((0, self.T)*br.ms, (90, 90), 'b--')
    for j in range(self.N_output):
        br.plot(self.net['monitor_v'][j].t, self.net['monitor_v'][j].v+70, label='v ' + str(j))
        #br.plot(self.net['monitor_c'][j].t, self.net['monitor_c'][j].c, label='c ' + str(j))
    #br.plot(self.net['monitor1'][1].t, self.net['monitor1'][1].c, 'g-')
        br.legend()
    if i != None and save == True:
        file_name = '../figs/'
        for j in range(4):
            if i < 10**(j+1):
                file_name += '0'
        file_name += str(i) + '.png'
        self.fig.savefig(file_name)
    if show==True:
        br.show()
