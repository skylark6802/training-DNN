import numpy as np
import theano
import theano.tensor as tt


class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None):
        if W is None:
            W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(0.6 / (n_in + n_out)),
                        high=np.sqrt(0.6 / (n_in + n_out)),
                        size=(n_in, n_out)
                        ),
                    dtype=theano.config.floatX
                    )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.asarray(rng.uniform(low=-np.sqrt(0.6 / (n_in + n_out)),high=np.sqrt(0.6 / (n_in + n_out)),size=(n_out)), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
    def run(self, input):
        self.output = tt.add(tt.dot(input, self.W), self.b)
    def update(self, gW, gb, learning_rate):
	self.W = tt.add(self.W ,-gW*learning_rate)
	self.b = tt.add(self.b , -gb*learning_rate)
# number of hidden layer can not be zero
class deepNeuralNetwork(object):
    def __init__(self, num_hidden_layer, n_in, n_hidden, n_out, batch_size):
        #input layer
        rng = np.random.RandomState(1234)
        self.InputLayer = HiddenLayer(
                            rng=rng,
                            n_in=n_in,
                            n_out=n_hidden)
        #hidden layer
        self.hiddenLayer = []
        for i in range(num_hidden_layer):
            h = HiddenLayer(rng=rng, n_in=n_hidden, n_out=n_hidden)
            self.hiddenLayer.append(h)
        #output layer
        self.OutputLayer = HiddenLayer(
                            rng=rng,
                            n_in=n_hidden,
                            n_out=n_out)
        #set dnn parameters
        self.num_hidden_layer = num_hidden_layer

        self.y_pred = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX),borrow=True)
    
    def forward(self, feature, index):
        self.a=[]#record output of layer
        self.z=[]#record input activation function
        self.output=[]
        for i in index:
            output= []
            self.InputLayer.run(tt.cast(feature[i], dtype=theano.config.floatX))
            output.append(self.InputLayer.output)
	    self.hiddenLayer[0].run( tt.log(1+tt.exp(self.InputLayer.output)) )
            output.append(self.hiddenLayer[0].output)
            for j in range(1,self.num_hidden_layer):
		self.hiddenLayer[j].run(tt.log(1+tt.exp(self.hiddenLayer[j-1].output)))
                output.append(self.hiddenLayer[j].output)
	    self.OutputLayer.run(tt.log(1+tt.exp(self.hiddenLayer[self.num_hidden_layer-1].output)))
            output.append(self.OutputLayer.output)
            self.z.append(output)
            inputs=[tt.cast(feature[i], dtype=theano.config.floatX)]
            for out in output:
                inputs.append(tt.log(1+tt.exp(out)))
            self.a.append(inputs)
        for i in range(len(index)):
            self.output.append(self.a[i][self.num_hidden_layer+2])
        self.y_pred = tt.argmax(self.output,axis=1)

    def calculate_error(self, y_pred, y):
        return tt.eq(y_pred, y)
    
    def backpropagation(self, y, index):
        self.delta=[]#store in reverse order (delta[0] is actually delta_L)
        for i in range(len(index)):
            dlt=[]
	    g = self.function_gradient(y[index[i]],self.a[i][self.num_hidden_layer+2])
            dlt.append(1/(1+tt.exp(-self.z[i][self.num_hidden_layer+1] ))*g)
	    dl = 1/(1+tt.exp(-self.z[i][self.num_hidden_layer]))*tt.dot(dlt[0],self.OutputLayer.W.transpose())
	    dlt.append(dl)
	    for j in range(self.num_hidden_layer):
	        dl = 1/(1+tt.exp(-self.z[i][self.num_hidden_layer-j-1]))*tt.dot(dlt[j+1],self.hiddenLayer[self.num_hidden_layer-j-1].W.transpose())
		dlt.append(dl)
            self.delta.append(dlt)
            
    def momentum_update(self, index):
        learning_rate = 0.001
	mu = 0.5
        gW = []
        gb = []
        for i in range(len(index)):
            gradient=[]
            gradb = []
            len_a = len(self.a[i])-1 #output doesn't caculate gradient
            for j in range(len_a):
		#print "delta: "+str(np.size(self.delta[i][len_a-1-j]))
		#print "a: "+str(np.size(self.a[i][j]))
                gradient.append(tt.reshape(self.a[i][j], [self.a[i][j].shape[0],1])*self.delta[i][len_a-1-j])
                gradb.append(self.delta[i][len_a-1-j])
		#print self.a[i][j].reshape(len(self.a[i][j]),1)*self.delta[i][len_a-1-j]
	    if gW == []:
		gW = gradient
		gb = gradb
	    else:
                for j in range(len(gradient)):
        	    gW[j] = tt.add(gW[j],gradient[j])
        	gb = np.add(gb,gradb)
        for i in range(len(gW)):
	    gW[i] = gW[i] / float(len(index))
	for i in range(len(gb)):
	    gb[i] = gb[i] / float(len(index))
	# momentum
	if self.momentum_W == []:
		self.momentum_W = gW
		self.momentum_b = gb
		for i in range(len(gW)):
		    self.momentum_W[i] = -learning_rate*self.momentum_W[i]
		for i in range(len(gb)):
		    self.momentum_b[i] = -learning_rate*self.momentum_b[i]
	else:
		for i in range(len(gW)):
		    self.momentum_W[i] = self.momentum_W[i]*mu - learning_rate*gW[i]
		for i in range(len(gb)):
		    self.momentum_b[i] = self.momentum_b[i]*mu - learning_rate*gb[i]
	self.InputLayer.update(self.momentum_W[0],self.momentum_b[0],-1.0)
	for i in range(self.num_hidden_layer):
		self.hiddenLayer[i].update(self.momentum_W[i+1],self.momentum_b[i+1],-1.0)
	#for j in range(10):
	    #print self.OutputLayer.W.eval()[j][y[index[i]]], self.OutputLayer.W.eval()[j][y[index[i]]+1]
	self.OutputLayer.update(self.momentum_W[self.num_hidden_layer+1],self.momentum_b[self.num_hidden_layer+1],-1.0)
	#for j in range(10):
	#    print self.OuitputLayer.W.eval()[j][y[index[i]]], self.OutputLayer.W.eval()[j][y[index[i]]+1]
    '''
    def predict(self, feature, label, index):
        y = []
        acc = 0
        for i in range(len(index)):
            pre = np.argmax(self.a[i][self.num_hidden_layer+2].eval())
            y.append(pre)
            if pre == label[index[i]]:
                acc += 1
        acc /= float(len(index))
        return y, acc
    '''
    def function_gradient(self, y, a):
        p = np.zeros(1943,dtype=theano.config.floatX)
        p[y] = 1.0
	s = tt.sum(a)
	a = a/s
	#print 2*(p[y]-a[y]),y,s.eval(),a[y]
	return (p-a)*-2
        

