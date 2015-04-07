import numpy as np
import theano
import theano.tensor as tt

class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, batch_size, W=None, b=None):
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
	self.a = theano.shared(np.zeros((batch_size,n_in), dtype=theano.config.floatX),borrow=True)
	self.z = theano.shared(np.zeros((batch_size,n_out), dtype=theano.config.floatX),borrow=True)
	self.delta = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX),borrow=True) 
    def run(self):
	self.z = tt.add(tt.dot(self.a, self.W), self.b)
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
                            n_out=n_hidden,batch_size = batch_size)
        #hidden layer
        self.hiddenLayer = []
        for i in range(num_hidden_layer):
            h = HiddenLayer(rng=rng, n_in=n_hidden, n_out=n_hidden,batch_size = batch_size)
            self.hiddenLayer.append(h)
        #output layer
        self.OutputLayer = HiddenLayer(
                            rng=rng,
                            n_in=n_hidden,
                            n_out=n_out,batch_size = batch_size)
        #set dnn parameters
        self.num_hidden_layer = num_hidden_layer
	self.output= theano.shared(np.zeros(batch_size, dtype=theano.config.floatX),borrow=True)
	self.y_pred = theano.shared(np.zeros(batch_size, dtype=theano.config.floatX),borrow=True)
    
    def forward(self, feature, index):
	self.InputLayer.a = tt.cast(feature[index], dtype=theano.config.floatX)
	self.InputLayer.run()
	self.hiddenLayer[0].a = tt.log(1+tt.exp(self.InputLayer.z))
	self.hiddenLayer[0].run()
	for i in range(1,self.num_hidden_layer):
		self.hiddenLayer[i].a = tt.log(1+tt.exp(self.hiddenLayer[i-1].z))
		self.hiddenLayer[i].run()
	self.OutputLayer.a = tt.log(1+tt.exp(self.hiddenLayer[self.num_hidden_layer-1].z))
	self.OutputLayer.run() 
	self.output = tt.log(1+tt.exp(self.OutputLayer.z))
	self.y_pred = tt.argmax(self.output,axis=1)



    def calculate_error(self, y_pred, y):
	return tt.eq(y_pred, y)
    
    def backpropagation(self, y, index):
	g = self.function_gradient(y,index,self.output)
	self.OutputLayer.delta = (1/(1+tt.exp(-self.OutputLayer.z)))*g
	print self.OutputLayer.delta.shape.eval()
	self.hiddenLayer[self.num_hidden_layer-1].delta = 1/(1+tt.exp(-self.hiddenLayer[self.num_hidden_layer-1].z))*tt.dot(self.OutputLayer.delta,self.OutputLayer.W.transpose())
        for i in range(1,self.num_hidden_layer):
		 self.hiddenLayer[self.num_hidden_layer-1-i].delta = 1/(1+tt.exp(-self.hiddenLayer[self.num_hidden_layer-1-i].z))*tt.dot(self.hiddenLayer[self.num_hidden_layer-i].delta,self.hiddenLayer[self.num_hidden_layer-i].W.transpose())
	self.InputLayer.delta = 1/(1+tt.exp(-self.InputLayer.z))*tt.dot(self.hiddenLayer[0].delta,self.hiddenLayer[0].W.transpose())
	print self.InputLayer.delta.shape.eval()
    def update(self, index):
        learning_rate = 0.01
        gW = []
        gb = []
	self.InputLayer.update(tt.dot(self.InputLayer.a.transpose(),self.InputLayer.delta)/float(len(index)),tt.mean(self.InputLayer.delta,axis=0),learning_rate)
	print (tt.dot(self.InputLayer.a.transpose(),self.InputLayer.delta)/float(len(index))).shape.eval()
	print tt.mean(self.InputLayer.delta,axis=0).eval()
	for i in range(self.num_hidden_layer):
		self.hiddenLayer[i].update(tt.dot(self.hiddenLayer[i].a.transpose(),self.hiddenLayer[i].delta)/float(len(index)),tt.mean(self.hiddenLayer[i].delta,axis=0),learning_rate)
	self.OutputLayer.update(tt.dot(self.OutputLayer.a.transpose(),self.OutputLayer.delta)/float(len(index)),tt.mean(self.OutputLayer.delta,axis=0),learning_rate)
    
    def function_gradient(self, y, index, a):
        p = np.zeros((len(index),1943),dtype=theano.config.floatX)
	s = tt.sum(a,axis=1)
	for i in range(len(index)):
		p[i][y[index[i]]] = 1.0
	a = a/s
	return (p-a)*-2
        

