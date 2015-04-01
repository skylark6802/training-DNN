import numpy as np
import theano
import theano.tensor as tt

x = tt.dscalar('x')
y = tt.log(1 + tt.exp(x))
#y = 1/(1+tt.exp(-x))
act = theano.function( [x], y )
g = tt.grad(y, x)
gact = theano.function( [x], g )

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
            b_values = np.asarray(rng.uniform(low=-np.sqrt(0.6 / (n_in + n_out)),high=np.sqrt(0.6 / (n_in + n_out)),size=(1,n_out)), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
    def run(self, input):
        self.output = tt.dot(input, self.W) + self.b 
    def update(self, gW, gb, learning_rate):
	self.W = tt.add(self.W ,-gW*learning_rate)
	self.b = tt.add(self.b , -gb*learning_rate)
# number of hidden layer can not be zero
class deepNeuralNetwork(object):
    def __init__(self, num_hidden_layer, n_in, n_hidden, n_out):
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
    
    def forward(self, feature, index):
        self.a=[]#record output of layer
        self.z=[]#record input activation function
        for i in index:
            output= []
            self.InputLayer.run(feature[i])
            output.append(self.InputLayer.output.eval()[0])
	    self.hiddenLayer[0].run( [act(j) for j in (self.InputLayer.output.eval()[0])])
            output.append(self.hiddenLayer[0].output.eval()[0])
            for j in range(1,self.num_hidden_layer):
		self.hiddenLayer[j].run([act(k) for k in self.hiddenLayer[j-1].output.eval()[0]])
                output.append(self.hiddenLayer[j].output.eval()[0])
	    self.OutputLayer.run([act(j) for j in self.hiddenLayer[self.num_hidden_layer-1].output.eval()[0]])
            output.append(self.OutputLayer.output.eval()[0])
            self.z.append(output)
            inputs=[np.asarray(feature[i])]
            for out in output:
                inputs.append(np.asarray([act(o) for o in out]))
            self.a.append(inputs)

    def calculate_error(self, y, index):
        self.err = 0
        for i in range(len(index)):
	    #print np.argmax(self.a[i][self.num_hidden_layer+2]),self.a[i][self.num_hidden_layer+2][np.argmax(self.a[i][self.num_hidden_layer+2])],self.a[i][self.num_hidden_layer+2][y[index[i]]]
            if not np.argmax(self.a[i][self.num_hidden_layer+2]) == y[index[i]]:
                self.err += 1
	print 1-float(self.err)/len(index)
    
    def backpropagation(self, y, index):
        self.delta=[]#store in reverse order (delta[0] is actually delta_L)
        for i in range(len(index)):
            dlt=[]
	    g = self.function_gradient(y[index[i]],self.a[i][self.num_hidden_layer+2])
            dlt.append(np.asarray([gact(j) for j in self.z[i][self.num_hidden_layer+1]]*g,dtype=theano.config.floatX ))
	    dl = np.asarray([gact(j) for j in self.z[i][self.num_hidden_layer]],dtype=theano.config.floatX)*tt.dot(dlt[0],self.OutputLayer.W.transpose())
	    dlt.append(dl.eval())
	    for j in range(self.num_hidden_layer):
	        dl = np.asarray([gact(k) for k in self.z[i][self.num_hidden_layer-j-1]],dtype=theano.config.floatX)*tt.dot(dlt[j+1],self.hiddenLayer[self.num_hidden_layer-j-1].W.transpose())
		dlt.append(dl.eval())
            self.delta.append(dlt)
            
    def update(self, index):
        learning_rate = 0.01
        gW = []
        gb = []
        for i in range(len(index)):
            gradient=[]
            gradb = []
            len_a = len(self.a[i])-1 #output doesn't caculate gradient
            for j in range(len_a):
		#print "delta: "+str(np.size(self.delta[i][len_a-1-j]))
		#print "a: "+str(np.size(self.a[i][j]))
                gradient.append(self.a[i][j].reshape(len(self.a[i][j]),1)*self.delta[i][len_a-1-j])
                gradb.append(self.delta[i][len_a-1-j])
		#print self.a[i][j].reshape(len(self.a[i][j]),1)*self.delta[i][len_a-1-j]
	    if gW == []:
		gW = gradient
		gb = gradb
	    else:
        	gW = np.add(gW,gradient)
        	gb = np.add(gb,gradb)
        for i in range(len(gW)):
		for j in range(len(gW[i])):
			gW[i][j] = gW[i][j] / float(len(index))
	for i in range(len(gb)):
		for j in range(len(gb[i])):
			gb[i][j] = gb[i][j] / float(len(index))
	self.InputLayer.update(gW[0],gb[0],learning_rate)
	for i in range(self.num_hidden_layer):
		self.hiddenLayer[i].update(gW[i+1],gb[i+1],learning_rate)
	#for j in range(10):
	    #print self.OutputLayer.W.eval()[j][y[index[i]]], self.OutputLayer.W.eval()[j][y[index[i]]+1]
	self.OutputLayer.update(gW[self.num_hidden_layer+1],gb[self.num_hidden_layer+1],learning_rate)
	#for j in range(10):
	#    print self.OutputLayer.W.eval()[j][y[index[i]]], self.OutputLayer.W.eval()[j][y[index[i]]+1]
    def predict(self, feature, label, index):
        y = []
        acc = 0
        for i in range(len(index)):
            y.append(np.argmax(self.a[i][self.num_hidden_layer+2]))
            if np.argmax(self.a[i][self.num_hidden_layer+2]) == label[i]:
                acc += 1
        acc /= float(len(index))
        return y, acc
    def function_gradient(self, y, a):
        p = np.zeros(1943)
        p[y] = 1.0
	s = tt.sum(a)
	a = a / (s.eval())
	#print 2*(p[y]-a[y]),y,s.eval(),a[y]
	return (p-a)*-2
        

