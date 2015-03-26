import numpy as np
import theano
import theano.tensor as tt

f = theano.function( [x],1 + tt.exp(x) )

class hiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_in, n_out)
                        ),
                    dtype=theano.config.floatX
                    )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        kobe_output = tt.dot(input, self.W) + self.b
        self.output = f(kobe_output)
        self.params = [self.W, self.b]

class dnn(object):
    def __init__(self, num_hidden_layer, n_int, n_hidden, n_out):
        #input layer
        self.InputLayer = HiddenLayer(
                            rng=rng,
                            input=None;
                            n_in=n_in,
                            n_out=n_hidden)
        #hidden layer
        self.hiddenLayer = []
        h = HiddenLayer(rng=rng, input=self.InputLayer.output, n_in=n_hidden, n_out=n_hidden)
        self.hiddenLayer.append(h)
        for i in range(1,num_hidden_layer):
            h = HiddenLayer(rng=rng, input=self.hiddenLayer[i-1].output, n_in=n_hidden, n_out=n_hidden)
            self.hiddenLayer.append(h)
        #output layer
        self.OutputLayer = HiddenLayer(
                            rng=rng,
                            input=self.hiddenLayer[num_hidden_layer-1].output,
                            n_in=n_hidden,
                            n_out=n_out)
        #set dnn parameters
        self.params = self.InputLayer.params
        for i in range(num_hidden_layer):
            self.params += self.hiddenLayer[i].params
        self.params = self.OutputLayer.params
        self.num_hidden_layer
    
    def forward(self, index):
        self.a=[]#record output of neuron in every layer (a)
        for i in index:
            output=[]
            self.InputLayer.input=dnn.feature[i]
            output.append(self.InputLayer.output)
            for i in range(self.num_hidden_layer):
                output.append(self.hiddenLayer[i].output)
            output.append(self.OutputLayer.output)
            self.a.append(output)

    def calculate_error(self, index):
        self.err = 0
        for i in len(index):
            if not max(self.a[i][self.num_hidden_layer+1]) == max(dnn.y[index[i]]):
                self.err += 1
    
    def backpropagation:
        gparams = [tt.grad(f, param) for param in classifier.params]
