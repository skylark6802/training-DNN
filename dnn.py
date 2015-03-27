import numpy as np
import theano
import theano.tensor as tt

y = tt.log(1 + tt.exp(x))
act = theano.function( [x], y )
g = T.grad(y, x)
gact = theano.function( [x], g )

class hiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W is None:
            W_values = np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (n_in + n_out)),
                        high=np.sqrt(6. / (n_in + n_out)),
                        size=(n_out, n_in)
                        ),
                    dtype=theano.config.floatX
                    )
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        self.output = tt.dot(input, self.W) + self.b
        self.params = [self.W, self.b]

# number of hidden layer can not be zero
class dnn(object):
    def __init__(self, num_hidden_layer, n_int, n_hidden, n_out):
        #input layer
        self.InputLayer = HiddenLayer(
                            rng=rng,
                            input=None,
                            n_in=n_in,
                            n_out=n_hidden)
        #hidden layer
        self.hiddenLayer = []
        h = HiddenLayer(rng=rng, input=act(self.InputLayer.output), n_in=n_hidden, n_out=n_hidden)
        self.hiddenLayer.append(h)
        for i in range(1,num_hidden_layer):
            h = HiddenLayer(rng=rng, input=act(self.hiddenLayer[i-1].output), n_in=n_hidden, n_out=n_hidden)
            self.hiddenLayer.append(h)
        #output layer
        self.OutputLayer = HiddenLayer(
                            rng=rng,
                            input=act(self.hiddenLayer[num_hidden_layer-1].output),
                            n_in=n_hidden,
                            n_out=n_out)
        #set dnn parameters
        self.params = self.InputLayer.params
        for i in range(num_hidden_layer):
            self.params += self.hiddenLayer[i].params
        self.params += self.OutputLayer.params
        self.num_hidden_layer = num_hidden_layer
    
    def forward(self, feature, index):
        self.a=[]#record output of layer
        self.z=[]#record input activation function
        for i in index:
            output=[]
            self.InputLayer.input=feature[i]
            output.append(self.InputLayer.output)
            for j in range(self.num_hidden_layer):
                output.append(self.hiddenLayer[i].output)
            output.append(self.OutputLayer.output)
            self.z.append(np.asarray(output,dtype=theano.config.floatX ))
            inputs=[feature[i]]
            for o in output:
                inputs.append(act(o))
            self.a.append(np.asarray(inputs,dtype=theano.config.floatX ))

    def calculate_error(self, y, index):
        self.err = 0
        for i in range(len(index)):
            if not max(self.a[i][self.num_hidden_layer+2]) == y[index[i]]:
                self.err += 1
    
    def backpropagation(self, y, index):
        self.delta=[]#store in reverse order (delta[0] is actually delta_L)
        for i in range(len(index)):
            dlt=[]
            if not max(self.a[i][self.num_hidden_layer+2]) == y[index[i]]:# update only when incorrect prediction
                dlt.append(np.asarray(gact(self.z[i][self.num_hidden_layer+1]),dtype=theano.config.floatX ))
                dl = gact(self.z[i][self.num_hidden_layer]*tt.dot(dlt[0],self.OutputLayer.W))
                dlt.append(np.asarray(dl,dtype=theano.config.floatX ))
                for j in range(self.num_hidden_layer):
                    dl = gact(self.z[i][self.num_hidden_layer-j-1]*tt.dot(dlt[j+1],self.hiddenLayer[self.num_hidden_layer-j-1].W))
                    dlt.append(np.asaray(dl),dtype=theano.config.floatX )
            self.delta.append(np.asarray(dlt,dtype=theano.config.floatX ))
            
    def update(self, index):
        learning_rate = 0.0001
        gparam = []
        for i in range(len(index)):
            gradient=[]
            len_a = len(self.a)
            for j in range(len_a):
                gradient.append(tt.dot(self.delta[i][len_a-1-j].transpose(),self.a[i][j]))
            gparam = tt.add(gparm,gradient)
        tt.add(self.param,-learning_rate*gparam)
    def predict(self, feature, label, index):
        y = []
        acc = 0
        for i in index:
            self.InputLayer.input = feature[i]
            output = self.OutputLayer.output
            y.append(max(output))
            if max(output) == label[i]:
                acc += 1
        acc /= float(len(index))
        return y, acc
        

