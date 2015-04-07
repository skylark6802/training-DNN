import numpy as np
import theano
import theano.tensor as tt

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, output = False):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),high=np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        output = tt.dot(input, self.W) + self.b
        if output == False:
            self.output = self.activation(output)
        else:
            self.output = output
        # parameters of the model
        self.params = [self.W, self.b]
    def activation(self,x):
        return tt.log(1+tt.exp(x))
