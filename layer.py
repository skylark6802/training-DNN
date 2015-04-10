import numpy as np
import theano
import theano.tensor as tt

def dropout(inputs, rng, p=0.5):
    srng = tt.shared_randomstreams.RandomStreams(rng.randint(567891))
    mask = srng.binomial(n=1, p=1-p, size=inputs.shape, dtype=theano.config.floatX)
    return inputs*tt.cast(mask, theano.config.floatX)

def softmax(x):
    p_x = tt.exp(x)
    psum = tt.sum(p_x)
    return (p_x / psum)


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, outputlayer = False, p=1.):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),high=np.sqrt(6. / (n_in + n_out)),size=(n_in, n_out)),dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        if p == 1.:
        	output = tt.dot(input, self.W) + self.b
        else:
            output = tt.dot(dropout(input, rng=rng, p=p), self.W)*(1./(1.-p)) + self.b
            #output = tt.dot(input, dropout(self.W, rng=rng, p=p)) + self.b
        self.output = self.activation(output,outputlayer)
        # parameters of the model
        self.params = [self.W, self.b]
    def activation(self,x,outputlayer):
        if outputlayer:
            return softmax(x)
            #return tt.nnet.softmax(x)
        else:
            return tt.log(1+tt.exp(x))
    
