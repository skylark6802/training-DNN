import os
import sys
import time
import numpy as np
import theano
import theano.tensor as tt
from layer import HiddenLayer
from inputs import dataset

BATCH_SIZE= 128
VAL_BATCH_SIZE = 10000
MAX_EPOCH = 100000
L2_weighting = 0.0001
learning_rate = 0.01
n_hidden = 1024

class DNN(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.InputLayer = HiddenLayer(rng=rng,input=input,n_in=n_in,n_out=n_hidden)
        self.hiddenLayer_1 = HiddenLayer(rng=rng,input=self.InputLayer.output,n_in=n_hidden,n_out=n_hidden)
        self.hiddenLayer_2 = HiddenLayer(rng=rng,input=self.hiddenLayer_1.output,n_in=n_hidden,n_out=n_hidden)
        self.OutputLayer = HiddenLayer(rng=rng,input=self.hiddenLayer_2.output,n_in=n_hidden,n_out=n_out,output=True)

        self.L2_reg = (self.InputLayer.W**2).sum() + (self.hiddenLayer_1.W**2).sum()+ (self.hiddenLayer_2.W**2).sum()+ (self.OutputLayer.W**2).sum()
        self.params = self.InputLayer.params + self.hiddenLayer_1.params+ self.hiddenLayer_2.params+ self.OutputLayer.params
        self.p_y_given_x = tt.nnet.softmax(self.OutputLayer.output)
        self.prediction = tt.argmax(self.p_y_given_x,axis=1)
    def errors(self,label):
        return tt.mean(tt.eq(self.prediction,label))
    def negative_log_likelihood(self, y):
        return -tt.mean(tt.log(self.p_y_given_x)[tt.arange(y.shape[0]), y])


def main():
#==============================================================================data loading==============================================================================
    print 'data loading'
    data = dataset('./mfcc/train_scale.ark','./state_label/train.lab','./state_48_39.map')
    training_data,training_label,valid_data,valid_label = data.GetInput()

    traing_size = len(training_label)
    valid_size = len(valid_label)

    training_data = theano.shared(np.asarray(training_data, dtype=theano.config.floatX),borrow=True)
    valid_data = theano.shared(np.asarray(valid_data, dtype=theano.config.floatX),borrow=True)

    training_label = tt.cast(theano.shared(np.asarray(training_label, dtype=theano.config.floatX),borrow=True),'int32')
    valid_label = tt.cast(theano.shared(np.asarray(valid_label, dtype=theano.config.floatX),borrow=True),'int32')


#===============================================================================build model===============================================================================
    print 'building the model'

    index = tt.lvector()
    x = tt.matrix('x')
    y = tt.ivector('y')

    rng = np.random.RandomState(1234)

    # construct the DNN class
    model = DNN(rng=rng,input=x,n_in=39,n_hidden=n_hidden,n_out=39)

    '''
    test_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: test_data[index],
            y: test_label[index]
        }
    )
    '''
    validate_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: valid_data[index],
            y: valid_label[index]
        }
    )

    cost_function = model.negative_log_likelihood(y) + model.L2_reg*L2_weighting
    gparams = [tt.grad(cost_function, param) for param in model.params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(model.params, gparams)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost_function,
        updates=updates,
        givens={
            x: training_data[index],
            y: training_label[index]
        }
    )

#===============================================================================train model===============================================================================
    print 'training'

    epoch = 0

    while (epoch < MAX_EPOCH):
        batch_index = np.random.randint(0,traing_size,BATCH_SIZE)
        epoch = epoch + 1
        minibatch_avg_cost = train_model(batch_index)
        if epoch % 100 == 0:
            batch_index = np.random.randint(0,valid_size,VAL_BATCH_SIZE)
            acc = validate_model(batch_index)
            print epoch
            print acc



if __name__ == '__main__':
   main()