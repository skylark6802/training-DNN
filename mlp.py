import os
import sys
import time
import numpy as np
import theano
import theano.tensor as tt
from layer import HiddenLayer
from inputs import dataset,shared_dataset

#Model params
BATCH_SIZE= 128
VAL_BATCH_SIZE = 10000
MAX_EPOCH = 100000
L2_weighting = 0.0001
learning_rate = 0.01
learning_rate_decay = 0.99
NUM_HIDDEN_NEURON = 1024
NUM_HIDDEN_LAYER = 5

class DNN(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hidden_layer):

        #construct layers
        self.InputLayer = HiddenLayer(rng=rng,input=input,n_in=n_in,n_out=n_hidden)
        self.hiddenLayer = []
        self.hiddenLayer.append(HiddenLayer(rng=rng,input=self.InputLayer.output,n_in=n_hidden,n_out=n_hidden))
        for i in range(1,n_hidden_layer):
            self.hiddenLayer.append(HiddenLayer(rng=rng,input=self.hiddenLayer[i-1].output,n_in=n_hidden,n_out=n_hidden))
        self.OutputLayer = HiddenLayer(rng=rng,input=self.hiddenLayer[n_hidden_layer-1].output,n_in=n_hidden,n_out=n_out,outputlayer=True)

        # set params and L2
        self.L2_reg = (self.InputLayer.W**2).sum()
        self.params = self.InputLayer.params
        for i in range(n_hidden_layer):
            self.L2_reg+= (self.hiddenLayer[i].W**2).sum()
            self.params+= self.hiddenLayer[i].params
        self.L2_reg+= (self.OutputLayer.W*2).sum()
        self.params+= self.OutputLayer.params

        self.prediction = tt.argmax(self.OutputLayer.output,axis=1)
    def errors(self,label):
        return tt.mean(tt.eq(self.prediction,label))
    def negative_log_likelihood(self, y):
        return -tt.mean(tt.log(self.OutputLayer.output)[tt.arange(y.shape[0]), y])


def main():
#==============================================================================data loading==============================================================================
    print 'data loading'
    data = dataset('./mfcc/train_scale.ark','./state_label/train.lab','./state_48_39.map')
    training_data,training_label,valid_data,valid_label = data.GetInput()

    traing_size = len(training_label)
    valid_size = len(valid_label)

    training_data,training_label = shared_dataset(training_data,training_label)
    valid_data,valid_label = shared_dataset(valid_data,valid_label)

#===============================================================================build model===============================================================================
    print 'building the model'

    index = tt.lvector()
    x = tt.matrix('x')
    y = tt.ivector('y')

    rng = np.random.RandomState(1234)

    # construct the DNN class
    model = DNN(rng=rng,input=x,n_in=39,n_hidden=NUM_HIDDEN_NEURON,n_out=39,n_hidden_layer=NUM_HIDDEN_LAYER)

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

    training_acc = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: training_data[index],
            y: training_label[index]
        }
    )

#===============================================================================train model===============================================================================
    print 'training'

    epoch = 0
    num_mini_batch = int(traing_size/float(BATCH_SIZE))
    print num_mini_batch

    while (epoch < MAX_EPOCH):
        random_index = np.random.permutation(traing_size)
        for i in range(num_mini_batch):
            #batch_index = np.random.randint(0,traing_size,BATCH_SIZE)
            batch_index = random_index[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            minibatch_avg_cost = train_model(batch_index)
        #global learning_rate
        #learning_rate = learning_rate*learning_rate_decay
        epoch = epoch + 1
        
        #valid_index = np.random.randint(0,valid_size,VAL_BATCH_SIZE)
        valid_index = np.asarray(range(valid_size))
        num_valid_batch = int(valid_size/float(VAL_BATCH_SIZE))
        acc = 0
        for i in range(num_valid_batch):
            acc += validate_model(valid_index[i*VAL_BATCH_SIZE:(i+1)*VAL_BATCH_SIZE])
        #t_acc = training_acc(np.asarray(range(training_size)))
        print epoch
        #print 'training acc = '+ str(t_acc)
        print 'valid acc = '+str(acc / num_valid_batch)



if __name__ == '__main__':
   main()