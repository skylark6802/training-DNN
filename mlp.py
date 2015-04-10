import os
import sys
import time
import numpy as np
import theano
import theano.tensor as tt
from layer import HiddenLayer
from modelTool import Load_Model, Save_Model
from inputs import dataset,shared_dataset
import math
import pickle
import time

#Model params
BATCH_SIZE= 256
VAL_BATCH_SIZE = 10000
TEST_BATCH_SIZE = 10000
MAX_EPOCH = 100
L2_weighting = 0.0001
learning_rate = 0.05
learning_rate_decay = 0.97
NUM_HIDDEN_NEURON = 512
NUM_HIDDEN_LAYER = 6

class DNN(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hidden_layer, p=1.):

        #construct layers
        self.InputLayer = HiddenLayer(rng=rng,input=input,n_in=n_in,n_out=n_hidden, p=p)
        self.hiddenLayer = []
        self.hiddenLayer.append(HiddenLayer(rng=rng,input=self.InputLayer.output,n_in=n_hidden,n_out=n_hidden, p=0.1))
        self.hiddenLayer.append(HiddenLayer(rng=rng,input=self.hiddenLayer[0].output,n_in=n_hidden,n_out=n_hidden, p=0.1))
        for i in range(2,n_hidden_layer):
            self.hiddenLayer.append(HiddenLayer(rng=rng,input=self.hiddenLayer[i-1].output,n_in=n_hidden,n_out=n_hidden, p=p))
        self.OutputLayer = HiddenLayer(rng=rng,input=self.hiddenLayer[n_hidden_layer-1].output,n_in=n_hidden,n_out=n_out,outputlayer=True, p=p)

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

def momentum(moment, theta, params, gparams, mu):
    updates=[]
    for param, gparam, mom, the in zip(params, gparams, moment, theta):
        mom=mom*mu - learning_rate*gparam
        the=the + mom
        updates.append((param,param + the))
    return updates
        

def main():
#==============================================================================data loading==============================================================================
    print 'data loading'
    if sys.argv[2] == 'MFCC':
        trainset = open('MFCCtrainset','r')
    elif sys.argv[2] == 'fbank':
        trainset = open('fbanktrainset345','r')
    elif sys.argv[2] == 'fbank69':
        trainset = open('fbanktrainset69','r')
    training_data = pickle.load(trainset)
    training_label39 = pickle.load(trainset)
    training_label = pickle.load(trainset)

    
    if sys.argv[2] == 'MFCC':
        validset = open('MFCCvalset','r')
    elif sys.argv[2] == 'fbank':
        validset = open('fbankvalset345','r')
    elif sys.argv[2] == 'fbank69':
        validset = open('fbankvalset69','r')
    valid_data = pickle.load(validset)
    valid_label39 = pickle.load(validset)
    valid_label = pickle.load(validset)
    
    traing_size = len(training_label)
    valid_size = len(valid_label)

    if sys.argv[1] == '1943':
        training_data,training_label = shared_dataset(training_data,training_label)
        valid_data,valid_label = shared_dataset(valid_data,valid_label)
    elif sys.argv[1] == '39':
        training_data,training_label = shared_dataset(training_data,training_label39)
        valid_data,valid_label = shared_dataset(valid_data,valid_label39)

    ph39_i,i_ph39,ph_dict = makelabel('./state_48_39.map')
    if sys.argv[2] == 'MFCC':
        testset = open('MFCCtestset','r')
    elif sys.argv[2] == 'fbank':
        testset = open('fbanktestset345','r')
    elif sys.argv[2] == 'fbank69':
        testset = open('fbanktestset69','r')
    test_data = pickle.load(testset)
    test_id = pickle.load(testset)
    test_size = len(test_data)
    test_data = theano.shared(np.asarray(test_data,dtype=theano.config.floatX),borrow=True)

#===============================================================================build model===============================================================================
    print 'building the model'

    index = tt.lvector()
    x = tt.matrix('x')
    y = tt.ivector('y')

    rng = np.random.RandomState(1234)

    if sys.argv[2] == 'MFCC':
        n_feature = 39
    elif sys.argv[2] == 'fbank':
        n_feature = 345
    elif sys.argv[2] == 'fbank69':
        n_feature = 69
    # construct the DNN class
    if sys.argv[1] == '1943':
        model = DNN(rng=rng,input=x,n_in=n_feature,n_hidden=NUM_HIDDEN_NEURON,n_out=1943,n_hidden_layer=NUM_HIDDEN_LAYER, p=1.)# using dropout if p < 1 
    elif sys.argv[1] == '39':
        model = DNN(rng=rng,input=x,n_in=n_feature,n_hidden=NUM_HIDDEN_NEURON,n_out=39,n_hidden_layer=NUM_HIDDEN_LAYER, p=1.)# using dropout if p < 1 

    
    test_model = theano.function(
        inputs=[index],
        outputs=model.prediction,
        givens={
            x: test_data[index],
        }
    )
    
    if sys.argv[1] == '1943':
        validate_model = theano.function(
            inputs=[index],
            outputs=model.prediction,
            givens={
                x: valid_data[index]
            }
        )
    elif sys.argv[1] == '39':
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
    #updates = [(param, param - learning_rate * gparam) for param, gparam in zip(model.params, gparams)]
    moment = [theano.shared(param.get_value()*0., broadcastable=param.broadcastable) for param in model.params]
    theta = [theano.shared(param.get_value()*0., broadcastable=param.broadcastable) for param in model.params]
    updates = momentum(moment, theta, model.params, gparams, 0.5)

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
    num_mini_batch = int(math.ceil(traing_size/float(BATCH_SIZE)))
    print num_mini_batch

    log = open('log','w')
    while (epoch < MAX_EPOCH):
        start_time = time.clock()
        random_index = np.random.permutation(traing_size)
        for i in range(num_mini_batch):
            batch_index = random_index[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            minibatch_avg_cost = train_model(batch_index)
        global learning_rate
        learning_rate = learning_rate*learning_rate_decay
        epoch = epoch + 1

        
        valid_index = np.random.randint(0,valid_size,VAL_BATCH_SIZE)
        
        valid_index = np.asarray(range(valid_size))
        num_valid_batch = int(math.ceil(valid_size/float(VAL_BATCH_SIZE)))
        acc = 0
        for i in range(num_valid_batch):
            if sys.argv[1] == '1943':
                if i == 0:
                    prediction = validate_model(valid_index[i*VAL_BATCH_SIZE:(i+1)*VAL_BATCH_SIZE])
                else:
                    prediction = np.append(prediction,validate_model(valid_index[i*VAL_BATCH_SIZE:(i+1)*VAL_BATCH_SIZE]))
            elif sys.argv[1] == '39':
                acc += validate_model(valid_index[i*VAL_BATCH_SIZE:(i+1)*VAL_BATCH_SIZE])
        
        print epoch
        
        if sys.argv[1] == '1943':
            print 'valid acc = '+ str(changeLabel(prediction,valid_label39,ph39_i,ph_dict))
        elif sys.argv[1] == '39':
            print 'valid acc = '+str(acc / num_valid_batch)
        
        end_time = time.clock()
        print 'each epoch time = '+ str(end_time - start_time)
        
        log.write(str(epoch)+'\n')
        log.write('valid acc = '+str(acc / num_valid_batch)+'\n')
        
        if epoch >= 40 and epoch % 10 == 0:
            # test
            test_index = np.asarray(range(test_size))
            num_test_batch = int(math.ceil(test_size/float(VAL_BATCH_SIZE)))
            for i in range(num_test_batch):
                if i == 0:
                    prediction = test_model(test_index[i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE])
                else:
                    prediction = np.append(prediction,test_model(test_index[i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE]))
            pre_label = []
            for i in range(test_size):
                if sys.argv[1] == '1943':
                    pre_label.append(ph39_i[ph_dict[str(prediction[i])]])
                elif sys.argv[1] == '39':
                    pre_label.append(i_ph39[prediction[i]])

            with open (str(epoch)+'.predict','w') as f:
                f.write('Id,Prediction\n')
                for i in range(test_size):
                    f.write(str(test_id[i])+','+str(pre_label[i])+'\n')
        
def makelabel(Mapfile):
    phlabel = np.loadtxt(Mapfile,dtype='str_',delimiter='\t')
    ph39_i = dict(zip(list(set(phlabel[:,2])),np.arange(0,39)))
    i_ph39 = dict(zip(np.arange(0,39),list(set(phlabel[:,2]))))
    ph_dict = dict(zip(phlabel[:,0],phlabel[:,2])) # Id -> 39 phonemes
    return (ph39_i,i_ph39,ph_dict)

def changeLabel(prediction,label,ph39_i,ph_dict):
    pre = []
    for i in range(len(label)):
        pre.append(ph39_i[ph_dict[str(prediction[i])]])
    pre = np.asarray(pre)
    return np.mean(np.equal(pre,label))



if __name__ == '__main__':
   main()
