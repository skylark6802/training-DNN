from dnn_momentum import deepNeuralNetwork
from modelTool import Save_Model 
import random
from inputs import dataset
from theano import tensor as tt
import theano
import numpy as np

MAX_EPOCH = 100000
BATCH_SIZE = 10

if __name__ == '__main__':
    data = dataset('/tmp2/r03922067/mfcc/train_scale.ark','/tmp2/r03922067/state_label/train.lab')
    
    training_data,training_label,val_data,val_label = data.GetInput()
    print 'training size='+str(len(training_label))
    print 'validation size='+str(len(val_label))
    epoch = 0
    print 'data load'

    index = tt.lscalar()
    y = tt.dscalar()
    y_pred = tt.dscalar()
    dnn = deepNeuralNetwork(1,39,10,1943,BATCH_SIZE)
    
    model = theano.function(
            inputs=[index, y],
            outputs=dnn.calculate_error(y_pred, y),
            givens={
                y_pred:dnn.y_pred[index],
            }
    )


    dnn.momentum_W = []
    dnn.momentum_b = []
    while epoch < MAX_EPOCH:
        batch = random.sample(range(len(training_label)),BATCH_SIZE)
        dnn.forward(training_data, batch)
        dnn.backpropagation(training_label, batch)
        dnn.momentum_update(batch)
        
        if epoch % 1000 == 0:
            acc = [model(i, training_label[batch[i]]) for i in range(len(batch))]
            print acc
            print np.mean(acc)
	epoch += 1

	if epoch % 10000 == 0:		
            batch = random.sample(range(len(val_label)),100)
            dnn.forward(val_data, batch)
            #pre, acc = dnn.predict(val_data, val_label, batch)
            acc = [model(i, val_label[batch[i]]) for i in range(len(batch))]
            print np.mean(acc)
        if epoch % 10000 == 0:
            Save_Model(dnn,str(epoch)+'.model.momentum')
	print epoch
    print 'validation data' 
    batch = range(len(val_label))
    #batch = random.sample(range(len(val_label)),10)
    dnn.forward(val_data, batch)
    #pre, acc = dnn.predict(val_data, val_label, batch)
    acc = [model(i, val_label[batch[i]]) for i in range(len(batch))]
    print np.mean(acc)
    Save_Model(dnn,'final.model.momentum')
