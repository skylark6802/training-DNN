from dnn import deepNeuralNetwork
from modelTool import Save_Model 
import random
from inputs import dataset
from theano import tensor as tt
import theano
import numpy as np

MAX_EPOCH = 100000
BATCH_SIZE = 10

if __name__ == '__main__':
    data = dataset('./mfcc/train_scale.ark','./state_label/train.lab')
    
    training_data,training_label,val_data,val_label = data.GetInput()
    print 'training size='+str(len(training_label))
    print 'validation size='+str(len(val_label))
    epoch = 0
    print 'data load'

    index = tt.lscalar()
    y = tt.fscalar()
    y_pred = tt.fscalar()
    dnn = deepNeuralNetwork(1,39,10,1943,BATCH_SIZE)

    model = theano.function(
            inputs=[index, y],
            outputs=dnn.calculate_error(y_pred, y),
            givens={
                y_pred:dnn.y_pred[index],
            }
    )
    while epoch < MAX_EPOCH:
    	batch = random.sample(range(len(training_label)),BATCH_SIZE)
        dnn.forward(training_data, batch)
        dnn.backpropagation(training_label, batch)
        dnn.update(batch)
	

	acc = [model(i, training_label[batch[i]]) for i in range(len(batch))]
	#print acc
	print np.mean(acc)
	epoch += 1

	if epoch % 10000 == 0:
		batch = random.sample(range(len(val_label)),10000)
    		dnn.forward(val_data, batch)
    		pre, acc = dnn.predict(val_data, val_label, batch)
		print acc
        if epoch % 10000 == 0:
        	Save_Model(dnn,str(epoch)+'.model')
	if epoch %10 == 0:
		print epoch
    print 'validation data' 
    batch = range(len(val_label))
    #batch = random.sample(range(len(val_label)),10)
    dnn.forward(val_data, batch)
    pre, acc = dnn.predict(val_data, val_label, batch)
    print acc
    Save_Model(dnn,'final.model')
