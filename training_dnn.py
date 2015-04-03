from dnn import deepNeuralNetwork
from modelTool import Save_Model 
import random
from inputs import dataset

MAX_EPOCH = 100000

if __name__ == '__main__':
    data = dataset('./mfcc/train_scale.ark','./state_label/train.lab')
    
    training_data,training_label,val_data,val_label = data.GetInput()
    print 'training size='+str(len(training_label))
    print 'validation size='+str(len(val_label))
    epoch = 0
    print 'data load'
    dnn = deepNeuralNetwork(1,39,10,1943)
    while epoch < MAX_EPOCH:
    	batch = random.sample(range(len(training_label)),10)
        dnn.forward(training_data, batch)
        dnn.backpropagation(training_label, batch)
        dnn.update(batch)
       
        #dnn.calculate_error(training_label, batch)
	epoch += 1
	
	if epoch % 10000 == 0:		
    		batch = random.sample(range(len(val_label)),10000)
    		dnn.forward(val_data, batch)
    		pre, acc = dnn.predict(val_data, val_label, batch)
		print acc
        if epoch % 1000 == 0:
        	Save_Model(dnn,str(epoch)+'.model')
        if epoch % 1000 == 0:
		print epoch
    print 'validation data' 
    batch = range(len(val_label))
    #batch = random.sample(range(len(val_label)),10)
    dnn.forward(val_data, batch)
    pre, acc = dnn.predict(val_data, val_label, batch)
    print acc
    Save_Model(dnn,'final.model')
