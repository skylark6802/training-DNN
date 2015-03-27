import dnn
import modelTool 
import random

MAX_EPOCH = 1

if __name__ == '__main__':
    data = dataset('/tmp2/r03922067/mfcc/train.ark','/tmp2/r03922067/state_label/train.lab')
    training_data,training_label,val_data,val_label = data.GetInput()
    while epoch < MAX_EPOCH:
        batch = random.sample(range(len(training_label)),10)
        dnn.foward(training_data, batch)
        dnn.calculate_error(training_label, batch)
        dnn.backpropagate(training_label, batch)
        dnn.update(batch)
        epoch += 1
    
    dnn.foward(val_data, range(len(val_label)))
    pre, acc = dnn.predict(val_data, val_label, range(len(val_label)))
    print acc
