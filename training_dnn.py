import dnn
import modelTool 

MAX_EPOCH = 1

if __name__ == '__main__':
    while epoch < MAX_EPOCH:
        while :
            dnn.foward(feature, batch)
            dnn.calculate_error(y, batch)
            dnn.backpropagate(y, batch)
            dnn.update(batch)
        epoch += 1
    while:
        dnn.foward(feature, batch)
        dnn.predict(feature, batch)
