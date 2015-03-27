import pickle

def Load_Model(filename):
    with open(filename,'rb') as imput:
        return pickle.load(imput)

def Save_Model(obj,filename):
    with open(filename,'wb') as output:
        pickle.dump(obj,output,pickle.HIGHEST_PROTOCOL)
