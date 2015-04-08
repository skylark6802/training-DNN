import sys
import random
import numpy
import theano
import theano.tensor as tt

def shared_dataset(data_x,data_y,borrow=True):
	shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
	return shared_x, tt.cast(shared_y, 'int32')
class dataset:	
	def __init__(self,a,b,Map): 
		self.train_MFCC = a
		self.labelset = b
		self.label_dic={}
		self.TrainFea = []
		self.TrainLabel =[]
		self.ValFea = []
		self.ValLabel = []
		self.mapfile = Map
	def InvertInput(self):
		phlabel = numpy.loadtxt(self.mapfile,dtype='str_',delimiter='\t')
		print phlabel
		ph39_i = dict(zip(list(set(phlabel[:,2])),numpy.arange(0,39)))
		ph_dict = dict(zip(phlabel[:,0],phlabel[:,2])) # Id -> 39 phonemes
		Label = []
		for line in open(self.labelset):
			instance,label = line.split(',')
			self.label_dic[instance]=int(label)
			Label.append(int(label))
		MFCC_x=[]
		MFCC_y=[]
		for line in open(self.train_MFCC):
			line = line.split(None,1)
			instance,features = line
			x1 = []
			for e in features.split():
				x1 += [float(e)]
			MFCC_x += [x1]
			MFCC_y += [instance]
		datasize = len(MFCC_x)
		dim = len(MFCC_x[0])
		shuffle_array = range(1,datasize+1)
		random.shuffle(shuffle_array)
		for i in range(int(4*datasize/5)):
			self.TrainLabel.append(ph39_i[ph_dict[str(self.label_dic[MFCC_y[shuffle_array[i]-1]])]])
			self.TrainFea.append(MFCC_x[shuffle_array[i]-1])
		for i in range(int(4*datasize/5),datasize):
			self.ValLabel.append(ph39_i[ph_dict[str(self.label_dic[MFCC_y[shuffle_array[i]-1]])]])
			self.ValFea += [MFCC_x[shuffle_array[i]-1]]
	def GetInput(self):
		self.InvertInput()
		return (self.TrainFea,self.TrainLabel,self.ValFea,self.ValLabel)
'''
if __name__ == '__main__':
	print sys.argv[1]
	data = dataset(sys.argv[1],sys.argv[2],sys.argv[3])
	training_data,training_label,val_data,val_label = data.GetInput()
	train_x,train_y = shared_dataset(training_data,training_label)
'''
