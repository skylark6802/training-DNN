import sys
import random
import numpy
from modelTool import Load_Model, Save_Model
'''import theano
import theano.tensor as T

def shared_dataset(datax,datay,borrow=True):
	shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
    	shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
    	return shared_x, T.cast(shared_y, 'int32')
'''
class dataset:	
	def __init__(self,trainset,testset,labelset,Map): 
		
		self.trainset = trainset
		self.testset = testset
		self.labelset = labelset
		self.label_dic={}
		self.TrainFea = []
		self.TrainLabel =[]
		self.TrainLabel39 =[]
		self.ValFea = []
		self.ValLabel = []
		self.ValLabel39 = []
		self.makelabel(Map)

	def makelabel(self,Mapfile):
		phlabel = numpy.loadtxt(Mapfile,dtype='str_',delimiter='\t')
		self.ph39_i = dict(zip(list(set(phlabel[:,2])),numpy.arange(0,39)))
		self.i_ph39 = dict(zip(numpy.arange(0,39),list(set(phlabel[:,2]))))
		self.ph_dict = dict(zip(phlabel[:,0],phlabel[:,2])) # Id -> 39 phonemes
	def InvertLabel(self,state_label):
		tmplist=[];
		for tmp in numpy.nditer(state_label):
			tmplist.append(self.ph39_i[self.ph_dict[str(tmp)]]);
		return numpy.array(tmplist)
	def MakeOutput(self,Label,Result):
		with open (Result,'w') as f:
			f.write('Id,Prediction\n')
			index = 0
			for line in open(self.testset):
				line = line.split(None,1)
				instance,fea = line
				f.write(str(instance)+','+str(self.i_ph39[Label[index]])+'\n')
				index = index + 1
	def ConvertTrain(self):
		Label = []
		for line in open(self.labelset):
			instance,label = line.split(',')
			self.label_dic[instance]=int(label)
			Label.append(int(label))
		MFCC_x=[]
		MFCC_y=[]
		for line in open(self.trainset):
			line = line.split(None,1)
			instance,features = line
			x1 = [float(e)for e in features.split()]
			MFCC_x += [x1]
			MFCC_y += [instance]
		for i in range(len(MFCC_x)):
			self.TrainLabel39.append(self.ph39_i[self.ph_dict[str(self.label_dic[MFCC_y[i]])]])
			self.TrainLabel.append(self.label_dic[MFCC_y[i]])
			self.TrainFea.append(MFCC_x[i])
		
	def InvertInput(self):
		Label = []
		for line in open(self.labelset):
			instance,label = line.split(',')
			self.label_dic[instance]=int(label)
			Label.append(int(label))
		MFCC_x=[]
		MFCC_y=[]
		for line in open(self.trainset):
			line = line.split(None,1)
			instance,features = line
			x1 = [float(e)for e in features.split()]
			MFCC_x += [x1]
			MFCC_y += [instance]
		datasize = len(MFCC_x)
		shuffle_array = range(1,datasize+1)
		random.shuffle(shuffle_array)
		for i in range(int(4*datasize/5)):
			self.TrainLabel39.append(self.ph39_i[self.ph_dict[str(self.label_dic[MFCC_y[shuffle_array[i]-1]])]])
			self.TrainLabel.append(self.label_dic[MFCC_y[shuffle_array[i]-1]])
			self.TrainFea.append(MFCC_x[shuffle_array[i]-1])
		for i in range(int(4*datasize/5),datasize):
			self.ValLabel39.append(self.ph39_i[self.ph_dict[str(self.label_dic[MFCC_y[shuffle_array[i]-1]])]])
			self.ValLabel.append(self.label_dic[MFCC_y[shuffle_array[i]-1]])
			self.ValFea.append(MFCC_x[shuffle_array[i]-1])
	def GetFullTrain(self):
		self.ConvertTrain()
		return (self.TrainFea,self.TrainLabel39,self.TrainLabel)
	def GetInput(self):
		self.InvertInput()
		return (self.TrainFea,self.TrainLabel39,self.TrainLabel,self.ValFea,self.ValLabel39,self.ValLabel)
if __name__ == '__main__':
	data = dataset('./mfcc/train_scale.ark','./mfcc/test_scale_small.ark','./state_label/train.lab','./phones/state_48_39.map')
	'''a = numpy.arange(9)
	b = data.InvertLabel(a)
	
	print b
	print max(b)
	print min(b)
	'''
	Save_Model(data,'dataset')
	newdata = Load_Model('dataset')
	training_data,training_label39,training_label = newdata.GetFullTrain()
	print len(training_data)
	'''
	print training_label[0:5];
	print training_label39[0:5];
	'''
	
	#data.MakeOutput(a,'./result/logan.csv')
	#train_x,train_y = shared_dataset(training_data,training_label)
