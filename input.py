import sys
import random

class dataset:	
	def __init__(self,a,b): 
		self.train_MFCC = a
		self.labelset = b
		print self.labelset
		self.label_dic={}
		self.TrainFea = []
		self.TrainLabel =[]
		self.ValFea = []
		self.ValLabel = []
	def InvertInput(self):
		for line in open(self.labelset):
			instance,label = line.split(',')
			self.label_dic[instance]=int(label)
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
			self.TrainLabel.append(self.label_dic[MFCC_y[shuffle_array[i]-1]])
			self.TrainFea.append(MFCC_x[shuffle_array[i]-1])
		for i in range(int(4*datasize/5),datasize):
			self.ValLabel += [self.label_dic[MFCC_y[shuffle_array[i]-1]]]
			self.ValFea += [MFCC_x[shuffle_array[i]-1]]
	def GetInput(self):
		self.InvertInput()
		return (self.TrainFea,self.TrainLabel,self.ValFea,self.ValLabel)
if __name__ == "__main__":
	a = dataset(sys.argv[1],sys.argv[2])
	a,b,c,d=a.GetInput()	
	print a,b,c,d
