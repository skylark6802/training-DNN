class Dataset:
    def __init__(self,dim):
	self.DIM = dim
	self.trainlist = []
	self.testlist = []
	self.mtrainlist=[]
	self.mtestlist=[]
    
    def ReadInput(self,trainfile,testfile):
	with open(trainfile, "r") as fark:
	    for line in fark:
		alist = []
		l = line.split()
		for ele in l:
		    try:
			alist.append(float(ele))
		    except:
			alist.append(ele)
		self.trainlist.append(alist)
	
	with open(testfile, "r") as fark:
	    for line in fark:
		alist = []
		l = line.split()
		for ele in l:
		    try:
			alist.append(float(ele))
		    except:
			alist.append(ele)
		self.testlist.append(alist)
	

    def Normalize(self,low,up):
	#scale to [-1,1]
	dist = up - low
	for i in range(1,self.DIM+1):
	    maxx = -float("inf")
	    minn = float("inf")
	    for lst in self.trainlist:
		if lst[i] > maxx:
		    maxx = lst[i]
		if lst[i] < minn:
		    minn = lst[i]
	    for lst in self.testlist:
		if lst[i] > maxx:
		    maxx = lst[i]
		if lst[i] < minn:
		    minn = lst[i]

	    ratio = float(maxx - minn)
	    for lst in self.trainlist:
		lst[i] = low + (lst[i]-minn)/ratio*dist
	    for lst in self.testlist:
		lst[i] = low + (lst[i]-minn)/ratio*dist
    
    def Merge(self,scope):
	self.scope = scope
	length = len(self.trainlist)
	for i in range(length):
	    if i-scope < 0 or i+scope > length-1:
		continue
	    try:
		mlist = mlist[self.DIM+1:]
		mlist = mlist + self.trainlist[i+scope][1:]
		mlist.insert(0,self.trainlist[i][0])
	    except NameError:
		mlist = [self.trainlist[i][0]]
		for j in range(-1*scope,scope):
		    mlist = mlist + self.trainlist[i+j][1:]
	    self.mtrainlist.append(mlist)
	
	length = len(self.testlist)
	for i in range(length):
	    if i-scope < 0 or i+scope > length-1:
		continue
	    try:
		mmlist = mmlist[self.DIM+1:]
		mmlist = mmlist + self.testlist[i+scope][1:]
		mmlist.insert(0,self.testlist[i][0])
	    except NameError:
		mmlist = [self.testlist[i][0]]
		for j in range(-1*scope,scope):
		    mmlist = mmlist + self.testlist[i+j][1:]
	    self.mtestlist.append(mmlist)

    def MWriteOutput(self,trainfile,testfile):
	with open(trainfile,"w") as fark:
	    for lst in self.mtrainlist:
		fark.write("%s " % lst[0])
		for num in lst[1:]:
		    fark.write("%f " % num)
		fark.write("\n")
	
	with open(testfile,"w") as fark:
	    for lst in self.mtestlist:
		fark.write("%s " % lst[0])
		for num in lst[1:]:
		    fark.write("%f " % num)
		fark.write("\n")

    def WriteOutput(self,trainfile,testfile):
	with open(trainfile,"w") as fark:
	    for lst in self.trainlist:
		fark.write("%s " % lst[0])
		for num in lst[1:]:
		    fark.write("%f " % num)
		fark.write("\n")
	with open(testfile,"w") as fark:
	    for lst in self.testlist:
		fark.write("%s " % lst[0])
		for num in lst[1:]:
		    fark.write("%f " % num)
		fark.write("\n")
		
if __name__ == "__main__":
    """
    6 arguments
    1.dimension
    2.input file
    3.scale range
    4.scope(if need merge)
    5.scope output file(if need merge)
    6.output file(no merge)
    """
    data = Dataset(69)
    data.ReadInput("../MLDS_HW1_RELEASE_v1/fbank/train.ark","../MLDS_HW1_RELEASE_v1/fbank/test.ark")
    #data.ReadInput("tmp.ark","tmp1.ark")
    data.Normalize(-1.0,1.0)
    
    #if need to merge
    data.Merge(4)
    data.MWriteOutput("train_fbank_m_scale.ark","test_fbank_m_scale.ark")
    
    #else
    #data.WriteOutput("MLDS_HW1_RELEASE_v1/mfcc/test_scale.ark")
    data.WriteOutput("train_fbank_scale.ark","test_fbank_scale.ark")
