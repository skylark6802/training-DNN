class Dataset:
    def __init__(self,dim):
	self.dlist = []
	self.DIM = dim
    
    def ReadInput(self,arkfile):
	with open(arkfile, "r") as fark:
	    for line in fark:
		alist = []
		l = line.split()
		for ele in l:
		    try:
			alist.append(float(ele))
		    except:
			alist.append(ele)
		self.dlist.append(alist)

    def Normalize(self,low,up):
	#scale to [-1,1]
	dist = up - low
	for i in range(1,self.DIM+1):
	    maxx = -float("inf")
	    minn = float("inf")
	    for lst in self.dlist:
		if lst[i] > maxx:
		    maxx = lst[i]
		if lst[i] < minn:
		    minn = lst[i]
	    ratio = float(maxx - minn)
	    for lst in self.dlist:
		lst[i] = low + (lst[i]-minn)/ratio*dist
    
    def Merge(self,scope):
	self.mergelist=[]
	self.scope = scope
	length = len(self.dlist)
	for i in range(length):
	    if i-scope < 0 or i+scope > length-1:
		continue
	    try:
		del mlist[0:69]
		mlist = mlist + self.dlist[i+scope][1:]
		mlist.insert(0,self.dlist[i][0])
	    except NameError:
		mlist = [self.dlist[i][0]]
		for j in range(-1*scope,scope):
		    mlist = mlist + self.dlist[i+j][1:]
	    self.mergelist.append(mlist)

    def MWriteOutput(self,arkfile):
	with open(arkfile,"w") as fark:
	    for lst in self.mergelist:
		fark.write("%s " % lst[0])
		for num in lst[1:]:
		    fark.write("%f " % num)
		fark.write("\n")

    def WriteOutput(self,arkfile):
	with open(arkfile,"w") as fark:
	    for lst in self.dlist:
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
    data.ReadInput("../MLDS_HW1_RELEASE_v1/fbank/test.ark")
    data.Normalize(-1.0,1.0)
    
    #if need to merge
    data.Merge(4)
    data.MWriteOutput("test_fbank_m_scale.ark")
    
    #else
    #data.WriteOutput("MLDS_HW1_RELEASE_v1/mfcc/test_scale.ark")
    data.WriteOutput("test_fbank_scale.ark")
