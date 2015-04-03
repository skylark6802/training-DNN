class Dataset:
    def __init__(self):
	self.dlist = []
	self.DIM = 39
    
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
    
    def WriteOutput(self,arkfile):
	with open(arkfile,"w") as fark:
	    for lst in self.dlist:
		fark.write("%s " % lst[0])
		for i in range(1,self.DIM+1):
		    fark.write("%f " % lst[i])
		fark.write("\n")
		
if __name__ == "__main__":
    data = Dataset()
    data.ReadInput("MLDS_HW1_RELEASE_v1/mfcc/tmp.ark")
    data.Normalize(-1.0,1.0)
    #data.WriteOutput("MLDS_HW1_RELEASE_v1/mfcc/test_scale.ark")
    data.WriteOutput("tmp_scale.ark")
