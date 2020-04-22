import sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from array import array
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def getFeatureData(featureFile, bias = 0):
	x = []
	dFile = open(featureFile, 'r')
	i = 0
	for line in dFile:
		row = line.split()
		rVec = [float(item) for item in row]
		if bias > 0:
			rVec.insert(0, bias)
		x.append(rVec)        
		i += 1
	dFile.close()
	return x

def getLabelData(labelFile, hyperPlaneClass = False):
	lFile = open(labelFile, 'r')
	lDict = {}
	for line in lFile:
		row = line.split()
	#print('label : {}'.format(lDict))
		if hyperPlaneClass and int(row[0]) <= 0:
			lDict[int(row[1])] = -1
		else:
			lDict[int(row[1])] = int(row[0])
	lFile.close()
	return lDict

def first_n_features(n,fs_col):
	min_feature,val=[],[]
	#fs_col.pop()

	for _ in range(n):
		min_feature.append(fs_col.index(max(fs_col)))
		val.append(max(fs_col))
		fs_col[fs_col.index(max(fs_col))]=0
	#print(val)	
	return min_feature

def varianc(XPs, XPs_mean):
	sum = 0
	for i in (XPs):
		sum += ((1 / (len(XPs) - 1))) * (i - XPs_mean) ** 2	
	sum = sum ** (1 / 2)
	return sum

def F_score(XP, XM):
	XM_mean = sum(XM) / len(XM)
	XP_mean = sum(XP) / len(XP)
	PM_mean = sum(XM + XP) / len(XM + XP)
	varXP = varianc(XP, XP_mean)
	varXM = varianc(XM, XM_mean)
    
	if varXP == 0:
		return 1
	else:		
		return ((XP_mean - PM_mean) ** 2 + (XM_mean - PM_mean) ** 2) / (varXP + varXM)

def connectLabels(lsi, lsl):
	checkList, lstest, lstrain = [i[1] for i in lsl], [], []

	dict = {}
	for i in (range(len(lsi))):
		for j in range(len(lsl)):
			if lsl[j][1] == i:
				lsi[i].append(lsl[j][0])
			
		if i not in checkList:
			lstest.append(lsi[i])
			dict[i] = None
		else:
			lstrain.append(lsi[i])
	return  lstrain, lstest, dict

inputList = getFeatureData('traindata')
labelList = getLabelData('trainingLabels.txt')
labelList = [[-1 if v == 0 else 1, k] for k, v in labelList.items()]

#connecting missing labels with it's features
trainList, testList, dict = connectLabels(inputList, labelList)
fs_colm=[]
ch_colm=[]

for j in range(len(trainList[1])):
	positive = []
	neg = []
	for i in trainList:
		if(i[-1]==1):
			positive.append(i[j])
		else:
			neg.append(i[j])
	fs_colm.append(F_score(positive, neg))
#dFile = open('fscore.txt', 'r')
print(fs_colm)
#feature=[]
#for line in dFile:
#	row=line.split(',')
#	feature.extend([float(item) for item in row])
#dFile.close()
#fs_colm=feature.copy()	
scores, X = [], []

for n in range(8, 11):
	scores.append(first_n_features(n, fs_colm.copy()))
y=[i[-1] for i in trainList]

model = SVC(random_state=0)
X_test = getFeatureData('testdata')
X_t=[[] for _ in range(len(X_test))]
X=[[] for _ in range(len(trainList))]

for score in scores:
	for ji in score:
		for m,k in enumerate(trainList):			
			X[m].append(k[ji])

	scored=[]	
	for _ in range(50):
		X_train, X_test, y_train, y_test = train_test_split(X,y)	
		model.fit(X_train,y_train)
		scored.append(model.score(X_test,y_test))
	print(scored)
	print('feature =', len(score), 'accuracy', sum(scored) / len(scored))
exit()

