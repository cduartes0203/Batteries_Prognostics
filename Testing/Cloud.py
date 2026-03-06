import numpy as np
import math
from Functions.RLS import *
from Functions.RTLO_R1 import *

def euclidian_dist(x1,x2):
    return np.linalg.norm(x1 - x2,ord=None)

class DataCloud:
	N=0
	def __init__(self,x,ID,nI=1,nR=1,nO=1,ηS=[1,1,1],tau=1,decay=1):
		self.ID = ID
		self.track = [ID]
		self.merged = False
		self.merge = None
		self.n=1
		self.mean=x
		self.meant=np.array(x).dot(np.array(x))
		self.variance=0
		self.pertinency=1
		self.tipicality=1e-12
		self.nS = len(x)
		self.nI = nI
		self.nR = nR
		self.nO = nO
		self.N1 = ηS[0]
		self.N2 = ηS[1]
		self.N3 = ηS[2]
		self.decay = decay
		self.tau = tau
		self.rnn = RTLO(self.nI, self.nR, self.nO,
						[self.N1, self.N2, self.N3], self.tau, self.decay)
		self.x = []
		self.t = []
		self.R = 0
		self.xI = None
		self.xF = None
		self.Rmax = 0
		self.specificity = 0
		self.coverage = 0
		self.cardinality = 1
		self.v = 0
		self.Dmax = -np.inf
		
	def addDataClaud(self,x):
		self.n=2
		self.mean=(self.mean+x)/2
		self.meant=((self.meant)/2) + (x.dot(x))/2
		self.variance=self.meant-self.mean.dot(self.mean)

	def updateDataCloud(self,n,mean,meant,variance,tipicality):
		self.n=n
		self.mean=mean
		self.meant=meant
		self.variance=variance
		self.tipicality=tipicality
		self.cardinality=self.cardinality + 1

	def calc_Rmax(self):
		R1 = euclidian_dist(self.mean, self.xI)
		R2 = euclidian_dist(self.mean, self.xF)
		if R1 > R2: Rmax = R1
		else: Rmax = R2
		if Rmax > self.Rmax: self.Rmax = Rmax
		if self.Rmax > self.R: self.R = self.Rmax
		#self.Rmax = Rmax

	def calc_Dmax(self,xR,xF):
		D1 = euclidian_dist(self.mean, xR)
		D2 = euclidian_dist(self.mean, xF)
		if D1 > D2: Dmax = D1
		else: Dmax = D2
		if Dmax > self.Dmax: self.Dmax = Dmax
		#print(Dmax)

	def calc_sp(self):
		n = len(self.mean)
		vR = ((math.pi**(n/2))/(math.gamma((n/2)+1)))*(self.R**n)
		vMax = ((math.pi**(n/2))/(math.gamma((n/2)+1)))*(self.Dmax**n)
		self.specificity = 1-(vR/vMax)
	
	def calc_cv(self,k):	
		self.coverage = self.cardinality/(k-1)

	def calc_v(self,k):
		self.calc_cv(k)
		self.calc_sp()
		self.v = self.coverage*self.specificity
	
	def calculate_specificity(self,k):
		sum = 0
		for X in self.x[:]:
			dsample = euclidian_dist(self.mean, X)
			sum = sum + dsample/self.dmax
		self.specificity = sum/k	

	def calc_cv(self,k):	
		self.coverage = self.cardinality/(k-1)	

	def calculate_SpCov(self,k):
		self.calc_cv(k)
		self.calculate_specificity(k)
		return self.specificity*self.coverage