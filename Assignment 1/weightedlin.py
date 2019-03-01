#                   Author: Suraj S                 #
#                      COL 774                      #
##################################################### 
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import linalg as LA
import sys
import csv


if(len(sys.argv)==4):
    xfilename=sys.argv[1]
    yfilename=sys.argv[2]
    tauin=float(sys.argv[3])
else:
    xfilename="linearX.csv"
    yfilename="linearY.csv"
    tauin=0.8
    
reader = csv.reader(open(xfilename, "rb"), delimiter=",")
x = list(reader)
xdata= np.array(x).astype("float")
[m,n]=xdata.shape
reader = csv.reader(open(yfilename, "rb"), delimiter=",")
x = list(reader)
ydata= np.array(x).astype("float")

def normalization(X):
    mean=np.mean(X)
    var=np.std(X)
    def norm(Z):
        return (Z-mean)/var
    return np.vectorize(norm)(X)

xdata=normalization(xdata)
maxval=np.amax(xdata)
minval=np.amin(xdata)
temp=np.zeros([m,n+1])
temp[:,-1:]=xdata
temp[:,0]=np.ones(m)
xdata=temp
theta=LA.inv(xdata.T.dot(xdata)).dot(xdata.T.dot(ydata)).T


def part1(theta):
    dataplot, = plt.plot(xdata[:,1],ydata,'go',label="Data plot")
    linemaking = list(map(lambda x: theta.dot(x), xdata))
    line, = plt.plot(xdata[:, 1], linemaking,'r',label="Hypothesis plot")
    plt.xlabel("X Data")
    plt.ylabel("Y Data")
    plt.legend(handles=[dataplot, line])
    plt.title('Unweighted Linear Regression')
    #plt.savefig(os.path.join("outputfolder/","wlinearplot.png"))
    plt.show()


part1(theta)


def plotfor2(taur):
    def weight_matrix(X, x, taur):
        return np.diag(np.exp(-1 * ((x - X[:, 1])**2 / (2 * taur *taur))))
    x_point=np.linspace(minval,maxval)
    recval=[]
    tau=0.16
    for x in x_point:
        wm=weight_matrix(xdata,x,taur)
        theta=LA.inv(xdata.T.dot(wm.dot(xdata))).dot(xdata.T).dot(wm).dot(ydata)
        recval.append(theta.T.dot(np.array([1,x])))
    plt.plot(xdata[:,1],ydata,'go',Label='data')
    plt.plot(x_point,recval,'r',Label='Hypothesis')
    plt.suptitle('value of tau='+str(taur))
    plt.legend()
    plt.xlabel("X Data")
    plt.ylabel("Y Data")
    #plt.savefig(os.path.join("outputfolder/","weightedt"+str(taur)+".png"))
    plt.show()



def part2():
    plotfor2(tauin)


part2()

def part3():
    for tau1 in [0.1, 0.3, 2 , 10]:
        try:
            plotfor2(tau1) 
            
        except np.linalg.LinAlgError:
            print "Inverse of X'WX does not exist"



part3() 

