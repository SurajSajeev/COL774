
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy.linalg as LA
import sys
import csv
if(len(sys.argv)==3):
    xfilename=sys.argv[1]
    yfilename=sys.argv[2]
else:
    xfilename="linearX.csv"
    yfilename="linearY.csv"
    
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
    



xdata1=normalization(xdata[:,0])


xdata2=normalization(xdata[:,1])


[m,n]=xdata.shape

xdata0=np.ones([m,])

xdata=np.c_[xdata0,xdata1,xdata2]
[m,n]=xdata.shape
gradient=np.zeros([n,])

theta=np.zeros([n,1])


def sigmoid(z):
    return 1/(1+np.exp(-z))


def gradient(xdata,ydata,theta):
    gradientval=np.zeros([n,1])
    for i in range(0,n):
            gradientval[i][0]+=np.squeeze((ydata.T-sigmoid(theta.T.dot(xdata.T))).dot(np.reshape(np.array(xdata[:,i]),[-1,1])))
    return gradientval


def hessian(X, theta):
    G = sigmoid(theta.T.dot(X.T))
    D = np.diag((G * (1 - G)).squeeze())
    return X.T.dot(D.dot(X))


def LL(theta):
    temp=sigmoid(theta.T.dot(xdata.T))
    return np.sum((ydata.T * np.log(temp) + (1 - ydata).T * np.log(1 - temp)))

def part1():
    temptheta=np.zeros([n,1])
    Logl=LL(temptheta)
    status=False
    iteration=0
    while(status!=True):
        
        temptheta+=LA.inv(hessian(xdata,temptheta)).dot(gradient(xdata,ydata,temptheta))
        olderror=Logl
        Logl=LL(temptheta)
        if abs(Logl-olderror) < 10**(-12):
            print "Converged"
            status=True
        if(iteration>900):
            print "Fail to Converge within the limit"
            break
        iteration=iteration+1
    if(status):
        print "Following are the details of the Newton's method"
        print "Stopping Criteria:absolute value of difference between the initial and final log-likelihood"
        print "Theta Values(Final Parameters)",temptheta
    return temptheta

theta=part1()

def part2():
    cls1x1=[]
    cls1x2=[]
    cls2x1=[]
    cls2x2=[]
    for x1,x2,y in zip(xdata[:,1],xdata[:,2],ydata):
        if y==0:
            cls1x1.append([x1])
            cls1x2.append([x2])
        else:
            cls2x1.append([x1])
            cls2x2.append([x2])
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Scatter Plot and decision boundary')
    cl1=plt.scatter(cls1x1,cls1x2,marker='o',label='class 1')
    cl2=plt.scatter(cls2x1,cls2x2,marker='x',label='class 2')
    def maptotheta(x,t=theta):
        return -(t[0]+t[1]*x[1])/t[2]
    yplot=list(map(maptotheta,xdata))
    line,=plt.plot(xdata[:, 1], yplot, 'g', label="Decision Boundary")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    #plt.savefig(os.path.join("outputfolder/","newton.png"))
    plt.show()



part2()





