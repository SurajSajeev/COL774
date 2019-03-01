import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy.linalg as LA
import sys

if(len(sys.argv)==4):
    xfilename=sys.argv[1]
    yfilename=sys.argv[2]
    flag=int(sys.argv[3])
else:
    xfilename='q4x.dat'
    yfilename='q4y.dat'
    flag=0
    
xdata=np.loadtxt(xfilename) 
ydata=np.loadtxt(yfilename,dtype='str')

[m,n]=np.shape(xdata)

def normalization(X):
    mean=np.mean(X)
    var=np.std(X)
    def norm(Z):
        return (Z-mean)/var
    return np.vectorize(norm)(X)
    
xdata0=normalization(xdata[:,0])


xdata1=normalization(xdata[:,1])

xdata=np.c_[xdata0.flatten(),xdata1.flatten()]

axlist=[]
cxlist=[]

for x,y in zip(xdata,ydata):
    if(y=='Alaska'):
        axlist.append(x)
    else:
        cxlist.append(x)


axlist=np.array(axlist)


cxlist=np.array(cxlist)

def part1():
    mu_0=np.mean(axlist,axis=0)
    mu_1=np.mean(cxlist,axis=0)
    fai=0.0
    fai=float(len(axlist))/m
    temp1=axlist-mu_0
    temp2=cxlist-mu_1
    final=np.concatenate([temp1,temp2])
    sigma=final.T.dot(final)/m
    if(flag==0):
        print 'the value of fai is:',fai
        print 'the value of mu0 is:',mu_0
        print 'the value of mu1 is:',mu_1
        print 'the value of sigma is:\n',sigma
    return fai,mu_0,mu_1,sigma


f,m0,m1,sg=part1()

def part2():
    fig=plt.figure()
    plt.plot(axlist[:,0],axlist[:,1],'ro',label='Alaska')
    plt.plot(cxlist[:,0],cxlist[:,1],'y+',label='Canada',markersize=10)
    plt.xlabel('fresh water')
    plt.ylabel('marine water')
    plt.title('Points plotting')
    plt.legend()
    #plt.savefig(os.path.join("outputfolder/","pointplot.png"))
    plt.show()
    plt.close()

part2()


def part3(f,m0,m1,sg):
    sigma_i=LA.inv(sg)
    fig=plt.figure()
    plt.plot(axlist[:,0],axlist[:,1],'ro',label='Alaska')
    plt.plot(cxlist[:,0],cxlist[:,1],'y+',label='Canada',markersize=10)
    A=2*np.reshape((m0-m1),[1,-1]).dot(sigma_i)
    B=((m0).T.dot(sigma_i.dot((m0)))-(m1).T.dot(sigma_i.dot((m1)))-2*np.log(1/f-1))
    xdata0=xdata[:,0]
    xdata1i=(B-A[0][0]*xdata0)/A[0][1]
    if(flag==0):
        plt.plot(xdata0,xdata1i,label='Decision Boundary',color='g')
        plt.xlabel('fresh water')
        plt.ylabel('marine water')
        plt.title('Gaussian Discriminant analysis')
        plt.legend()
        #plt.savefig(os.path.join("outputfolder/","gline.png"))
        plt.show()
        plt.close()
    return xdata0,xdata1i


xdata0i,xdata1i=part3(f,m0,m1,sg)


def part4():
    fi=float(len(axlist))/m
    m0=np.mean(axlist,axis=0)
    m1=np.mean(cxlist,axis=0)
    sigma0=(axlist-m0).T.dot((axlist-m0))/len(axlist)
    sigma1=(cxlist-m1).T.dot((cxlist-m0))/len(cxlist)
    print "fi=",fi
    print "mu0=",m0
    print "mu1=",m1
    print "sigma0=\n",sigma0
    print "sigma1=\n",sigma1
    return fi,m0,m1,sigma0,sigma1


if(flag==1):
    f,mu0,mu1,sm0,sm1=part4()

def part5(f,mu0,mu1,sm0,sm1,xdata0,xdata1i,m0,m1,sg):
    sigma0_inv=LA.inv(sm0)
    sigma1_inv=LA.inv(sm1)
    det1=LA.det(sm0)
    det2=LA.det(sm1)
    sigma_i=LA.inv(sg)
    A=sigma0_inv-sigma1_inv
    B=2*(mu1.T.dot(sigma1_inv)-mu0.T.dot(sigma0_inv))
    C=((mu0).T.dot(sigma0_inv.dot((mu0)))-(mu1).T.dot(sigma1_inv.dot((mu1)))-2*np.log((1/f-1)*(det1/det2)))
    A1=2*np.reshape((m0-m1),[1,-1]).dot(sigma_i)
    B1=((m0).T.dot(sigma_i.dot((m0)))-(m1).T.dot(sigma_i.dot((m1)))-2*np.log(1/f-1))
    
    xa=np.linspace(-2.5,2.5,50)
    ya=np.linspace(-3,3,50)
    p,q=np.meshgrid(xa,ya)
    s=np.c_[p.flatten(),q.flatten()]
    def quadbdry(x):
            return x.T.dot(A.dot(x)) + B.dot(x) + C
    boundaryval=np.array([quadbdry(sa) for sa in s]).reshape(np.shape(p))
    xdata0=xdata[:,0]
    xdata1=xdata[:,1]
    xdata1i=(B1-A1[0][0]*xdata0)/A1[0][1]
    plt.contour(p,q,boundaryval,[0])
    dec,=plt.plot(xdata0i,xdata1i,label='Decision Boundary',color='g')
    plt.xlabel('fresh water')
    plt.ylabel('marine water')
    plt.xlim(np.min(xdata0),np.max(xdata0))
    plt.ylim(np.min(xdata1),np.max(xdata1))
    plt.plot([-10],[-10],'-',label="quadratic boundary",color='m')
    plt.legend()
    #plt.savefig(os.path.join("outputfolder/","gaussianwline.png"))
    plt.show()
    plt.close()

if(flag==1):
    part5(f,mu0,mu1,sm0,sm1,xdata0,xdata1,m0,m1,sg)


