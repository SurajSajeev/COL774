#                   Author: Suraj S                 #
#                      COL 774                      #
##################################################### 
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import csv

#read the data and  keep record of the dimension

if(len(sys.argv)==5):
    xfilename=sys.argv[1]
    yfilename=sys.argv[2]
    time=float(sys.argv[4])
    etain=float(sys.argv[3])
else:
    xfilename="linearX.csv"
    yfilename="linearY.csv"
    time=0.02
    etain=0.1
reader = csv.reader(open(xfilename, "rb"), delimiter=",")
x = list(reader)
xdata= np.array(x).astype("float")
[m,n]=xdata.shape
reader = csv.reader(open(yfilename, "rb"), delimiter=",")
x = list(reader)
ydata= np.array(x).astype("float")


#module for normalization

def normalization(X):
    mean=np.mean(X)
    var=np.std(X)
    def norm(Z):
        return (Z-mean)/var
    return np.vectorize(norm)(X)
    


# normalized X-data


xdata=normalization(xdata)


#Theta values with adding intercepts to x


theta=np.zeros([1,n+1])
temp=np.zeros([m,n+1])
temp[:,-1:]=xdata
temp[:,0]=np.ones(m)
xdata=temp


# J(theta) function that is needed to be calculated

def J(theta):
    return np.sum((np.dot(xdata,theta.T)-ydata)**2)/(2*m)


temptheta=np.zeros([1,n+1])

#initial learning rate
eta=etain


#part 1 of the problem
#algorithm calculates limited number of iterations as the number of iterations exceeds the algorithm stops automatically
def part1(eta,limit):
    temptheta=np.zeros([1,n+1])
    error=J(temptheta)
    status=False
    iteration=0
    recordvalues=np.array([]);
    while(status!=True):
        
        temptheta=temptheta-eta*(np.dot((np.dot(xdata,temptheta.T)-ydata).T,xdata))/(m)
        #print "Theta:",temptheta,"Error On:Iteration",iteration,":",error
        olderror=error
        error=J(temptheta)
        recordvalues=np.append(recordvalues,[temptheta[0,0],temptheta[0,1],error]);
        if abs(error-olderror) < 10**(-9):
            print "Converged"
            status=True
        if(iteration>limit):
            print "Fail to Converge within the limit"
            break
        iteration=iteration+1
    if(status):
        print "Following are the details of the gradient descent"
        print "Learning Rate:",eta
        print "Stopping Criteria:absolute value of difference between the initial and final error"
        print "Theta Values(Final Parameters)",temptheta
        print "The number of iterations took in order to converge is",iteration
    return temptheta,recordvalues    


# recvalue is the variable used for storing the current value of the theta which we will use later

theta,recval=part1(eta,2000)


# part 2 of the problem


def part2(theta,eta):

    dataplot, = plt.plot(xdata[:,1],ydata,'bo',label="Data plot")
    linemaking = list(map(lambda x: theta.dot(x), xdata))
    line, = plt.plot(xdata[:, 1], linemaking,'y',label="Hypothesis plot")
    plt.xlabel("Acidity of wine")
    plt.ylabel("Density of wine")
    plt.title("Plotting Line")
    tit='eta='+str(eta)
    plt.suptitle(tit,fontsize=15)
    plt.legend(handles=[dataplot, line])
    #plt.savefig(os.path.join("outputfolder/","linearplot"+str(eta)+".png"))
    plt.show()
    plt.close()


part2(theta,eta)


# part 3 of problem;here the recval is the previously used value


def part3(recval,eta):
    arr=np.squeeze(np.asarray(recval))
    arr=np.reshape(arr,[-1,3])
    theta0=arr[:,0]
    theta1=arr[:,1]
    error=arr[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('cost')
    plt.title("3D plot between theta and cost")
    tit='eta='+str(eta)
    plt.suptitle(tit,fontsize=15)
    x11=np.linspace(0,2,50)
    y11=np.linspace(-1,1,50)
    T0, T1 = np.meshgrid(x11,y11)
    mesh = np.c_[T0.flatten(), T1.flatten()]
    J_values = (
            np.array([J(point) for point in mesh])
            .reshape(50, 50)
        )
    surf=ax.plot_surface(T0, T1, J_values, cmap='plasma')
    fig.colorbar(surf,shrink=0.5,aspect=5)
    iteration=0
    for i in arr:
        ax.plot([i[0]], [i[1]], [J(np.array([i[0],i[1]]))], color='r', marker='+',label='points')
        plt.pause(time)
        if(iteration>50):
		break
	iteration=iteration+1
    #plt.savefig(os.path.join("outputfolder/","meshplot"+str(eta)+".png"))
    plt.show()
    plt.close()


part3(recval,eta)


#plotting the contour using the similar module just change the plot_surface to contour


def part4(recval,eta):
    arr=np.squeeze(np.asarray(recval))
    arr=np.reshape(arr,[-1,3])
    theta0=arr[:,0]
    theta1=arr[:,1]
    error=arr[:,2]
    x11=np.linspace(0,2,50)
    y11=np.linspace(-1,1,50)
    T0, T1 = np.meshgrid(x11,y11)
    mesh = np.c_[T0.flatten(), T1.flatten()]
    J_values = (
                np.array([J(point) for point in mesh])
                .reshape(50, 50)
            )
    fug=plt.figure()
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.contour(T0,T1,J_values)
    tit='eta='+str(eta)
    iteration=0
    plt.suptitle(tit,fontsize=15)
    plt.title("contour Plot between cost and theta")
        
    for i in arr:
            
        plt.plot([i[0]], [i[1]], color='r', marker='+')
        plt.pause(time)
	if(iteration>50):
		break
        temp2=i	
	iteration=iteration+1;
    #plt.savefig(os.path.join("outputfolder/","contourplot"+str(eta)+".png"))
    plt.show()
    plt.close()



part4(recval,eta)


# Trying for different values of eta


def part5():
    etavalues=[0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5]
    for i in etavalues:
        theta,reco=part1(i,500)
        part2(theta,i)
        part3(reco,i)
        part4(reco,i)


part5()






