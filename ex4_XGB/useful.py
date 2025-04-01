import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='copper')

# not used
from matplotlib.colors import ListedColormap
# custom colormap
def mycmap_colors():
    colors = [(0.0, 0.0, 0.0, 1),(1.0, 0.66, 0.0, 1)]
    return ListedColormap(colors)
mycmap = mycmap_colors()


# a nonlinear function of a 2d array x
def NF(x,B,c=1):
    r=0
    if c==3:
        if np.sin((0.5*x[0]+0.5*x[1])*4*np.pi/B) * np.cos((-0.5*x[0]+0.5*x[1])*2*np.pi/B)>0:
            r=1
    if c==4:
        if np.sin((0.5*x[0]+0.5*x[1])*8*np.pi/B) * np.cos((-0.5*x[0]+0.5*x[1])*6*np.pi/B)>0:
            r=1
    return r

def plot_data(x,y):  
    plt.figure(figsize = (6,6))
    plt.scatter(x[:,0], x[:,1],s=6,c=y)
    plt.show()


def filename(s,L,TYPE=1):
    return "./DATA/"+s+"-for-DNN_type"+str(TYPE)+"_L"+str(L)+".dat"

def Standardize(x, m, s):
    """
    rescale each component using its mean and standard deviation
    """
    N = len(x)
    # assuming len(m)=len(s)=len(x[0])
    mm, ss = np.tile(m,(N,1)), np.tile(s,(N,1))
    return (x - mm) / ss


def PlotPrediction(x, y, xM, xS, Model, whichData, whichModel):

    L = 8
    B = 10

    dX = .05
    X1 = np.arange(0, 10+dX, dX)
    LG = len(X1)
    X, Y = np.meshgrid(X1, X1)
    allXY = np.reshape((np.array((X,Y)).T), (LG**2,2))
    grid = np.random.rand(LG**2, L)*B
    grid[:,:2] = allXY
    grid_r = Standardize(grid, xM, xS)

    pred = Model.predict(grid_r)

    fig, AX = plt.subplots(1, 2, figsize = (16,8))

    ax = AX[0]
    ax.scatter(x[:,0], x[:,1], c=y, s=6)
    ax.set_title("Data", fontsize = 12)

    ax = AX[1]
    pred01=np.copy(pred)
    pred01[pred>0.5]=1
    pred01[pred<=0.5]=0


    ax.pcolormesh(X1, X1, pred01.reshape((LG, LG)))
    ax.set_title(f"Result of the fit with {whichData} data and {whichModel} model", fontsize = 13)

    plt.show()

def LossAccPlot(fit, whichData, whichModel):
   fig, AX = plt.subplots(1, 2, figsize = (16,8))

   ax = AX[0]
   ax.plot(fit.history['accuracy'], label="train", c="b", ls="--")
   ax.plot(fit.history['val_accuracy'], label="valid.", c="r")
   ax.set_xlabel('epoch')
   ax.set_ylabel("Accuracy")
   ax.set_title(f"Accuracy trend for {whichData} data and {whichModel} model", fontsize = 13)
   ax.legend()

   ax = AX[1]
   ax.plot(fit.history['loss'],label="train",c="b",ls="--")
   ax.plot(fit.history['val_loss'],label="valid.",c="r")
   ax.set_xlabel('epoch')
   ax.set_ylabel("Loss")
   ax.set_title(f"Loss trend for {whichData} data and {whichModel} model", fontsize = 13)

   ax.legend()


def GenerateData(N, L, B, TYPE):
    np.random.seed(12345)

    x, y = (np.random.random((N, L)))*B, np.zeros(N)
    for i in range(N):
        # label data according to a nonlinear function "f"
        y[i] = NF(x[i], B, TYPE)

    return x, y
