from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


# Get data
data = pd.read_csv("./data/Wines.csv")
Y= data[["quality"]].values
A = data.drop(['quality','Id'], axis=1)
# print(A)
print(type(Y))
print(Y.shape)
# print(type(A))
# df=pd.DataFrame(A)
df =A
# Scaling data
mins=df.min()
maxs=df.max()
df=(df-mins)/(maxs-mins)
#nom des variables
keys=df.keys()
#un vin bien noté (trouver le max par exemple)
center=df.iloc[1004]
niter=0
#Nombre minimum de données locales
ndata=25
#pas de montée
pas=pasinit=0.1
#nombre max d'itérations
nitermax=30
notemax=0
while (True):
    #définition d'un fenêtre autour du centre
    #toutes les données dans cette fenêtre sont sélectionnées
    #S'il n'y a pas assez de données on augmente la taille eps de la fenêtre
    eps=0.1
    while (True):
        mydata=df
        myY=Y
        for i,key in enumerate(keys):
            malist=abs(mydata[key]-center[i])<eps*(maxs[i]-mins[i])
            mydata=mydata[malist]
            myY=myY[malist]
        if len(malist)>ndata and min(myY)!=max(myY):break
        #augmentation de la taille de la fenêtre
        eps=eps*1.5   
    #regression locale
    print(type(myY))
    print(myY.shape)

    print(mydata.to_numpy())
    regr = LinearRegression().fit(mydata.to_numpy(), myY)
    #première prédiction
    firstpredict=regr.predict([center.to_numpy()])
    if (niter>0):
        if (firstpredict<newvalue): pas=1-(1-pas)/2
        else: pas=pasinit
    n2=sum(regr.coef_[0]**2)
    print(regr.coef_)
    #mise à jour du centre par déplacement dans la direction du gradient (donné par les coefficients)
    center+=pas*(10-firstpredict[0])*regr.coef_[0]/n2
   
    niter+=1
    print("*******************************")
    print("Itération :", niter)
    newvalue= regr.predict([center])
    print("Note estimée:" ,newvalue[0][0])
    if (notemax<newvalue[0][0]):
        centermax=center
        notemax=newvalue[0][0]
    if newvalue>9.9:break
    if (niter==nitermax):break
print("***************************")
print("Meilleure Composition:")
centermax=centermax*(maxs-mins)+mins
centermax[centermax<mins]=mins[centermax<mins]
centermax[centermax>maxs]=maxs[centermax>maxs]
print(centermax)
print("Note estimée:")
print(notemax)
