#Linear regression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from objects.wine_manager import Datasets
import pandas as pd

class RandomForestModel:
    """
        Create a Random Forest Model object and methods to interact with it
    """

    model : RandomForestClassifier
    model_score : float
    filepath : str

    def __init__(self,save_file_name : str):
        """
            Initialize the Random Forest
            args : model file path 
        """
        self.filepath = save_file_name
        self.model = RandomForestClassifier(max_depth=6, n_estimators=15, min_samples_leaf=3, min_samples_split=4)

    def train(self,data : Datasets):
        """
            Trains the model on the training set
            
            args : data is a Datasets object

            returns : the trained model
        """ 
        self.model = self.model.fit(data.X_train, data.y_train)
        return self.model

    def predict(self, data : Datasets):
        """
            Predict the wine quality score of the test set
            
            args : data is a Datasets object

            returns : array of scores
        """ 
        return self.model.predict(data.X_test)

    def predict_one(self, wine: pd.DataFrame):
        """
            Returns the quality score of a wine based on its features

            Args:
                wine (Wine): An object of type wine to predict the quality score
            Returns:
                int: The quality score of the wine
        """ 
        return self.model.predict(wine)

    def score(self,data : Datasets):
        """
            returns the score of the model on the test set 

            args : data is a Datasets object

            returns : float
        """ 
        self.model_score = self.model.score(data.X_test, data.y_test)
        return self.model_score

    def save(self):
        """
            Try to save the model in the filepath
        """
        try:
            dump(self.model, self.filepath)
            return self.filepath
        except :
            return 'None'
            
    def load(self):
        """
            Loads the model from the filepath
        """ 
        try : 
            self.model = load(self.filepath)
            return self.model
        except :
            return 'None'

    def best_wine(self, wines : pd.DataFrame, qualities : pd.DataFrame):
        """
            Returns the best wine of the test set

            args : data is a Datasets object

            returns : the best wine of the test set
        """ 
        X = wines.copy()
        Y = qualities.copy()
        #Scale dataset
        mins=X.min()
        maxs=X.max()
        X=(X-mins)/(maxs-mins)
        keys=X.keys()
        # Find a good starting point : a wine with a good score
        # center=X.iloc[1104]
        center=X.iloc[Y.argmax()]
        # Parameters : maximum number of local data, step size, maximum number of iterations
        ndata=25
        pas=pasinit=0.1
        nitermax=30
        notemax=0
        niter=0
        while (True):
            # Definition of a window around the center. 
            # All the data in this window are selected. If not enough data, the window size (eps) is increased.
            eps=0.1
            while (True):
                mydata=X
                myY=Y
                for i,key in enumerate(keys):
                    malist=abs(mydata[key]-center[i]) < eps*(maxs[i]-mins[i])
                    mydata=mydata[malist]
                    myY=myY[malist]
                if len(malist)>ndata and min(myY)!= max(myY):break
                # Increase of window size
                eps=eps*1.5   
            # Local regression
            regr = LinearRegression().fit(mydata.to_numpy(), myY)
        
            # First prediction
            firstpredict=regr.predict([center.to_numpy()])
            if (niter>0):
                if (firstpredict<newvalue): pas=1-(1-pas)/2
                else: pas=pasinit
            n2=sum(regr.coef_[0]**2)

            #Update of the centre by moving in the directions of the gradient (given by the coefficients)
            center+=pas*(10-firstpredict[0])*regr.coef_[0]/n2
            niter+=1
            # print("*******************************")
            # print("Itération :", niter)
            newvalue= regr.predict([center])
            # print("Note estimée:" ,newvalue[0][0])
            if (notemax<newvalue[0]):
                centermax=center
                notemax=newvalue[0][0]
            if newvalue>9.9:break
            if (niter==nitermax):break
        # print("***************************")
        # print("Meilleure Composition:")
        centermax = centermax*(maxs-mins)+mins
        centermax[centermax<mins]=mins[centermax<mins]
        centermax[centermax>maxs]=maxs[centermax>maxs]
        centermax = dict(centermax)
        centermax['quality'] = notemax
        return centermax