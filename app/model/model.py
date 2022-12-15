#Linear regression 
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from objects.wine_manager import Wine, Datasets
import pandas as pd

  

# Create a Random Forest Model object
class RandomForestModel:

    model : RandomForestClassifier
    model_score : float
    filepath : str

    def __init__(self,save_file_name : str):
        self.filepath = save_file_name
        self.model = RandomForestClassifier()

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
        return self.model.predict(wine.values)

    def score(self,data : Datasets):
        """
            returns th e score of the model on the test set 

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
