from objects.wine_manager import FileManager, Wine,Datasets
from model.model import RandomForestModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import os

class Launcher:

    file_manager : FileManager
    datasets : Datasets
    model : RandomForestModel
    data_file_name: str
    predict_file_name: str

    def __init__(self):
        """
            Initialize the launcher object and train the model if it doesn't exist
        """ 
        self.file_manager = FileManager("/data/Wines.csv")
        self.datasets = Datasets(self.file_manager.read_data())
        self.model = RandomForestModel()
        if os.path.exists(self.model.filepath) :
            print("Load model...")
            self.model.load()
        else: 
            print("Train model...")
            self.model.train(self.datasets)
            print("Save model...")
            self.model.save()
        # print(self.datasets.X_train.iloc[0])


    def predict_score(self,wine : Wine):
        """
            Predict the quality of a wine based on its features

            Args: wine is a Wine object

            Returns: the quality score of the wine
        """ 
        return int(self.model.predict_one(wine.df_for_prediction()))


    def add_data(self, wine : Wine):
        """
            Adds a line of data to the csv file (and update the datasets??)

            Args:
                wine (Wine): An object of type wine to add to the csv file
            
            Returns:
                bool: True if the data was added, False otherwise.
        """
        return self.file_manager.write_data(wine)


    def serialize(self):
        """
            Serializes the model (saves it)
        """ 
        print("Save model...")
        return os.path.exists(self.model.save())


    def describe(self):
        """
            Gets the prameters of the model and the performance score and return it
        """
        parameters = self.model.model.get_params(deep=True)
        score = self.model.score(self.datasets)
        return [parameters, score]


    def retrain(self):
        """
            Retrains the model and save it
        """
        print("Train model...")
        return isinstance(self.model.train(self.datasets),RandomForestClassifier)
