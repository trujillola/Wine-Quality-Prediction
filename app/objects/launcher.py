from objects.wine_manager import FileManager, Wine,Datasets
from model.model import RandomForestModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

import os

class Launcher:

    file_manager : FileManager
    datasets : Datasets
    model : RandomForestModel

    def __init__(self,data_file_name : str = "./app/data/Wines.csv", save_file_name : str = "./app/data/random_forest.joblib"):
        """
            Initialize the launcher object and train the model if it doesn't exist
        """ 
        self.file_manager = FileManager(data_file_name)
        data = self.file_manager.read_data()

        if isinstance(data, pd.DataFrame) :
            if data.empty :
                print("The given file is empty. Please add some data to it.")
                exit("Empty file")
            self.datasets = Datasets(self.file_manager.read_data())
        else :
            print("The given file is unknown. Please check the file name.")
            exit("Unknown file name")
            
        self.model = RandomForestModel(save_file_name)

        if os.path.exists(self.model.filepath) :
            print("Load model...")
            self.model.load()
        else: 
            print("Train model...")
            self.model.train(self.datasets)
            print("Save model...")
            self.model.save()


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
        result = self.file_manager.write_data(wine)
        self.datasets = Datasets(self.file_manager.read_data())
        return result


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
