
from objects.wine import FileManager
from model.model import LinearRegressionModel, Datasets, RandomForestModel
from sklearn.ensemble import RandomForestClassifier

import os

class Launcher:

    file_manager : FileManager
    datasets : Datasets
    model : RandomForestModel

    def __init__(self):
        self.file_manager = FileManager("./app/data/Wines.csv")
        self.datasets = Datasets(self.file_manager.read_data())
        self.model = RandomForestModel()
        print(self.model.filepath)
        if os.path.exists(self.model.filepath) :
            print("Load model...")
            self.model.load()
        else: 
            print("Train model...")
            self.model.train(self.datasets)
            print("Save model...")
            self.model.save()

    
    def run(self):
        if os.path.exists(self.model.filepath) :
            print("Load model...")
            self.model.load()
        else: 
            print("Train model...")
            self.model.train(self.datasets)
            print("Save model...")
            self.model.save()
        print("Model Score :", self.model.score(self.datasets))
        return 1

    def describe(self):
        """
            Get the prameters of the model and the performance score and return it
        """
        parameters = self.model.model.get_params(deep=True)
        score = self.model.score(self.datasets)
        return [parameters, score]

    def retrain(self):
        """
            Retrain the model and save it
        """
        print("Train model...")
        model = self.model.train(self.datasets)
        print("Save model...")
        result_save = self.model.save()
        return os.path.exists(result_save[0]) and isinstance(model,RandomForestClassifier)


#
#
#
#