
from objects.wine import FileManager
from model.model import LinearRegressionModel, Datasets, RandomForestModel
import os

class Launcher:
    def __init__(self):
        pass
    
    def launch(self):
        file_manager = FileManager("./app/data/Wines.csv")
        datasets = Datasets(file_manager.read_data())
        model = RandomForestModel()
        if os.path.exists(model.filepath) :
            print("Load model...")
            model.load()
        else: 
            print("Train model...")
            model.train(datasets)
            print("Save model...")
            model.save()
        print("Model Score :", model.score(datasets))
        



#
#
#
#