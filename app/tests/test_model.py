# Test Launcher :
import sys
import os

sys.path.append('.')
sys.path.append('../')

import unittest
import numpy as np
from model.model import RandomForestModel
from sklearn.ensemble import RandomForestClassifier
from objects.wine_manager import FileManager,Datasets, Wine


class TestModel(unittest.TestCase) :

    def test_init(self):
        model= RandomForestModel("./tests/save_model_test.joblib")
        self.assertEqual(model.filepath,"./tests/save_model_test.joblib")

    def test_train(self):
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,5\n")
        self.file_manager = FileManager("./tests/Wines_tests.csv")
        datasets =  Datasets(self.file_manager.read_data())
        model= RandomForestModel("./tests/save_model_test.joblib")
        self.assertIsInstance(model.train(datasets),RandomForestClassifier)
        os.remove("./tests/Wines_tests.csv")


    def test_predict(self):
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,5\n")
        self.file_manager = FileManager("./tests/Wines_tests.csv")
        datasets =  Datasets(self.file_manager.read_data())
        model= RandomForestModel("./tests/save_model_test.joblib")
        model.train(datasets)
        self.assertIsInstance(model.predict(datasets),np.ndarray)
        os.remove("./tests/Wines_tests.csv")

    def test_predict_one(self):
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,5\n")
        self.file_manager = FileManager("./tests/Wines_tests.csv")
        datasets =  Datasets(self.file_manager.read_data())
        model= RandomForestModel("./tests/save_model_test.joblib")
        model.train(datasets)
        wine = Wine(7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4)
        self.assertEqual(model.predict(datasets)[0],5)
        os.remove("./tests/Wines_tests.csv")

    def test_score(self):
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,5\n")
        self.file_manager = FileManager("./tests/Wines_tests.csv")
        datasets =  Datasets(self.file_manager.read_data())
        model= RandomForestModel("./tests/save_model_test.joblib")
        model.train(datasets)
        self.assertEqual(model.score(datasets),1)
        os.remove("./tests/Wines_tests.csv")


    def test_save(self):
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,5\n")
        self.file_manager = FileManager("./tests/Wines_tests.csv")
        datasets =  Datasets(self.file_manager.read_data())
        model= RandomForestModel("./tests/save_model_test.joblib")
        model.train(datasets)
        self.assertIsInstance(model.save(),str)
        self.assertEqual(model.save(),"./tests/save_model_test.joblib")
        self.assertEqual(os.path.exists(model.save()),True)
        os.remove("./tests/Wines_tests.csv")
        os.remove("./tests/save_model_test.joblib")


    def test_load(self):
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,5\n")
        self.file_manager = FileManager("./tests/Wines_tests.csv")
        datasets =  Datasets(self.file_manager.read_data())
        model= RandomForestModel("./tests/save_model_test.joblib")
        model.train(datasets)
        model.save()
        self.assertIsInstance(model.load(),RandomForestClassifier)
        os.remove("./tests/Wines_tests.csv")
        os.remove("./tests/save_model_test.joblib")