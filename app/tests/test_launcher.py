# Test Launcher :
import sys
import os

sys.path.append('.')
sys.path.append('../')

import unittest
import pandas as pd
from objects.launcher import Launcher
from objects.wine_manager import FileManager, Datasets, Wine
from model.model import RandomForestModel
class TestLauncher(unittest.TestCase) :


    def test_init_no_file(self):
        """
            Test the initialization of the Launcher object when the file doesn't exist
        """
        with self.assertRaises(SystemExit) as cm:
            launcher = Launcher("./tests/Wines_tests.csv","./tests/save_model_test.joblib")
        self.assertEqual(cm.exception.code, "Unknown file name")

    def test_init_file_empty(self):
        """
            Test the initialization of the Launcher object when the file is empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            pass
        with self.assertRaises(SystemExit) as cm:
            launcher = Launcher("./tests/Wines_tests.csv","./tests/save_model_test.joblib")
        self.assertEqual(cm.exception.code, "Empty file")
        os.remove("./tests/Wines_tests.csv")

    def test_init_file_not_empty(self):
        """
            Test the initialization of the Launcher object when the file is not empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,4\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,5\n")
        launcher = Launcher("./tests/Wines_tests.csv","./tests/save_model_test.joblib")
        self.assertIsInstance(launcher.file_manager,FileManager)
        self.assertEqual(launcher.file_manager.file_name,"./tests/Wines_tests.csv")
        self.assertIsInstance(launcher.datasets,Datasets)
        self.assertIsInstance(launcher.model,RandomForestModel)
        os.remove("./tests/Wines_tests.csv")


    def test_predict_score(self):
        """
            Test the predict_score method
        """
        with open("./data/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,3\n")
        launcher = Launcher("./data/Wines_tests.csv")
        wine= Wine(7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4)
        wine.__dict__['__pydantic_initialised__'] = True
        self.assertEqual(launcher.predict_score(wine),5)
        os.remove("./data/Wines_tests.csv")

    def test_add_data(self):
        """
            Test the add_data function
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n") 
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
        launcher = Launcher("./tests/Wines_tests.csv","./tests/save_model_test.joblib")
        wine= Wine(7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0)
        self.assertEqual(launcher.add_data(wine),True)
        os.remove("./tests/Wines_tests.csv")

    def test_serialize(self):
        """
            Test the serialization of the model
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n") 
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
        launcher = Launcher("./tests/Wines_tests.csv","./tests/save_model_test.joblib")
        self.assertEqual(launcher.serialize(),True)
        os.remove("./tests/Wines_tests.csv")
        os.remove("./tests/save_model_test.joblib")

    def test_describe(self):
        """
            Test the describe function
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n") 
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
        launcher = Launcher("./tests/Wines_tests.csv","./tests/save_model_test.joblib")
        self.assertEqual(len(launcher.describe()),2)

    def test_retrain(self):
        """
            Test the retrain function
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n") 
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,2\n")
        launcher = Launcher("./tests/Wines_tests.csv","./tests/save_model_test.joblib")
        self.assertEqual(launcher.retrain(),True)

    def test_get_best_wine(self):
        """
            Test the function that calls best_wine to give the best wine composition to the api
        """
        launcher = Launcher("./data/Wines.csv","./tests/save_model_test.joblib")
        self.assertIsInstance(launcher.get_best_wine(),pd.DataFrame)