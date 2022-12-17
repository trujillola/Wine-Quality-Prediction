# Test Datasets :
import sys
import os

sys.path.append('.')
sys.path.append('../')

import unittest
from objects.wine_manager import FileManager, Datasets


class TestDatasets(unittest.TestCase) :
    """
        Used to test Datasets Class
    """

    def test_init(self):
        """
            Test the initialization of the Datasets object
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            for i in range(5):
                creating_new_csv_file.write("7.0,0.0,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,"+str(i)+"\n")
        manager = FileManager("./tests/Wines_tests.csv")
        data = manager.read_data()
        datasets = Datasets(data)
        self.assertNotEqual(len(datasets.X_test),0)
        self.assertNotEqual(len(datasets.X_train),0)
        self.assertNotEqual(len(datasets.y_test),0)
        self.assertNotEqual(len(datasets.y_train),0)
        os.remove("./tests/Wines_tests.csv")