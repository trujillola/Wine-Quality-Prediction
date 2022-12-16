# Test Wine : :
import sys
import os

sys.path.append('.')
sys.path.append('../')

import unittest
from objects.wine_manager import FileManager, Datasets , Wine
import pandas as pd

class TestDatasets(unittest.TestCase) :

    def test_init(self):
        """
            Test the initialization of the Datasets object
        """
        wine = Wine("7.4","0.7","0.0","1.9","0.076","11.0","34.0","0.9978","3.51","0.56","9.4","5")
        self.assertEqual(wine.fixed_acidity, "7.4")
        self.assertEqual(wine.volatile_acidity, "0.7")
        self.assertEqual(wine.citric_acid, "0.0")
        self.assertEqual(wine.residual_sugar, "1.9")
        self.assertEqual(wine.chlorides, "0.076")
        self.assertEqual(wine.free_sulfur_dioxide, "11.0")
        self.assertEqual(wine.total_sulfur_dioxide, "34.0")
        self.assertEqual(wine.density, "0.9978")
        self.assertEqual(wine.pH, "3.51")
        self.assertEqual(wine.sulphates, "0.56")
        self.assertEqual(wine.alcohol, "9.4")
        self.assertEqual(wine.quality, "5") 
        self.assertEqual(wine.Id,None)

    # def test_df_for_prediction(self):
    #     #Dans cet objet wine il n'y a pas '__pydantic_initialised__' parce qu'il n'a pas été sérialisé avec Json 
    #     wine = Wine(7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4).toJson()
    #     self.assertEqual( wine.df_for_prediction().equals(pd.DataFrame(data = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]], columns =  ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])), True)

    def test_to_csv(self) :
        """
            Test the to_csv method
        """
        wine = Wine(7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4,5.0)
        self.assertEqual(wine.to_csv(),"7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5.0")

