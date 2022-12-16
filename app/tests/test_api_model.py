# Test Datasets :
import sys
import os

sys.path.append('.')
sys.path.append('../')

import endpoints.main as api 
#from objects.launcher import Launcher

import unittest

class TestApiModel(unittest.TestCase):

    def test_get_serialized_model(self):
        print(os.path.exists("../data/Wines.csv"))
        #api.launcher = Launcher(data_file_name="../data/Wines.csv", save_file_name = "../data/random_forest.joblib")
        #self.assertEqual(api.get_serialized_model(), {"message" : "Model saved in /data/random_forest.joblib"})
