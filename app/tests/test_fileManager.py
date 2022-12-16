# Test File manager :
import sys
import os

sys.path.append('.')
sys.path.append('../')

import unittest
from objects.wine_manager import FileManager, Wine


class TestFileManager(unittest.TestCase) :

    #Test constructor
    def test_init(self):
        """
            Test the initialization of the FileManager object
        """
        manager = FileManager("./app/data/Wines.csv")
        self.assertEqual(manager.file_name,"./app/data/Wines.csv")

    #Test get_next_Id
    def test_get_next_Id_no_file(self):
        """
            Test the get_next_Id method when the file doesn't exist
        """
        manager = FileManager("./app/data/Wines_tests.csv")
        self.assertEqual(manager.get_next_Id(),-1)


    def test_get_next_Id_file_empty(self):
        """
            Test the get_next_Id method when the file is empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            pass
        manager = FileManager("./tests/Wines_tests.csv")
        self.assertEqual(manager.get_next_Id(),0)
        os.remove("./tests/Wines_tests.csv")

    
    def test_get_next_Id_file_not_empty(self):
        """
            Test the get_next_Id method when the file is not empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
        manager = FileManager("./tests/Wines_tests.csv")
        self.assertEqual(manager.get_next_Id(),1)
        os.remove("./tests/Wines_tests.csv")


    #Test read_data
    def test_read_data_no_file(self):
        """
            Test the read_data method when the file doesn't exist
        """
        manager = FileManager("./app/data/Wines_tests.csv")
        self.assertEqual(manager.read_data(),False)
    
    def test_read_data_file_empty(self):
        """
            Test the read_data method when the file is empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            pass
        manager = FileManager("./tests/Wines_tests.csv")
        self.assertEqual(manager.read_data().empty,True)
        os.remove("./tests/Wines_tests.csv")
    
    def test_read_data_file_not_empty(self):
        """
            Test the read_data method when the file is not empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
        manager = FileManager("./tests/Wines_tests.csv")
        self.assertEqual(manager.read_data().empty,False)
        os.remove("./tests/Wines_tests.csv")

    #Test write_data
    def test_write_data_no_file(self):
        """
            Test the write_data method when the file doesn't exist
        """
        wine = Wine("7","0.0","0.0","1.9","0.076","11.0","34.0","0.9978","3.51","0.56","9.4","5")
        manager = FileManager("./tests/Wines_tests.csv")
        self.assertEqual(manager.write_data(wine),True)
        os.remove("./tests/Wines_tests.csv")


    def test_write_data_file_empty(self):
        """
            Test the write_data method when the file is empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            pass
        wine = Wine("7.4","0.7","0.0","1.9","0.076","11.0","34.0","0.9978","3.51","0.56","9.4","5")
        manager = FileManager("./tests/Wines_tests.csv")
        self.assertEqual(manager.write_data(wine),True)
        with open("./tests/Wines_tests.csv", 'r') as reading_csv_file: 
            line = reading_csv_file.readline().strip('\n')
        self.assertEqual(line,"fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id")
        os.remove("./tests/Wines_tests.csv")

    def test_write_data_file_not_empty(self):
        """
            Test the write_data method when the file is not empty
        """
        with open("./tests/Wines_tests.csv", 'w') as creating_new_csv_file: 
            creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
            creating_new_csv_file.write("7.0,0.0,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,0\n")
        wine = Wine("7.4","0.7","0.0","1.9","0.076","11.0","34.0","0.9978","3.51","0.56","9.4","5")
        manager = FileManager("./tests/Wines_tests.csv")
        self.assertEqual(manager.write_data(wine),True)
        with open("./tests/Wines_tests.csv", 'r') as reading_csv_file: 
            line = reading_csv_file.readlines()
        self.assertEqual(line[2],'7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5,1\n')
        os.remove("./tests/Wines_tests.csv")