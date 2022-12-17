from dataclasses import dataclass
import pandas as pd
import os
from sklearn.model_selection import train_test_split


@dataclass
class Wine():
    """
        Describe the Wine Class
    """

    fixed_acidity : float
    volatile_acidity : float
    citric_acid : float
    residual_sugar : float
    chlorides : float
    free_sulfur_dioxide : float
    total_sulfur_dioxide : float
    density : float
    pH : float
    sulphates : float
    alcohol : float
    quality : int = None
    Id : int =  None

    def __init__(self, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol,quality=None,Id=None):
        """
            Initailize a new Wine Class with its parameters
        """
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        self.quality = quality
        self.Id = Id
    
    def df_for_prediction(self):
        """
            Creates the dataframe for prediction

            Returns: the dataframe for prediction with the wine features
        """    
        return pd.DataFrame([self.__dict__]).drop(['quality', 'Id','__pydantic_initialised__'], axis=1)

    def to_csv(self):
        """
        Reads the csv file and returns the data in a pandas dataframe

        Returns:
            The read data if the csv file has been read, False otherwise.
        """
        csv_string =  f"{self.fixed_acidity},{self.volatile_acidity},{self.citric_acid},{self.residual_sugar},{self.chlorides},{self.free_sulfur_dioxide},{self.total_sulfur_dioxide},{self.density},{self.pH},{self.sulphates},{self.alcohol},{self.quality}"
        return csv_string


class Datasets:
    """
        Describe the Datasets Class
    """

    X_train : list
    y_train : list
    X_test : list
    y_test : list

    def __init__(self, data):
        """
            Initialize a new Dataset with its parameters
        """
        data = data.rename(columns={"fixed acidity": "fixed_acidity", "volatile acidity": "volatile_acidity", "citric acid": "citric_acid", "residual sugar": "residual_sugar", "free sulfur dioxide": "free_sulfur_dioxide", "total sulfur dioxide": "total_sulfur_dioxide"})

        # Select the target variable column    
        y = data["quality"]

        # Select the columns corresponding to the features describing the wine
        X = data.drop(['quality', 'Id'], axis=1)

        # Split the dataset into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
      


@dataclass
class FileManager:
    """
        Class used to interact with files
    """

    file_name: str

    def __init__(self, file_name):
        """
            Initialize a new Class
            params : Wine List csv file path
        """
        self.file_name = file_name

    def get_next_Id(self):
        """
            Gets the Id of the new wine 

        Returns:
            The next highest Id of the list of wines
        """
        data = self.read_data()
        if isinstance(data, pd.DataFrame) :
            if not data.empty :
                return int(data["Id"].max()) + 1 
            else :
                return 0
        else :
            return -1
            

    def read_data(self):
        """
        Reads the csv file and returns the data in a pandas dataframe

        Returns:
            The read data if the csv file has been read, False otherwise.
        """
        try :
            if os.path.exists(self.file_name) :
                if os.stat(self.file_name).st_size == 0:
                    return pd.DataFrame()
                else :
                    data = pd.read_csv(self.file_name)
                    return data
            else :
                return False
        except :
            print("Error while reading csv file.")
            exit()


    def write_data(self, wine : Wine):
        """

        If the file Wine.csv exists, we add the wine to the file. If not, we create the file and add the wine to it.

        Args:
            wine (Wine): object of type Wine that we want to add to a csv file

        Returns:
            True if the wine has been added to the csv file, False otherwise

        """
        if os.path.exists(self.file_name) and os.stat(self.file_name).st_size != 0:
            try :
                f = open(self.file_name, "a")
                f.write(wine.to_csv()+','+str(self.get_next_Id())+'\n')
                f.close()
                print("Wine added to the csv file")
                return True
            except :
                print("Error while writing data")
                return False
        else :
            try :
                print(self.file_name)
                with open(self.file_name, 'w') as creating_new_csv_file: 
                    creating_new_csv_file.write("fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality,Id\n")
                    creating_new_csv_file.write(wine.to_csv()+','+str(self.get_next_Id())+'\n')
                return True
            except :
                print("Error while writing data")
                return False
        