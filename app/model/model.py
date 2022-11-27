#Linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load



class Datasets:

    X_train : list
    y_train : list
    X_test : list
    y_test : list

    def __init__(self, data):
        # Select the columns corresponding to the features describing the wine
        X = data.drop(['quality', 'Id'], axis=1)

        # Select the target variable column    
        y = data["quality"]

        # Split the dataset into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        

# Create a linear regression object
class LinearRegressionModel:

    model : LinearRegression
    score : float
    filepath : str

    def __init__(self, data : Datasets):
        self.filepath = "./app/data/regression.joblib"
        self.model = LinearRegression().fit(data.X_train, data.y_train)
        self.score = self.model.score(data.X_test, data.y_test)
        print("Score : ",self.score)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.score

    def save(self):
        print(f"Saving model in {self.filepath}...")
        dump(self.model, self.filepath)

    def load(self):
        print(f"Loading model from {self.filepath}...")
        self.model = load(self.filepath)



# Create a linear regression object
class RandomForestModel:

    model : RandomForestClassifier
    model_score : float
    filepath : str

    def __init__(self):
        self.filepath = "./app/data/random_forest.joblib"
        self.model = RandomForestClassifier()

    def train(self,data : Datasets):
        self.model.fit(data.X_train, data.y_train)

    def predict(self, data : Datasets):
        return self.model.predict(data.X_test)

    def score(self,data : Datasets):
        self.model_score = self.model.score(data.X_test, data.y_test)
        return self.model_score

    def save(self):
        # print(f"Saving model in {self.filepath}...")
        dump(self.model, self.filepath)

    def load(self):
        # print(f"Loading model from {self.filepath}...")
        self.model = load(self.filepath)
