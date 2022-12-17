from fastapi import APIRouter
from objects.wine_manager import Wine
from objects.launcher import Launcher

"""
    Describe the parameters of this api file
"""
router = APIRouter(
    prefix='/api',
    tags = ['api']
)

launcher = Launcher()

@router.get("/predict")
async def identify_best_wine():
    """
        Send the characteristic of the best wine found by the model
    """
    wine = launcher.get_best_wine()
    return {"wine" : wine}

@router.post("/predict")
async def get_wine_quality(wine : Wine):
    """
        Returns the score of the wine sent in parameters
        wine : An example of Wine() object
        Returns : the score of the wine (int)
    """
    score : int = launcher.predict_score(wine)
    if isinstance(score,int) and score >= 0 and score <= 10 :
         return {"score" : score}
    else : 
        return {"message" : "An error occured while predicting the score."}


@router.get("/model")
async def get_serialized_model():
    """
        Save the model in a file
    """
    if  launcher.serialize():
        return {"message" : "Model saved in "+launcher.model.filepath}
    else :
        return {"message" : "An error occured while saving the model. Please check the paths."}


@router.get("/model/description")
async def get_model_description():
    """
        Get the description of the model and its training score
    """
    parameters,score = launcher.describe()
    if isinstance(parameters,dict) and  isinstance(score,float) and score >= 0 and score <= 1 :
        parameters['score_model'] = score
        parameters['model'] = "RandomForest"
        return parameters
    else : 
        return {"message" : "An error occured while retrieving the parameters."}


@router.put("/model")
async def add_new_entry(wine : Wine):
    """
        Add a new Wine Entry in the CSV
    """
    if launcher.add_data(wine):
        return {"message" : "Succeed"}
    else : 
        return {"message" : "An error occured adding the new entry"}
        

@router.post("/model/retrain")
async def train_model():
    """
        Train the model
    """
    if  launcher.retrain(): 
        return {"message" : "Succeed"}
    else : 
        return {"message" : "An error occured training the model"}