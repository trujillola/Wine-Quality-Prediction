from fastapi import APIRouter, Header, Response
from objects.wine_manager import Wine
import os
from objects.launcher import Launcher


"""
    Describe the parameters of this api file
"""
router = APIRouter(
    prefix='/api/model',
    tags = ['api/model']
)

launcher = Launcher()

# vin = Wine(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0)

# async def read_items(x_token : Optional[List[str]] = Header(None)):
#     """
#         Get the token
#     """
#     return {"X-Token values" : x_token}

@router.get("/")
async def get_serialized_model():
    """
        Save the model in a file
    """
    if  launcher.serialize():
        return {"message" : "Model saved in "+launcher.model.filepath}
    else :
        return {"message" : "An error occured while saving the model. Please check the paths."}


@router.get("/description")
async def get_model_description():
    """
        Get the description of the model and its training score
    """
    parameters,score = launcher.describe()
    if isinstance(parameters,dict) and  isinstance(score,float) and score >= 0 and score <= 1 :
        parameters['score_model'] = score
        return parameters
    else : 
        return {"message" : "An error occured while retrieving the parameters."}


@router.put("/")
async def add_new_entry(wine : Wine):
    """
        Add a new Wine Entry in the CSV
    """
    # add a new wine entry
    if launcher.add_data(wine):
        return {"message" : "Succeed"}
    else : 
        return {"message" : "An error occured adding the new entry"}

@router.post("/retrain")
async def train_model():
    ##train model
    """
        Train the model
    """
    
    if  launcher.retrain(): 
        return {"message" : "Succeed"}
    else : 
        return {"message" : "An error occured training the model"}