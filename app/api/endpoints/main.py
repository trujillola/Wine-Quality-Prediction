from fastapi import APIRouter, Header, Response
from objects.wine import Wine
import os
from objects.launcher import Launcher

"""
    Describe the parameters of this api file
"""
router = APIRouter(
    prefix='/api',
    tags = ['api']
)

launcher = Launcher()

vin = Wine(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1)

# async def read_items(x_token : Optional[List[str]] = Header(None)):
#     """
#         Get the token
#     """
#     return {"X-Token values" : x_token}

@router.post("/predict")
async def get_wine_quality(wine : Wine):
    """
        Send the score of the wine sent in parameters
        wine : An example of Wine() object
    """
    score : int = 0
    return {"score" : score}

@router.get("/predict")
async def identify_best_wine():
    """
        Send the characteristic of the best wine found by the model
    """
    return {"wine" : vin}

@router.get("/model")
async def get_serialized_model():
    """
        Send the content of the data model
    """
    if not os.path.exists('./data/model.txt'):
        #train model
        print("pas de fichier !!!!!!!!!!!!!!!!!!!!")
    with open('./data/model.txt', 'r') as f:
        data = f.read()
    return Response(content=data, media_type="PlainTextResponse")


@router.get("/model/description")
async def get_model_description():
    """
        Get the description of the model
    """
    parameters,score = launcher.describe()
    data = 'Model parameters : \n' + str(parameters) + '\nScore : ' + str(score)
    return Response(content=data, media_type="PlainTextResponse")

import json

@router.put("/model")
async def add_new_entry(wine : Wine):
    """
        Add a new Wine Entry in the CSV
    """
    ##add a new wine
    if True:
        return {"message" : "Succeed"}
    else : 
        return {"message" : "An error occured adding the new entry"}

@router.post("/model/retrain")
async def train_model():
    ##train model
    """
        Train the model
    """
    
    if  launcher.retrain(): 
        return {"message" : "Succeed"}
    else : 
        return {"message" : "An error occured training the model"}