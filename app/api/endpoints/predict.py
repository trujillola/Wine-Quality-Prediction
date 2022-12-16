from fastapi import APIRouter
from objects.wine_manager import Wine
import os
from objects.launcher import Launcher

"""
    Describe the parameters of this api file
"""
router = APIRouter(
    prefix='/api/predict',
    tags = ['predict']
)

launcher = Launcher()

@router.post("/")
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

@router.get("/")
async def identify_best_wine():
    """
        Send the characteristic of the best wine found by the model
    """
    return {"wine" : "oui"}