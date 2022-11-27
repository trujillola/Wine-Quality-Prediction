from fastapi import FastAPI
import json

app = FastAPI()

help = [
    { 
        "method" : "POST",
        "path" : "/api/predict",
        "params" : "Wine params",
        "result" : "get a result of the wine quality sent between 0 and 10"
    },
    { 
        "method" : "GET",
        "path" : "/api/predict",
        "result" : "get the combination to identify the best wine"
    },
    { 
        "method" : "GET",
        "path" : "/api/model",
        "result" : "get the serialized model"
    },
    { 
        "method" : "GET",
        "path" : "/api/model/description",
        "result" : "Get Informations about the model used"
    },
    { 
        "method" : "PUT",
        "path" : "/api/model",
        "params" : "A wine model",
        "result" : "Add an object to the model"
    },
    { 
        "method" : "POST",
        "path" : "/api/model/retrain",
        "result" : "Train another time the model"
    }
]

@app.get("/")
async def root():
    """
        Get the Root of the project
        Describe the functionalities of the API
    """
    return {"title": "Welcome on our Wine API Quality Prediction",
            "Informations" : "Please check the folowing content to know how to use the different functionalities of the app",
            "help" : help}

{ 
    "method" : "POST",
    "path" : "/api/predict",
    "params" : "Wine params",
    "result" : "get a result of the wine quality sent between 0 and 10"
}

@app.post("/api/predict")
async def get_wine_quality(wine : Wine):
    score : int = 0
    return score