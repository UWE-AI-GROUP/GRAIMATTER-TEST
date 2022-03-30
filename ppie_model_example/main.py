'''
fastapi wrapper for dummy stroke model
'''
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import pandas as pd

random_forest = load("ppie_rf.joblib")
random_forest_bad = load("ppie_rf_bad.joblib")
app = FastAPI()


class InputData(BaseModel):
    '''
    Model for Post request data
    '''
    age: int
    blood_pressure: int
    smoker: int
    total_cholestorol: int
    model: int


@app.get("/")
async def root():
    '''
    Hello world!
    '''
    return {"message": "Hello World"}

@app.get("/predict/")
async def predict(age: int, blood_pressure:int, smoker:int, total_cholestorol: int, model:int):
    '''
    Predict from query args
    '''
    input_data = {
        'age': age,
        'blood_pressure': blood_pressure,
        'smoker': smoker,
        'total_cholestorol': total_cholestorol
    }

    input_df = pd.DataFrame(input_data, index=[1])
    if model == 1:
        probs = random_forest.predict_proba(input_df)
    else:
        probs = random_forest_bad.predict_proba(input_df)
    return {'stroke_risk': probs[0, 1]}

@app.post("/post_predict/")
async def post_predict(input_data: InputData):
    '''
    POST request prediction method
    '''
    input_df = pd.DataFrame(input_data.dict(), index=[1])
    model = input_df.loc[1, 'model']
    input_df.drop("model", axis=1, inplace=True)
    if model == 1:
        probs = random_forest.predict_proba(input_df)
    else:
        probs = random_forest_bad.predict_proba(input_df)
    return {'stroke_risk': probs[0, 1]}
