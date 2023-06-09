from fastapi import FastAPI
from pydantic import BaseModel
import pickle   
import pandas as pd
import numpy as np

app = FastAPI()

    
with open('./linear.pkl', 'rb') as f:
    model = pickle.load(f)


class humidity(BaseModel):
    Humidity: float

@app.post('/')
async def scoring_endpoint(item:humidity):
    df = []
    df.append(item.Humidity)
    for i in range(1, 28):
        if(i == 21):
            df.append(1)
        else:
            df.append(0)
    df = np.asarray(df)
    yhat = model.predict([df])
    return {"prediction": int(yhat)}