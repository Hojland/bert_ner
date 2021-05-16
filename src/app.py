import json
import sys
from pathlib import Path
from zipfile import ZipFile

import boto3
import mlflow.pyfunc
from botocore.exceptions import ClientError
from cachetools import LRUCache, cached
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, EmailStr, Field

from settings import settings
from utils import data_utils

app = FastAPI()
mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)
cache = LRUCache(maxsize=4)


class EmailInput(BaseModel):
    email_sender: EmailStr = Field(None, title="The email-address of the sender")
    email_subject: str = Field(None, title="The subject line of the email", max_length=300)
    email_body: str = Field(None, title="The body of the email", max_length=1000)


class ModelOut(BaseModel):
    label: str = Field(None, title="The model prediction label")
    score: float = Field(None, title="The probability the model places on <label> being the correct output")


class HealthResponse(BaseModel):
    ready: bool


class ModelHealth(BaseModel):
    live: bool = False
    sensible_results: bool = False
    ready: bool = False
    # TODO check if results are any good, fx if it outputs the expected labels and if score is in (0,1]


@app.post("/predict", response_model=ModelOut)
async def predict(email_input: EmailInput):

    # Fetch model
    try:
        model = get_model(settings.MODEL_NAME, settings.MODEL_STAGE)
    except Exception as e:
        cache.clear()
        raise HTTPException(status_code=500, detail=str(e))

    # Predict on text
    bert_text_input = data_utils.stitch_bert_string(email_input.email_subject, email_input.email_body)
    bert_out = model.predict(bert_text_input)

    # Return prediction result
    res = ModelOut(label=bert_out[0]["label"], score=bert_out[0]["score"])
    return res


@app.get("/health")
async def get_health(response: Response):
    status = await get_health_info()
    response.status_code = 200 if status.ready else 503
    return status


@app.get("/clear_cache")
async def clear_cache():
    cache.clear()
    return True


async def get_health_info() -> HealthResponse:
    model_check = get_model_health(settings.MODEL_NAME, settings.MODEL_STAGE)

    model_check = await model_check

    return HealthResponse(ready=model_check.ready)


async def get_model_health(model_name: str, model_stage: str) -> ModelHealth:
    result = ModelHealth()

    model = get_model(model_name, model_stage)

    if model:
        result.live = True
        result.sensible_results = True

    result.ready = result.live and result.sensible_results
    return result


@cached(cache=cache)
def get_model(model_name: str, model_stage: str):
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
    return model
