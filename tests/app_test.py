import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from transformers import PreTrainedModel

import app
from settings import settings


@pytest.fixture()
def random_email_input(random_email_sender, random_email_subject, random_email_body):
    return app.EmailInput(email_sender=random_email_sender, email_subject=random_email_subject, email_body=random_email_body)


@pytest.fixture()
def test_client() -> Generator:
    with TestClient(app.app) as fastapi_app:
        yield fastapi_app


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Do not run on github actions runner")
def test_api_predict_output(test_client: TestClient, random_email_input: app.EmailInput):
    response = test_client.post("/predict", json=random_email_input.dict())
    assert response.status_code == 200
    assert "score" in response.json().keys()
    assert "label" in response.json().keys()
    assert 0 < response.json()["score"] < 1


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Do not run on github actions runner")
def test_api_non_input(test_client: TestClient):
    response = test_client.post("/predict")
    assert response.status_code != 200
    assert "detail" in response.json().keys()


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Do not run on github actions runner")
def test_api_wrong_method(test_client: TestClient):
    response = test_client.get("/predict")
    assert response.status_code != 200
    assert "detail" in response.json().keys()


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Do not run on github actions runner")
def test_is_transformer_model():
    mlflow_model = app.get_model(settings.MODEL_NAME, settings.MODEL_STAGE)
    model = mlflow_model._model_impl.python_model.model
    assert isinstance(model, PreTrainedModel)


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Do not run on github actions runner")
def test_api_health(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
