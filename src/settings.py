import os
from typing import List

import pytz
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseSettings, HttpUrl, SecretStr

load_dotenv()


class Settings(BaseSettings):
    MLFLOW_TRACKING_USERNAME: SecretStr = "MLFLOW_USERNAME"
    MLFLOW_TRACKING_PASSWORD: SecretStr = "MLFLOW_PSW"
    MLFLOW_S3_UPLOAD_EXTRA_ARGS: str = '{"ACL": "bucket-owner-full-control"}'
    MLFLOW_S3_ENDPOINT_URL: str = "https://s3.eu-central-1.amazonaws.com/"
    LOCAL_TZ = pytz.timezone("Europe/Copenhagen")
    MODEL_NAME: str = "bert_email_router"
    MODEL_STAGE: str = "production"
    MLFLOW_URI: HttpUrl = "https://mlflow.nuuday-ai.cloud/"
    ENDPOINT_PORT: int = 5000
    ENDPOINT_URL: HttpUrl = "https://emailrouter.martech.non-prod.managed-eks.aws.nuuday.nu"
    DATA_S3_BUCKET: str = "maria-db-dump"
    AWS_ACCESS_KEY_ID: SecretStr = "AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY: SecretStr = "AWS_SECRET_ACCESS_KEY"
    TOKENIZERS_PARALLELISM: str = "false"
    N_TRIALS: int = 40


class DatabaseSettings(BaseSettings):
    MARIADB_USR: str = "MARIADB_USR"
    MARIADB_PSW: SecretStr = "MARIADB_PSW"
    HOST: str = "cubus.cxxwabvgrdub.eu-central-1.rds.amazonaws.com"
    PORT: int = 3306
    DB: str = "input"
    CONN_DCT: dict = {}
    CONN_STR: str = "mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8"


db_settings = DatabaseSettings()
db_settings.CONN_DCT = {
    "user": db_settings.MARIADB_USR,
    "psw": db_settings.MARIADB_PSW.get_secret_value(),
    "host": db_settings.HOST,
    "port": db_settings.PORT,
    "db": db_settings.DB,
}
db_settings.CONN_STR = db_settings.CONN_STR.format(*db_settings.CONN_DCT.values())
settings = Settings()
