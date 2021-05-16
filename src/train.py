import mlflow
import pandas as pd

from model import bert, model_utils
from settings import settings
from utils import data_utils, utils

logger = utils.get_logger(__name__)


def preprocess_data():
    folder_names = ["ml-yousee-nlp-email-labelling", "Etray_data"]
    data_utils.download_from_s3(folder_names=folder_names, bucket_name=settings.DATA_S3_BUCKET, to_path="data")
    data_utils.unzip_files(to_path="data")

    df = data_utils.get_etray_email_data(data_path="data")
    df_tech = pd.read_csv("data/Etray_emails_technology/Etray_emails_technology.csv")
    df = data_utils.preprocess_email(df, df_tech, transfertable_path="utils/resources/transfertable.csv")
    return df


def main():
    df = preprocess_data()
    texts, tags = df["final_text_bert"].to_list(), df["TARGET"].to_list()
    del df
    mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)

    train_texts, test_texts, train_tags, test_tags = model_utils.train_test_split(texts, tags, test_size=0.07)

    unique_tags = list(set(tag for tag in tags))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    params = bert.BertParams(unique_tags=unique_tags, tag2id=tag2id, id2tag=id2tag, epochs=3)
    trainer = bert.BertTokenTrainer(params=params)
    hp_space = model_utils.hp_space_optuna
    trainer.train(
        texts=train_texts,
        tags=train_tags,
        model_name=settings.MODEL_NAME,
        experiment_name=f"{settings.MODEL_NAME}-marthyperparams",
        last_checkpoint=None,
        hp_space=hp_space,
    )
    trainer.eval(texts=test_texts, tags=test_tags, experiment_name=f"{settings.MODEL_NAME}-marthyperparams")


if __name__ == "__main__":
    main()
