# Ingest email classifier performance to DB for dashboard
import mlflow
import pandas as pd
import sqlalchemy

from settings import db_settings, settings
from utils import data_utils, utils


def performance_data(model, lower_bound):
    folder_names = ["ml-yousee-nlp-email-labelling"]
    data_utils.download_from_s3(folder_names=folder_names, bucket_name=settings.DATA_S3_BUCKET, to_path="data")
    data_utils.unzip_files(to_path="data")

    df = data_utils.get_etray_email_data(data_path="data")
    df_tech = pd.read_csv("data/Etray_emails_technology/Etray_emails_technology.csv")
    df = data_utils.preprocess_email(df, df_tech, transfertable_path="utils/resources/transfertable.csv")

    df = df.loc[:, ["final_text_bert", "CLOSED_DATE", "TARGET"]]

    df["CLOSED_DATE"] = df["CLOSED_DATE"].astype("datetime64").apply(lambda x: x.date()).astype("datetime64")

    df = df.loc[df.CLOSED_DATE.astype("datetime64") > lower_bound]

    if len(df) > 0:
        df[["pred_label", "pred_proba"]] = df.final_text_bert.apply(lambda x: pd.Series(list(model.predict(x)[0].values())))

        df["true_positive"] = (df.TARGET == df.pred_label).astype("int")

        return df
    else:
        logger.info("No new data today")
        return pd.DataFrame()


def main():
    mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)
    MODEL_NAME = "bert_email_router"
    MODEL_STAGE = "production"

    mariadb_engine = sqlalchemy.create_engine(db_settings.CONN_STR)

    # Loading model
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    date_lower_bound = (
        pd.read_sql("SELECT max(CLOSED_DATE) FROM output.bert_email_router_performance", mariadb_engine).astype("str").values[0][0]
    )
    df = performance_data(model, date_lower_bound)

    if not df.empty:
        df.to_sql("bert_email_router_performance", mariadb_engine, schema="output", if_exists="append")


if __name__ == "__main__":
    logger = utils.get_logger(__name__)

    main()
