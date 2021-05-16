"""This module contains functions that move tables from maria-db to S3 and EBS.
"""
import logging
import os
import re
import sys
from pathlib import Path
from typing import List

import boto3
import jmespath
import numpy as np
import pandas as pd
import sqlalchemy
from tqdm import tqdm

from settings import db_settings
from utils import utils

logger = logging.getLogger(__name__)


def download_from_s3(folder_names: List[str], to_path: str = "data", bucket_name="maria-db-dump"):
    """Copies all objects from an s3 bucket to mounted EBS volume.
    Args:
        folder_names (list[str], optional): List of table names to copy. Defaults to [](all).
        to_path: str: relative path to download folder to
        bucket_name (str, optional): Bucket name. Defaults to "maria-db-dump".
    """
    session = boto3.Session()
    s3_client = session.client("s3")

    s3_paths = []
    for folder_name in folder_names:
        objects_res = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
        s3_paths.extend(jmespath.search("Contents[?Size > `0`].Key", objects_res))

    # remove old data
    if os.path.isdir(f"{to_path}"):
        utils.remove_contents_of_dir(f"{to_path}")

    for s3_path in tqdm(s3_paths):
        os.makedirs(f"{to_path}/{s3_path.split('/')[0]}", exist_ok=True)
        s3_client.download_file(bucket_name, s3_path, f"{to_path}/{s3_path}")


def preprocess_email(df, df_tech, transfertable_path):
    """Preprocesses emails
    - Remaps email target using the transfertable
    - Removes emails with missing body or target
    - Append columns with technology of customers TV and BB to emails
    - Creates a 'final_text' column with technology of TV, BB, email subject and email body.
    Arguments
    ---------
    df: pandas.DataFrame
        emails
    df_tech: pandas.DataFrame
        technology lookup
    transfertable_path: str
        path of csv file for remapping labels
    Returns
    -------
    pandas.DataFrame
        processed emails
    """
    # Create target column by mapping INITIAL_PROCESSED_CT_TO using the transfertable csv
    df_map = pd.read_csv(transfertable_path, sep=",", header=None)
    # Create dictionary for target mapping
    dictionary = dict(zip(df_map[0], df_map[1]))
    # Map routing targets
    df["TARGET"] = df["INITIAL_PROCESSED_CT_TO"].map(dictionary)

    # E-mails with missing body:
    logger.info("Emails with missing body: {0:.2f}%".format(100 * len(df[(pd.isnull(df["YS_EMAIL_MSG"]))]) / len(df)))
    # Fraction of emails with nan target
    logger.info("Emails with missing/incorrect target:  {0:.2f}%".format(100 * len(df[(pd.isnull(df["TARGET"]))]) / len(df)))

    # Select non empty mails and where Target is not null or Ignore
    df = df[(pd.notnull(df["YS_EMAIL_MSG"])) & (pd.notnull(df["TARGET"])) & (df["TARGET"] != "Ignore")].reset_index()

    # Remove autorouted inkasso e-mails
    df = df.loc[df["ORIG_EMAIL_TO"] != "inkasso@tdc.dk"]

    # Fill nan values with empty string
    df_tech.fillna("none", inplace=True)

    # Combine mailbox and YS emails in one column
    df["email_address"] = df["ORIG_EMAIL_FROM"].combine_first(df["YS_EMAIL_FROM"]).apply(email_from_str)
    df["email_subject"] = df["ORIG_EMAIL_SUBJECT"].combine_first(df["YS_SUBJECT"]).fillna("")
    df["YS_EMAIL_MSG"] = df["YS_EMAIL_MSG"].fillna("")

    # Create TV and BB maps for email_address -> technology
    df_tech["Email_adresse"] = df_tech["Email_adresse"].str.strip()
    df_tech = df_tech[~df_tech.duplicated("Email_adresse", "last")]
    df_tech = df_tech.set_index("Email_adresse")

    # Assign TV and BB technology
    df["tech_tv"] = df["email_address"].map(df_tech["TV_technology"]).str.lower()
    df["tech_bb"] = df["email_address"].map(df_tech["BB_technology"]).str.lower()

    # stich_bert_string some way as in fastapi
    df["final_text_bert"] = df.apply(
        lambda x: stitch_bert_string(x["email_subject"], x["YS_EMAIL_MSG"], x["tech_tv"], x["tech_bb"]),
        axis=1,
    )
    return df


def stitch_bert_string(
    email_subject: str, email_body: str, tech_tv: str, tech_bb: str
):  # tech_tv: str, tech_bb: str TODO these should be added when data is available again
    stitch_str = f" tv_{tech_tv} | bb_{tech_bb} | {email_subject} | {email_body}"
    stitch_str = clean_string_bert(stitch_str)
    return stitch_str


def clean_string_bert(input_string: str):
    """Cleans a string of links, digits.
    Arguments
    ---------
    input_string: string
        The input string to be processed
    Returns
    -------
    string
        Returns string where these have been cleaned
    """
    # Links
    input_string = re.sub(r"http\S+", "", input_string)
    # Remove excess space
    input_string = input_string.strip()
    # Remove specific subject string
    input_string = input_string.replace(
        "UK/DK: Be aware, this is an external email and may not be genuine / Vær opmærksom på, at dette er en ekstern e-mail og muligvis ikke ægte.",
        "",
    )
    return input_string


def email_from_str(line):
    """Extracts email from a string
    Arguments
    ---------
    input_string: string
        The input string to be processed
    Returns
    -------
    string
        Returns string containing only the email or None if no email is
        identified
    """
    try:
        match = re.search(r"[\w\.-]+@[\w\.-]+", line).group(0)
    except (AttributeError, TypeError):
        match = None
    return match


def unzip_files(to_path: str = "data"):
    import zipfile
    import glob

    for file_path in glob.glob("**/*.zip", recursive=True):
        file_path = Path(file_path)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(f"{to_path}/{file_path.stem}")


def get_etray_email_data(data_path: str = "data"):
    mariadb_engine = sqlalchemy.create_engine(db_settings.CONN_STR)
    df = (
        pd.read_sql(
            """SELECT CREATED_DATE,
            INITIAL_PROCESSED_CT_TO,
            YS_EMAIL_MSG,
            ORIG_EMAIL_FROM,
            YS_EMAIL_FROM,
            ORIG_EMAIL_SUBJECT,
            YS_SUBJECT,
            CASE_ID,
            ORIG_EMAIL_TO,
            CLOSED_DATE FROM input.Etray_data""",
            mariadb_engine,
        )
    ).astype({"CASE_ID": "int", "CLOSED_DATE": "datetime64", "CREATED_DATE": "datetime64"})

    df_msg = pd.read_csv(f"{data_path}/Yousee_etray_data_msg/Yousee_etray_data_msg.csv")

    # Merge msg data in to etray_data dataframe
    df_msg = df_msg.set_index("CASE_ID")
    df["YS_EMAIL_MSG_TMP"] = df["CASE_ID"].map(df_msg["YS_EMAIL_MSG"])
    df["YS_EMAIL_MSG"] = df["YS_EMAIL_MSG_TMP"].combine_first(df["YS_EMAIL_MSG"])
    df = df.drop(columns=["CREATED_DATE", "YS_EMAIL_MSG_TMP"])
    return df
