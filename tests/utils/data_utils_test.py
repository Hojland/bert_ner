import os

import pandas as pd
import pytest
from faker import Faker

from utils import data_utils

fake = Faker("dk_DK")


def test_preprocess_email(etray_email_data: pd.DataFrame, df_tech: pd.DataFrame):
    transfertable_path = os.path.join(os.path.dirname(__file__), "../../src/", "utils/resources/transfertable.csv")
    df = data_utils.preprocess_email(etray_email_data, df_tech, transfertable_path=transfertable_path)
    assert "final_text_bert" in list(df)
    assert "TARGET" in list(df)


def test_stitch_bert_string(random_email_subject: str, random_email_body: str, random_tech_bb: str, random_tech_tv: str):
    stitched_string = data_utils.stitch_bert_string(random_email_subject, random_email_body, random_tech_tv, random_tech_bb)
    assert isinstance(stitched_string, str)
    return stitched_string


def test_clean_string_bert(random_email_body: str):
    cleaned_string = data_utils.clean_string_bert(random_email_body)
    assert isinstance(cleaned_string, str)


def test_email_from_str():
    test_mail = fake.email()
    string = f"Here is an email in a line {test_mail}"
    assert data_utils.email_from_str(string) == test_mail
