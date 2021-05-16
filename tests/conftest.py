import os

import numpy as np
import pandas as pd
import pytest
from faker import Faker

from utils import data_utils

fake = Faker("dk_DK")


@pytest.fixture()
def random_email_sender() -> str:
    return fake.email()


@pytest.fixture(params=["Mit internet driller", "Hej Yousee", "Mit tv virker ikke længere hvad gør jeg dog yousee hjææælp"])
def random_email_subject(request):
    return request.param


@pytest.fixture(
    params=[
        "Kære Yousee. Mit internet er ikke særlig godt. Jeg har målt det på jeres hjemmeside, og der er kun meeeget dårligt signal, og i hvert fald langt under de 50 mbps, jeg er blevet lovet",
        "Hej. \nJeg kan ikke forstå, hvorfor I kun viser sæson 5 af Alene i vildmarken, jeg vil gerne kunne se det hele i træk, og det kan jeg ikke, når I opfører jer sådan",
        "Hej. Jeg fatter ikke mit tv. Fuck jer",
    ]
)
def random_email_body(request):
    return request.param


@pytest.fixture(params=["DSL", "COAX"])
def random_tech_tv(request):
    return request.param


@pytest.fixture(params=["DSL", "COAX"])
def random_tech_bb(request):
    return request.param


@pytest.fixture()
def etray_email_data(random_email_sender, random_email_subject, random_email_body):
    d = {
        "CASE_ID": fake.random_number(),
        "INITIAL_PROCESSED_CT_TO": "Customer Service - Mobile",
        "ORIG_EMAIL_SUBJECT": random_email_subject,
        "ORIG_EMAIL_FROM": random_email_sender,
        "ORIG_EMAIL_TO": random_email_sender,
        "YS_SUBJECT": random_email_subject,
        "YS_EMAIL_FROM": random_email_sender,
        "YS_EMAIL_MSG": random_email_body,
    }
    return pd.DataFrame(data=d, index=[0])


@pytest.fixture()
def df_tech(random_email_sender):
    d = {
        "Email_adresse": random_email_sender,
        "TV_technology": "DSL",
        "BB_technology": "DSL",
    }
    return pd.DataFrame(data=d, index=[0])


@pytest.fixture()
def texts_tags_train(etray_email_data: pd.DataFrame, df_tech: pd.DataFrame):
    transfertable_path = os.path.join(os.path.dirname(__file__), "../src/", "utils/resources/transfertable.csv")
    df = data_utils.preprocess_email(etray_email_data, df_tech, transfertable_path=transfertable_path)
    texts, tags = df["final_text_bert"].to_list(), df["TARGET"].to_list()
    return texts, tags


@pytest.fixture()
def text_score(random_email_subject, random_email_body):
    bert_text_input = data_utils.stitch_bert_string(random_email_subject, random_email_body)
    return bert_text_input


@pytest.fixture()
def label_list():
    label_list = [
        "Billing - DSL TV",
        "Billing - KASS",
        "Billing - Mobile mBilling",
        "Customer Service - COAX TV",
        "Customer Service - DSL TV",
        "Customer Service - Fiber",
        "Customer Service - Inkasso",
        "Customer Service - Inkasso - Urgent",
        "Customer Service - Mobile mbilling",
        "Tech Support - COAX 3. level",
        "Tech Support - COAX TV",
        "Tech Support - DSL 3. level",
        "Tech Support - DSL TV",
        "Tech Support - KASS",
        "Tech Support - Mobile mBilling",
    ]
    return label_list


@pytest.fixture()
def pred_labels():
    l = [
        "Customer Service - DSL TV",
        "Tech Support - COAX TV",
        "Tech Support - DSL TV",
        "Billing - DSL TV",
        "Customer Service - DSL TV",
    ]
    return l


@pytest.fixture()
def true_labels():
    l = [
        "Customer Service - DSL TV",
        "Tech Support - COAX TV",
        "Tech Support - DSL TV",
        "Billing - DSL TV",
        "Customer Service - COAX TV",
    ]
    return l


@pytest.fixture()
def pred_labels_proba():
    arr = [
        [
            5.72351681e-04,
            1.12928925e-02,
            1.23532765e-04,
            9.35752687e-05,
            3.53073934e-04,
            7.86063969e-02,
            4.89173224e-04,
            1.37649355e-02,
            8.83477092e-01,
            5.95592835e-04,
            2.38710386e-03,
            6.61268772e-04,
            1.59983942e-03,
            8.60694636e-05,
            5.89712802e-03,
        ],
        [
            8.10859680e-01,
            1.02699831e-01,
            2.71131605e-04,
            1.35472714e-04,
            1.81815965e-04,
            1.20694600e-02,
            1.23416190e-04,
            5.53697050e-02,
            1.51211722e-02,
            6.33870877e-05,
            1.19213262e-04,
            1.43777623e-04,
            6.26863970e-04,
            5.24439442e-04,
            1.69068284e-03,
        ],
        [
            2.38721445e-01,
            7.55397022e-01,
            2.19262700e-04,
            1.14590766e-04,
            8.34097562e-04,
            1.58518646e-03,
            3.58296311e-05,
            1.26625760e-03,
            7.55490328e-04,
            2.43659961e-05,
            3.99964119e-05,
            9.29192247e-05,
            1.80865361e-04,
            3.67844332e-04,
            3.64915031e-04,
        ],
        [
            6.95931230e-05,
            1.18692929e-04,
            8.31441594e-06,
            6.04542538e-06,
            7.56648569e-06,
            9.94418085e-01,
            2.53908511e-04,
            8.77826533e-04,
            7.58015376e-04,
            2.15340784e-04,
            7.70529790e-04,
            2.35162699e-03,
            3.25566616e-05,
            9.09921710e-06,
            1.02827653e-04,
        ],
        [
            7.84299057e-03,
            2.84843403e-03,
            1.02072911e-04,
            3.06061484e-05,
            2.41961989e-05,
            9.11540468e-04,
            2.55177765e-05,
            5.13337851e-01,
            4.71813411e-01,
            2.59935150e-05,
            6.31329676e-05,
            1.21228959e-05,
            8.19027249e-04,
            1.27558887e-04,
            2.01544585e-03,
        ],
    ]
    return np.array(arr)
