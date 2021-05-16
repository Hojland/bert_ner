import sys
from typing import List, Tuple

import pandas as pd
import pytest
from faker import Faker

from model import bert
from settings import settings

fake = Faker("dk_DK")


def test_bert_trainer_init(texts_tags_train: Tuple[List[str]]):
    texts, tags = texts_tags_train
    unique_tags = list(set(tag for tag in tags))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    params = bert.BertParams(unique_tags=unique_tags, tag2id=tag2id, id2tag=id2tag, epochs=3)
    trainer = bert.BertTokenTrainer(params=params)


class TestBertTrainer:
    def __init__(self, texts_tags_train: Tuple[List[str]]):
        self.texts_tags_train = texts_tags_train

    def setup(self):
        texts, tags = self.texts_tags_train
        unique_tags = list(set(tag for tag in tags))
        tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        id2tag = {id: tag for tag, id in tag2id.items()}

        params = bert.BertParams(unique_tags=unique_tags, tag2id=tag2id, id2tag=id2tag, epochs=3)
        trainer = bert.BertTokenTrainer(params=params)
        self.trainer = trainer
        self.texts = texts
        self.tags = tags

    def test_eval():
        self.trainer.eval(self.texts, self.tags)
