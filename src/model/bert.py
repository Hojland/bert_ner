import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from itertools import chain
from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import mlflow.pyfunc
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from optuna import pruners, trial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import (
    AdamW,
    ConvBertConfig,
    ConvBertForSequenceClassification,
    ConvBertTokenizerFast,
    TextClassificationPipeline,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback, TensorBoardCallback

from model import model_utils, text_utils
from settings import settings
from utils import eval_utils

logger = logging.getLogger(__name__)


@dataclass
class BertParams:
    """Class for keeping track of an params."""

    unique_tags: list = field(default_factory=list)
    id2tag: dict = field(default_factory=dict)
    tag2id: dict = field(default_factory=dict)
    epochs: int = 3
    n_cpu: int = 8
    max_len: int = 192
    learn_rate: float = 5e-5
    batch_size: int = 128  # would set to 64 if not for fp16=True
    warmup_proportion: float = 0.1

    def to_dict(self):
        return asdict(self)

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)


class BertTokenBase:
    def __init__(self, params: BertParams = BertParams()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.tokenizer = self.load_tokenizer()
        self.sentencizer = self.load_sentencizer()
        self.config = self.load_config()

    def load_tokenizer(self):
        tokenizer = ConvBertTokenizerFast.from_pretrained("sarnikowski/convbert-medium-small-da-cased")
        return tokenizer

    def load_sentencizer(self):
        nlp = spacy.load("da_core_news_sm", exclude=["parser"])
        nlp.enable_pipe("senter")  # can disable later when speed is not important
        return nlp

    def load_config(self):
        config = ConvBertConfig(
            label2id=self.params.tag2id, id2label=self.params.id2tag, max_position_embeddings=1580
        ).from_pretrained("sarnikowski/convbert-medium-small-da-cased", finetuning_task="ner")
        return config

    class NerDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    def preprocess(self, text: str):
        text = str(text.encode("utf-8"), "utf-8")
        text = text_utils.clean_text(text)
        sent_encoding_list = []
        sent_offset_mapping_list = []
        sent_text_list = []
        for sent in self.sentencizer(text).sents:
            encodings = self.tokenizer(
                sent.text, is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True
            )
            offset_mapping = encodings.pop("offset_mapping")
            sent_offset_mapping_list.append(offset_mapping)
            sent_text_list.append([sent.text[start:stop] for start, stop in offset_mapping])
            sent_encoding_list.append(encodings)
        return sent_encoding_list, sent_offset_mapping_list, sent_text_list


class BertTokenTrainer(BertTokenBase):
    # https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
    def __init__(self, params: BertParams = BertParams()):
        super().__init__(params)
        self.model = self.load_model()

    def load_model(self):
        model = ConvBertForSequenceClassification(self.config).from_pretrained(
            "sarnikowski/convbert-medium-small-da-cased",
            num_labels=len(self.params.tag2id),
            label2id=self.params.tag2id,
            id2label=self.params.id2tag,
        )
        return model

    def from_mlflow(self):
        mlflow_load = mlflow.pyfunc.load_model(f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}")
        self.model = mlflow_load._model_impl.python_model.model
        self.tokenizer = mlflow_load._model_impl.python_model.tokenizer

    def encode_tags(self, tags, encodings):
        """ THIS IS FOR TOKEN OR (SUB) SEQUENCE CLASSIFICATION"""
        labels = [[self.params.tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels

    def set_trainings_args(self, model_name: str = "bert_email_router", num_warmup_steps: int = 400, hp_tun: bool = True):
        fp16 = True if self.device.type == "cuda" else False
        if hp_tun:
            save_strategy = "no"
        else:
            save_strategy = "steps"

        self.training_args = TrainingArguments(
            output_dir=model_name,
            num_train_epochs=self.params.epochs,
            per_device_train_batch_size=self.params.batch_size,
            warmup_steps=num_warmup_steps,
            logging_steps=20,
            fp16=fp16,  # activate bit 16 trainer - allowing  a double of batch size
            learning_rate=self.params.learn_rate,
            disable_tqdm=False,
            logging_dir=model_utils.default_logdir_tf(),  # logs for tensorboard
            evaluation_strategy="steps",
            prediction_loss_only=False,
            do_eval=True,
            eval_steps=500,
            save_strategy=save_strategy,
            save_steps=500,
            save_total_limit=10,
        )
        return self.training_args

    def train(
        self,
        texts: List,
        tags: List,
        model_name: str = "bert_email_router",
        last_checkpoint: Union[str, bool] = None,
        experiment_name: str = None,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
    ):
        """[train the model]

        Args:
            texts (List): [list of text sequences]
            tags (List): [list of list of tags to train  on]
            model_name (str, optional): [name  of the model, used for outputdir and mlflow name]. Defaults to "bert_ner_test".
            last_checkpoint (Union[str, bool], optional): [if boolean true then keep modelling from last checkpoint or if modelname string  if we want to continue training on that checkpoint]. Defaults to None.
            experiment_name (str, optional): [name of mlflow experiment to start]
        """
        train_texts, val_texts, train_tags, val_tags = model_utils.train_test_split(texts, tags, test_size=0.1)
        train_encodings = self.tokenizer(
            train_texts,
            is_split_into_words=False,
            return_offsets_mapping=False,
            padding="max_length",
            max_length=self.params.max_len,
            truncation=True,
        )
        train_labels = [self.params.tag2id[tag] for tag in train_tags]
        train_data = self.NerDataset(train_encodings, train_labels)

        val_encodings = self.tokenizer(
            val_texts,
            is_split_into_words=False,
            return_offsets_mapping=False,
            padding="max_length",
            max_length=self.params.max_len,
            truncation=True,
        )
        val_labels = [self.params.tag2id[tag] for tag in val_tags]
        val_data = self.NerDataset(val_encodings, val_labels)

        num_train_steps = int(len(train_data) / self.params.batch_size * self.params.epochs)
        num_warmup_steps = int(num_train_steps * self.params.warmup_proportion)
        training_args = self.set_trainings_args(model_name=model_name, num_warmup_steps=num_warmup_steps)

        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

        if hp_space:
            self.trainer = Trainer(
                model_init=self.load_model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=training_args,
                tokenizer=self.tokenizer,
                compute_metrics=model_utils.compute_metrics,
                callbacks=[model_utils.TensorBoardCallbackWrap, model_utils.MLflowCallbackWrap],
            )
            self.trainer.remove_callback(TensorBoardCallback)
            self.trainer.remove_callback(MLflowCallback)
            best_trial = self.trainer.hyperparameter_search(
                hp_space=hp_space,
                direction="minimize",  # probably should be minimize (now it's maximize)
                backend="optuna",
                n_trials=settings.N_TRIALS,  # number of trials
                pruner=pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
                hp_name=model_utils.hp_name_optuna,
            )
            metrics = {}
            metrics["loss"] = best_trial.objective
            logger.info(f"Hyperparameter optimization finished: Best hyperparameters are {best_trial.hyperparameters}")
        else:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            else:
                checkpoint = None

            self.trainer = Trainer(
                model=self.model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=training_args,
                tokenizer=self.tokenizer,
                compute_metrics=model_utils.compute_metrics,
                callbacks=[model_utils.TensorBoardCallbackWrap, model_utils.MLflowCallbackWrap],
            )
            self.trainer.remove_callback(TensorBoardCallback)
            self.trainer.remove_callback(MLflowCallback)
            train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics

        self.model.save_pretrained(f"{model_name}/model")
        self.tokenizer.save_pretrained(f"{model_name}/tokenizer")
        self.save_model_mlflow(
            model_name=model_name,
            model_out_dir=f"{model_name}/model",
            tokenizer_out_dir=f"{model_name}/tokenizer",
        )

        max_train_samples = len(train_data)
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def eval(self, texts: List[str], tags: List[str], experiment_name: str = None):
        logger.info("*** Evaluate ***")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = -1 if self.device.type == "cpu" else torch.cuda.current_device()
        label_list = list(self.model.config.label2id.keys())

        ### Only label predictions
        pipeline = model_utils.TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

        batch_size = 4  # see how big you can make this number before OOM
        results = [
            res
            for i in range(0, len(texts), batch_size)
            for res in pipeline(
                texts[i : i + batch_size],
                is_split_into_words=False,
                return_offsets_mapping=False,
                padding="max_length",
                max_length=self.params.max_len,
                truncation=True,
            )
        ]
        pred_labels = [res["label"] for res in results]
        precision, recall, f1, support = eval_utils.multilabel_precision_recall_fscore_support(
            preds=pred_labels, labels=tags, label_list=label_list
        )
        confusion_matrix_ax, confusion_matrices = eval_utils.confusionmatrix_confusionmatrices(
            preds=pred_labels, labels=tags, label_list=label_list
        )
        ### all category predictions
        pipeline = model_utils.TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer, device=device, return_all_scores=True
        )
        results = [
            res
            for i in range(0, len(texts), batch_size)
            for res in pipeline(
                texts[i : i + batch_size],
                is_split_into_words=False,
                return_offsets_mapping=False,
                padding="max_length",
                max_length=self.params.max_len,
                truncation=True,
            )
        ]
        y_pred = np.array([[dct["score"] for dct in result] for result in results])
        roc_ax = eval_utils.plot_roc(tags, y_pred, classes=label_list, figsize=(12, 12))

        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

            mlflow.log_dict(precision, "precision.json")
            mlflow.log_dict(recall, "recall.json")
            mlflow.log_dict(f1, "f1.json")
            mlflow.log_dict(support, "support.json")
            mlflow.log_dict(confusion_matrices, "confusion_matrices.json")

            confusion_matrix_ax.get_figure().savefig("confusion_matrix.svg", transparent=True)
            mlflow.log_artifact(local_path="confusion_matrix.svg")

            roc_ax.get_figure().savefig("roc_ax.svg", transparent=True)
            mlflow.log_artifact(local_path="roc_ax.svg")

        return {"precision": precision, "recall": recall, "f1": f1, "support": support, "confusion_matrices": confusion_matrices}

    def save_model_mlflow(self, model_name: str, model_out_dir: str, tokenizer_out_dir: str, **kwargs):
        artifacts = {"tokenizer_dir": tokenizer_out_dir, "model_dir": model_out_dir}
        mlflow.pyfunc.log_model(
            artifact_path="model",
            registered_model_name=model_name,
            python_model=BertTokenPredictor(model_name=model_name, params=self.params),
            artifacts=artifacts,
            conda_env="utils/resources/conda-env.json",
            **kwargs,
        )


class BertTokenPredictor(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name: str = "bert_email_router", params: BertParams = BertParams()):
        self.model_name = model_name
        self.params = params

    def load_context(self, context):
        """ function to enable loading flow mlflow """
        self.tokenizer = ConvBertTokenizerFast.from_pretrained(
            context.artifacts["tokenizer_dir"],
            config=ConvBertConfig.from_pretrained(os.path.join(context.artifacts["tokenizer_dir"], "tokenizer_config.json")),
        )
        self.model = ConvBertForSequenceClassification.from_pretrained(context.artifacts["model_dir"], return_dict=True)
        self.model.eval()  # Put model in evaluation mode.

    def pipeline(self, text: str):
        # TODO maybe this needs senctenizing
        device = -1 if self.device.type == "cpu" else 0  # set to other device id if more (see how when you have a gpu)
        nlp = model_utils.TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, device=device)
        res = nlp(
            inputs=text,
            is_split_into_words=False,
            return_offsets_mapping=False,
            padding="max_length",
            max_length=self.params.max_len,
            truncation=True,
        )
        return res

    def eval(self, texts: List[str], tags: List[str]):
        # TODO not done, and should be made in correspondance with loading the model when instantiating this class
        # could be a new class built on Bertbase
        logger.info("*** Evaluate ***")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = -1 if self.device.type == "cpu" else torch.cuda.current_device()
        pipeline = model_utils.TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

        batch_size = 4  # see how big you can make this number before OOM
        results = [
            res
            for i in range(0, len(texts), batch_size)
            for res in pipeline(
                texts[i : i + batch_size],
                is_split_into_words=False,
                return_offsets_mapping=False,
                padding="max_length",
                max_length=self.params.max_len,
                truncation=True,
            )
        ]
        pred_labels = [res["label"] for res in results]
        precision, recall, f1, support = eval_utils.multilabel_precision_recall_fscore_support(
            preds=pred_labels, labels=test_tags, label_list=list(self.model.config.label2id.keys())
        )
        confusion_matrices = eval_utils.multilabel_confusion_matrix_labels(
            pred_labels, test_tags, label_list=list(self.model.config.label2id.keys())
        )

    def predict(self, context, text: str):
        """
        Uses mlflow to load and predict model
        Given text, returns dictionary containing predictions from the BERT model
        in various useful forms. Explicitly,
        dict = {
            'processed_text':       Text cleaned by the BERTtokenizer pipe. I.e. the input fed to BERT.
            'entities':             List of entities found. The are triples (start, end, label).
            'entity_dict':          Dictionary of entities. Keys are labels, values are lists of entities of labels.
            'raw_bert_output':      The output from BertPredictor.predict(text).
            'token_label_pairs':    The BERT output, where wordpieces are compressed back into tokens. Easier on the eyes.
        }
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text = str(text.encode("utf-8"), "utf-8")
        text = text_utils.clean_text(text)
        res = self.pipeline(text)
        return res


if __name__ == "__main__":
    from settings import settings

    mlflow.set_tracking_uri(uri=settings.MLFLOW_URI)
    text = "hello niels you little stupid fuck. Hi Emma you seem nice.  "
    text = "Nicklas, vores kære PO, har en rigtig god dag i dag, hvor der er rigtig dejligt vejr"
    MODEL_NAME = "bert_email_router"
    MODEL_STAGE = "production"

    # Loading model
    bert_pred_load = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

    # Setting model
    bert_pred = BertTokenPredictor()
    bert_pred.model = bert_pred_load._model_impl.python_model.model
    bert_pred.tokenizer = bert_pred_load._model_impl.python_model.tokenizer

    bert_pred.predict(None, text)

    bert_pred_load.predict(text)
