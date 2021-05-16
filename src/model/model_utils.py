import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
from optuna import trial
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainerCallback
from transformers.integrations import MLflowCallback, TensorBoardCallback
from transformers.pipelines import Pipeline, PipelineException
from transformers.trainer_utils import default_hp_space_optuna

logger = logging.getLogger(__name__)


def default_logdir_tf() -> str:
    """
    Same default as PyTorch
    """

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("logs", current_time)


def train_test_split(texts: List[List], labels: List[List], test_size: float = 0.2, random_state: int = None):
    if 0 <= test_size <= 1:
        test_size = int(np.floor(test_size * len(texts)))

    test_idx = np.random.choice(len(texts), size=test_size, replace=False)
    try:
        texts = np.array(texts)
        labels = np.array(labels)
        texts_train, texts_test, labels_train, labels_test = (
            list(texts[~test_idx]),
            list(texts[test_idx]),
            list(labels[~test_idx]),
            list(labels[test_idx]),
        )
    except MemoryError:
        train_idx = np.setdiff1d(np.arange(0, len(texts)), test_idx)
        texts_train = [texts[idx] for idx in train_idx]
        texts_test = [texts[idx] for idx in test_idx]
        labels_train = [labels[idx] for idx in train_idx]
        labels_test = [labels[idx] for idx in test_idx]

    return texts_train, texts_test, labels_train, labels_test


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class TensorBoardCallbackWrap(TensorBoardCallback):
    def rewrite_logs(self, d):
        new_d = {}
        eval_prefix = "eval_"
        eval_prefix_len = len(eval_prefix)
        for k, v in d.items():
            if k.startswith(eval_prefix):
                new_d["eval/" + k[eval_prefix_len:]] = v
            else:
                new_d["train/" + k] = v
        return new_d

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = args.logging_dir

        try:
            argv = [
                "tensorboard",
                "--logdir",
                f"{os.getcwd()}/{log_dir}",
                "--port=6006",
                "--reload_multifile=True",
                "--reload_interval=10",
                "--bind_all",
            ]
            res = subprocess.Popen(argv)
            logger.info(f"Tensorboard started in port 6006")
        except Exception as e:
            logger.info(f"tensorboard not started because of error {e}")

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        self._init_summary_writer(args, log_dir)

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    self.tb_writer.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(self.tb_writer, "add_hparams"):
                self.tb_writer.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            if self.tb_writer is None:
                self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = self.rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()


class MLflowCallbackWrap(MLflowCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
        self._ml_flow.end_run()
        self._initialized = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not self._initialized:
            self.setup(args, state, model)
            if state.is_hyper_param_search:
                self._ml_flow.set_tags({"state": state.trial_name})

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                logger.info("Logging artifacts. This may take time.")
                self._ml_flow.log_artifacts(args.output_dir)

        self._ml_flow.end_run()
        self._initialized = False


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using any :obj:`ModelForSequenceClassification`. See the `sequence classification
    examples <../task_summary.html#sequence-classification>`__ for more information.

    This text classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"sentiment-analysis"` (for classifying sequences according to positive or negative
    sentiments).

    If multiple classification labels are available (:obj:`model.config.num_labels >= 2`), the pipeline will run a
    softmax over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=text-classification>`__.
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_all_scores = return_all_scores

    def _parse_and_tokenize(self, inputs, **kwargs):
        """
        THIS overwrites the Pipeline function, and should enable kwargs in tokenizer relative the huggingface pipeline
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self.tokenizer(inputs, return_tensors=self.framework, **kwargs)

        return inputs

    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) to classify.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (:obj:`str`) -- The label predicted.
            - **score** (:obj:`float`) -- The corresponding probability.

            If ``self.return_all_scores=True``, one such dictionary is returned per label.
        """
        outputs = super().__call__(*args, **kwargs)

        if self.model.config.num_labels == 1:
            scores = 1.0 / (1.0 + np.exp(-outputs))
        else:
            scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)
        if self.return_all_scores:
            return [
                [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(item)] for item in scores
            ]
        else:
            return [{"label": self.model.config.id2label[item.argmax()], "score": item.max().item()} for item in scores]


def hp_space_optuna(trial: trial.Trial) -> Dict[str, float]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 7),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [64, 128]),
        "fp16": trial.suggest_categorical("fp16", [True, False]),
        "weight_decay": trial.suggest_float("weight_decay", 0.005, 0.02, log=True),
    }


def hp_name_optuna(trial: trial.Trial) -> str:
    return f"run-{trial.number}"
