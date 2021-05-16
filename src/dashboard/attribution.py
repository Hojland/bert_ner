import logging

import mlflow
import torch
from cachetools import LRUCache, cached
from captum.attr import ShapleyValueSampling
from captum.attr import visualization as viz

from settings import settings
from utils import data_utils

cache = LRUCache(maxsize=4)
logger = logging.getLogger(__name__)


@cached(cache=cache)
def get_model(model_name: str, model_stage: str):
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_stage}")
    tokenizer = model._model_impl.python_model.tokenizer
    model = model._model_impl.python_model.model
    return model, tokenizer


def construct_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    # construct input token ids
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    # construct reference token ids
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

    return (
        torch.tensor([input_ids], device=device),
        torch.tensor([ref_input_ids], device=device),
        len(text_ids),
    )


def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
    return token_type_ids, ref_token_type_ids


def construct_input_ref_pos_id_pair(input_ids):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
    ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
    return position_ids, ref_position_ids


def construct_attention_mask(input_ids):
    return torch.ones_like(input_ids)


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def custom_forward(self, inputs):
        preds = self.model(inputs)[0]
        return torch.softmax(preds, dim=1)


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def get_shap_attributions(text, tech_tv, tech_bb, true_label):
    try:
        model, tokenizer = get_model(settings.MODEL_NAME, settings.MODEL_STAGE)
    except Exception as e:
        logger.info(f"wtf is going on here: {e}")
    ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
    sep_token_id = (
        tokenizer.sep_token_id
    )  # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence

    bert_string = data_utils.stitch_bert_string("", text, tech_tv, tech_bb)

    input_ids, ref_input_ids, _ = construct_input_ref_pair(tokenizer, bert_string, ref_token_id, sep_token_id, cls_token_id)

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)

    pred = model(input_ids)[0]
    pred_proba = torch.softmax(pred, dim=1)[0]
    model_custom = ModelWrapper(model)

    shap = ShapleyValueSampling(model_custom.custom_forward)
    attributions = shap.attribute(
        inputs=input_ids,
        baselines=ref_input_ids,
        target=torch.argmax(pred[0]),
    )

    score_vis = viz.VisualizationDataRecord(
        attributions[0, :],
        torch.softmax(pred, dim=1)[0][torch.argmax(pred[0]).cpu().numpy().item()],
        model.config.id2label[torch.argmax(pred[0]).cpu().numpy().item()],
        true_label,
        model.config.id2label[torch.argmax(pred[0]).cpu().numpy().item()],
        attributions.sum(),
        all_tokens,
        0,
    )

    labels = list(model.config.id2label.values())

    return score_vis, pred_proba, labels
