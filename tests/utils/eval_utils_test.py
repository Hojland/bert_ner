import numpy as np
import pytest

from settings import settings
from utils import eval_utils


def test_plot_roc(true_labels: list, pred_labels_proba: np.ndarray, label_list: list):
    roc_ax = eval_utils.plot_roc(true_labels, pred_labels_proba, classes=label_list, figsize=(12, 12))


def test_confusionmatrix_confusionmatrices(pred_labels: list, true_labels: list, label_list: list):
    confusion_matrix_img, confusion_matrices = eval_utils.confusionmatrix_confusionmatrices(pred_labels, true_labels, label_list)


def test_multilabel_precision_recall_fscore_support(pred_labels: list, true_labels: list, label_list: list):
    precision, recall, f1, support = eval_utils.multilabel_precision_recall_fscore_support(pred_labels, true_labels, label_list)
