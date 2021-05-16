from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import interp
from sklearn.metrics import (
    auc,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def plot_roc(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    classes: List[str],
    title="ROC Curves",
    plot_micro=True,
    plot_macro=True,
    classes_to_plot=None,
    ax=None,
    figsize=None,
    cmap="nipy_spectral",
    title_fontsize="large",
    text_fontsize="medium",
):
    """Generates the ROC curves from labels and predicted scores/probabilities
    Args:
        y_true (array-like, shape (n_samples, n_classes)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".
        plot_micro (boolean, optional): Plot the micro average ROC curve.
            Defaults to ``True``.
        plot_macro (boolean, optional): Plot the macro average ROC curve.
            Defaults to ``True``.
        classes_to_plot (list-like, optional): Classes for which the ROC
            curve should be plotted. e.g. [0, 'cold']. If given class does not exist,
            it will be ignored. If ``None``, all classes will be plotted. Defaults to
            ``None``
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    probas = y_probas

    if classes_to_plot is None:
        classes_to_plot = classes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    fpr_dict = dict()
    tpr_dict = dict()

    indices_to_plot = np.in1d(classes, classes_to_plot)
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(
                fpr_dict[i],
                tpr_dict[i],
                lw=2,
                color=color,
                label="ROC curve of class {0} (area = {1:0.2f})" "".format(classes[i], roc_auc),
            )

    if plot_micro:
        binarized_y_true = label_binarize(y_true, classes=classes)
        if len(classes) == 2:
            binarized_y_true = np.hstack((1 - binarized_y_true, binarized_y_true))
        fpr, tpr, _ = roc_curve(binarized_y_true.ravel(), probas.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            label="micro-average ROC curve " "(area = {0:0.2f})".format(roc_auc),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

    if plot_macro:
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[x] for x in range(len(classes))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(
            all_fpr,
            mean_tpr,
            label="macro-average ROC curve " "(area = {0:0.2f})".format(roc_auc),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=text_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc="lower right", fontsize=text_fontsize)
    return ax


def plot_cm(cm: np.ndarray, label_list: List[str], figsize=(10, 10)):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)
    cm = pd.DataFrame(cm, index=label_list, columns=label_list)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=figsize)
    return sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt="", ax=ax)


def confusionmatrix_confusionmatrices(preds: List[str], labels: List[str], label_list: List[str]):
    def add_label(*args, label_list: List[str]):
        outs = []
        for arg in args:
            outs.append(dict(zip(label_list, arg)))
        return tuple(out for out in outs)

    confusion_matricks = confusion_matrix(labels, preds, labels=label_list)
    confusion_matrix_img = plot_cm(confusion_matricks, label_list=label_list)
    confusion_matrices = multilabel_confusion_matrix(labels, preds, labels=label_list)
    confusion_matrices = confusion_matrices.tolist()
    confusion_matrices = add_label(confusion_matrices, label_list=label_list)
    return confusion_matrix_img, confusion_matrices[0]


def multilabel_precision_recall_fscore_support(preds: List[str], labels: List[str], label_list: List[str]):
    def add_label(*args, label_list: List[str]):
        outs = []
        for arg in args:
            outs.append(dict(zip(label_list, arg)))
        return tuple(out for out in outs)

    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=label_list)
    support = [int(sup) for sup in support]
    precision, recall, f1, support = add_label(precision, recall, f1, support, label_list=label_list)
    return precision, recall, f1, support
