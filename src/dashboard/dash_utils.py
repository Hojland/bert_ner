import dash_html_components as html
import numpy as np


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def divide_chunks(l, n):
    result = []
    for i in range(0, len(l), n):
        result.append(l[i : i + n])
    return result


def generate_shap_table(words, importances, pred_class):
    word = [format_special_tokens(token) for token in words][1:-1]
    color = [get_color(imp) for imp in importances][1:-1]
    body = [html.Td(w, style={"background-color": c, "border": "0px"}) for (w, c) in zip(word, color)]
    body = divide_chunks(body, 10)
    table = html.Table(
        children=[
            html.Thead(html.Th(f"Word importances for {pred_class}", colSpan="7")),
            html.Tbody(
                [html.Tr(row) for row in body],
            ),
        ],
    )
    return table


def generate_pred_table(pred_proba, labels, pred_class, true_class_label):
    styles = [{"background-color": f"hsl({120}, {75}%, {80}%)"} if label == pred_class else {} for label in labels]
    if true_class_label != "":
        true_class_index = labels.index(true_class_label)
        for i, d in enumerate(styles):
            if i == true_class_index:
                d.update({"font-weight": "bold"})

    return html.Table(
        children=[
            html.Thead([html.Th("Predicted label"), html.Th("Probability")]),
            html.Tbody(
                [
                    html.Tr([html.Td(label), html.Td(np.round(proba.cpu().item(), 2))], style=style)
                    for label, proba, style in zip(labels, pred_proba, styles)
                    if np.round(proba.cpu().item(), 2) != 0  # and label != true_class_label
                ]
            ),
        ],
    )
