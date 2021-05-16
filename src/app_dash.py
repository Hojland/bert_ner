import dash
import dash_core_components as dcc
import dash_html_components as html
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import sqlalchemy
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from dashboard import attribution, dash_utils, graphs, performance_data
from settings import db_settings, settings

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash("attributions", external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

mlflow.set_tracking_uri(settings.MLFLOW_URI)

mariadb_engine = sqlalchemy.create_engine(db_settings.CONN_STR)

server = app.server

app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            value="tab-1",
            children=[
                dcc.Tab(label="Model performance", value="tab-1"),
                dcc.Tab(label="Model attributions", value="tab-2"),
            ],
        ),
        html.Div(id="tabs-content"),
    ]
)

tab_1 = html.Div(
    [
        html.Div(
            [graphs.performance_graph(mariadb_engine)],
            className="twelve columns",
        ),
        html.Div(
            [graphs.precision_recall_graph(mariadb_engine)],
            className="six columns",
        ),
        html.Div(
            [graphs.pie_chart(mariadb_engine)],
            className="five columns",
        ),
    ],
    style={
        "marginTop": "10px",
        "marginRight": "10px",
        "marginBottom": "30px",
        "marginLeft": "40px",
    },
)


md_intro = """
### Model attributions
This dashboard uses the email classifier to generate attributions for predictions. More information about the model and source code can be found at https://github.com/nuuday/bert_email_router.

"""

md_post = """

"""


tab_2 = html.Div(
    children=[
        html.Div(
            children=[
                dcc.Markdown(children=md_intro),
            ],
            style={"width": "100%", "marginBottom": "1.5em"},  # "height": 290,
            className="twelve columns",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Textarea(
                            id="input_text",
                            placeholder="Email body ...",
                            style={"width": "100%", "height": 290, "marginBottom": "1.5em"},
                        ),
                    ],
                    className="six columns",
                ),
                html.Div(
                    children=[
                        html.Label("BB technology"),
                        dcc.Dropdown(
                            id="tech_bb",
                            options=[
                                {"label": "COAX", "value": "COAX"},
                                {"label": "DSL", "value": "DSL"},
                                {"label": "Fiber", "value": "Fiber"},
                            ],
                            value="",
                            style={"width": "80%", "marginBottom": "1.5em"},
                        ),
                        html.Label("TV technology"),
                        dcc.Dropdown(
                            id="tech_tv",
                            options=[
                                {"label": "COAX", "value": "COAX"},
                                {"label": "DSL", "value": "DSL"},
                                {"label": "Fiber", "value": "Fiber"},
                            ],
                            value="",
                            style={"width": "80%", "marginBottom": "1.5em"},
                        ),
                        html.Label("True label"),
                        dcc.Dropdown(
                            id="true_label_input",
                            options=[
                                {
                                    "label": "Customer Service - COAX TV",
                                    "value": "Customer Service - COAX TV",
                                },
                                {
                                    "label": "Customer Service - DSL TV",
                                    "value": "Customer Service - DSL TV",
                                },
                                {
                                    "label": "Customer Service - Mobile mbilling",
                                    "value": "Customer Service - Mobile mbilling",
                                },
                                {
                                    "label": "Customer Service - Inkasso",
                                    "value": "Customer Service - Inkasso",
                                },
                                {
                                    "label": "Customer Service - Inkasso - Urgent",
                                    "value": "Customer Service - Inkasso - Urgent",
                                },
                                {
                                    "label": "Customer Service - Fiber",
                                    "value": "Customer Service - Fiber",
                                },
                                {"label": "Billing - DSL TV", "value": "Billing - DSL TV"},
                                {"label": "Billing - KASS", "value": "Billing - KASS"},
                                {
                                    "label": "Billing - Mobile mBilling",
                                    "value": "Billing - Mobile mBilling",
                                },
                                {
                                    "label": "Tech Support - COAX TV",
                                    "value": "Tech Support - COAX TV",
                                },
                                {
                                    "label": "Tech Support - COAX 3. level",
                                    "value": "Tech Support - COAX 3. level",
                                },
                                {
                                    "label": "Tech Support - DSL TV",
                                    "value": "Tech Support - DSL TV",
                                },
                                {
                                    "label": "Tech Support - DSL 3. level",
                                    "value": "Tech Support - DSL 3. level",
                                },
                                {
                                    "label": "Tech Support - Mobile mBilling",
                                    "value": "Tech Support - Mobile mBilling",
                                },
                                {"label": "Tech Support - KASS", "value": "Tech Support - KASS"},
                            ],
                            value="",
                            style={"width": "80%", "marginBottom": "1.5em"},
                        ),
                        html.Button(
                            "Submit",
                            id="submit_button",
                            n_clicks=0,
                            style={"marginBottom": "1.5em"},
                        ),
                    ],
                    className="six columns",
                ),
            ]
        ),
        html.Div(
            children=[
                dcc.Markdown(children=md_post),
            ],
            className="twelve columns",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Table(
                            id="pred_table",
                        ),
                    ],
                    className="three columns",
                ),
                html.Div(
                    children=[
                        html.Table(
                            id="importances",
                        ),
                    ],
                    className="nine columns",
                ),
            ],
            className="twelve columns",
        ),
    ],
    style={
        "marginTop": "10px",
        "marginRight": "10px",
        "marginBottom": "30px",
        "marginLeft": "40px",
    },
)


@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "tab-1":
        return html.Div(tab_1)
    elif tab == "tab-2":
        return html.Div(tab_2)


@app.callback(
    [
        Output(component_id="importances", component_property="children"),
        Output(component_id="pred_table", component_property="children"),
    ],
    [
        Input("submit_button", "n_clicks"),
        Input("input_text", "value"),
        Input("tech_tv", "value"),
        Input("tech_bb", "value"),
        Input("true_label_input", "value"),
    ],
)
def update_metrics(submit_button, input_text, tech_tv, tech_bb, true_label_input):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]  # To determine if n_clicks is changed.
    if "submit_button" not in changed_id:
        return ["", ""]
    else:
        attr, pred_proba, labels = attribution.get_shap_attributions(input_text, tech_tv, tech_bb, true_label_input)
        pred_class = attr.pred_class
        importances = dash_utils.generate_shap_table(attr.raw_input, attr.word_attributions, pred_class)
        pred_table = dash_utils.generate_pred_table(pred_proba, labels, pred_class, true_label_input)

        return [
            importances,
            pred_table,
        ]


if __name__ == "__main__":
    app.run_server(debug=True, port=8787, host="0.0.0.0")
