import dash_core_components as dcc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import sqlalchemy

from settings import db_settings, settings


def performance_graph(mariadb_engine: sqlalchemy.engine):
    df = pd.read_sql(
        """
        SELECT distinct t.CLOSED_DATE, 
	        t1.total as num_emails,
	        t1.true_positive / t1.total as recall
        FROM output.bert_email_router_performance t
        LEFT JOIN (
	        SELECT CLOSED_DATE, 
		        sum(true_positive) as true_positive, 
		        count(true_positive) as total 
	        from output.bert_email_router_performance group by CLOSED_DATE
	    ) t1
        ON t1.CLOSED_DATE = t.CLOSED_DATE
        order by CLOSED_DATE
    """,
        mariadb_engine,
    )

    return dcc.Graph(
        id="performance-graph",
        figure={
            "data": [
                go.Bar(
                    x=df.CLOSED_DATE,
                    y=df.num_emails,
                    name="Count",
                    yaxis="y",
                    marker_color="rgba(34, 139, 34, .3)",
                ),
                go.Scatter(
                    x=df.CLOSED_DATE,
                    y=df.recall,
                    name="Recall",
                    yaxis="y2",
                    mode="lines",
                    marker_color="rgba(34, 139, 34, 1)",
                ),
            ],
            "layout": go.Layout(
                yaxis=dict(title="Number of emails", zeroline=False, showgrid=False),
                yaxis2=dict(
                    title="Recall",
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    zeroline=False,
                    showgrid=False,
                ),
                xaxis=dict(zeroline=False, showgrid=False),
                # margin=dict(l=0),  # , r=20, t=20, b=20)
            ),
        },
    )


def precision_recall_graph(mariadb_engine: sqlalchemy.engine):

    df_recall = pd.read_sql(
        """
        SELECT distinct t.pred_label as pred_label,
	        t1.true_positive / t1.total as recall
        FROM output.bert_email_router_performance t
        LEFT JOIN (
	        SELECT pred_label, 
		        sum(true_positive) as true_positive, 
		        count(true_positive) as total 
	        from output.bert_email_router_performance group by pred_label
	    ) t1
        ON t1.pred_label = t.pred_label
    """,
        mariadb_engine,
    )

    df_precision = pd.read_sql(
        """
        SELECT distinct t.TARGET as true_label,
	        t1.true_positive / t1.total as `precision`
        FROM output.bert_email_router_performance t
        LEFT JOIN (
	        SELECT TARGET, 
		        sum(true_positive) as true_positive, 
		        count(true_positive) as total 
	        from output.bert_email_router_performance group by TARGET
	    ) t1
        ON t1.TARGET = t.TARGET
    """,
        mariadb_engine,
    )

    df = df_recall.set_index("pred_label").join(df_precision.set_index("true_label"), how="outer").sort_index(ascending=False)

    return dcc.Graph(
        id="recall-graph",
        figure={
            "data": [
                go.Bar(
                    x=df.recall.values,
                    y=df.index.values,
                    orientation="h",
                    marker=dict(
                        color="rgba(34, 139, 34, 1)",
                    ),
                    # text=df_grouped.num_emails.values,
                    # textposition="auto",
                    name="Recall",
                ),
                go.Bar(
                    x=df.precision.values,
                    y=df.index.values,
                    orientation="h",
                    marker_color="rgba(34, 139, 34, .5)",
                    name="Precision",
                ),
            ],
            "layout": go.Layout(
                xaxis=dict(
                    overlaying="x",
                    side="bottom",
                    zeroline=False,
                    showgrid=False,
                ),
                yaxis=dict(
                    zeroline=False,
                    showgrid=False,
                    automargin=True,
                    side="right",
                ),
                barmode="group",
                bargap=0.25,  # gap between bars of adjacent location coordinates.
                bargroupgap=0.1,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=5),
                height=800,
                width=750,
                hovermode="y",
            ),
        },
    )


def pie_chart(mariadb_engine: sqlalchemy.engine):

    df = pd.read_sql(
        """
        SELECT TARGET as true_label, 
            count(*) as num_emails
        FROM output.bert_email_router_performance
        group by TARGET
    """,
        mariadb_engine,
    )

    return dcc.Graph(
        id="recall-graph",
        figure={
            "data": [
                go.Pie(
                    labels=df.true_label.values,
                    values=df.num_emails.values,
                    textposition="inside",
                    textinfo="label+percent",
                )
            ],
            "layout": go.Layout(
                margin=dict(l=5),
                height=800,
                width=750,
                uniformtext_minsize=12,
                uniformtext_mode="hide",
            ),
        },
    )
