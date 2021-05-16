import logging

import sqlalchemy


def get_experiment_tag(db_engine: sqlalchemy.engine, experiment_name_prefix: str):
    experiment_tag = db_engine.execute(
        f"""
        SELECT MAX(experiment_id)
        FROM models.experiments
        WHERE name like '{experiment_name_prefix}%%'
        """
    ).scalar()
    if experiment_tag:
        experiment_tag = int(experiment_tag)
    else:
        experiment_tag = 0
    return experiment_tag
