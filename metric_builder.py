import yaml
import config
import abc
import pandas as pd
import numpy as np
from yaml.loader import SafeLoader
from os import listdir


def _load_yaml_preset(preset="todo"):
    preset_path = config.PATH_METRIC_CONFIGS + preset
    metrics_to_load = listdir(preset_path)
    metrics = []
    for metric in metrics_to_load:
        with open(preset_path + "/" + metric) as f:
            metrics.append(yaml.load(f, Loader=SafeLoader))
    return metrics


class Metric:
    def __init__(self, metric_config: dict):
        self._config = metric_config

    @property
    def name(self) -> str:
        return self._config.get("name", "default_value")

    @property
    def type(self) -> str:
        return self._config.get("type", "default_metric_type")

    @property
    def level(self) -> str:
        return self._config.get("level", "default_unit_level")

    @property
    def estimator(self) -> str:
        return self._config.get("estimator", "default_estimator")

    @property
    def numerator(self) -> dict:
        return self._config.get("numerator", {"aggregation_field": "default_value"})

    @property
    def denominator(self) -> dict:
        return self._config.get("denominator", {"aggregation_field": "default_value"})

    @property
    def numerator_aggregation_field(self) -> str:
        return self.numerator.get("aggregation_field", "default_value")

    @property
    def denominator_aggregation_field(self) -> str:
        return self.denominator.get("aggregation_field", "default_value")

    @property
    def numerator_aggregation_function(self) -> callable:
        return self._map_aggregation_function(self.numerator.get("aggregation_function"))

    @property
    def denominator_aggregation_function(self) -> callable:
        return self._map_aggregation_function(self.denominator.get("aggregation_function"))

    @staticmethod
    def _map_aggregation_function(aggregation_function: str) -> callable:
        mappings = {
            "count_distinct": pd.Series.nunique,
            "sum": np.sum
        }
        if aggregation_function not in mappings:
            raise ValueError(f"{aggregation_function} not found in mappings")
        return mappings[aggregation_function]
    
    @property
    def numerator_conditions(self) -> dict:
        return self._config.get("numerator_conditions", {"condition_field": "default_value", 
                                                         "comparison_value": "default_value"})
    
    @property
    def numerator_conditions_condition_field(self) -> str:
        return self.numerator_conditions.get("condition_field", "default_value")
    
    @property
    def numerator_conditions_comparison_sign(self) -> callable:
        return self._map_comparison_sign_function(self.numerator_conditions.get("comparison_sign"))
    
    @staticmethod
    def _map_comparison_sign_function(comparison_sign: str) -> callable:
        mappings = {
            "equal": "==",
            "not_equal": "!="
        }
        if comparison_sign not in mappings:
            raise ValueError(f"{comparison_sign} not found in mappings")
        return mappings[comparison_sign]

    @property
    def numerator_conditions_comparison_value(self) -> str:
        return self.numerator_conditions.get("comparison_value", "default_value")
    


    @property
    def denominator_conditions(self) -> dict:
        return self._config.get("denominator_conditions", {"condition_field": "default_value", 
                                                         "comparison_value": "default_value"})
    
    @property
    def denominator_conditions_condition_field(self) -> str:
        return self.denominator_conditions.get("condition_field", "default_value")
    
    @property
    def denominator_conditions_comparison_sign(self) -> callable:
        return self._map_comparison_sign_function(self.denominator_conditions.get("comparison_sign"))

    @property
    def denominator_conditions_comparison_value(self) -> str:
        return self.denominator_conditions.get("comparison_value", "default_value")
    


class CalculateMetric:
    def __init__(self, metric: Metric):
        self.metric = metric

    def __call__(self, df):
        if self.metric.numerator_conditions_condition_field == 'default_value' and self.metric.denominator_conditions_condition_field == 'default_value':
            df = df.dropna(subset = [self.metric.numerator_aggregation_field, self.metric.denominator_aggregation_field])
        
        elif self.metric.numerator_conditions_condition_field != 'default_value' and self.metric.denominator_conditions_condition_field == 'default_value':
            df = df.dropna(subset = [self.metric.numerator_aggregation_field, self.metric.denominator_aggregation_field])\
                .query(f'{self.metric.numerator_conditions_condition_field} {self.metric.numerator_conditions_comparison_sign} "{self.metric.numerator_conditions_comparison_value}"')
        
        elif self.metric.numerator_conditions_condition_field == 'default_value' and self.metric.denominator_conditions_condition_field != 'default_value':
            df = df.dropna(subset = [self.metric.numerator_aggregation_field, self.metric.denominator_aggregation_field])\
                .query(f'{self.metric.denominator_conditions_condition_field} {self.metric.denominator_conditions_comparison_sign} "{self.metric.denominator_conditions_comparison_value}"')
        
        elif self.metric.numerator_conditions_condition_field != 'default_value' and self.metric.denominator_conditions_condition_field != 'default_value':
             df = df.dropna(subset = [self.metric.numerator_aggregation_field, self.metric.denominator_aggregation_field])\
                .query(f'{self.metric.numerator_conditions_condition_field} {self.metric.numerator_conditions_comparison_sign} "{self.metric.numerator_conditions_comparison_value}" & \
                       {self.metric.denominator_conditions_condition_field} {self.metric.denominator_conditions_comparison_sign} "{self.metric.denominator_conditions_comparison_value}"')
        
        return df.dropna(subset = [self.metric.numerator_aggregation_field, self.metric.denominator_aggregation_field])\
            .groupby([config.VARIANT_COL, self.metric.level]).apply(
                lambda df: pd.Series({
                    "num": self.metric.numerator_aggregation_function(df[self.metric.numerator_aggregation_field]),
                    "den": self.metric.denominator_aggregation_function(df[self.metric.denominator_aggregation_field]),
                    "n": pd.Series.nunique(df[self.metric.level])
                    })
                    ).reset_index()