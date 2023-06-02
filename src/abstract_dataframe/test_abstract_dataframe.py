from dataclasses import dataclass

import polars as pl

from abstract_dataframe.abstract_dataframe import Columns, StaticDataframe


@dataclass(frozen=True)
class TestDataframeColumns(Columns):
    predicted_class_scores: pl.Expr = pl.col("predicted_class_scores")  # noqa: RUF009
    true_label: pl.Expr = pl.col("true_label")  # noqa: RUF009


class TestDataframe(StaticDataframe):
    def __init__(
        self, df: pl.DataFrame, cols: TestDataframeColumns, validate_on_init: bool
    ):
        self.__df = df
        self.__cols = cols

        if validate_on_init:
            self._validate()

    def unpack(self) -> tuple[pl.DataFrame, TestDataframeColumns]:
        return self.__df, self.__cols


def test_abstract_dataframe():
    input_df = pl.DataFrame(
        {"predicted_class_scores": [0.1, 0.2, 0.3], "true_label": [1, 2, 3]}
    )

    general_df = TestDataframe(
        df=input_df, cols=TestDataframeColumns(), validate_on_init=True
    )

    df, cols = general_df.unpack()
    result = df.filter(cols.predicted_class_scores > 0.2)