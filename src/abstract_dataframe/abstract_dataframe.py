from abc import abstractmethod

import polars as pl


class Columns:
    ...


class StaticDataframe:
    @abstractmethod
    def __init__(self, df: pl.DataFrame, cols: Columns, validate_on_init: bool = False):
        self._df = df
        self._cols = cols

        if validate_on_init:
            self._validate()

    def _validate(self) -> None:
        # For attribute in self.cols, check if it is in self.df
        non_private_attributes = [
            attr for attr in dir(self._cols) if not attr.startswith("_")
        ]

        for attr in non_private_attributes:
            value = getattr(self._cols, attr)
            attr_exists_in_df = attr in self._df.schema
            col_exists_in_df = self._df.select(value).shape[1] == 1

            if not attr_exists_in_df or not col_exists_in_df:
                raise AttributeError(f"{attr} not in self.df")

    def unpack(self) -> tuple[pl.DataFrame, Columns]:
        return self._df, self._cols
