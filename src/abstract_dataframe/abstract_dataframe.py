from abc import abstractmethod

import polars as pl


class Columns:
    ...


class StaticDataframe:
    @abstractmethod
    def __init__(self, df: pl.DataFrame, cols: Columns, validate_on_init: bool = False):
        self.__df = df
        self.__cols = cols

        if validate_on_init:
            self._validate()

    @abstractmethod
    def _validate(self) -> None:
        # For attribute in self.cols, check if it is in self.df
        for attr, value in vars(self.__cols).items():
            attr_is_private = attr.startswith("_")
            attr_exists_in_df = attr in self.__df.schema
            col_exists_in_df = self.__df.select(value).shape[1] == 1

            if not attr_is_private and not (attr_exists_in_df and col_exists_in_df):
                raise AttributeError(f"{attr} not in self.df")

    def unpack(self) -> tuple[pl.DataFrame, Columns]:
        return self.__df, self.__cols
