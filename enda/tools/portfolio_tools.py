"""This module contains utility functions for transforming portfolio data"""

import pandas as pd
from enda.tools.decorators import handle_multiindex
from enda.tools.timeseries import TimeSeries


class PortfolioTools:
    """
    This class contains utility methods used on portfolio DataFrames (typically DataFrames where each row
    represents a contract with either clients or producers
    """

    @staticmethod
    def portfolio_to_events(
            portfolio_df: pd.DataFrame, date_start_col: str, date_end_exclusive_col: str
    ) -> pd.DataFrame:
        """
        Converts a portfolio DataFrame where each line has a start and an and date to an event DataFrame, where
        each line is a start or end of contract event, except if the end date is null

        Given portfolio_df:
        station         start_date      excl_end_date     value_col
        station1        2023-01-01      2024-01-01        10
        station2        2023-06-01      None              30

        _portfolio_to_events(portfolio_df, 'start_date', 'excl_end_date'):
        event_type      event_date      station     value_col
        start           2023-01-01      station1    1O
        start           2023-06-01      station2    30
        end             2024-01-01      station1    10

        :param portfolio_df: The portfolio DataFrame to modify
        :param date_start_col: The column containing contracts start dates
        :param date_end_exclusive_col: The column containing contracts exclusive end date (can be null)
        :return: A DataFrame with one row per start/end event, order by event date and event type
        """
        # check that no column is named "event_type" or "event_date"
        for col in ["event_type", "event_date"]:
            if col in portfolio_df.columns:
                raise ValueError(
                    f"contracts has a column named {col}, but this name is reserved in this"
                    "function; rename your column."
                )

        columns_to_keep = [
            col
            for col in portfolio_df.columns
            if col not in [date_start_col, date_end_exclusive_col]
        ]
        events_columns = ["event_type", "event_date"] + columns_to_keep

        # compute "contract start" and "contract end" events
        start_contract_events_df = portfolio_df.copy(
            deep=True
        )  # all contracts must have a start date
        start_contract_events_df["event_type"] = "start"
        start_contract_events_df["event_date"] = start_contract_events_df[
            date_start_col
        ]
        start_contract_events_df = start_contract_events_df[events_columns]

        # for "contract end" events, only keep contracts with an end date (NaT = contract is not over)
        end_contract_events_df = portfolio_df[
            portfolio_df[date_end_exclusive_col].notna()
        ].copy(deep=True)
        end_contract_events_df["event_type"] = "end"
        end_contract_events_df["event_date"] = end_contract_events_df[
            date_end_exclusive_col
        ]
        end_contract_events_df = end_contract_events_df[events_columns]

        # concat all events together and sort them chronologically
        all_events_df = pd.concat([start_contract_events_df, end_contract_events_df])
        all_events_df.sort_values(by=["event_date", "event_type"], inplace=True)

        return all_events_df

    @staticmethod
    @handle_multiindex(arg_name="portfolio_df")
    def get_portfolio_between_dates(
            portfolio_df: pd.DataFrame,
            start_datetime: pd.Timestamp,
            end_datetime_exclusive: pd.Timestamp,
            freq: str = None,
    ) -> pd.DataFrame:
        """
        Keeps portfolio data between the specified dates.
        If the first date in portfolio is after start_datetime, we add missing dates with 0 as value.
        If the last date in portfolio is before end_datetime_exclusive, we forward fill the last present values
        until end_datetime_exclusive
        :param portfolio_df: The portfolio DataFrame. It must have a pd.DatetimeIndex with a frequency
        :param start_datetime: The start datetime from which to keep the portfolio
        :param end_datetime_exclusive: The exclusive end datetime until which to keep the portfolio
        :param freq: The frequency of the DataFrame (ex: '30min'). If not specified, the function will try to infer it
        :return: A portfolio DataFrame with values between specified dates
        """

        result_df = portfolio_df.copy(deep=True)

        if not isinstance(result_df.index, pd.DatetimeIndex):
            raise TypeError(
                f"The index of portfolio should be a pd.DatetimeIndex, but given {result_df.index.dtype}"
            )

        if freq is None:
            freq = TimeSeries.find_most_common_frequency(result_df.index)
            if freq is None:
                raise ValueError(
                    "No freq has been provided, and it could not be inferred"
                    "from the index itself. Please set it or check the data."
                )

        if start_datetime and result_df.index.min() > start_datetime:
            # add empty portfolio at the beginning with correct frequency
            new_index = pd.date_range(
                start_datetime,
                result_df.index.min(),
                inclusive='left',
                freq=freq,
                name=result_df.index.name
            )
            new_df = pd.DataFrame(index=new_index,
                                  data=[[0 for _ in range(len(result_df.columns))] for _ in range(len(new_index))],
                                  columns=result_df.columns)
            result_df = pd.concat([new_df, result_df])

        if (
                end_datetime_exclusive is not None
                and result_df.index.max() < end_datetime_exclusive
        ):
            # add days at the end, with the same portfolio as the last available day
            new_index = pd.date_range(
                result_df.index.max(),
                end_datetime_exclusive,
                inclusive='neither',
                freq=freq,
                name=result_df.index.name
            )
            new_df = pd.DataFrame(index=new_index,
                                  data=[list(result_df.loc[result_df.index.max()]) for _ in range(len(new_index))],
                                  columns=result_df.columns)
            result_df = pd.concat([result_df, new_df])

        # remove dates outside desired range
        result_df = result_df[(result_df.index >= start_datetime) & (result_df.index < end_datetime_exclusive)]

        return result_df
