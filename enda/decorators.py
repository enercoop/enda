"""This module contains various utility functions to be used as decorators"""

import functools
import warnings
import pandas as pd


def handle_multiindex(arg_name):
    """
    This function is meant to be used as a decorator. It is a wrapper around functions defined
    for a single-indexed dataframe so that they also work for multi-indexed dataframes.
    The wrapped function will be applied to all single-indexed last-level dataframes
    which constitute the multi-index dataframe
    :param arg_name: name of the dataframe in the wrapped function signature.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper_handle_multiindex(*args, **kwargs):
            # check whether a multiindex has been given
            multi_df = None
            is_multiindex = False
            is_arg_name_in_kwargs = False

            if arg_name in kwargs:
                is_arg_name_in_kwargs = True
                if not isinstance(kwargs[arg_name], pd.DataFrame):
                    raise ValueError(
                        f"Object for argument '{arg_name}' passed to {func.__name__} is not a dataframe"
                    )

                if isinstance(kwargs[arg_name].index, pd.MultiIndex):
                    is_multiindex = True
                    multi_df = kwargs[arg_name]

            if args and not is_arg_name_in_kwargs:
                if not isinstance(args[0], pd.DataFrame):
                    raise ValueError(
                        f"Object for argument '{arg_name}' passed to {func.__name__} is not a dataframe"
                    )

                if isinstance(args[0].index, pd.MultiIndex):
                    is_multiindex = True
                    multi_df = args[0]

            if not is_multiindex:
                # call the function directly, as the arg_name-related object is a single-indexed dataframe
                return func(*args, **kwargs)

            last_level_name = multi_df.index.levels[-1].name
            other_levels_names = [_.name for _ in multi_df.index.levels[0:-1]]

            if last_level_name is None or pd.isna(other_levels_names).sum() > 0:
                raise ValueError(
                    f"Cannot use the function {func.__name__} with a multiindex dataframe having "
                    "unnamed indexes"
                )

            # we will build a new dataframe looping on all sub-dataframes, and applying func to all of them
            new_df = pd.DataFrame()
            for other_level_values, last_level_df in multi_df.groupby(
                level=other_levels_names, sort=False
            ):
                # turn last level to a single-index dataframe
                last_level_df = last_level_df.reset_index(
                    level=other_levels_names, drop=True
                )

                # define arguments of the function, to use it with the new last-level single-index dataframe
                args_decorator = args
                kwargs_decorator = kwargs
                if not is_arg_name_in_kwargs:
                    args_decorator = (last_level_df,) + args[1:]
                else:
                    kwargs_decorator = {arg_name: last_level_df} | {
                        kw: kwarg for kw, kwarg in kwargs.items() if kw != arg_name
                    }

                # call function with sub-dataframe
                result = func(*args_decorator, **kwargs_decorator)

                # organize result
                if isinstance(result, pd.DataFrame):
                    new_col_name = result.index.name
                    result[other_levels_names] = other_level_values
                    result = result.reset_index().set_index(
                        other_levels_names + [new_col_name]
                    )
                else:
                    # it should be convertible to a pd.Series
                    result = pd.Series(result).to_frame().T
                    result[other_levels_names] = other_level_values
                    result = result.set_index(other_levels_names)

                new_df = pd.concat([new_df, result], axis=0)

            return new_df

        return wrapper_handle_multiindex

    return decorator


def handle_series_as_datetimeindex(arg_name, return_input_type):
    """
    This function is meant to be used as a decorator over functions which process timeseries
    given as datetimeIndex, so that they can process time_series given as pd.Series too.
    :param arg_name: name of the datetimeindex in the wrapped function signature.
    :param return_input_type: boolean, if true, for a decorated function that returns a datetime index,
                              it will return a datetimeindex if a datetimeindex is passed, but a series
                              if a series is passed.
                              If False, it will return the object the function would if un-decorated.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            is_series = False

            # check if the argument is in kwargs
            if arg_name in kwargs:
                if not isinstance(kwargs[arg_name], pd.Series) and not isinstance(
                    kwargs[arg_name], pd.DatetimeIndex
                ):
                    raise ValueError(
                        f"Object for argument '{arg_name}' passed to {func.__name__} must "
                        f"be a datetimeindex or a series convertible to a datetimeindex"
                    )

                if isinstance(kwargs[arg_name], pd.Series):
                    is_series = True
                    kwargs[arg_name] = pd.DatetimeIndex(kwargs[arg_name])

            elif args:
                if not isinstance(args[0], pd.Series) and not isinstance(
                    args[0], pd.DatetimeIndex
                ):
                    raise ValueError(
                        f"Object for argument '{arg_name}' passed to {func.__name__} must "
                        f"be a datetimeindex or a series convertible to a datetimeindex"
                    )

                if isinstance(args[0], pd.Series):
                    is_series = True
                    args = (pd.DatetimeIndex(args[0]),) + args[1:]

            # call the function designed for datetimeindex
            result = func(*args, **kwargs)

            # turn result (supposed to be a datetimeindex else error) back to series
            if is_series and return_input_type:
                result = result.to_series().reset_index(drop=True)

            return result

        return wrapper

    return decorator


def warning_deprecated_name(
    namespace_name, new_namespace_name=None, new_function_name=None
):
    """
    This decorator with a parameter is meant to be used to issue a specific warning, namely that a function
    has been renamed, or moved in a new namespace. This is useful for classes that changed named, or will be deleted
    :param namespace_name: the namespace that contains the function. Might be the one to replace.
    :param new_namespace_name: the new namespace that contains the function
    :param new_function_name: the new name of the function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper_warning_deprecated_name(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} in {namespace_name} is deprecated, use "
                f"{new_function_name if new_function_name is not None else 'it'}"
                f" from {namespace_name if new_namespace_name is None else new_namespace_name}"
                f" instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        # noinspection PyDeprecation
        return wrapper_warning_deprecated_name

    return decorator
