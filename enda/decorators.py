import functools
import pandas as pd


def handle_multiindex(func):
    '''
    This function is a wrapper around functions defined for a single dataframe with a datetime index so that 
    they also work for multiindexed dataframe. More specifically, functions designed for a 
    dataframe with a DatetimeIndex also work for a two-levels dataframe defined with a first
    index that defines a group, and a second index which is a DatetimeIndex.
    This function is meant to be used as a decorator. 
    :param func: the function to decorate
    '''

    @functools.wraps(func)
    def wrapper_handle_multiindex(*args, **kwargs):

        if 'df' in kwargs.keys(): 
            df = kwargs['df']
        else:
            df = args[0]
        
        # if it is a single indexed dataframe, call the function directly 
        if not isinstance(df.index, pd.MultiIndex):
            return func(*args, **kwargs)

        # the multiindex must be a two-level
        if len(df.index.levels) != 2:
            raise TypeError("The provided multi-indexed dataframe must be a two-levels one, the "
                            "second one being the date index.")

        if not isinstance(df.index.levels[1], pd.DatetimeIndex):
            raise TypeError("The second index of the dataframe should be a pd.DatetimeIndex, but given {}"
                            .format(df.index.levels[1].dtype))

        key_col = df.index.levels[0].name
        date_col = df.index.levels[1].name
        
        if 'index_name' not in kwargs.keys(): 
            index_name = date_col
        else: 
            index_name = kwargs['index_name']        
        
        df_new = pd.DataFrame()
        for key, data in df.groupby(level=0):
            data = data.reset_index().set_index(date_col).drop(columns=[key_col])
            args_decorator = (data,) 
            kwargs_decorator = {x: kwargs[x] for x in kwargs.keys() if x != 'df'}
            data = func(*args_decorator, **kwargs_decorator)
            data[key_col] = key
            data = data.reset_index().set_index([key_col, index_name])
            df_new = pd.concat([df_new, data], axis=0)
        
        return df_new
    
    return wrapper_handle_multiindex
