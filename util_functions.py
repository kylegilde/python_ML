# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
import plotly.express as px


def get_feature_names(column_transformer, verbose = False):  

    """

    Get the column names from the a ColumnTransformer containing transformers & pipelines

    Parameters
    ----------
    column_transformer : a fit ColumnTransformer instance
    verbose : a boolean indicating whether to print summaries. default = False


    Returns
    -------
    a list of the correct feature names
    
    IMPORTANT NOTE: If a transformer is adding net new columns, it must come LAST in the pipeline, 
    e.g. SimpleImputer(add_indicator=True) must come after MinMaxScaler. The OH encoder must be last also.
    
    Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525 

    """

    assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
    check_is_fitted(column_transformer)

    new_feature_names = []

    for transformer in column_transformer.transformers_: 

        if verbose: print('\n\ntransformer: ', transformer[0], type(transformer[1]))

        orig_feature_names = list(transformer[2])
        
        if isinstance(transformer[1], Pipeline):
            # if pipeline, get the last transformer in the Pipeline
            transformer = transformer[1].steps[-1][1]

        if hasattr(transformer, 'get_feature_names'):

            if 'input_features' in transformer.get_feature_names.__code__.co_varnames:

                names = list(transformer.get_feature_names(orig_feature_names))

            else:

                names = list(transformer.get_feature_names())
                if verbose: print(names)
  

        elif hasattr(transformer,'indicator_') and transformer.add_indicator:
            # is this transformer one of the imputers & did it call the MissingIndicator?

            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

        elif hasattr(transformer,'features_'):
            # is this a MissingIndicator class? 
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' for idx in missing_indicator_indices]

        else:
   
            names = orig_feature_names

        if verbose: print(names)

        new_feature_names.extend(names)

    return new_feature_names



def get_selected_features(pipeline, verbose = False):
    """
    
    Get the Feature Names that were retained after Feature Selection (sklearn.feature_selection)
    
    Parameters
    ----------
    
    pipeline : a fit pipeline instance where the first step is a ColumnTransformer
    verbose : a boolean indicating whether to print summaries. default = False

    Returns
    -------
    a list of the correct feature names

    
    """

    
    assert isinstance(pipeline, Pipeline), "Input isn't a Pipeline"
    assert isinstance(pipeline[0], ColumnTransformer), "First step isn't a ColumnTransformer"

    features = get_feature_names(pipeline[0], verbose=verbose)

    for i, step in enumerate(pipeline.steps[1:]):
        if verbose: print(i, ": ", step[0])
        
        if hasattr(step[1], 'get_support'):
            
            check_is_fitted(step[1])

            retained_cols = step[1].get_support()
            if verbose: print(sum(retained_cols), "of", len(retained_cols), "retained, ",\
                round(sum(retained_cols) / len(retained_cols) * 100, 1), "%")

            features = [feature for is_retained, feature in zip(retained_cols, features) if is_retained]      

    return features


def plot_feature_importance(pipeline, top_n_features=100, rank_features=True, orientation='h', width=500, height=None):
    """
    
    Plot the Feature Names & Importances 
    
    
    Parameters
    ----------
    
    pipeline : a fit pipeline instance where the first step is a ColumnTransformer
    top_n_features : the number of features to plot, default is 100
    rank_features : whether to rank the features with integers, default is True
    orientation : the plot orientation, 'h' (default) or 'v'
    width :  the width of the plot, default is 500
    height : the height of the plot, the default is top_n_features * 10

    Returns
    -------
    plot
    
    """
    assert isinstance(pipeline, Pipeline), "Input isn't a Pipeline"
    
    if height is None:
        height = top_n_features * 10
    
    features = get_selected_features(pipeline)
    importance_values = pipeline[-1].get_feature_importance()

    assert len(features) == len(importance_values), "The number of feature names & importance values doesn't match"
    
    importances = pd.Series(importance_values, 
                            index=features)\
                            .nlargest(top_n_features)\
                            .sort_values()
    
    
    if rank_features:
        existing_index = importances.index.to_series().reset_index(drop=True)
        ranked_index = pd.Series(range(1, len(importances) + 1)[::-1])\
                    .astype(str)\
                    .str.cat(existing_index, sep='. ')
                    
        importances.index = ranked_index
    
    fig = px.bar(importances, orientation=orientation, width=width, height=height)
    fig.update(layout_showlegend=False)
    fig.show()


def reduce_mem_usage(df, n_unique_object_threshold=0.30, verbose=True):
    """
    Downcasts the data type when possible in order to reduce memory usage
    Inspiration: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    
    Parameters
    ----------
    df : a DataFrame

    Returns
    -------
    returns a smaller df if possible
    """
    
    assert isinstance(df, pd.DataFrame), "This isn't a DataFrame!"
    
    start_mem_usg = df.memory_usage().sum() / 1024**2

    # record the dtype changes
    dtype_df = pd.DataFrame(df.dtypes.astype('str'), columns=['original'])

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # If no infinite/NaNs values, proceed to reduce the int
            if np.isfinite(df[col]).all():
                
                # make variables for max, min
                mx, mn = df[col].max(), df[col].min()

                # test if column can be converted to an integer
                as_int = df[col].astype(np.int64)
                delta = (df[col] - as_int).sum()

                # Make Integer/unsigned Integer datatypes
                if delta == 0:
                    if mn >= 0:
                        if not isinstance(df[col], np.uint8) and mx < np.iinfo(np.uint8).max:
                            df[col] = df[col].astype(np.uint8)
                        elif not isinstance(df[col], np.uint16) and mx < np.iinfo(np.uint16).max:
                            df[col] = df[col].astype(np.uint16)
                        elif not isinstance(df[col], np.uint32) and mx < np.iinfo(np.uint32).max:
                            df[col] = df[col].astype(np.uint32)
                        elif not isinstance(df[col], np.uint64):
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if not isinstance(df[col], np.int8) and mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif not isinstance(df[col], np.int16) and mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif not isinstance(df[col], np.int32) and mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif not isinstance(df[col], np.int64) and mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)

                # Make float datatypes 32 bit
                else:
                    if not isinstance(df[col], np.float32) and sum(df[col] - df[col].astype(np.float32)) == 0:
                        df[col] = df[col].astype(np.float32)

        elif pd.api.types.is_object_dtype(df[col]):
            if df[col].nunique() / len(df) < n_unique_object_threshold:
                df[col] = df[col].astype('category')


    if verbose:
        
        # Print final result
        dtype_df['new'] = df.dtypes.astype('str')
        dtype_changes = dtype_df.original != dtype_df.new

        print("------------------------------------")

        if dtype_changes.sum():

            print("Starting memory usage is %s MB" % "{0:}".format(start_mem_usg))
            print(dtype_df.loc[dtype_changes])
            new_mem_usg = df.memory_usage().sum() / 1024**2
            print("Ending memory usage is %s MB" % "{0:}".format(new_mem_usg))
            print("Reduced by", int(100 * (1 - new_mem_usg / start_mem_usg)), "%")

        else:
            print('No reductions possible')

        print("------------------------------------")
        
    return df



def get_object_name(obj):
    """
    Get the name of a variable in the calling namespace as a string
    
    Parameters
    ----------
    obj: any variable
    Returns
    -------
    a string
    
    """

    namespace = dict(globals(), **locals()) 
    return [name for name in namespace if namespace[name] is obj][0]

