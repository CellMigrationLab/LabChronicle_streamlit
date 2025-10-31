###############################################
# This file has been taken from https://github.com/wjbmattingly/streamlit-pandas/blob/main/streamlit_pandas/streamlit_pandas.py
#
# Due to incompatibilities with their pandas and streamlit outdated versions, the pip installation is not posible.
# And as the project seems to be dead, I have copied their necessary code here.
###############################################
from datetime import datetime
import streamlit as st
import pandas as pd

def filter_string(df, column, selected_list):
    final = []
    df = df[df[column].notna()]
    for idx, row in df.iterrows():
        row_value = row[column]
        if isinstance(row_value, str) and row_value in selected_list:
            final.append(row)
        elif isinstance(row_value, list) and any(item in selected_list for item in row_value):
            final.append(row)
    res = pd.DataFrame(final)
    return res

def date_widget(df, column, ss_name, prettify_function=None):
    df = df[df[column].notna()]
    min_date = df[column].apply(pd.to_datetime).min().date()
    max_date = df[column].apply(pd.to_datetime).max().date()
    
    # Ensure that min and max are not the same (only one unique date)
    if min_date != max_date:
        temp_input = st.slider(f"{prettify_function(column) if prettify_function else column.title()}",
                               min_date, max_date, (min_date, max_date), key=ss_name)
        all_widgets.append((ss_name, "date", column))

def number_widget(df, column, ss_name, prettify_function=None):
    df = df[df[column].notna()]
    max = float(df[column].max())
    min = float(df[column].min())
    temp_input = st.slider(f"{prettify_function(column) if prettify_function else column.title()}", 
                           min, max, (min, max), key=ss_name)
    all_widgets.append((ss_name, "number", column))

def number_widget_int(df, column, ss_name, prettify_function=None):
    df = df[df[column].notna()]
    max = int(df[column].max())
    min = int(df[column].min())
    temp_input = st.slider(f"{prettify_function(column) if prettify_function else column.title()}",
                           min, max, (min, max), key=ss_name)
    all_widgets.append((ss_name, "number", column))

def create_select(df, column, ss_name, multi=False, prettify_function=None):
    df = df[df[column].notna()]
    options = df[column]
    # Check of the options are strings or lists:
    if options.apply(lambda x: isinstance(x, list)).any():
        options = options.explode()
        options = options[options.notna()].unique()
    else:
        options = options.unique()
    options.sort()
    if multi==False:
        temp_input = st.selectbox(f"{prettify_function(column) if prettify_function else column.title()}", 
                                  options, key=ss_name)
        all_widgets.append((ss_name, "select", column))
    else:
        temp_input = st.multiselect(f"{prettify_function(column) if prettify_function else column.title()}", 
                                     options, key=ss_name)
        all_widgets.append((ss_name, "multiselect", column))

def text_widget(df, column, ss_name, prettify_function=None):
    temp_input = st.text_input(f"{prettify_function(column) if prettify_function else column.title()}", 
                               key=ss_name)
    all_widgets.append((ss_name, "text", column))

def create_widgets(df, create_data={}, ignore_columns=[], prettify_function=None):
    """
    This function will create all the widgets from your Pandas DataFrame and return them.
    df => a Pandas DataFrame
    create_data => Optional dictionary whose keys are the Pandas DataFrame columns
        and whose values are the type of widget you wish to make.
        supported: - multiselect, select, text
    ignore_columns => columns to entirely ignore when creating the widgets.
    """
    for column in ignore_columns:
        df = df.drop(column, axis=1)
    global all_widgets
    all_widgets = []
    for ctype, column in zip(df.dtypes, df.columns):
        if column in create_data:
            if create_data[column] == "text":
                text_widget(df, column, column.lower(), prettify_function=prettify_function)
            elif create_data[column] == "select":
                create_select(df, column, column.lower(), multi=False, prettify_function=prettify_function)
            elif create_data[column] == "multiselect":
                create_select(df, column, column.lower(), multi=True, prettify_function=prettify_function)
            elif create_data[column] == "date":
                date_widget(df, column, column.lower(), prettify_function=prettify_function)
        else:
            if ctype == "float64":
                number_widget(df, column, column.lower(), prettify_function=prettify_function)
            elif ctype == "int64":
                number_widget_int(df, column, column.lower(), prettify_function=prettify_function)
            elif ctype == "object":
                if str(type(df[column].tolist()[0])) == "<class 'str'>":
                    text_widget(df, column, column.lower(), prettify_function=prettify_function)
    return all_widgets


def filter_df(df, all_widgets):
    """
    This function will take the input dataframe and all the widgets generated from
    Streamlit Pandas. It will then return a filtered DataFrame based on the changes
    to the input widgets.

    df => the original Pandas DataFrame
    all_widgets => the widgets created by the function create_widgets().
    """
    res = df
    for widget in all_widgets:
        ss_name, ctype, column = widget
        data = st.session_state[ss_name]
        if data:
            if ctype == "text":
                if data != "":
                    # Check if the column contains lists or strings
                    if res[column].apply(lambda x: isinstance(x, list)).any():
                        res = res.loc[res[column].apply(lambda x: any(data in str(item) for item in x) if isinstance(x, list) else False)]
                    else:
                        res = res.loc[res[column].str.contains(data)]
            elif ctype == "select":
                res = filter_string(res, column, data)
            elif ctype == "multiselect":
                res = filter_string(res, column, data)
            elif ctype == "number":
                min, max = data
                res = res.loc[(res[column] >= min) & (res[column] <= max)]
            elif ctype == "date":
                start, end = data
                # Convert start and end to pandas Timestamp for valid comparison
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)
                res = res.loc[
                    (res[column].apply(pd.to_datetime) >= start) &
                    (res[column].apply(pd.to_datetime) <= end)
                ]
    return res