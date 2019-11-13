import pandas as pd
import numpy as np
import logging
import sys

logger2 = logging.getLogger(__name__)
def create_star_schema(df_pre_processed,**kwargs):
    logger2.info("***********************Creating star schema started**************************")
    try:
        # Dimension tables
        
        # Item dimension
        df_item = df_pre_processed.loc[ : , ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content',  'Item_Type' ]]
        print(df_item.head())

        # Outlet dimension
        df_outlet = df_pre_processed.loc[ : , ['Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size','Outlet_Location_Type',
                                            'Outlet_Type'] ]
        print(df_outlet.head())

        # Fact tables

        # Item_outlet fact  
        df_fact = df_pre_processed.loc[ : , ['Item_Identifier','Outlet_Identifier','Item_Visibility','Item_MRP','Item_Outlet_Sales'] ]
        print(df_fact.head())

        # Unique ids for the dimension tables

        # Sort the df_outlet dataframe in ascending order based on Item_Identifier and remove duplicates
        df_item = df_item.sort_values(by = ['Item_Identifier'], ascending = True, na_position = 'last').drop_duplicates(keep='first')
        print(df_item.head())

        # Sort the df_outlet dataframe in ascending order based on Outlet_Identifier and remove duplicates
        df_outlet = df_outlet.sort_values(by = ['Outlet_Identifier'], ascending = True, na_position = 'last').drop_duplicates(keep = 'first')
        print(df_outlet.head())

        # check if any composite key ['Item_Identifier', 'Outlet_Identifier'] is getting repeated 
        df_fact.groupby(['Item_Identifier', 'Outlet_Identifier']).size().apply(lambda x: 0 if x>1 else 1).value_counts()

        df_fact.sort_values(by = ['Item_Identifier', 'Outlet_Identifier'], ascending = True, na_position = 'last').drop_duplicates(
            ['Item_Identifier', 'Outlet_Identifier'],keep = 'first')

        # Export the Star Schema and save them as .csv files
        df_fact.to_csv(kwargs["fact_output"], header=True,index=False)
        df_item.to_csv(kwargs["item_dim_output"], header=True,index=False)
        df_outlet.to_csv(kwargs["outlet_dim_output"], header=True,index=False)

        logger2.info("***********************Creating star schema completed**************************")
    except Exception as e:
        logger2.error(e.with_traceback())
        print(e)


