import numpy as np
import pandas as pd
import logging 

logger1 = logging.getLogger(__name__)

def preprocess_data(df_raw):
    logger1.info("***********************Preprocessing started**************************")
    df_preprocessed = df_raw.copy()
    print(df_preprocessed.apply(lambda x: sum(x.isnull())))

    """
    Item_Weight:
    The missing values are filled using the cross
    join with Item_identifier & item_wt and filling 
    the empty item_wt with the value existing for
    same Item_identifier
    """

    # https://stackoverflow.com/questions/56310440/comparing-two-columns-in-a-pandas-dataframe-and-filling-in-missing-values-for-on

    # df_raw["Item_Weight"].value_counts(dropna=False)
    mapping = dict(df_preprocessed.dropna()[["Item_Identifier","Item_Weight"]].values)
    # mapping = df.set_index('code')['name'].dropna().to_dict()

    df_preprocessed['Item_Weight'] = df_preprocessed['Item_Weight'].fillna(df_preprocessed['Item_Identifier'].map(mapping))
    # df_raw["Item_Weight"].value_counts(dropna=False)

    df_raw[df_raw["Item_Identifier"]=="FDN15"]

    """
    Item_Fat_Content: 
    LF replaced with Low Fat
    reg replaced with Regular
    low fat is converted to Low Fat (title case )
    """
    print(df_preprocessed["Item_Fat_Content"].value_counts(dropna=False))
    df_preprocessed["Item_Fat_Content"].replace({"LF": "Low Fat","reg":"Regular","low fat":"Low Fat"},inplace=True)
    print(df_preprocessed["Item_Fat_Content"].value_counts(dropna=False))

    """
    Item_Visibility:
    The values 0 are filled by mode/mean
    as the valu zero does not mean anything
    """
    #Find average visibility 
    visibility_avg = df_preprocessed.pivot_table(values='Item_Visibility', index='Item_Identifier')

    #Impute 0 values with mean visibility of that product:
    zero_bool = (df_preprocessed['Item_Visibility'] == 0)

    print('Number of 0 values initially: %d'%sum(zero_bool))
    df_preprocessed.loc[zero_bool,'Item_Visibility'] = df_preprocessed.loc[zero_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
    print('Number of 0 values after modification: %d'%sum(df_preprocessed['Item_Visibility'] == 0))

    """
    Outlet_Establishment_Year:
    The value is checked to be less than or equal
    to 2019
    """
    (df_preprocessed["Outlet_Establishment_Year"] >2019).value_counts()

    """
    Outlet_Size:
    outlet_size can be filled with the mode of outlet_size for the particular
    outlet_location. 

    Medium for Tier 3(OUT010) and Small for Tier 2(OUT017, OUT045).

    """
    df_preprocessed["Outlet_Size"]=np.where(((df_preprocessed['Outlet_Size'].isnull()) & (df_preprocessed["Outlet_Identifier"] == "OUT010")),\
                                   "Medium",df_preprocessed['Outlet_Size'])

    df_preprocessed["Outlet_Size"] = np.where(((df_preprocessed['Outlet_Size'].isnull()) & (df_preprocessed["Outlet_Identifier"]  ==  "OUT017")),\
                                   "Small",df_preprocessed['Outlet_Size'])

    df_preprocessed["Outlet_Size"] = np.where(((df_preprocessed['Outlet_Size'].isnull()) & (df_preprocessed["Outlet_Identifier"]  ==  "OUT045")),\
                                   "Small",df_preprocessed['Outlet_Size'])

    
    """
    Item_Outlet_Sales:
    The decimal places’ precison is kept at 2 as
    the currency value should not contain partial
    cents
    """
    print(df_preprocessed["Item_Outlet_Sales"].head())
    df_preprocessed["Item_Outlet_Sales"] = df_preprocessed["Item_Outlet_Sales"].round(2)
    print(df_preprocessed["Item_Outlet_Sales"].head())

    logger1.info("***********************Preprocessing completed**************************")
    return df_preprocessed
