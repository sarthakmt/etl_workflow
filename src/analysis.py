import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn import metrics
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)

logger3 = logging.getLogger(__name__)

def feature_engineering(df_preprocessed):
    logger3.info("************************Feature_engineering started*********************\n")
    # Generating new features from the existing ones

    """
    Item_Type_Combined:
    Using 1st 2 characters of Item_Identifier,
    create combined food type
    """
    #Get the first two characters of ID:
    df_preprocessed['Item_Type_Combined'] = df_preprocessed['Item_Identifier'].apply(lambda x: x[0:2])
    #Rename them to more intuitive categories:
    df_preprocessed['Item_Type_Combined'] = df_preprocessed['Item_Type_Combined'].map({'FD':'Food',
                                                                'NC':'Non-Consumable',
                                                                'DR':'Drinks'})
    df_preprocessed['Item_Type_Combined'].value_counts()

    """
    # Years of operation:
    No- of years the outlet has been in operation
    """
    df_preprocessed['Outlet_Years'] = 2019 - df_preprocessed['Outlet_Establishment_Year']

    
    # Hot Encoding
    """
    One-Hot-Coding refers to creating dummy variables, one for each category of a categorical variable. 
    Example: Item_Fat_Content has 2 categories – 'Low Fat' and 'Regular'. 
    One hot coding will remove this variable and generate 2 new variables. 
    """
    labelenc = LabelEncoder()
    data = df_preprocessed.copy()

    #New variable for outlet
    data['Outlet'] = labelenc.fit_transform(data['Outlet_Identifier'])
    var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']

    labelenc = LabelEncoder()
    for i in var_mod:
        data[i] = labelenc.fit_transform(data[i])

    data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type_Combined','Outlet'])
    logger3.info("feature engineered data\n")
    print(data.head())
    logger3.info("*******************Feature_engineering completed**********************\n")
    return data

def model(alg, dtrain, dtest, predictors, target, id_cols, filename):
    logger3.info("****************Modelling & evaluation started**************************\n")
    
    # NaN  removal check
    dtrain.dropna(inplace=True)
    dtest.dropna(inplace=True)
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20,scoring="neg_mean_squared_error")
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    logger3.info("\nModel Report")
    logger3.info("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    logger3.info("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    id_cols.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in id_cols})
    submission.to_csv(filename, index=False)

    logger3.info("******************Modelling & evaluation completed*************************\n")

