import pandas as  pd
import preprocessing
import star_schema
import logging
from pathlib import Path
import analysis
from models import linear_regression_model
from sklearn.model_selection import train_test_split
import inspect

base_path = Path(__file__).parent
# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=(base_path / "../myapp.log").resolve(),
                    filemode='w')

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


#####################################
#           extract                 #
#####################################

# Now, we can log to the root logger, or any other logger. First the root...
logging.info('*******************Extracting********************')

# raw file
file_input = (base_path / "../data/source/walmart_raw.csv").resolve()

# Output File : Pre-Processed  CSV file after input file is cleaned and prepared
file_pre_processed = (
    base_path / "../data/staging/walmart_cleaned.csv").resolve()

# Output File : item_data.csv file will hold only the Item related Information.
#               outlet_data.csv file will hold only the Outlet related Information.
#               walmart_fact.csv file will hold the factual information of
#               Sales by Walmart.

fact_output = (base_path / "../data/destination/walmart_fact.csv")
item_dim_output = (base_path / "../data/destination/item_data.csv")
outlet_dim_output = (base_path / "../data/destination/outlet_data.csv")
test_output = (base_path / "../data/algo_files/test.csv")


df_raw = pd.read_csv(file_input)
print(df_raw.head())

# create connection

if __name__ == "__main__":

    #####################################
    #       staging/transform area      #
    #####################################

    # preprocess the data
    df_preprocessed = preprocessing.preprocess_data(df_raw)

    #####################################
    #        transform & load           #
    #####################################

    # Create star schema and load to destination
    star_schema.create_star_schema(df_preprocessed, **{
                                "fact_output": fact_output, "item_dim_output": item_dim_output, "outlet_dim_output": outlet_dim_output})


    #####################################
    #         Analysis                  #
    #####################################

    # feature engineering
    df_prepared = analysis.feature_engineering(df_preprocessed)

    # Data ready for Model building
    # Drop the columns which have been converted to different types:
    df_prepared.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

    #Divide into test and train:
    train,test = train_test_split(df_prepared, test_size=0.2, random_state=10)

    # We have to predict Item_Outlet_Sales
    test.to_csv(test_output,header=True,index=False)
    test.drop(['Item_Outlet_Sales',],axis=1,inplace=True)


    # Modelling
    #Define target and ID columns:
    target = 'Item_Outlet_Sales'
    id_cols = ['Item_Identifier','Outlet_Identifier']

    # Linear Regression
    linear_regression_model("lin_reg",id_cols,target,train,test)