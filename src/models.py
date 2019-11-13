import analysis
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
import  logging
from pathlib import Path

logger4 = logging.getLogger(__name__)
base_path = Path(__file__).parent

def linear_regression_model(algorithm_name, idcols, target, train, test):
    logger4.info("Linear Regression modelling started")

    predictors = [x for x in train.columns if x not in [target]+idcols]
    # print predictors
    filename = "../data/algo_files/"+algorithm_name+'.csv'
    alg = LinearRegression(normalize=True)
    alg_file_path = (base_path / filename).resolve()

    print(alg_file_path)
    analysis.model(alg, train, test, predictors, target,
                   idcols,alg_file_path)
    coef1 = pd.Series(alg.coef_, predictors).sort_values()
    coef1.plot(kind='bar', title='Model Coefficients')

    logger4.info("Linear Regression modelling completed")
