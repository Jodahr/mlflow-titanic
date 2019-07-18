import pandas as pd
import argparse
import mlflow
import scipy

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

from sklearn_pandas import DataFrameMapper
import mlflow
import mlflow.sklearn


# 's3://jodahr-mlflow/data/titanic.csv'
def read_data(training_data):
    return pd.read_csv(training_data, sep='\t')


def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_data", type=str,
                        help="Path to training data csv file.")
    parser.add_argument("--n_iter", type=int,
                        help="Number of iterations (defaults to 10).",
                        default=10)
    parser.add_argument("--rstate", type=int,
                        help="Random seed", default=42)
    args = parser.parse_args()
    return args


def setup_model():
    mapper = DataFrameMapper([
        (['Sex', 'Pclass'], OneHotEncoder(sparse=False)),
        (['Age', 'Fare'], [SimpleImputer(strategy='median'), MinMaxScaler()])
    ], df_out=True)

    pipe = Pipeline([
        ('prep', mapper),
        ('model', SVC(gamma='scale'))
    ])

    return pipe


def setup_space():

    # Set the parameters by cross-validation
    return {'model__C': scipy.stats.expon(scale=100),
            'model__gamma': scipy.stats.expon(scale=.1),
            'model__kernel': ['rbf', 'linear'],
            'model__class_weight': ['balanced', None]}


if __name__ == "__main__":

    print("Start Random Search ...")

    # argparse
    args = parse_cl_args()

    # read data
    df = read_data(args.training_data)

    # setup model
    clf = setup_model()

    # X,y
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']

    # setup 5-fold cv search
    param_dist = setup_space()
    n_iter_search = args.n_iter
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, iid=False,
                                       random_state=args.rstate,
                                       return_train_score=True)

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id

        print("start experiment: {}".format(experiment_id))

        # start search
        random_search.fit(X, y)

        # retrieve results
        df_cv = pd.DataFrame(random_search.cv_results_)[[
            'mean_test_score',
            'std_test_score',
            'param_model__C',
            'param_model__gamma',
            'param_model__kernel',
            'param_model__class_weight'
        ]]

        best_run = df_cv.loc[df_cv['mean_test_score'].idxmax, :]

        # log params of best run
        mlflow.log_metric(key='mean_acc',
                          value=best_run['mean_test_score'])
        mlflow.log_metric(key='std_acc',
                          value=best_run['std_test_score'])
        mlflow.log_param(key='C',
                         value=best_run['param_model__C'])
        mlflow.log_param(key='gamma',
                         value=best_run['param_model__gamma'])
        mlflow.log_param(key='kernel',
                         value=best_run['param_model__kernel'])
        mlflow.log_param(key='class_weight',
                         value=best_run['param_model__class_weight'])
        mlflow.log_param(key='n_iters',
                         value=args.n_iter)
        mlflow.log_param(key='random_state',
                         value=args.rstate)

        # log best model
        mlflow.sklearn.log_model(random_search, 'svc')

        print("... Done")
