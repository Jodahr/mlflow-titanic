# mlflow-titanic
A simple example to test mlflow functionalities.

## Requirements
- Anaconda Python Environment
- mlflow v. 1.0.0
- data file is provided but can actually be stored anywhere, e.g. S3.

## Run the code
You can run the project by using the following command:

`mlflow run git@github.com:jodahr/mlflow-titanic.git`

If you want to change some of the parameters you can use:

`mlflow run git@github.com:jodahr/mlflow-titanic.git {training_data_uri} -P [rstate=rstate] -P [n_iter=n_iter]`

You can also add the run to a specific experiment:

`mlflow experiments create -n hyper_param_runs`

returns a run exp id which can be used:

`mlflow run -e main --experiment-id exp_id project_uri`

After the run is done you can investigate the results by using

`mlflow ui`

in the folder of the project or where you issued the above commands.

If you need to keep different versions of you data use s3 versioning! This should lead to a different s3 uri
and will be recognized by mlflow.