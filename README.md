# d100project

This project attempts to model and predict the revnue of movies

Create and activate Conda environment:

conda env create -f environment.yml
conda activate d100_project_movies_ds_env

I've used a kaggle dataset. Given it's relatively small size  (roughly 20x5000), the easiest way to access this it to put the dataframe within the data folder.

To clean this data and save it as a parquet, please run _clean_data.py which is in the cleaning folder in d100project.

Then once you have created the cleaned parquet you will be able to run model_training.py in the analyses folder.

Once you have run this script, you should be able to run the code in evaluation.ipynb