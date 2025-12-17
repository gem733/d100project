# d100project

This project attempts to model and predict the revenue of films

Create and activate Conda environment:

# cd inside where you have saved this repo then:

conda env create -f environment.yml
conda activate d100_project_movies_ds_env

This is also the kernel for Jupyter Notebooks

I've used a kaggle dataset. Given it's relatively small size (roughly 20x5000), the easiest way to access this it to put the dataframe within the data folder.

Exploratory data analysis is in eda_cleaning.ipynb

Please run \_clean_data.py which is in the cleaning folder in d100project, to create the cleaned parquet.

Then once you have created the cleaned parquet you will be able to run the models in untuned_model_training.py and tuned_model_training.py. Both are in the analyses folder.

Once you have run these scripts, you should be able to run the code in evaluation.ipynb in the evaluation folder in d100project.
