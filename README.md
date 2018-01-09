# Multitask Variable Importance #

The Multitask Variable Importance (Multitask VI) is a modified version of the permuted variable importance score for Random Forests. Essentially, for a Random Forest trained simultaneously for multiple response vectors, it allows the inference of variable importance scores per variable and per task. For more information, see our manuscript (Aben et al, submitted, [https://doi.org/10.1101/243568](https://doi.org/10.1101/243568)), where we applied this score to a dataset where each tasks corresponds to a drug combination (and hence the Multitask VI is called the Drug combination specific Variable Importance, or DVI, there).

### Dependencies ###

The easiest way of making sure all dependencies are present is by using [conda](https://conda.io/docs/user-guide/install/download.html). In the main directory of the repository, run:
```
conda env create --file environment.yml
source activate multitask_vi
pip install .
```

### Example run using the command line ###

First, activate the conda environment:
```
source activate multitask_vi
```

Next, we'll train the Random Forest using mt-train. This command has the following parameters:

* --x\_train: the input matrix **X** used for training (required).
* --y\_train: the response vector y used for training (required).
* --output\_base: where should a pickled version of the Random Forest model be saved? (required)
* --n\_estimators: number of trees to use in the Random Forest (default: 500).
* --cpus: number of threads to use (default: 1).
* --seed: random seed to use (default: 1).

We'll briefly describe the format of **X** and y here, for more details we refer to our manuscript. Let n be the number of samples, p the number of features and q the number of tasks. The matrix **X** is the nq x p input matrix that is constructed by 1) taking a 'regular' n x p input matrix and repeating it q times; and 2) adding q features (whose column name starts with 'task\_') with indicator variables that associate each row to the relevant task. We obtain the nq x 1 response vector y by repeating the regular n x 1 response vector q times.
```
mt-train --x_train example/x_train.csv --y_train example/y_train.csv --output_base example/trained
```

We can then use mt-vi to determine the variable importance. This command has the following parameters:

* --x\_train: the input matrix **X** used for training (required).
* --y\_train: the response vector y used for training (required).
* --model: the trained Random Forest model, resulting from mt-train (required).
* --design: a design matrix, indicating which samples correspond to which task (required).
* --output: where should the resulting variable importances be saved? (required)
* --threads: number of threads to use (default: 1).

```
mt-vi --x_train example/x_train.csv --y_train example/y_train.csv --model example/trained.model.pkl --output example/vi.csv --design example/design.csv
```
The example data has been designed such that

* for the first five tasks the importance of the first variable will be high; and
* for the next five tasks the importance of the second variable will be high.

### Example run using the API ###
An example using the API is given in example/api\_example.ipynb. To run this example, first install the dependencies as described above and then run these additional commands.
```
source activate multitask_vi
conda install jupyter
conda install matplotlib
conda install seaborn
jupyter notebook
```
From the Jupyter Notebook, example/api\_example.ipynb can be opened and executed.
