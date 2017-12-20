# Drug combination specific Variable Importance (DVI) #

The Drug combination specific Variable Importance (DVI) is a modified version of the permuted variable importance score for Random Forests. Essentially, for a Random Forest trained simultaneously for multiple response vectors (multiple drug combinations in our case), it allows the inference of variable importance scores per variable and per drug combination. For more information, see our manuscript (Aben et al, submitted, TODO: DOI).

### Dependencies ###

The easiest way of making sure all dependencies are present is by using [conda](https://conda.io/docs/user-guide/install/download.html). In the main directory of the repository, run:
```
conda env create --file environment.yml
```

### Example run ###

First, activate the conda environment:
```
source activate DVI
```

Next, we'll compute the DVI using calculate\_weighted\_vi.py. This script requires an input matrix $\textbf{X}$, a response vector $y$ and a file to write the output to. We'll briefly describe the format of $\textbf{X}$ and $y$ here, for more details we refer to our manuscript. Let $n$ be the number of samples, $p$ the number of features and $q$ the number of drug combinations. The matrix $\textbf{X}$ is the $nq \times p$ input matrix that is constructed by 1) taking a 'regular' $n \times p$ input matrix and repeating it $q$ times; and 2) adding $q$ features (whose column name ends in '\_combi') with indicator variables that associate each row to the relevant drug combination. We obtain the $nq \times 1$ response vector $y$ by repeating the regular $n \times 1$ response vector $q$ times.
```
python calculate_weighted_vi.py -x example/x.csv -y example/y.csv -o example/output.csv
```
The example data has been designed such that
* for the first five drug combinations the importance of the first variable will be high; and
* for the next five drug combinations the importance of the second variable will be high.

Besides the mandatory -x, -y and -o parameters, there are a number of optional parameters:
* --output\_model: where should a pickled version of the Random Forest model be saved?
* --n\_estimators: number of trees to use in the Random Forest (default: 500).
* --threads: number of threads to use (default: 1).
* --seed: random seed to use (default: 1).

If we'd prefer to train a Random Forest without computing the DVI, we can use
```
python train_rf_regressor.py --x_train example/x.csv --y_train example/y.csv --x\_test example/x.csv --output_pred example/y_hat.csv
```
This script has the following parameters:
* --x\_train: the input matrix $\textbf{X}$ used for training (required).
* --y\_train: the response vector $y$ used for training (required).
* --x\_test: the input matrix $\textbf{X}$ used for testing.
* --output\_pred: where should the prediction using x_test be saved?
* --output\_model: where should a pickled version of the Random Forest model be saved?
* --n\_estimators: number of trees to use in the Random Forest (default: 500).
* --threads: number of threads to use (default: 1).
* --seed: random seed to use (default: 1).
