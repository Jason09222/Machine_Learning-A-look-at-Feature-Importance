# z5270589
import numpy as np
# from google.colab import drive
from sklearn.model_selection import *
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import log_loss

# (e)
def bestsubset_selection(X, y):
    # record the 20 features shuffled index at first.
    all_shuffled_feature = X.shape[1]
    # we initilize the outcome best_subset first.
    best_subset = None
    # We define a infinity to used at first which is bigger than all possible log loss value.
    possitive_inf = float('inf')
    # We inirialize smallest_log_loss to infinity at the begining.
    smallest_log_loss = possitive_inf
    # We find all subsets which size is 3 by using the itertools:(x1, x2, x3)
    size_3_possible_subset = itertools.combinations(range(all_shuffled_feature), 3)
    # Logistic regrression same as (c)
    for sub in size_3_possible_subset:
        LR = LogisticRegression(penalty=None, random_state=4)
        LR.fit(X[:, sub], y)
        # I read the forum, Omar said the best model is the one that achives the lowest training error. We do not train on 'absolute value of the coefficients'.
        # So in (e), we select the bets by using log loss:
        # Also from forum, Omar said Since we are not distinguishing between train/test for the purposes of this assignment, you would just check which one has best train error.
        y_training_proba = LR.predict_proba(X[:, sub])[:, 1]
        # we calculate the log-loss for train error:
        train_error = log_loss(y, y_training_proba)
        # Since the smaller the log loss is, the better model we can get, so we aims to find the minimum log loss.
        if  smallest_log_loss > train_error:
            # updata and choose the bestsubset(lowest overall log loss value).
            smallest_log_loss = train_error
            best_subset = list(sub)

    return best_subset

# (c)
# Repeat the experiment a total of 1000 times
experiment_count = 1000
trial_1000_important_feature_identify = []

# i = 1,2,....,1000
for i in range(1, experiment_count+1):
    X, y = make_classification(
        # set all parameters as in Q1 part (a), but with only 7 features, 3 of which are to be taken to be informative, and the rest to be redundant. 
        n_samples=1000,   # 1000 observations.
        n_features=7,    # 7 features.
        n_informative=3,  # Set 3 of those features to be informative.
        n_redundant=4,   # n_redundant = 7-3=4.
        shuffle=False,    # Be sure to set the shuffle parameter to False.
        random_state=i    # Use a random seed of i .
    )
    # Normalize your data using sklearn.StandardScaler()
    standard_scaler_normalizer = StandardScaler()
    normalize_X = standard_scaler_normalizer.fit_transform(X)

    # Before fit a decision tree (using entropy as the criteria for splits) to a shuffled version of the data1, we need to shuffled data1 here.
    # Use a random seed of 0 when shuffling the data, we can use shuffled idxs = np.random.default rng(seed=0).permutation(X.shape[1]).
    shuffled_index = np.random.default_rng(seed=0).permutation(X.shape[1])
    shuffled_X = normalize_X[:, shuffled_index]

    # Apply best subset selection algorithm:
    best_subset = bestsubset_selection(shuffled_X, y)
    # same as previous, use the shuffled index to get the original index.
    top_3_cal_feature_index_original = shuffled_index[best_subset]
    # Calculate the number of the remaining features are actually important features at the end:
    actual_top_3_feature = np.sum(top_3_cal_feature_index_original < 3)
    trial_1000_important_feature_identify.append(actual_top_3_feature)
    print(i)
# Provide a histogram of this metric over the 1000 trials.
np_output = np.array(trial_1000_important_feature_identify)
# Report the average number of good features recovered over the 1000 trials.
average_recovered = np_output.mean()
print(f"Q2e The average number of good features recovered over the 1000 trials: {average_recovered}")

plt.figure(figsize=(10, 6))
plt.hist(np_output, bins=range(3+2), align='left', rwidth=0.7)
plt.xlabel("Actually important features are found in the top 3")
plt.ylabel("Count")
plt.title("Q2e Best Subset Top 3 good features recovered over the 1000 trials")
plt.xticks(range(0, 4))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()