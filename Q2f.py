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
from sklearn.inspection import permutation_importance
# (f)

# Repeat the experiment a total of 1000 times
experiment_count = 1000
trial_1000_important_feature_identify = []
# i = 1,2,....,1000
for i in range(1, experiment_count + 1):
    # Same as before.
    X, y = make_classification(
        n_samples=1000,   # 1000 observations.
        n_features=20,    # 20 features.
        n_informative=5,  # Set 5 of those features to be informative.
        n_redundant=15,   # n_redundant = 20-5=15.
        shuffle=False,    # Be sure to set the shuffle parameter to False.
        random_state=i    # Use a random seed of i .
    )
    print(i)
    # Normalize your data using sklearn.StandardScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Before fit a decision tree (using entropy as the criteria for splits) to a shuffled version of the data1, we need to shuffled data1 here.
    # Use a random seed of 0 when shuffling the data, we can use shuffled idxs = np.random.default rng(seed=0).permutation(X.shape[1]).
    shuffled_index = np.random.default_rng(seed=0).permutation(X.shape[1])
    shuffled_X = X[:, shuffled_index]
    # We choose to use the logistic regression with no penalty here:
    LR = LogisticRegression(penalty=None, random_state=4)
    LR.fit(shuffled_X, y)

    # We calculate the permutation importance score here, we use the default method as the scoring method
    # Meanwhile, we will repeat the permutation of a feature for 10 times to aviod the random.
    permutation_importance_score = permutation_importance(LR, shuffled_X, y, n_repeats=10, random_state=4)

    # Found in the top 5 important features by the pfi.
    # np.argsort gives us the index of small value to large value, we use [::-1] to convert and [:5] to get the top 5.
    top_5_cal_feature_index_shuffled = np.argsort(permutation_importance_score.importances_mean)[::-1][:5]
    # After we get the shuffled top 5 index, then get the current top 5 original index:
    top_5_cal_feature_index_original = shuffled_index[top_5_cal_feature_index_shuffled]

    # Sicne the original informative features have the index 0-4, we use < 5 to filter that:
    actual_top_5_feature = np.sum(shuffled_index[top_5_cal_feature_index_shuffled] < 5)
    trial_1000_important_feature_identify.append(actual_top_5_feature)

np_output = np.array(trial_1000_important_feature_identify)
# Report the average number of good features recovered over the 1000 trials.
average_recovered = np_output.mean()
print(f"Q2(f) The average number of good features recovered over the 1000 trials: {average_recovered}")

plt.figure(figsize=(10, 6))
plt.hist(np_output, bins=range(5+2), align='left', rwidth=0.7)
plt.xlabel("Actually important features are found in the top 5")
plt.ylabel("Count")
plt.title("Q2(f) Top 5 good features recovered over the 1000 trials")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()