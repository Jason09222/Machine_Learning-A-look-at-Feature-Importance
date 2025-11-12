# z5270589
import numpy as np
# from google.colab import drive
from sklearn.model_selection import *
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# (c)
def backward_selection(X, y):
    # record the 20 features shuffled index at first.
    remain_feature_index = list(range(X.shape[1]))
    #  At each round, we remove the j-th feature from the model based on the drop in the value of a certain metric. We eliminate the feature corresponding to the smallest drop in the metric.
    # Firstly we will get the first metric value.
    LR_begin = LogisticRegression(penalty=None, random_state=4)
    LR_begin.fit(X[:, remain_feature_index], y)
    # we record the first metric value, which is the sum of the coefficients.
    metric_value = np.sum(np.abs(LR_begin.coef_[0]))
    # End the loop when 5 features left:
    while len(remain_feature_index) > 5:
        # create a list to record the drop in coefficient metric for each remaining feature in each turn.
        drop_remain_coefficient = []
        # we will calculate the drop in the coefficient metric for the remaining features (each turn delete one feature):
        for i in range(len(remain_feature_index)):
            # create the temp_remain to perform each step where we delete one feature and then calculate the drop in the metric.
            temp_remain = remain_feature_index.copy()
            # delete one feature.
            temp_remain.pop(i)     
            # After delete one feature, we apply the logistic regression model to calculate the current metric value.
            LR = LogisticRegression(penalty=None, random_state=4)
            LR.fit(X[:, temp_remain], y)
            # Get the current metric value.
            curr_metric_value = np.sum(np.abs(LR.coef_[0]))
            # Then we calculate and record the drop value.
            drop_value = metric_value - curr_metric_value
            # we record each drop value.
            drop_remain_coefficient.append(drop_value)

        # We eliminate the feature corresponding to the smallest drop in the metric, first we find the samllest drop:
        # We first find the index of the feature which delete it we obtain the samllest drop.
        cal_delete_feature_index = np.argmin(drop_remain_coefficient)
        # After we get the index of that feature, we pop it.
        remain_feature_index.pop(cal_delete_feature_index)

        # After we delete the feature, before move to next loop, we need pre calculate the metric value for getiing the drop in the next loop:
        LR_next = LogisticRegression(penalty=None, random_state=4)
        LR_next.fit(X[:, remain_feature_index], y)
        metric_value = np.sum(np.abs(LR_next.coef_[0]))
    # Finally left 5 features.
    return remain_feature_index

# (c)
# Repeat the experiment a total of 1000 times
experiment_count = 1000
trial_1000_important_feature_identify = []

# i = 1,2,....,1000
for i in range(1, experiment_count+1):
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
    standard_scaler_normalizer = StandardScaler()
    normalize_X = standard_scaler_normalizer.fit_transform(X)

    # Before fit a decision tree (using entropy as the criteria for splits) to a shuffled version of the data1, we need to shuffled data1 here.
    # Use a random seed of 0 when shuffling the data, we can use shuffled idxs = np.random.default rng(seed=0).permutation(X.shape[1]).
    shuffled_index = np.random.default_rng(seed=0).permutation(X.shape[1])
    shuffled_X = normalize_X[:, shuffled_index]

    # Apply backward_selection to accomplish the backward elimination algorithm:
    remaining_index_at_round_15 = backward_selection(shuffled_X, y)
    # same as previous, use the shuffled index to get the original index.
    top_5_cal_feature_index_original = shuffled_index[remaining_index_at_round_15]
    # print("The original remaining features index at round 15: ", top_5_cal_feature_index_original.tolist())

    # Calculate the number of the remaining features are actually important features at round 15:
    actual_top_5_feature = np.sum(top_5_cal_feature_index_original < 5)
    # print("The number of the remaining features are actually important features at round 15: ", actual_top_5_feature)
    trial_1000_important_feature_identify.append(actual_top_5_feature)

# Provide a histogram of this metric over the 1000 trials.
np_output = np.array(trial_1000_important_feature_identify)
# Report the average number of good features recovered over the 1000 trials.
average_recovered = np_output.mean()
print(f"Q2c The average number of good features recovered over the 1000 trials: {average_recovered}")

plt.figure(figsize=(10, 6))
plt.hist(np_output, bins=range(5+2), align='left', rwidth=0.7)
plt.xlabel("Actually important features are found in the top 5")
plt.ylabel("Count")
plt.title("Q2c Top 5 good features recovered over the 1000 trials")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()