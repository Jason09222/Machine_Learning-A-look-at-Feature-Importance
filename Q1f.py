# z5270589
from matplotlib import pyplot as plt
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
from sklearn.linear_model import LogisticRegression

# (f)
experiment_count = 1000
experiment_count = 1000
trial_1000_important_feature_identify = []
scaled_feature_identify = []
overlap_trial = []
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

    # Normalize your data using sklearn.StandardScaler()
    standard_scaler_normalizer = StandardScaler()
    normalize_X = standard_scaler_normalizer.fit_transform(X)
    # Use a random seed of 0 when shuffling the data, we can use shuffled idxs = np.random.default rng(seed=0).permutation(X.shape[1]).
    shuffled_index = np.random.default_rng(seed=0).permutation(X.shape[1])
    shuffled_X = normalize_X[:, shuffled_index]

    # Fit a decision tree (using entropy as the criteria for splits) to a shuffled version of the data1(X).
    DT_classifier = DecisionTreeClassifier(criterion="entropy", random_state=4)
    DT_classifier.fit(shuffled_X, y)

    # And using its feature importances method, report how many of the actually important features are found in the top 5 important features by the decision tree.
    # We get 20 values here, which represents 20 normarlized entropy valeus of 20 features.
    importances_feature = DT_classifier.feature_importances_

    # Found in the top 5 important features by the decision tree.
    # np.argsort gives us the index of small value to large value, we use [::-1] to convert and [:5] to get the top 5.
    top_5_cal_feature_index_shuffled = np.argsort(importances_feature)[::-1][:5]
    # After we get the shuffled top 5 index, then get the current top 5 original index:
    top_5_cal_feature_index_original = shuffled_index[top_5_cal_feature_index_shuffled]

    # Sicne the original informative features have the index 0-4, we use < 5 to filter that:
    actual_top_5_feature = np.sum(shuffled_index[top_5_cal_feature_index_shuffled] < 5)
    trial_1000_important_feature_identify.append(actual_top_5_feature)


    # Use logistic regression with no penalty
    logistic_classifier = LogisticRegression(penalty=None, random_state=4)
    # With scaling:
    logistic_classifier.fit(shuffled_X, y)
    # use the absolute value of the coefficient of that feature
    scaled_log_output = np.abs(logistic_classifier.coef_[0])
    # np.argsort gives us the index of small value to large value, we use [::-1] to convert and [:5] to get the top 5.
    top_5_scaled_log_output_index = np.argsort(scaled_log_output)[::-1][:5]
    # After we get the shuffled top 5 index, then get the current top 5 original index:
    top_5_scaled_log_output_index_original = shuffled_index[top_5_scaled_log_output_index]
    scaled_log_actual_top_5_feature = np.sum(shuffled_index[top_5_scaled_log_output_index] < 5)
    scaled_feature_identify.append(scaled_log_actual_top_5_feature)

    # Record the number of overlaps for the top-5 ranked features for each of the two models.
    # We use '&' to match the overlap:
    overlap_DT_scaled_log = len(set(top_5_cal_feature_index_original) & set(top_5_scaled_log_output_index_original))
    overlap_trial.append(overlap_DT_scaled_log)

np_scaled_log_output = np.array(scaled_feature_identify)
np_output = np.array(trial_1000_important_feature_identify)

# Report the average number of good features recovered over the 1000 trials.
log_average_recovered = np_output.mean()
scaled_log_average_recovered = np_scaled_log_output.mean()
print(f"The average number of good features recovered over the 1000 trials(Decision Tree version): {log_average_recovered}")
print(f"The average number of good features recovered over the 1000 trials(Scaled version): {scaled_log_average_recovered}")

# Provide a histogram of this metric over the 1000 trials.
np_overlap_trial = np.array(overlap_trial)
# Report the average number of good features recovered over the 1000 trials.
average_overlapped = np_overlap_trial.mean()
print(f"The average number of informative features overlapped of DT and scaled-LR: {average_overlapped}")

plt.figure(figsize=(10, 6))
plt.hist(np_overlap_trial, bins=range(5+2), align='left', rwidth=0.7)
plt.xlabel("Actually important features are found in the top 5")
plt.ylabel("Count")
plt.title("Q1f The number of informative features overlapped of DT and scaled-LR")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()