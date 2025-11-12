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

# (d)
# Repeat part(c)
experiment_count = 1000
log_feature_identify = []
scaled_feature_identify = []
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

    # Use a random seed of 0 when shuffling the data, we can use shuffled idxs = np.random.default rng(seed=0).permutation(X.shape[1]).
    shuffled_index = np.random.default_rng(seed=0).permutation(X.shape[1])
    shuffled_X = X[:, shuffled_index]

# Use logistic regression with no penalty
    logistic_classifier = LogisticRegression(penalty=None, random_state=4)

    # Do this once with and once without scaling the feature matrix
    # Without scaling:
    logistic_classifier.fit(shuffled_X, y)
    # use the absolute value of the coefficient of that feature
    log_output = np.abs(logistic_classifier.coef_[0])
    # np.argsort gives us the index of small value to large value, we use [::-1] to convert and [:5] to get the top 5.
    top_5_log_output_index = np.argsort(log_output)[::-1][:5]
    # After we get the shuffled top 5 index, then get the current top 5 original index:
    top_5_log_output_index_original = shuffled_index[top_5_log_output_index]
    log_actual_top_5_feature = np.sum(shuffled_index[top_5_log_output_index] < 5)
    log_feature_identify.append(log_actual_top_5_feature)

    # With scaling:
    # Normalize your data using sklearn.StandardScaler()
    standard_scaler_normalizer = StandardScaler()
    normalize_X = standard_scaler_normalizer.fit_transform(shuffled_X)
    logistic_classifier.fit(normalize_X, y)
    # use the absolute value of the coefficient of that feature
    scaled_log_output = np.abs(logistic_classifier.coef_[0])
    # np.argsort gives us the index of small value to large value, we use [::-1] to convert and [:5] to get the top 5.
    top_5_scaled_log_output_index = np.argsort(scaled_log_output)[::-1][:5]
    # After we get the shuffled top 5 index, then get the current top 5 original index:
    top_5_scaled_log_output_index_original = shuffled_index[top_5_scaled_log_output_index]
    scaled_log_actual_top_5_feature = np.sum(shuffled_index[top_5_scaled_log_output_index] < 5)
    scaled_feature_identify.append(scaled_log_actual_top_5_feature)

np_scaled_log_output = np.array(scaled_feature_identify)
np_log_output = np.array(log_feature_identify)
# Report the average number of good features recovered over the 1000 trials.
log_average_recovered = np_log_output.mean()
scaled_log_average_recovered = np_scaled_log_output.mean()
print(f"The average number of good features recovered over the 1000 trials(Non-Scaled version): {log_average_recovered}")
print(f"The average number of good features recovered over the 1000 trials(Scaled version): {scaled_log_average_recovered}")


# Plot a histogram as before and report the average number of features recovered over the 1000 trials.
# Compare the scaled and non-scaled versions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(np_log_output, bins=range(5+2), align='left', rwidth=0.7)
plt.xlabel("Actually important features are found in the top 5")
plt.ylabel("Count")
plt.title("Q1d Non-scaled Top 5 good features recovered over the 1000 trials")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
plt.hist(np_scaled_log_output, bins=range(5+2), align='left', rwidth=0.7)
plt.xlabel("Actually important features are found in the top 5")
plt.ylabel("Count")
plt.title("Q1d Scaled Top 5 good features recovered over the 1000 trials")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()