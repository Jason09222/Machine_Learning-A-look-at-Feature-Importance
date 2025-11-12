# z5270589
from matplotlib import pyplot as plt
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

# Question 1
# (a)
# Generate a dataset of two classes using sklearn.datasets.make classification.
X, y = make_classification(
    n_samples=1000,   # 1000 observations.
    n_features=20,    # 20 features.
    n_informative=5,  # Set 5 of those features to be informative.
    n_redundant=15,   # n_redundant = 20-5=15.
    shuffle=False,    # Be sure to set the shuffle parameter to False.
    random_state=0    # Use a random seed of 0.
)

# Normalize your data using sklearn.StandardScaler()
standard_scaler_normalizer = StandardScaler()
normalize_X = standard_scaler_normalizer.fit_transform(X)

# Before fit a decision tree (using entropy as the criteria for splits) to a shuffled version of the data1, we need to shuffled data1 here.
# Use a random seed of 0 when shuffling the data, we can use shuffled idxs = np.random.default rng(seed=0).permutation(X.shape[1]).
shuffled_index = np.random.default_rng(seed=0).permutation(X.shape[1])
shuffled_X = normalize_X[:, shuffled_index]

# Fit a decision tree (using entropy as the criteria for splits) to a shuffled version of the data1(X).
# For fitting models, always use a random seed (or random state) of 4 for reproducibility.
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
# Report how many of the actually important features are found in the top 5 important features by the decision tree.
print(f"Original index for the calculated 5 important features: {top_5_cal_feature_index_original}")
print(f"The actually important features are found in the top 5 important features by the decision tree: {actual_top_5_feature}")

# Similarily to before, now we do not just get the top 5, we sort all 20 values from large to small and gets their index.
cal_feature_index_shuffled = np.argsort(importances_feature)[::-1]

# Get the 20 reordered important features value from large to small.
reorder_importances_feature = importances_feature[cal_feature_index_shuffled]

# Similarly, point the index of shuffled feature to its original index:
cal_feature_index_original = shuffled_index[cal_feature_index_shuffled]

# Plot a histogram with x-axis showing the features ranked in decreasing order of importance, and the y-axis showing the feature importance score.
plt.figure(figsize=(10, 5))
plt.bar(range(20), reorder_importances_feature)
plt.xticks(ticks=range(20), labels=cal_feature_index_original, rotation=30)
plt.xlabel("Features ranked in decreasing order of importance(original index)")
plt.ylabel("Feature importance score")
plt.title("Q1a Feature importances score ranked by DT")
plt.tight_layout()
plt.show()