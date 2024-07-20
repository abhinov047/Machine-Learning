import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

filename = "ApneaData.pkl"
features = []
classes = []

f = open(filename, 'rb')
data = pickle.load(f)
f.close()

for row in data:
    features.append(row[:-1])
    classes.append(row[-1])

# Convert features list to numpy array
features = np.array(features)

pca = PCA(n_components=3)
reduced_features = pca.fit_transform(features)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# Create a boolean mask to separate the positive and negative cases
positive_mask = np.array(classes) == 1
negative_mask = np.array(classes) == 0

# Plot the positive and negative cases with the correct colors
ax1.scatter(reduced_features[positive_mask, 0], reduced_features[positive_mask, 1], reduced_features[positive_mask, 2], c='r', marker='o', label='Sleep Apnea Positive')
ax1.scatter(reduced_features[negative_mask, 0], reduced_features[negative_mask, 1], reduced_features[negative_mask, 2], c='g', marker='^', label='Sleep Apnea Negative')


ax1.legend()
plt.title('3D Visualization of Sleep Apnea Data')
plt.show()

# Interpret the plotted data
print("Interpretation of the Plotted Data:")

# Calculate the percentage of variance explained by each principal component
pca_variance = pca.explained_variance_ratio_
print("Percentage of variance explained by each principal component:")
for i, variance in enumerate(pca_variance):
    print(f"Principal Component {i+1}: {variance*100:.2f}%")

# Identify the most important features
most_important_features = np.argsort(pca.components_, axis=1)[:, -3:]
print("Most important features:")
for i in range(3):
    print(f"Principal Component {i+1}: {', '.join(map(str, most_important_features[i]))}")

# Analyze the separation between positive and negative cases
positive_centroid = reduced_features[positive_mask].mean(axis=0)
negative_centroid = reduced_features[negative_mask].mean(axis=0)
distance = np.linalg.norm(positive_centroid - negative_centroid)
print(f"Distance between positive and negative case centroids: {distance:.2f}")

