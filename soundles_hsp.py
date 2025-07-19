from modules.db_soundless import Db
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

db = Db()

df = db.getCluster()
df.loc[df["dB"].isnull(), "dB"] = 0

range_n_clusters = [2, 3, 4, 5, 6]

# X = df[["se", "light", "deep", "rem"]].to_numpy()

pca = PCA(n_components=2).fit(df[["se", "light", "deep", "rem"]])
print(pca.explained_variance_ratio_)
X = pca.transform(df[["se", "light", "deep", "rem"]])

# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)

#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)

#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print(
#         "For n_clusters =",
#         n_clusters,
#         "The average silhouette_score is :",
#         silhouette_avg,
#     )

#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(
#             np.arange(y_lower, y_upper),
#             0,
#             ith_cluster_silhouette_values,
#             facecolor=color,
#             edgecolor=color,
#             alpha=0.7,
#         )

#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples

#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")

#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(
#         X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
#     )

#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(
#         centers[:, 0],
#         centers[:, 1],
#         marker="o",
#         c="white",
#         alpha=1,
#         s=200,
#         edgecolor="k",
#     )

#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")

#     plt.suptitle(
#         "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
#         % n_clusters,
#         fontsize=14,
#         fontweight="bold",
#     )

# plt.show()

clusterer = KMeans(n_clusters=3, random_state=10)
cluster_labels = clusterer.fit_predict(X)

df["cluster"] = cluster_labels

print("SE")
print(df.loc[df["cluster"] == 0, "se"].mean())
print(df.loc[df["cluster"] == 1, "se"].mean())
print(df.loc[df["cluster"] == 2, "se"].mean())
print("LIGHT")
print(df.loc[df["cluster"] == 0, "light"].mean())
print(df.loc[df["cluster"] == 1, "light"].mean())
print(df.loc[df["cluster"] == 2, "light"].mean())
print("DEEP")
print(df.loc[df["cluster"] == 0, "deep"].mean())
print(df.loc[df["cluster"] == 1, "deep"].mean())
print(df.loc[df["cluster"] == 2, "deep"].mean())
print("REM")
print(df.loc[df["cluster"] == 0, "rem"].mean())
print(df.loc[df["cluster"] == 1, "rem"].mean())
print(df.loc[df["cluster"] == 2, "rem"].mean())
print("SOUNDLESS")
print(len(df[(df["dataset"] == "soundless") & (df["cluster"] == 0)]))
print(len(df[(df["dataset"] == "soundless") & (df["cluster"] == 1)]))
print(len(df[(df["dataset"] == "soundless") & (df["cluster"] == 2)]))
print("DECIBELS")
print(df.loc[(df["dataset"] == "soundless") & (df["cluster"] == 0), "dB"].mean())
print(df.loc[(df["dataset"] == "soundless") & (df["cluster"] == 1), "dB"].mean())
print(df.loc[(df["dataset"] == "soundless") & (df["cluster"] == 2), "dB"].mean())


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3)

ax1.set_ylabel('Light sleep percentage of cluster 0')
ax2.set_ylabel('Light sleep percentage of cluster 1')
ax3.set_ylabel('Light sleep percentage of cluster 2')

ax4.set_ylabel('Deep sleep percentage of cluster 0')
ax5.set_ylabel('Deep sleep percentage of cluster 1')
ax6.set_ylabel('Deep sleep percentage of cluster 2')

ax7.set_ylabel('REM sleep percentage of cluster 0')
ax8.set_ylabel('REM sleep percentage of cluster 1')
ax9.set_ylabel('REM sleep percentage of cluster 2')

bplot = ax1.boxplot(df.loc[df["cluster"] == 0, "light"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax2.boxplot(df.loc[df["cluster"] == 1, "light"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax3.boxplot(df.loc[df["cluster"] == 2, "light"],
                   patch_artist=True)  # will be used to label x-ticks


bplot = ax4.boxplot(df.loc[df["cluster"] == 0, "deep"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax5.boxplot(df.loc[df["cluster"] == 1, "deep"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax6.boxplot(df.loc[df["cluster"] == 2, "deep"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax7.boxplot(df.loc[df["cluster"] == 0, "rem"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax8.boxplot(df.loc[df["cluster"] == 1, "rem"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax9.boxplot(df.loc[df["cluster"] == 2, "rem"],
                   patch_artist=True)  # will be used to label x-ticks

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
ax6.set_xticks([])
ax7.set_xticks([])
ax8.set_xticks([])
ax9.set_xticks([])
ax1.grid(visible=True)
ax2.grid(visible=True)
ax3.grid(visible=True)
ax4.grid(visible=True)
ax5.grid(visible=True)
ax6.grid(visible=True)
ax7.grid(visible=True)
ax8.grid(visible=True)
ax9.grid(visible=True)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_ylabel('Sleep efficiency for cluster 0')
ax2.set_ylabel('Sleep efficiency for cluster 1')
ax3.set_ylabel('Sleep efficiency for cluster 2')

bplot = ax1.boxplot(df.loc[df["cluster"] == 0, "se"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax2.boxplot(df.loc[df["cluster"] == 1, "se"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax3.boxplot(df.loc[df["cluster"] == 2, "se"],
                   patch_artist=True)  # will be used to label x-ticks

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax1.grid(visible=True)
ax2.grid(visible=True)
ax3.grid(visible=True)
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.set_ylabel('Decibels for cluster 0')
ax2.set_ylabel('Decibels for cluster 1')
ax3.set_ylabel('Decibels for cluster 2')

bplot = ax1.boxplot(df.loc[(df["dataset"] == "soundless") & (df["cluster"] == 0), "dB"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax2.boxplot(df.loc[(df["dataset"] == "soundless") & (df["cluster"] == 1), "dB"],
                   patch_artist=True)  # will be used to label x-ticks

bplot = ax3.boxplot(df.loc[(df["dataset"] == "soundless") & (df["cluster"] == 2), "dB"],
                   patch_artist=True)  # will be used to label x-ticks

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax1.grid(visible=True)
ax2.grid(visible=True)
ax3.grid(visible=True)
plt.show()

# fig, ax = plt.subplots()
# ax.set_ylabel('Sleep Efficiency Soundless')

# bplot = ax.boxplot(df.loc[df["dataset"] == "soundless", "se"],
#                    patch_artist=True)  # will be used to label x-ticks
# ax.set_xticks([])
# plt.grid(visible=True)
# plt.show()

# fig, ax = plt.subplots()
# ax.set_ylabel('Sleep Efficiency HSP')

# bplot = ax.boxplot(df.loc[df["dataset"] == "hsp", "se"],
#                    patch_artist=True)  # will be used to label x-ticks
# ax.set_xticks([])
# plt.grid(visible=True)
# plt.show()

# fig, ax = plt.subplots()
# ax.set_ylabel('Decibels Soundless >0.6')

# bplot = ax.boxplot(df.loc[(df["dataset"] == "soundless") & (df["se"] >= 0.6), "dB"],
#                    patch_artist=True)  # will be used to label x-ticks
# ax.set_xticks([])
# plt.grid(visible=True)
# plt.show()

# fig, ax = plt.subplots()
# ax.set_ylabel('Decibels Soundless <0.6')

# bplot = ax.boxplot(df.loc[(df["dataset"] == "soundless") & (df["se"] < 0.6), "dB"],
#                    patch_artist=True)  # will be used to label x-ticks
# ax.set_xticks([])
# plt.grid(visible=True)
# plt.show()