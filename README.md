## Module-10-Challenge-on-Machine-Learning
Clustering and Optimizing Cluster for Prediction

### Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### Load the data into a Pandas DataFrame
df_market_data=pd.read_csv(Path("./Resources/crypto_market_data.csv"),index_col="coin_id")
### Display sample data
df_market_data.head()

### Generate summary statistics
df_market_data.describe()

# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(withdth =400,
                          length =600,rot =90)

### Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)
#Review Transformed Data
scaled_data

# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)
#Review Data
df_market_data_scaled.head()


### Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

### Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

### Display sample data
df_market_data_scaled.head()

# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empy list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. KMeans model using the loop counter for the n_clusters
# 2. Fitting the model to the data using `df_market_data_scaled`
# 3. Appending the model.inertia_ to the inertia list
for i in k:
    k_model = KMeans(n_clusters=i, random_state=1)
    k_model.fit(df_market_data_scaled)
    inertia.append(k_model.inertia_)

### Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, "inertia": inertia}

### Create a DataFrame with the data to plot the Elbow curve
df_elbow = pd.DataFrame(elbow_data)

### Review the DataFrame
df_elbow.head()

### Plot a line chart with all the inertia values computed with 
### the different values of k to visually identify the optimal value for k.
df_elbow_curve = df_elbow.hvplot.line(
    x="k", 
    y="inertia", 
    title="Elbow Curve", 
    xticks=k
)
df_elbow_curve

### printing the best K-Vlaue
print(f"The Best K-Value is 4")

###Cluster Cryptocurrencies with K-means Using the Original Data

### Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, random_state=0)

### Fit the model
model.fit(df_market_data_scaled)


### Make predictions
kmeans_predictions = model.predict(df_market_data_scaled)

### View the resulting array of cluster values.
kmeans_predictions

### Create a copy of the DataFrame to add the prediction data 
df_market_prediction = df_market_data_scaled.copy()
df_market_prediction.head()
df_market_prediction.tail()

### Add a new column to the DataFrame with the predicted clusters
df_market_prediction["predicted clusters"] =kmeans_predictions
df_market_prediction.head()


### Create a scatter plot using hvPlot by setting 
### `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
### Color the graph points with the labels found using K-Means and 
### add the crypto name in the `hover_cols` parameter to identify 
### the cryptocurrency represented by each data point.
### by="predicted clusters" has not worked
df_market_data_scaled.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    hover_cols = ["coin_id"], 
    title = "Scatter Plot by Stock Segment - k=3"
)

### Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

### Use the PCA model with `fit_transform` to reduce to 
### three principal components.
stocks_pca_data = pca.fit_transform(df_market_data_scaled)

### View the first five rows of the DataFrame. 
stocks_pca_data[:5]

### Retrieve the explained variance to determine how much information 
### can be attributed to each principal component.
pca.explained_variance_ratio_


### Creating a DataFrame with the PCA data
df_stocks_pca =pd.DataFrame(stocks_pca_data,columns=["PC1","PC2","PC3"])

### Copy the crypto names from the original data
df_stocks_pca["coin_id"] = df_market_data_scaled.index

### Set the coinid column as index
df_stocks_pca = df_stocks_pca.set_index("coin_id")

### Display sample data
df_stocks_pca.head()

### Find the Best Value for k Using the PCA Data

### Create a list with the number of k-values to try
k =list(range(1,11))

### Create an empy list to store the inertia values
inertia = []

### Create a for loop to compute the inertia with each possible value of k
### Inside the loop:
### 1. Create a KMeans model using the loop counter for the n_clusters
### 2. Fit the model to the data using `df_market_data_pca`
### 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_stocks_pca)
    inertia.append(model.inertia_)

### Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {
    "k": k,
    "inertia": inertia
}

### Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)

### Plot a line chart with all the inertia values computed with 
### the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow_pca.hvplot.line(x="k", y="inertia", title="Elbow Curve Using PCA Data", xticks=k)
elbow_plot_pca

### Initialize the K-Means model using the best value for k
k_model = KMeans(n_clusters=4)
### Fit the K-Means model using the PCA data
model.fit(df_stocks_pca)

### Predict the clusters to group the cryptocurrencies using the PCA data
coin_clusters = model.predict(df_stocks_pca)

### View the resulting array of cluster values.
coin_clusters

### Create a copy of the DataFrame with the PCA data
df_stocks_pca_predictions = df_stocks_pca.copy()
### Add a new column to the DataFrame with the predicted clusters
df_stocks_pca_predictions["predicted clusters"] = coin_clusters

### Display sample data
df_stocks_pca_predictions.head()
df_coin_clusters_pca =pd.DataFrame(coin_clusters, columns=["PC1"])

### Create a scatter plot using hvPlot by setting 
### `x="PC1"` and `y="PC2"`. 
### Color the graph points with the labels found using K-Means and 
### add the crypto name in the `hover_cols` parameter to identify 
### the cryptocurrency represented by each data point.
hv_plot_coin= df_stocks_pca_predictions.hvplot.scatter(
    x="PC1",
    y="PC2",
    by="predicted clusters",
    hover_cols = ["coin_id"], 
)
hv_plot_coin

### Visualize and Compare the Results
### Composite plot to contrast the Elbow curves
elbow_plot_pca*df_elbow_curve

### Compoosite plot to contrast the clusters
hv_plot_coin