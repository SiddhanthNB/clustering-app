import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
plt.style.use('fivethirtyeight')

st.title("CLUSTERING ANALYSIS APP")
st.write(""" ### This app is powered by Machine Learning and uses *K-Means clustering* from *sci-kit learn* for cluster analysis. """)
st.write('---')

st.sidebar.title('User Input Features')
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file  = st.sidebar.file_uploader("Upload your input CSV file (NUMERIC DATA ONLY)", type=["csv"])
st.set_option('deprecation.showfileUploaderEncoding', False)

dataset = st.sidebar.selectbox(" OR Choose from these datasets", ["Cars Dataset","Credit Card Dataset","Iris Dataset","Wine Dataset"])
VAL = st.sidebar.slider('Number of Clusters', 2, 6)


if uploaded_file is not None:
    X = pd.read_csv(uploaded_file)
    X = X.dropna()

elif dataset == "Iris Dataset" :
	dataset = pd.read_csv('https://github.com/SiddhanthNB/clustering-app/raw/master/Dataset/iris.csv')
	X = dataset.copy()
	X = X.dropna()
	X = X.drop(['Species'], axis = 1)

	st.write(""" 
				#### About the dataset:
				The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.

				It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.
				""")
	st.write('---')

elif dataset == "Wine Dataset" :
	data = pd.read_csv("https://github.com/SiddhanthNB/clustering-app/raw/master/Dataset/wine.csv")
	X = data.copy()
	y = X.pop("Proline") 

	st.write(""" 
				#### About the dataset:
				The data set that we are going to analyze in this post is a result of a chemical analysis of wines grown in a particular region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines. The attributes are: Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, and Proline. The data set has 178 observations and no missing values.
				""")
	st.write('---')

elif dataset == "Cars Dataset" :
	dataset = pd.read_csv('https://github.com/SiddhanthNB/clustering-app/raw/master/Dataset/cars.csv')
	X = dataset.copy()
	y = X.pop("brand")

	st.write(""" 
				#### About the dataset:
				Cars Data has Information about 3 brands/make of cars. Namely US, Japan, Europe. Target of the data set to find the brand of a car using the parameters such as horsepower, Cubic inches, Make year, etc.
				""")
	st.write('---')

elif dataset == "Credit Card Dataset" :
	dataset = pd.read_csv('https://github.com/SiddhanthNB/clustering-app/raw/master/Dataset/card%20transactions.csv')
	X = dataset.copy()
	X = X.dropna()
	X = X.drop(['CUST_ID'], axis = 1)

	st.write(""" 
				#### About the dataset:
				This Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables. And it is used to develop a customer segmentation to define marketing strategy.
				""")
	st.write('---')


## Reducing the components of data
from sklearn.decomposition import PCA
pca= PCA(3)
X_projected = pca.fit_transform(X)
data = pd.DataFrame(data = X_projected, index=None, columns=["x1", "x2", "x3"])


## Using silhouette score to find optimum no. of clusters
from sklearn.metrics import silhouette_score
range_clusters = [2, 3, 4, 5, 6]
silhouette_dict= {}

for num_clusters in range_clusters:
	kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
	kmeans.fit(data)
    
	cluster_labels = kmeans.labels_
    
	score = silhouette_score(data, cluster_labels)
	silhouette_dict.update({num_clusters:score}) 

inverse = [(value, key) for key, value in silhouette_dict.items()]
recommend = max(inverse)[1]


## Credits
st.sidebar.title("Credits")
st.sidebar.info(
		"This work is greatly inspired by: \n [Roshan's Kaggle kernel](https://www.kaggle.com/roshansharma/mall-customers-clustering-analysis) \n [DataProfessor Youtube Channel](https://www.youtube.com/channel/UCV8e2g4IWQqK71bbzGDEI4Q) \n\n"
		"This project is maintained by [Siddhanth]() and [click here]() to check the code."
)


## Plotting
st.write(""" ### 3-D Scatter Plot: """)
x = data[['x1', 'x2', 'x3']].values
km = KMeans(n_clusters = VAL, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
km.fit(x)
labels = km.labels_

data['labels'] =  labels
trace1 = go.Scatter3d(
    x= data['x1'],
    y= data['x2'],
    z= data['x3'],
    mode='markers',
     marker=dict(
        color = data['labels'], 
        size= 10,
        line=dict(
            color= data['labels'],
            width= 12
        ),
        opacity=0.8
     )
)
df = [trace1]

layout = go.Layout(
    title = 'Clustering Analysis',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    ),
    scene = dict(
            xaxis = dict(title  = 'Principal Component 1'),
            yaxis = dict(title  = 'Principal Component 2'),
            zaxis = dict(title  = 'Principal Component 3')
        )
)

fig = go.Figure(data = df, layout = layout)
st.write(fig)

st.write("This app takes multivariate dataset as input and performs Principal Component Analysis(PCA) to find 3 Principal eigen values as axes and uses them for a 3-dimensional Scatter plot.")
st.write("Algorithm analyses silhouette score and recommends", recommend,"or", recommend+1, "Clusters for this Dataset. User can vary *Number of Clusters* upto 6 clusters using the slider in the *User Input Interface*." )

