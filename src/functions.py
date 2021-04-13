import pandas as pd
import numpy as np
import datetime
import os
import ast
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def encoding_dic(data, variable, liste):

   serie_col = data[variable]
   #création de la colonne total : liste des catégories appartenant à la liste pour chaque ligne
   def add(x, liste_col):
       total = []
       if type(x) == str and x[0] == "[":
           a = ast.literal_eval(x)
           if len(a) > 0:
               for j in range(len(a)):
                   comp = a[j]["name"]
                   if comp in liste_col:
                       total.append(comp)
               if len(total) == 0:
                   total.append("autre")
           else:
               total.append("autre")
       return total
   
   total = serie_col.apply(lambda x : add(x, liste_col = liste))
   df = serie_col.to_frame()
   df["total"] = total
   return df

def simulation_output_folder(output, timenow):
	output_dir = output+"output_"+ str(timenow) +'/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	return output_dir
	



def filter_ratings (ratings, trunc_movie_high, trunc_movie_low, trunc_user_high, trunc_user_low):
	
	
	#########Filtrer de ratings les film vus trop ou pas assez
	#########Filtrer de ratings les utilisateurs qui ont vus trop ou pas assez de films
	movie_views = ratings.groupby("movieId")["movieId"].count().reset_index(name= "movie_view_count")
	ratings = pd.merge(ratings, movie_views, left_on="movieId", right_on='movieId', how='inner')
	ratings = ratings[ratings["movie_view_count"]>trunc_movie_low]
	ratings = ratings[ratings["movie_view_count"]<trunc_movie_high]
	ratings = ratings.drop("movie_view_count", axis= 1)
	movie_views = ratings.groupby("movieId")["movieId"].count().reset_index(name= "movie_view_count")
	movie_views = movie_views.sort_values(by=['movie_view_count']) #utilisé pour le graphique
	
	data_user_votes = ratings.groupby(["userId"])["rating"].count().reset_index(name = 'voteCount')
	data_user_votes = data_user_votes[ trunc_user_low  < data_user_votes['voteCount'] ]
	data_user_votes = data_user_votes[  data_user_votes['voteCount'] < trunc_user_high]

	df = data_user_votes.sort_values(by=['voteCount']) #utilisé pour le graphique
	ratings = ratings[np.isin(ratings['userId'], data_user_votes['userId'])]
	
		
	return {'ratings' : ratings, 'movie_views' : movie_views, 'user_votes' : df}


def apply_pca(dimensions, table):

	pca = PCA(n_components=dimensions, random_state=80)
	pca.fit(table)
	table = pd.DataFrame(pca.transform(table))
	
	return table
	
# def add_figure_pdf( x, y, title, xlabel, ylabel):
	# name = plt.figure(figsize=(16, 9))
	# plt.plot(x, y)
	# plt.title(title)
	# plt.xlabel(xlabel)
	# plt.ylabel(ylabel)
	# pp.savefig(name)

def create_kmeans_user_input(ratings,kmeans_centroid_movies, a):
	user_movies = ratings.groupby(['userId', 'Kmeans_movies_cluster'])
	df_score = user_movies['rating'].apply(lambda x : (1-a)*float(5) + a*sum(list(x))/len(list(x)) if float(5) in list(x) else sum(list(x))/len(list(x))).reset_index(name = 'score')
	df_users = df_score.pivot(index = 'userId', columns = 'Kmeans_movies_cluster').reset_index()
	df_users = df_users.replace(np.nan, 0)

	names = []
	for i in range(kmeans_centroid_movies):
		name = "k"+str(i)
		names.append(name)
	df_users.columns= ["userId"] + names # flatten multi-index columns from pivot operation

	#On calcule la moyenne des notes par utilisateur afin de ne pas se retrouver avec un clustering de type :
	# groupe des utilisteurs qui notent bien, groupe des utilisateurs qui notent mal ,ect

	moy_by_user = ratings.groupby("userId")["rating"].mean().reset_index(name="note_moy_user")
	data =  pd.merge(df_users, moy_by_user, left_on = "userId", right_on="userId")

	#data = data.drop("userId", axis=1)
	for col in data.drop("note_moy_user", axis=1).columns:
		data[col] = data[col]/data["note_moy_user"]
	data = data.drop("note_moy_user", axis=1)
	data = data.replace(0, 1)


	return data
	


def generate_kmeans_inertia(n_centroids, data_kmeans_fit):
	inertie = []
	#n_centroids = coude_centroid_users
	for i in range(1, n_centroids):
		kmeans = KMeans(n_clusters=i).fit(data_kmeans_fit)
		inertie.append(kmeans.inertia_)
	
	return inertie
