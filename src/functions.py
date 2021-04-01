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
	

def filter_ratings (ratings, tableau_movies_full, trunc_movie_high, trunc_movie_low, trunc_user_high, trunc_user_low):
	
	
	#########Filtrer de ratings les film vus trop ou pas assez
	nbr_votes_movie = ratings.groupby("movieId")["movieId"].count().reset_index(name= "count_movie")
	ratings = pd.merge(ratings, nbr_votes_movie, left_on="movieId", right_on='movieId', how='inner')
	ratings = ratings[ratings["count_movie"]>trunc_movie_low]
	ratings = ratings[ratings["count_movie"]<trunc_movie_high]
	ratings = ratings.drop("count_movie", axis= 1)
	
	return ratings


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
