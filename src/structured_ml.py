#!/usr/bin/python3

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
from variables import input_dir, output_dir
from functions import encoding_dic, simulation_output_folder, filter_ratings, create_kmeans_user_input, generate_kmeans_inertia

################### Parametres simulation ##############################
remove_col_kmeans_movies = ['id','title','vote_average', 'vote_count']
trunc_user_high = 80 #nombre max de vues total par user
trunc_user_low = 20 #nombre min de vues total par user
trunc_movie_low = 100
trunc_movie_high = 1200
coude_centroid_movies = 13
kmeans_centroid_movies = 4
coude_centroid_users =9
kmeans_centroid_users = 5
n = 5 #nombre de films à recommander

p_c_a = True # Activer ou pas la Principal Component analysis
acp_dim = 20 # Principal Component analysis

###### Modelisation Kmeans utlisateur #########
a=1
# valeur de a doit être comprise entre zero et un!!!
# Pour a = 0,
# Si l'utilisateur a offert un rating de 5 a un des films. score vaut 5
# Si l'utilisateur n'a offert un rating de 5 a aucun un des films. score vaut le score moyen qu'il a offert aux films
# Pour a =1,
# Le score vaut toujours le score moyen qu'il a offert aux films
#plus a augmente et se rapproche de 1, plus l'ecart entre avoir son film préfére dans un cluster ou pas diminue
b = 0.99 #définir un score de film pour chaque cluster d'utilisateur  : b*moyenne_note + (1-b)*part



##########################################################################
date_time = datetime.datetime.now()
################### fichier output ##############################

################### fichier input et output ###############################
_raw_input = input_dir
_processed_input = '../data/processed/'
output_dir = simulation_output_folder(output = output_dir, timenow = date_time)
################################################################

    
content_file = 'Colonnes retirées de tableau_movies'  + str(remove_col_kmeans_movies) +' \n'
content_file += 'On retire les utilisateurs ayant vu plus de ' + str(trunc_user_high) +'  films' +' \n'
content_file += 'On retire les utilisateurs ayant vu moins de ' + str(trunc_user_low) +'  films' +' \n'
content_file += 'On retire les films vus plus de ' + str(trunc_movie_high) +'  fois' +' \n'
content_file += 'On retire les films vus moins de ' + str(trunc_movie_low) +'  fois' +' \n'
content_file += 'Kmeans utilisateur avec parametre de combinaison linéaire ' + str(a) +'  ' +' \n'
content_file += ' Nombre de centroides Kmeans Movies ' + str(kmeans_centroid_movies) +'  ' +' \n'
content_file += ' Nombre de centroides Kmeans utilisateur ' + str(kmeans_centroid_users) +'  ' +' \n'
if p_c_a:
    content_file += 'Principal Component Analysis activée et réduit à  ' + str(acp_dim ) + ' dimensions' + '\n'
else:
    content_file += 'Principal Component Analysis inactive  '



################### fichier pdf ##############################
pp = PdfPages(output_dir+ ' Récapitulatif graphiques '+ str(date_time)+ ' .pdf')
firstPage = plt.figure(figsize=(11.69,8.27))
firstPage.clf()
txt = content_file
firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=12, ha="center")
pp.savefig()


def add_figure_pdf( x, y, title, xlabel, ylabel):
	name = plt.figure(figsize=(16, 9))
	plt.plot(x, y)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	pp.savefig(name)





################ Lecture et tri de ratings et tableau_movies ########################
tableau_movies_full = pd.read_csv(_processed_input + "final_data_movie.csv")
ratings = pd.read_csv(_raw_input + "ratings.csv")


################ Filtrer ratings et tableau_movies ########################

ratings  = ratings.drop(['timestamp'], axis = 1)
tableau_movies_full = tableau_movies_full.drop_duplicates("id")
all_movies = list(tableau_movies_full["id"])
ratings = ratings[ratings["movieId"].isin(all_movies)]
del all_movies
filtered_dict = filter_ratings (ratings, trunc_movie_high, trunc_movie_low, trunc_user_high, trunc_user_low)
ratings = filtered_dict['ratings']
movie_views = filtered_dict['movie_views']
user_votes = filtered_dict['user_votes']
del filtered_dict




#####  Add movie titles to ratings
ratings = pd.merge(ratings, tableau_movies_full[["id", "title"]], left_on = "movieId", right_on="id")

#########Filtrer de tableau_movies les colonnes inutiles pour le Kmeans
tableau_movies = tableau_movies_full.drop(tableau_movies_full[remove_col_kmeans_movies], axis = 1)

#del data_user_votes

######################################################################
######################## FIGURE ##############################



add_figure_pdf(x= pd.Series(range(0, len(user_votes['userId']))),y= user_votes['voteCount'], title = "Repartition du nombre total de vues par utilisateur", xlabel = "utilisateurs", ylabel = "Nombre de vues au total par utilisateur")
#del df
add_figure_pdf(x= pd.Series(range(0, len(movie_views['movieId']))),y= movie_views['movie_view_count'], title = "Repartition du nombre total de vues par film", xlabel = "movies", ylabel ="Nombre de vues par film")



#####################################################################################




# ####################### Principle Component Analysis #####################################

if p_c_a:
    pca = PCA(n_components=acp_dim, random_state=80)
    pca.fit(tableau_movies)
    tableau_movies = pd.DataFrame(pca.transform(tableau_movies))



####################### Kmeans sur le récapiptulatif de films #############################


######################## Generate Elbow curve Kmeans Movies ##############################
Inertie = generate_kmeans_inertia( coude_centroid_movies, tableau_movies)

# n_centroids = coude_centroid_movies
# for i in range(1, n_centroids):
    # kmeans = KMeans(n_clusters=i).fit(tableau_movies)
    # Inertie.append(kmeans.inertia_)

add_figure_pdf(x = range(1, coude_centroid_movies), y=Inertie, title = 'Critere de Coude Kmeans movies', xlabel ='Nombre de clusters', ylabel = 'Inertie')



######################## Apply Kmeans on Movies with number of clusters as a defined parameter #######################
kmeans = KMeans(n_clusters=kmeans_centroid_movies).fit(tableau_movies)
centroids = kmeans.cluster_centers_

######################## On ajoute le numero de cluster movies à ratings et à la table tableau_movies_full

movies = pd.DataFrame({'id': tableau_movies_full['id'], 'Kmeans_movies_cluster': kmeans.labels_})

ratings = pd.merge(ratings, movies, left_on = "movieId", right_on = "id")
tableau_movies_full = pd.merge(tableau_movies_full, movies, left_on = "id", right_on = "id")




######################## Preparer input kmeans users ##############################################
##permet d'avoir une info sur le cluster de film en complément du tableau ratings

df_kmeans_users_ids = create_kmeans_user_input(ratings,kmeans_centroid_movies, a)
#df_kmeans_users = df_kmeans_users_ids.drop("userId", axis=1)
# user_movies = ratings.groupby(['userId', 'Kmeans_movies_cluster'])
# df_score = user_movies['rating'].apply(lambda x : (1-a)*float(5) + a*sum(list(x))/len(list(x)) if float(5) in list(x) else sum(list(x))/len(list(x))).reset_index(name = 'score')
# df_users = df_score.pivot(index = 'userId', columns = 'Kmeans_movies_cluster').reset_index()
# df_users = df_users.replace(np.nan, 0)

# names = []
# for i in range(kmeans_centroid_movies):
    # name = "k"+str(i)
    # names.append(name)
# df_users.columns= ["userId"] + names # flatten multi-index columns from pivot operation

# #On calcule la moyenne des notes par utilisateur afin de ne pas se retrouver avec un clustering de type :
# # groupe des utilisteurs qui notent bien, groupe des utilisateurs qui notent mal ,ect

# moy_by_user = ratings.groupby("userId")["rating"].mean().reset_index(name="note_moy_user")
# data =  pd.merge(df_users, moy_by_user, left_on = "userId", right_on="userId")

# data = data.drop("userId", axis=1)
# for col in data.drop("note_moy_user", axis=1).columns:
    # data[col] = data[col]/data["note_moy_user"]
# data = data.drop("note_moy_user", axis=1)
# data = data.replace(0, 1)


# df_kmeans_users = data

# del moy_by_user
# del df_score
# del data
# del names
#################################################################################################





############################## Kmeans utilisateurs #######################################

######################## Generate Elbow curve Kmeans Users ##############################
Inertie = generate_kmeans_inertia( coude_centroid_users, df_kmeans_users_ids.loc[ : , df_kmeans_users_ids.columns != 'userId'] ) 
# n_centroids = coude_centroid_users
# for i in range(1, n_centroids):
    # kmeans = KMeans(n_clusters=i).fit(df_kmeans_users.loc[ : , df_kmeans_users.columns != 'userId'])
    # Inertie.append(kmeans.inertia_)

add_figure_pdf(x = range(1, coude_centroid_users), y=Inertie, title = 'Critere de Coude Kmeans utilisateurs', xlabel ='Nombre de clusters', ylabel = 'Inertie')



######################## Apply Kmeans on Users with number of clusters as a defined parameter #######################
kmeans = KMeans(n_clusters=kmeans_centroid_users).fit(df_kmeans_users_ids.loc[ : , df_kmeans_users_ids.columns != 'userId'])
centroids = kmeans.cluster_centers_

#user_clusters = pd.DataFrame({'userId': df_users["userId"], 'Kmeans_user_cluster': kmeans.labels_})
user_clusters = pd.DataFrame({'userId': df_kmeans_users_ids["userId"], 'Kmeans_user_cluster': kmeans.labels_}) #use of function 
# On ajoute le clustering à la tale ratings
ratings = pd.merge(ratings, user_clusters, left_on="userId", right_on="userId")



# nombre d'utilisateurs par cluster
nb_users_cluster = ratings.drop_duplicates("userId").groupby('Kmeans_user_cluster')["rating"].count().reset_index()


del user_clusters
#del df_kmeans_users
#del df_users
del movies
###########################################################################################





############################### Stats descriptives ############################################

### moyenne et compte de chaque duo film/cluster user
# links = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].count().reset_index(name="count")
# moyenne = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].mean().reset_index(name="mean")
# variance = ratings.groupby(["Kmeans_user_cluster","movieId"])["rating"].var().reset_index(name="variance")
# links["mean"] = moyenne["mean"]
# links["variance"] = variance["variance"]

# del moyenne
# del variance

# ### récupérer les meilleurs films par cluster user selon la mean
# parmi_combien = 300
# best_movies_per_cluster = links.sort_values(["Kmeans_user_cluster",'mean'],ascending=False).groupby("Kmeans_user_cluster").head(parmi_combien).reset_index(drop=True)
# best_movies_per_cluster["nb_user_cluster"] = pd.merge(best_movies_per_cluster, nb_users_cluster, left_on="Kmeans_user_cluster", right_on = "Kmeans_user_cluster")["rating"]
# best_movies_per_cluster["part"] = (best_movies_per_cluster["count"] / best_movies_per_cluster["nb_user_cluster"]) * 100
# best_movies_per_cluster["score"] = b*best_movies_per_cluster["mean"] + (1-b)*best_movies_per_cluster["part"]
# best_movies_per_cluster = pd.merge(best_movies_per_cluster, tableau_movies_full, left_on = "movieId", right_on = "id")[["title", "Kmeans_user_cluster","Kmeans_movies_cluster", "mean", "part" ,"count", "variance", "nb_user_cluster"]].sort_values(["Kmeans_user_cluster", "mean"], ascending = False).reset_index()

# # Lien cluster movies, cluster user
# contingence_clusteruser_clustermovie = best_movies_per_cluster.groupby(["Kmeans_user_cluster", "Kmeans_movies_cluster"])["mean"].count().reset_index(name="count_per_cluster")
# contingence_clusteruser_clustermovie = contingence_clusteruser_clustermovie.pivot(index = "Kmeans_user_cluster", columns = "Kmeans_movies_cluster", values = "count_per_cluster")

# # visualisation des meilleurs films par cluster
# visualisation_meilleurs_film_par_cluster = best_movies_per_cluster.pivot( columns = 'Kmeans_user_cluster', values ="title" )

# for i in range(kmeans_centroid_users):
    # index = (kmeans_centroid_users - 1 - i) *parmi_combien
    # values = visualisation_meilleurs_film_par_cluster.loc[index:index+parmi_combien-1,i].reset_index(drop=True)
    # visualisation_meilleurs_film_par_cluster[i] = values
    
# visualisation_meilleurs_film_par_cluster = visualisation_meilleurs_film_par_cluster[0:parmi_combien-1]
# ####################################################################################################




# ############################ Recommendations pour chaque utilisateur ###############################
# # On veut maintenant recommander n films  à chaque utilisateur
# # On commence par le meilleur film selon son groupe et on descend jusqu'à qu'il y ait n films à lui recommander

# def delete_if_in_other(row):
    # return [x for x in row["recommended"] if x not in row["title"]][0:n]

# def recommendations():
    # for i in range(kmeans_centroid_users):
        # films = list(best_movies_per_cluster[best_movies_per_cluster["Kmeans_user_cluster"]==i]["title"])
        # users = ratings[ratings["Kmeans_user_cluster"]==i]
        # user_movie_list = users.groupby("userId")["title"].apply(list).reset_index()
        # user_movie_list["recommended"] = [films for _ in range(len(user_movie_list))]
        # user_movie_list["recommended"] = user_movie_list.apply(delete_if_in_other, axis=1)
        # user_movie_list = user_movie_list.drop("title", axis=1)
        
        # if i == 0:
            # toutes_les_recommendations = user_movie_list
        # else:
            # toutes_les_recommendations = pd.concat([toutes_les_recommendations, user_movie_list])
    
    # del user_movie_list
    # del films
    # del users
    # return toutes_les_recommendations

# recommendations = recommendations()




# # Sauvegarde des recommendations
# recommendations.to_csv(output_dir + "recommendations.csv", index= False)











































# #Graphs (à mettre en commentaires)

# a = pd.read_csv(_raw_input+"/metadata_carac_speciaux.csv")

# a=a.dropna(subset=['id'])
# a=a.dropna(subset=['title'])
# a=a.loc[a['status']== 'Released']
# a=pd.get_dummies(a, columns=["adult"]) 
# a=a.drop_duplicates()
# a=a.drop_duplicates(subset='id', keep="first")
# a=a.drop_duplicates(subset='title', keep="first")
# a=a.reset_index(drop=True)



# liste_genre = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Documentary']
# liste_prod_comp = ['WarnerBros.', 'Metro-Goldwyn-MayerMGM', 'ParamountPictures', 'TwentiethCenturyFoxFilmCorporation', 'UniversalPictures', 'ColumbiaPicturesCorporation', 'Canal', 'ColumbiaPictures', 'RKORadioPictures']
# liste_prod_count = ['UnitedStatesofAmerica', 'null', 'UnitedKingdom', 'France', 'Germany', 'Italy', 'Canada', 'Japan', 'Spain', 'Russia']


# genres = encoding_dic(a, "genres", liste_genre)
# genres["genre"] = genres["total"].apply(lambda x :  x[0])
# prod_comp = encoding_dic(a, "production_companies", liste_prod_comp)
# prod_comp["prod_comp"] = prod_comp["total"].apply(lambda x :  x[0])
# prod_count = encoding_dic(a, "production_countries", liste_prod_count)
# prod_count["prod_count"] = prod_count["total"].apply(lambda x :  x[0])

# b = pd.concat([a["title"], genres["genre"], prod_comp["prod_comp"], prod_count["prod_count"]], axis = 1)

# c = pd.merge(best_movies_per_cluster, b, left_on = "title", right_on = "title").sort_values(["Kmeans_user_cluster", "mean"], ascending=False)

# ##Save for graphs
# c.to_csv(output_dir+"best_movies_per_cluster.csv", index= False)
# #




##################  FIN DE SIMULATION ###########
pp.close()
date_timeend = datetime.datetime.now()
runtime = date_timeend - date_time

