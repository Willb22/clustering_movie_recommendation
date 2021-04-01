import os
import ast

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
	
