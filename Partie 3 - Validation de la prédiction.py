#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#je commence par importer le DataFrame
df = pd.read_csv('/Users/octoberone/Desktop/DataFrame GitHub/DataFrame/Prédiction des ventes liées au marketing.csv')


# In[3]:


#affichage des 5 premières lignes
df.head()


# In[4]:


#voici le nombre de lignes et de colonnes sans le détail
df.shape


# In[5]:


#voici les détails des colonnes avec beaucoup plus d'informations (variables, types, le nombre de valeurs nulles)
df.info()


# In[6]:


#par souci du détail, j'ai utilisé cette commande pour être sûr qu'il n'y a pas de valeurs manquantes
df.isnull().sum()
#par conséquent, j'en déduis que ce DataFrame est plutôt agréable à utiliser dans un premier temps


# In[7]:


#Afin d'avoir plus de détails statistiques, j'affiche les colonnes numériques

#count%:nombre de valeurs nulles
#mean%:la moyenne des valeurs
#std%:l'écart type (dispersion valeurs)
#min%:le min de la colonne 
#25%:1er quatile à moins de 25
#50%:2er quatile à moins de 25
#75%:3er quatile à moins de 25
#max:le maxi de la colonne 

print(df.describe())


# In[8]:


#premier affichage sous forme de nuage de points afin de visualiser la tendance et son évolution dans le temps des ventes liées à la publicité sur la TV 
#conclusion : j'observe une tendance progressive avec des données progressives d'un taux d'engagement lié à la publicité sur la TV

plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x="TV", y="Sales", color= "r", data=df)
plt.title("tendance entre les dépenses TV et les ventes")


# In[ ]:





# In[9]:


#deuxième affichage sous forme de nuage de points afin de visualiser la tendance et son évolution dans le temps des ventes liées à la publicité sur la Radio
#conclusion : nous n'avons pas de tendance, mais les publicités émises ont un impact relatif sur le CA

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 2)
sns.scatterplot(x="Radio", y="Sales",color= "g", data=df)
plt.title("tendance entre les dépenses radio et les ventes")


# In[10]:


#troisième affichage sous forme de nuage de points afin de visualiser la tendance et son évolution dans le temps des ventes liées à la publicité sous forme de papier 
#conclusion : nous n'avons pas de tendance, mais les publicités émises ont moins d'impact sur le CA

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 2)
sns.scatterplot(x="Newspaper", y="Sales",color= "black",  data=df)
plt.title("tendance entre les dépenses newspaper et les ventes")


# In[11]:


#afin d'avoir une visualisation plus mathématique,
#j'ai pris la décision de regarder les tendances avec l'aide de matrices de corrélation
#pour pouvoir avoir des données exploitables, j'ai préféré les afficher également sous forme de chiffres ci-dessous 
sns.pairplot(df, vars=["TV", "Radio", "Newspaper"])
plt.show()


# In[12]:


#dans cette matrice, j'observe
#TV: a une corrélation plutôt positive avec un coefficient de += 1
#Radio : a une corrélation plutôt négative par rapport à la TV avec un coef de -= 0 
#Newspaper : a une corrélation également plutôt négative par rapport à la TV avec un coef de -= 0

corr = df[["TV", "Radio", "Newspaper"]].corr()
print(corr)


# In[13]:


#premièrement, je vais séparer les colonnes dans la variable X sauf "Sales" dans Y
#pour isoler les caractéristiques de l'entraînement
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# In[14]:


#maintenant je vais fractionner le jeu de données
#pour tester la performance de la prédiction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


#ensuite une normalisation
#pour améliorer les résultats
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[24]:


#par la suite je crée une régression linéaire pour orienter le model avec ce parametre 
#(un parametre qui aide a construire une prédiction)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# In[34]:


#et enfin de calculer la performance de la régression sur l'entraînement.
score_train = linear_model.score(X_train , y_train)
print(score_train)


# In[33]:


#et aussi la régression sur les tests
score_test = linear_model.score(X_test , y_test)
print(score_test)


# In[40]:


#afin de confirmer la prédiction, je vais calculer l'erreur quadratique moyenne et le coefficient
y_pred_test = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(mse)
print(r2)
#j'observe une performance MSE beaucoup trop élevée
#j'observe une performance R² plutôt performante avec un 90% cela se rapproche fortement de 100%


# In[65]:


#dans cette partie, je regarde les coefficients qui ont été prédits par rapport aux différents
#canaux de campagne publicitaire
#de plus, pour une meilleure visualisation possible des effectifs, je l'affiche sous forme d'histogramme
features = X.columns
coef = linear_model.coef_

print(features)
print(coef)
plt.bar(features, coefficients)


# In[64]:


#pour terminer les résidus je décide d'observer les erreurs possibles du modèle par rapport à la prédiction
#dans ce graphique j'observe que l'erreur est modérée avec une ligne linéaire
#cela indique une bonne prédiction pour déterminer
#l'axe sur lequel il est préférable de croître le marketing (TV)
residu = y_test - y_pred_test

sns.residplot(x=y_pred_test, y=residu, lowess=True)
plt.xlabel("Valeurs prédite")
plt.ylabel("Le résidu")
plt.title("Graphique des résidus")
plt.show()

