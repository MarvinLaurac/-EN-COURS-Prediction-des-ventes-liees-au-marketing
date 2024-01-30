#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
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
sns.scatterplot(x='TV', y='Sales', color= "r", data=df)
plt.title('Relation entre les dépenses TV et les ventes')


# In[10]:


#deuxième affichage sous forme de nuage de points afin de visualiser la tendance et son évolution dans le temps des ventes liées à la publicité sur la Radio
#conclusion : nous n'avons pas de tendance, mais les publicités émises ont un impact relatif sur le CA

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 2)
sns.scatterplot(x="Radio", y="Sales",color= "g", data=df)
plt.title("Relation entre les dépenses radio et les ventes")


# In[16]:


#troisième affichage sous forme de nuage de points afin de visualiser la tendance et son évolution dans le temps des ventes liées à la publicité sous forme de papier 
#conclusion : nous n'avons pas de tendance, mais les publicités émises ont moins d'impact sur le CA

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 2)
sns.scatterplot(x="Newspaper", y="Sales",color= "black",  data=df)
plt.title("Relation entre les dépenses newspaper et les ventes")


# In[17]:


#afin d'avoir une visualisation plus mathématique,
#j'ai pris la décision de regarder les tendances avec l'aide de matrices de corrélation
#pour pouvoir avoir des données exploitables, j'ai préféré les afficher également sous forme de chiffres ci-dessous 
sns.pairplot(df, vars=["TV", "Radio", "Newspaper"])
plt.show()


# In[18]:


#dans cette matrice, j'observe
#TV: a une corrélation plutôt positive avec un coefficient de += 1
#Radio : a une corrélation plutôt négative par rapport à la TV avec un coef de -= 0 
#Newspaper : a une corrélation également plutôt négative par rapport à la TV avec un coef de -= 0

corr = df[["TV", "Radio", "Newspaper"]].corr()
print(corr)

