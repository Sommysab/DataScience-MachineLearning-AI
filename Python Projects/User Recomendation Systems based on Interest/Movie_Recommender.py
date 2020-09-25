import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt



movie_titles_df = pd.read_csv('Movie_Id_Title')

movies_rating_df = pd.read_csv('u.data', sep = '\t', names = ['user_id', 'item_id', 'rating', 'timestamp'])

movies_rating_df.drop(['time_stamp', axis = 1, inplace=True]) 

movies_rating_df = pd.merge(movies_rating_df, movies_titles_df, on='item_id')


ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']

ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']

ratings_mean_count_df = pd.concat([ratings_df_count, ratings_df_mean], axis = 1)

ratings_mean_count_df.reset_index()

ratings_mean_count_df['mean'].plot(bins=100, kind='hist', color='r') 

ratings_mean_count_df['count'].plot(bins=100, kind='hist', color='r') 

ratings_mean_count_df[ ratings_mean_count_df['mean'] ==5 ]


# FILTER
user_id_movietitle_matrix = movies_rating_df.pivot_table(index='user_id', column='title', values='rating')
user_id_movietitle_matrix

titanic = userid_movietitle_matrix['Titanic (1997)']

titanic_correlations = pd.DataFram(userid_movietitle_matrix.corrwith(titanic), columns = ['Correlation'])
titanic_correlations = titanic_correlations.join(ratings_mean_count_df['count'])
titanic_correlations

titanic_correlations.dropna(inplace=True)

titanic_correlations.sort_values('Correlation', ascending = False)

titanic_correlations[titanic_correlations['count'] > 80].sort_values('Correlation', ascending = False).head(5)

# All
movie_correlations = user_movietitle_matrix.corr(method='pearson', min_periods=80)


# Test
myRatings = pd.read_csv('My_Ratings.csv')

myRatings['Movie Name'][0] 


similar_movies_list = pd.Series()

for i in rang(0,2):
    similar_movie = movie_correlations[ myRatings['Movie Name'][i] ].dropna()
    similar_movies = similar_movies.map(lambda x: x* myRatings ['Ratings'][i])
    similar_movies_list = similar_movies_list.append(similar_movie)
    
similar_movies_list.sort_values(inplace = True, ascending = False)

print(similar_movies_list.head(10))


