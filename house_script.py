import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  

from sklearn.linear_model import LinearRegression  # Modelo de 
model = LinearRegression(
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score  

!git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/Ronaldolvf/projetos.git

df_price_23 = pd.read_csv('/content/projetos/house_price_model_2.csv', sep=',')

df_price_23 = df_price_23.drop(df_price_23[(df_price_23['property_type'] == 'Room')].index)

df_price_23.shape

df_price_23 = df_price_23.drop(['purpose', 'location', 'city', 'new_price'], axis = 1)
df_price_23.head()

df_price_23['Farm_House'] = df_price_23['property_type'].apply(lambda x : 1 if x == 'Farm House' else 0 )
df_price_23['Flat'] = df_price_23['property_type'].apply(lambda x : 1 if x == 'Flat' else 0 )
df_price_23['House'] = df_price_23['property_type'].apply(lambda x : 1 if x == 'House' else 0 )
df_price_23['Lower_Portion'] = df_price_23['property_type'].apply(lambda x : 1 if x == 'Lower Portion' else 0 )
df_price_23['Penthouse'] = df_price_23['property_type'].apply(lambda x : 1 if x == 'Penthouse' else 0 )
df_price_23['Upper_Portion'] = df_price_23['property_type'].apply(lambda x : 1 if x == 'Upper Portion' else 0 )

df_price_23 = df_price_23.drop('property_type', axis = 1)
df_price_23.head()

dados_para_treino = df_price_23[['baths', 'bedrooms', 'Area_in_Marla', 'Farm_House', 'Flat',  'House', 'Lower_Portion', 'Penthouse', 'Upper_Portion']]

baths = 2
bedrooms = 2
Area_in_Marla = 75.0

Farm_House = 0
Flat  = 0
House = 1
Lower_Portion = 0
Penthouse = 0
Upper_Portion = 0

def proj_price(a: np.array):

    predictors_train, predictors_test, target_train, target_test = train_test_split(
        dados_para_treino,
        df_price_23['price'],
        test_size = 0.25,
        random_state = 2811)

    modelo = model.fit(predictors_train, target_train)

    peso = modelo.predict(a.reshape(1, -1))
    print(f'Predição: {(peso).round(2)}')

    target_predicted = modelo.predict(predictors_test)
    r_squared = r2_score(target_test, target_predicted)
    print(f'R²: {(r_squared).round(2)} %')

    return

proj_price(np.array([ baths,  bedrooms, Area_in_Marla, Farm_House, Flat,  House,  Lower_Portion,  Penthouse,Upper_Portion]))