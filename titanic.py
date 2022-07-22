import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://raw.githubusercontent.com/terranigmark/curso-analisis-exploratorio-datos-platzi/main/train_titanic.csv'

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df = pd.read_csv(url, error_bad_lines = False)
print(df.head())

print(df.dtypes)
print(df.shape)

# Cuantos datos nulos hay?
nulls = df.isnull().sum().sort_values(ascending=False) # Saber si el df posee valores nulos y los suma
print(nulls)
print('='*158)

# Porcentaje de mujeres sobrevivientes

women = df.loc[df['Sex'] == 'female']["Survived"]
print(women)
rate_women_surv = sum(women)/len(women)*100
print(f'El porcentaje de la mujeres que sobrevivieron fue del {rate_women_surv}%')
print('='*158)

# Porcentaje de hombres sobrevivientes

men = df.loc[df['Sex'] == 'male']['Survived']
rate_men_surv = sum(men)/len(men)*100
print(f'El porcentaje de la hombres que sobrevivieron fue del {rate_men_surv}%')
print('='*158)

# Numero de pasajeros por sexo
cant_tot = df['Sex'].value_counts()
cant_men = len(men)
cant_women = len(women)
print(cant_tot)
print(f'La cantidad de pasajeros que iban en el titanic era de {cant_men} hombres y {cant_women} mujeres')
print('='*158)

# Creo variable categorica generando nueva columna indicando si sobrevivio o no

df['Name Survived'] = df['Survived'].map({0:"not_survived", 1:"survived"})   
print(df)
print('='*158)

#Graficación de numero de pasajeros por edad y numero de muertos y sobrevivientes por sexo

fig, ax = plt.subplots(1, 2, figsize = (10, 8))
df["Sex"].value_counts().plot.bar(color = "skyblue", ax = ax[0])
ax[0].set_title("Número de Pasajeros por sexo")
ax[0].set_ylabel("# pasajeros")
sns.countplot("Sex", hue = "Survived", data = df, ax = ax[1])
ax[1].set_title("Sexo: Sobrevivientes vs Muertes")
plt.show()

#distribución de la edad de los pasajeros
sns.distplot(df['Age'].dropna()) # mediante dropna no tomo en cuenta los datos nulos para evitar sesgo
plt.show()

# Se puede observar que la mayoria de los pasajeros tenian entre 20 y 40 anos

# Generacion de graficas por tipo de clase y de edad
sns.set(style="ticks", color_codes=True)
sns.pairplot(df,vars = [ 'Pclass','Age'], hue="Survived")
plt.show()

# Generacion de graficas de correlacion entre las variables

corr_titanic = df.corr(method = 'pearson')
print(corr_titanic)

sns.heatmap(corr_titanic, annot = True, cmap = 'coolwarm')
plt.title('Seaborn heatmap - Correlation between Variables',fontsize= 15, fontweight='bold', pad='50.0',fontstyle='italic')
plt.show()
