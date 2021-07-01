import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

#df = pd.read_csv('https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv')

df = pd.read_csv('cars.csv')
# It is necessary to remove whitespace (and the fullstop)

df['continent'] = df['continent'].str.strip()

df['continent'] = df['continent'].str.replace(".", "")

st.title('Welcome to the car correlation app')

st.sidebar.header('User Input Parameters')

continent = st.sidebar.selectbox('Choose the continent',
                                 ('US', 'Europe', 'Japan'))

sns.set_theme(style='darkgrid')

# Compute the correlation matrix

df_c = df.loc[(df['continent'] == continent)]

corr = df_c.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0, square=True, linewidths=.5,
            cbar_kws={"shrink": .5})

plt.title("Diagonal Correlation Matrix for your input choice", fontsize=30)
# plt.show() this is for matplotlib
st.pyplot(f)
