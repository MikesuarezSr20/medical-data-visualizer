import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importar datos
df = pd.read_csv('medical_examination.csv')

# Agregar columna 'overweight'
# Fórmula IMC: peso / altura^2
# La altura viene en cm, por eso se divide entre 100
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# Normalizar los datos:
# Si cholesterol o gluc es 1, queda como 0.
# Si es mayor que 1, queda como 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    # Crear DataFrame para el gráfico categórico
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # Agrupar y contar los valores
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Crear el gráfico categórico
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    # Guardar imagen
    fig.savefig('catplot.png')

    return fig


def draw_heat_map():
    # Limpiar datos incorrectos
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calcular matriz de correlación
    corr = df_heat.corr()

    # Crear máscara para ocultar la mitad superior del mapa
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 12))

    # Crear mapa de calor
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .5},
        ax=ax
    )

    # Guardar imagen
    fig.savefig('heatmap.png')

    return fig
