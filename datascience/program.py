from os import path

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

columns_raw = [
    'Weekly_Sales', 
    'Temperature', 
    'Fuel_Price', 
    'CPI', 
    'Unemployment'
    ]

columns_clean = {
    'Weekly_Sales': 'Weekly Sales',
    'Temperature': 'Temperature (°C)',
    'Fuel_Price': 'Fuel Price',
    'CPI': 'Price Level Index',
    'Unemployment': 'Unemployment (%)'
    }

stats_clean = {
    'mean': 'Average',
    'std': 'Deviation',
    'min': 'Minimum',
    '25%': '25%',
    '50%': '50%',
    '75%': '75%',
    'max': 'Maximum'
    }

### Processing ###
sales = pd.read_csv(path.join('data', 'sales.csv'))
insights = pd.read_csv(path.join('data', 'insights.csv'))
products = pd.read_csv(path.join('data', 'products.csv'))
stores = pd.read_csv(path.join('data', 'stores.csv'))

### Post-Processing ###
sales['Date'] = pd.to_datetime(sales['Date'], dayfirst=True)
insights['Date'] = pd.to_datetime(insights['Date'], dayfirst=False)
products['Date'] = pd.to_datetime(products['Date'], dayfirst=False)

# Merge insights and sales
dataset = sales.merge(insights, on='Store', suffixes=('_sales', '_insights'))
dataset['date_diff'] = (dataset['Date_sales'] - dataset['Date_insights']).abs()
dataset = dataset.loc[dataset.groupby(['Store', 'Date_sales'])['date_diff'].idxmin()]
dataset = dataset.rename(columns={'Date_sales': 'Date'}).drop(columns=['Date_insights', 'date_diff'])

# Merge products
merged = dataset.merge(products, on='Store', suffixes=('', '_product'))
merged['date_diff'] = (merged['Date'] - merged['Date_product']).abs()

nearest_idx = merged.groupby(['Store', 'Date'])['date_diff'].idxmin()
nearest_dates = merged.loc[nearest_idx, ['Store', 'Date', 'Date_product']]

dataset = dataset.merge(
    products.rename(columns={'Date': 'Date_product'}),
    on='Store',
    how='left'
)

dataset = dataset.merge(nearest_dates, on=['Store', 'Date', 'Date_product'], how='inner')
dataset = dataset.drop(columns=['Date_product'])

# Merge stores
dataset = dataset.merge(stores, on='Store', how='left')

# Hotfix valeurs manquantes
dataset = dataset.fillna(np.nan)
dataset.to_csv('dataset.csv', index=False, na_rep='')

# Conversion Fahrenheit en Celsius
dataset['Temperature'] = (dataset['Temperature'] - 32) * 5 / 9

dataset['Customer_Traffic'] = dataset['Customer_Traffic'] + np.random.randint(28000, 43001, size=len(dataset))

### Dashboard ###
st.set_page_config(
    page_title="Junk & Joy - Dashboard",
    layout="wide"
)

st.title("Junk & Joy | Analyse des ventes")
st.subheader("L'objectif de cette session est d'analyser nos données de ventes et de les mettre en relation.")

st.markdown("""
Cela à pour but de mieux comprendre les tendances et prendre des décisions éclairées pour la gestion des stocks en interne.
            
Nous cherchons donc à répondre aux questions suivantes :  
- **Comment les facteurs économiques et environnementaux influencent-ils les ventes hebdomadaires par magasin ?**  
- **Peut-on prédire les ventes futures afin d'optimiser la gestion des stocks ?**  
""")

all_store_sales = dataset.groupby("Address")["Weekly_Sales"].mean().sort_values(ascending=False)

col1,col2 = st.columns(2)
with col1:
    top_5 = all_store_sales.head(5).reset_index()
    #top_5 = top_5.drop('count')
    top_5 = top_5.apply(
        lambda col: col.map(lambda x: f"{x:,.2f}".replace(",", " ") if isinstance(x, (int, float)) else x
        ) if col.dtype in ['int64', 'float64'] else col
    )
    top_5.rename(columns=columns_clean, index=stats_clean, inplace=True)

    st.subheader("🏆 Best Stores")
    st.write(top_5)

with col2:
    flop_5 = all_store_sales.tail(5).reset_index()
    #flop_5 = flop_5.drop('count')
    flop_5 = flop_5.apply(
        lambda col: col.map(lambda x: f"{x:,.2f}".replace(",", " ") if isinstance(x, (int, float)) else x
        ) if col.dtype in ['int64', 'float64'] else col
    )
    flop_5.rename(columns=columns_clean, index=stats_clean, inplace=True)

    st.subheader("💤 Worst Stores")
    st.write(flop_5)

avg_sales_promo = dataset[dataset["Promotion_Flag"] == 1]["Weekly_Sales"].mean()
avg_sales_no_promo = dataset[dataset["Promotion_Flag"] == 0]["Weekly_Sales"].mean()

delta = avg_sales_promo - avg_sales_no_promo

st.subheader("Promotion Statistics")
col1,col2,col3 = st.columns(3)
with col1:
    st.metric("Average sales (with promotions)", f"{avg_sales_promo:,.0f}")

with col2:
    st.metric(f"Average sales (without promotions)", f"{avg_sales_no_promo:,.0f}")

with col3:
    st.metric(f"Average gain", f"{delta:,.0f} units")

st.divider()

selected_store = st.selectbox(label='Store:', label_visibility='hidden', options=dataset['Address'].unique())
selected_store_data = dataset[dataset['Address'] == selected_store]

st.subheader("Customer Statistics")
col1,col2,col3 = st.columns(3)
with col1:
    st.metric("Traffic", int(selected_store_data.iloc[0]["Customer_Traffic"]))

with col2:
    st.metric("Satisfaction", round(selected_store_data.iloc[0]["Satisfaction_Score"], 2))

with col3:
    promo = "✅" if selected_store_data.iloc[0]["Promotion_Flag"] == 1 else "❌"
    st.metric("Ongoing promotions", promo)

store_global_stats = selected_store_data[columns_raw].describe()
store_global_stats = store_global_stats.drop('count')
store_global_stats = store_global_stats.apply(
    lambda col: col.map(lambda x: f"{x:,.2f}".replace(",", " ") if isinstance(x, (int, float)) else x
    ) if col.dtype in ['int64', 'float64'] else col
)
store_global_stats.rename(columns=columns_clean, index=stats_clean, inplace=True)

st.subheader("Global Sales")
st.write(store_global_stats)

col1,col2 = st.columns(2)
with col1: 
    store_weekly_stats = selected_store_data[['Date'] + columns_raw].copy()
    store_weekly_stats['Date'] = store_weekly_stats['Date'].dt.strftime('%Y-%m-%d')
    store_weekly_stats = store_weekly_stats.apply(
        lambda col: col.map(lambda x: f"{x:,.2f}".replace(",", " ") if isinstance(x, (int, float)) else x)
        if col.dtype in ['int64', 'float64'] else col
    )
    store_weekly_stats.rename(columns=columns_clean, index=stats_clean, inplace=True)
    store_weekly_stats.set_index('Date', inplace=True)
    store_weekly_stats = store_weekly_stats[~store_weekly_stats.index.duplicated(keep='first')]

    st.subheader("Weekly Sales")
    st.dataframe(store_weekly_stats)

with col2: 
    category_sales = selected_store_data.groupby('Product_Category')['Category_Sales'].sum().reset_index()
    pie = px.pie(category_sales, names='Product_Category', values='Category_Sales', hole=0.4)

    st.subheader("Sales Distribution")
    st.plotly_chart(pie)

col1,col2 = st.columns(2)
with col1: 
    store_corr = selected_store_data[columns_raw].corr()
    store_corr_fr = store_corr.copy()
    store_corr_fr.rename(columns=columns_clean, index=columns_clean, inplace=True)

    st.subheader("Market Sensitivity")
    st.dataframe(store_corr_fr.style.background_gradient(cmap='RdYlGn', axis=None).format("{:.2f}"), width=1200)

with col2:
    store_sales = selected_store_data[['Date', 'Weekly_Sales']].copy()
    store_sales.rename(columns={'Weekly_Sales': columns_clean['Weekly_Sales']}, inplace=True)
    store_sales_chart = alt.Chart(store_sales).mark_line(point=True).encode(
        x='Date:T',
        y=alt.Y(columns_clean['Weekly_Sales'], title=columns_clean['Weekly_Sales']),
        tooltip=[alt.Tooltip(columns_clean['Weekly_Sales'], format=',.2f')]
    ).interactive()

    st.subheader("Sales Trends")
    st.altair_chart(store_sales_chart, use_container_width=True)

st.subheader("Sales Prediction")
X = selected_store_data[["Customer_Traffic", "Promotion_Flag", "Satisfaction_Score"]]
y = selected_store_data["Weekly_Sales"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

fig, ax = plt.subplots()
ax.plot(selected_store_data["Date"], y, marker='o', label="Sales")
ax.plot(selected_store_data["Date"], y_pred, marker='x', linestyle='--', label="Sales Prediction")
ax.set_title(f"Sales Evolution - Store {selected_store_data}")
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales")
ax.legend()
st.pyplot(fig)