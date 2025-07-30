#  Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  Load dataset
df = pd.read_csv("C:\\Users\\laksh\\OneDrive\\Desktop\\Internship\\Oasis Infobyte\\retail_sales_dataset.csv")




#  Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

#  Basic info
print(df.info())
print(df.describe())
print(df.isnull().sum())

#  Descriptive statistics
print("Mode:\n", df.mode())

# Boxplot for Total Amount
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Total Amount'])
plt.title("Boxplot of Total Amount")
plt.show()

#  Time Series Analysis
# Sales over time (monthly)
monthly_sales = df.resample('M', on='Date')['Total Amount'].sum()

plt.figure(figsize=(12,6))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# Rolling average
plt.figure(figsize=(12,6))
monthly_sales.rolling(window=3).mean().plot(label='3-Month Rolling Average')
monthly_sales.plot(alpha=0.4, label='Monthly Sales')
plt.title("Monthly Sales with Rolling Average")
plt.legend()
plt.show()

#  Customer Analysis
# Top 10 customers by total spend
top_customers = df.groupby('Customer ID')['Total Amount'].sum().sort_values(ascending=False).head(10)
print("Top 10 Customers by Revenue:\n", top_customers)

plt.figure(figsize=(10,5))
sns.barplot(x=top_customers.index, y=top_customers.values)
plt.title("Top 10 Customers by Total Spend")
plt.xlabel("Customer ID")
plt.ylabel("Total Spend")
plt.xticks(rotation=45)
plt.show()

# Gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df)
plt.title("Customer Gender Distribution")
plt.show()

# Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title("Customer Age Distribution")
plt.show()

#  Product Analysis
# Sales by product category
product_sales = df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)
print("Product Sales:\n", product_sales)

plt.figure(figsize=(10,5))
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.show()

# Heatmap: Sales by Month & Product Category
df['Month'] = df['Date'].dt.month
sales_pivot = df.pivot_table(values='Total Amount', index='Month', columns='Product Category', aggfunc='sum')

plt.figure(figsize=(12,6))
sns.heatmap(sales_pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Heatmap of Sales by Month and Product Category")
plt.show()

#  Recommendations
print("\nSample Recommendations:")
print("- Focus on top product categories for promotions.")
print("- Retarget high-value customers for loyalty programs.")
print("- Prepare inventory for peak months based on seasonal trends.")
print("- Consider segmenting marketing by age/gender profiles.")

