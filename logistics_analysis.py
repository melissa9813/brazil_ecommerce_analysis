#!/usr/bin/env python
# coding: utf-8

# # Logistics Analysis
# 
# ### Tasks
# ###### The effect of the following four on shipping quality will be analyzed.
# 1. Distance between sellers and customers
# 2. Product's size and volume
# 3. Freight value
# 4. Product category
# 
# - The meaning of 'shipping_limit_date' column is not clear. It will be studied by comparing with 'order_approved_at' and 'order_delivered_carrier_date'
# 
# ### Data
# - olist_orders_dataset.csv
# - olist_sellers_dataset.csv
# - olist_customers_dataset.csv
# - olist_geolocation_dataset.csv
# - olist_order_items_dataset.csv

# In[1]:


# 
get_ipython().system('pip install missingno==0.4.2')
get_ipython().system('pip install squarify==0.4.3')
get_ipython().system('pip install jedi==0.17.2')
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


# import libraries

import pandas as pd
import numpy as np
from zipfile import ZipFile


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import squarify #treemap


import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Data Load
df_customers = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_customers_dataset.csv?raw=true')
df_order_payments = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_order_payments_dataset.csv?raw=true')
df_order_reviews = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_order_reviews_dataset.csv?raw=true')
df_orders = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_orders_dataset.csv?raw=true')
df_products = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_products_dataset.csv?raw=true')
df_sellers = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/raw/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_sellers_dataset.csv')
df_category_translation = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/raw/main/Brazilian%20E-Commerce%20Public%20Dataset/product_category_name_translation.csv')
df_order_items = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_order_items_dataset.csv?raw=true')

# 'olist_geolocation_dataset.csv' is too big. Stored by splitting into 2. 
df_geolocation_1 = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_geolocation_dataset_1.csv?raw=true')
df_geolocation_2 = pd.read_csv('https://github.com/melissa9813/brazil_ecommerce_analysis/blob/main/Brazilian%20E-Commerce%20Public%20Dataset/olist_geolocation_dataset_2.csv?raw=true')
df_geolocation = pd.concat([df_geolocation_1, df_geolocation_2], ignore_index=True)


# ## [Pre-Task] What is 'shipping_limit_date'?
# #### Compare with order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date

# In[4]:


# df_order_itmes
df_order_items.head()


# In[5]:


# df_orders
df_orders.head()


# In[6]:


# merge two tables to check datetime info
temp = pd.merge(df_orders, df_order_items, how='inner', on=['order_id'])
temp.head()


# In[7]:


# check no. of null
temp.isnull().sum()


# In[8]:


# remove all null
temp = temp.dropna(axis=0)
temp.isnull().sum()


# In[9]:


# Change the order of columns to easily compare datetime columns
temp = temp[['order_id',
 'order_status',
 'shipping_limit_date',
 'order_delivered_carrier_date',
 'order_delivered_customer_date',
 'order_estimated_delivery_date',
 'order_purchase_timestamp',
 'order_approved_at',
 'customer_id',
 'order_item_id',
 'product_id',
 'seller_id',
 'price',
 'freight_value']]

temp.head()


# In[10]:


# Convert to datetime
temp['shipping_limit_date'] = pd.to_datetime(temp['shipping_limit_date'])
temp['order_delivered_carrier_date'] = pd.to_datetime(temp['order_delivered_carrier_date'])
temp['order_delivered_customer_date'] = pd.to_datetime(temp['order_delivered_customer_date'])
temp['order_estimated_delivery_date'] = pd.to_datetime(temp['order_estimated_delivery_date'])
temp['order_purchase_timestamp'] = pd.to_datetime(temp['order_purchase_timestamp'])
temp['order_approved_at'] = pd.to_datetime(temp['order_approved_at'])

temp.info()


# In[11]:


temp.sample(30)


# #### From the above table, 'shipping_limit_date' is the due date for sellers to ship the ordered items.
# 
# 'shipping_limit_date' is mostly earlier than 'order_delivered_carrier_date', which means that most sellers meet the shipping limit date well. However, breaking shipping deadline does not always lead to late delivery to customers.

# ## [Task 1] Distance between sellers and customers
# #### What is the impact of distance between sellers and customers on delivery?
# - Does it affect the accuracy of predicting delivery time??
# - Does it affect delivery lead time?

# In[12]:


# To calculate the distance between sellers and customers
    # Get customer location data (customer_id, order_id, customer_zip_code_prefix, lat, lng)
    # Get seller location data (seller_id, order_id, seller_zip_code_prefix, lat, lng)
    # Get table with distance between customer and seller (order_id, customer_id, seller_id, distance, order_estimated_delivery_date, order_delivered_customer_date, order_approved_at)


# In[13]:


# one zipcode has multiple combinations of latitude, longtitude
# one zipcode will get only one combination of mean latitude and mean longtitude for convenience

df_geolocation_simple = pd.DataFrame(df_geolocation.groupby(by=['geolocation_zip_code_prefix'])['geolocation_lat', 'geolocation_lng'].mean().reset_index())
df_geolocation_simple.head()


# In[ ]:





# <b> Customer's Location Table <b>

# In[14]:


# Get customer location data
print(df_customers.shape)
df_customers.head()


# In[15]:


temp_cust_zip = df_customers[['customer_id', 'customer_zip_code_prefix']]
temp_cust_zip.columns = ['customer_id', 'geolocation_zip_code_prefix']
temp_cust_zip


# In[16]:


temp_cust_geo = pd.merge(temp_cust_zip, df_geolocation_simple, how='inner', on=['geolocation_zip_code_prefix'])
temp_cust_geo


# In[17]:


df_order_delivered = pd.DataFrame(df_orders.loc[df_orders['order_status'] == 'delivered'])
df_order_delivered


# In[18]:


get_cust_geo = pd.merge(temp_cust_geo, df_order_delivered, how='right', on=['customer_id'])


# In[19]:


get_cust_geo


# In[20]:


get_cust_geo.isnull().sum()


# In[21]:


get_cust_geo = get_cust_geo.dropna(axis=0)
get_cust_geo


# 

# <b> Seller's Location Table <b>

# In[22]:


# Get seller location data
print(df_sellers.shape)
df_sellers.head()


# In[23]:


temp_seller_zip = df_sellers[['seller_id', 'seller_zip_code_prefix']]
temp_seller_zip.columns = ['seller_id', 'geolocation_zip_code_prefix']
temp_seller_zip


# In[24]:


temp_seller_geo = pd.merge(temp_seller_zip, df_geolocation_simple, how='inner', on=['geolocation_zip_code_prefix'])
temp_seller_geo


# In[25]:


get_seller_geo = pd.merge(temp_seller_geo, df_order_items, how='inner', on=['seller_id'])
get_seller_geo


# In[26]:


get_seller_geo.isnull().sum()


# In[27]:


get_seller_geo.shape


# <b> Distance between Customer and Seller <b>

# In[28]:


# Distance formula
    # 6371.01 * acos(sin(cust_lat)*sin(seller_lat) + cos(cust_lat)*cos(seller_lat)*cos(cust_lng - seller_lng))


# In[29]:


temp1 = get_cust_geo[['customer_id', 'order_id', 'geolocation_lat', 'geolocation_lng']]
temp1.columns = ['customer_id', 'order_id', 'cust_lat', 'cust_lng']
temp2 = get_seller_geo[['seller_id', 'order_id', 'geolocation_lat', 'geolocation_lng']]
temp2.columns = ['seller_id', 'order_id', 'seller_lat', 'seller_lng']
df_cust_seller_dist = pd.merge(temp1, temp2, how='inner', on=['order_id'])


# In[30]:


df_cust_seller_dist.head()


# In[31]:


from math import radians, sin, cos, acos


# In[32]:


df_cust_seller_dist['rad(cust_lat)'] = df_cust_seller_dist['cust_lat'].apply(lambda x: radians(x))
df_cust_seller_dist['rad(cust_lng)'] = df_cust_seller_dist['cust_lng'].apply(lambda x: radians(x))
df_cust_seller_dist['rad(seller_lat)'] = df_cust_seller_dist['seller_lat'].apply(lambda x: radians(x))
df_cust_seller_dist['rad(seller_lng)'] = df_cust_seller_dist['seller_lng'].apply(lambda x: radians(x))
                                                                                 


# In[33]:


df_cust_seller_dist['sin(r_cust_lat)'] = df_cust_seller_dist['rad(cust_lat)'].apply(lambda x: sin(radians(x)))
df_cust_seller_dist['sin(r_seller_lat)'] = df_cust_seller_dist['rad(seller_lat)'].apply(lambda x: sin(radians(x)))
df_cust_seller_dist['cos(r_cust_lat)'] = df_cust_seller_dist['rad(cust_lat)'].apply(lambda x: cos(radians(x)))
df_cust_seller_dist['cos(r_seller_lat)'] = df_cust_seller_dist['rad(seller_lat)'].apply(lambda x: cos(radians(x)))


# In[34]:


df_cust_seller_dist['clng-slng'] = df_cust_seller_dist['rad(cust_lng)'] - df_cust_seller_dist['rad(seller_lng)']


# In[35]:


df_cust_seller_dist['cos(clng-slng)'] = df_cust_seller_dist['clng-slng'].apply(lambda x: cos(x))


# In[36]:


# 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
df_cust_seller_dist['distance'] = 6371.01*np.arccos((df_cust_seller_dist['sin(r_cust_lat)']*df_cust_seller_dist['sin(r_seller_lat)']) + (df_cust_seller_dist['cos(r_cust_lat)']*df_cust_seller_dist['cos(r_seller_lat)']*df_cust_seller_dist['cos(clng-slng)']))


# In[37]:


col = ['order_id', 'customer_id', 'seller_id', 'distance']
df_cust_seller_dist = df_cust_seller_dist[col]
df_cust_seller_dist.head()


# <b> Gather meaningful information <b>

# In[158]:


distance_delivery = pd.merge(df_cust_seller_dist, get_cust_geo, how='inner', on=['order_id'])
distance_delivery.dropna(axis=0, inplace=True)


# In[159]:


col = ['order_id', 'distance', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'order_approved_at']
distance_delivery = distance_delivery[col]
distance_delivery.drop_duplicates(inplace = True)
distance_delivery.reset_index(drop=True, inplace = True)
distance_delivery.shape


# In[160]:


distance_delivery['order_delivered_customer_date'] = pd.to_datetime(distance_delivery['order_delivered_customer_date'])
distance_delivery['order_estimated_delivery_date'] = pd.to_datetime(distance_delivery['order_estimated_delivery_date'])
distance_delivery['order_approved_at'] = pd.to_datetime(distance_delivery['order_approved_at'])


# In[161]:


distance_delivery.info()


# In[162]:


# Accuracy of estimated delivery date
    ## negative value : late delivery
    ## positive value : early delivery
distance_delivery['estimated_delivery_date_error(d)'] = distance_delivery['order_estimated_delivery_date'] - distance_delivery['order_delivered_customer_date']
distance_delivery['estimated_delivery_date_error(d)'] = distance_delivery['estimated_delivery_date_error(d)'] / pd.Timedelta(1, unit='d')
distance_delivery['estimated_delivery_date_error(d)'] = distance_delivery['estimated_delivery_date_error(d)'].astype('timedelta64[D]')
distance_delivery['estimated_delivery_date_error(d)'] = distance_delivery['estimated_delivery_date_error(d)'].apply(lambda x: x.days)
distance_delivery.describe()


# In[150]:


# Check the min, max, accurate case
print("No. of Cases that %ddirelivered on the exact Date")
distance_delivery[distance_delivery['estimated_delivery_date_error(d)']==0]


# In[163]:


print("Case of the biggest error in the estimated delivery date (Late Delivery)")
distance_delivery[distance_delivery['estimated_delivery_date_error(d)']==-188]


# In[164]:


print("Case of the biggest error in the estimated delivery date (Early Delivery)")
distance_delivery[distance_delivery['estimated_delivery_date_error(d)']==146]


# In[165]:


# Delivery lead time
    ## from payment approved date to delivery date
distance_delivery['delivery_leadtime(d)'] = distance_delivery['order_delivered_customer_date'] - distance_delivery['order_approved_at']
distance_delivery['delivery_leadtime(d)'] = distance_delivery['delivery_leadtime(d)'] / pd.Timedelta(1, unit='d')
distance_delivery['delivery_leadtime(d)'] = distance_delivery['delivery_leadtime(d)'].astype('timedelta64[D]')
distance_delivery['delivery_leadtime(d)'] = distance_delivery['delivery_leadtime(d)'].apply(lambda x: x.days)
distance_delivery.describe()


# In[166]:


# negative value for delivery_leadtime does not make sense. 
distance_delivery[distance_delivery['delivery_leadtime(d)']==-6]


# - order_id 'bc4854efd86d9f42140c951c595d20c1' had been delivered before payment was approved.
# - Or, it could be just data input error

# <b> Interpretation and Visualization <b>

# In[169]:


# Distance vs Estimation Error
plt.figure(figsize=(12,6))
sns.scatterplot(data = distance_delivery, x = 'distance', y = 'estimated_delivery_date_error(d)', alpha=0.3)
plt.title('Distance vs Delivery Estimation Error', fontsize=20)
plt.xlabel('Distance btw Seller and Customer (km)', fontsize=13)
plt.ylabel('Delivery Estimation Error (days)', fontsize=13)
plt.ticklabel_format(style='plain')
plt.show()


# * <b> Horizontal scatterplot. </b>
#     - There is no significant relationship between seller-customer distance and accuracy of estimated delivery date
#     - The company provides the estimated delivery date with a similar level of accuracy, no matter how far away the seller and buyer are.
#     
# * <b> Note. </b>
#     - Delivery Estimation Error = 0 days means that the ordered item is delivered on the exact same day as the estimated delivery date. Most cases are concentrated on 0 days while ranging from 50 to -50 days. 
#     - Even some cases with distances over 2000km are maintained along the 0 days level.

# In[171]:


# Distance vs Delivery Lead Time
plt.figure(figsize=(12,6))
sns.scatterplot(data = distance_delivery, x = 'distance', y = 'delivery_leadtime(d)', alpha=0.3)
plt.title('Distance vs Delivery Leadtime', fontsize=20)
plt.xlabel('Distance btw Seller and Customer (km)', fontsize=13)
plt.ylabel('Delivery Leadtime (days)', fontsize=13)
plt.show()


# * <b> Slight positive scatterplot. </b>
#     - There is positive relationship between seller-customer distance and delivery leadtime
#     - Longer distance between seller and buyer is more likely to result in longer delivery leadtime.
#     
# * <b> Note. </b>
#     - The longest distance with nearly 5000km has less than 50 days of delivery lead time. Nevertheless, there are many cases that have shorter than 1000km of distance but took more than 50 days to be delivered to customers. 
#     - It reflects that other factors other than seller-customer distance have more significant impacts on delivery lead time. Those could be the seller's personal issue, weather, logistics company's issue, etc.

# In[174]:


distance_delivery.head()


# In[175]:


corr = distance_delivery.corr(method='pearson')
corr


# In[179]:


plt.figure(figsize=(12,6))
sns.heatmap(corr, annot=True)
plt.title('Correlation', fontsize=20)


# * <b> Correlation </b>
#     - Distance vs Estimated Delivery Error: No relationship
#     - Distance vs Delivery Leadtime: Slight positive relationship
#     - Estimated Delivery Error vs Delivery Leadtime: <b> Negative relationship </b>
#         - Long delivery leadtime tends to have more accurate estimated delivery date.

# In[ ]:




