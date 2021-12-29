#!/usr/bin/env python
# coding: utf-8

# # <font color='gray'> Capstone Project </font> 
# 
# # <font color='gray'> Gözde Gözütok      28-12-2021 </font> 
# 
# ## <font color='gray'> Dataset Description : Brazilian E-Commerce Public Dataset by Olist </font> 
# 
# * The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil.
# 
# * This is real commercial data, it has been anonymised.
# 
# * This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners.
# 
# * All text identifying stores and partners where replaced by the names of Game of Thrones great houses.
# 
# ### <span style='color:gray '> Customers Dataset </span> <br>
# #### This dataset has information about the customer and its location. 
# 
# * At our system each order is assigned to a unique customer_id. This means that the same customer will get different ids for different orders. The purpose of having a customer_unique_id on the dataset is to allow you to identify customers that made repurchases at the store. Otherwise you would find that each order had a different customer associated with.
# 
# * customer_id : key to the orders dataset. Each order has a unique customer_id.
# * customer_unique_id : unique identifier of a customer.
# * customer_city : customer city name
# * customer_state : customer state
# * customer_zip_code : first five digits of customer zip code
# 
# ### <span style='color:gray '> Geolocation Dataset </span> <br> 
# #### This dataset has information Brazilian zip codes and its lat/lng coordinates <br>
# 
# * geolocation_zip_code : first 5 digits of zip code
# * geolocation_state: state
# * geolocation_city: city
# * geolocation_lat : latitude
# * geolocation_lng: longitude
# 
# ### <span style='color:gray '> Order Items Dataset </span> <br> 
# #### This dataset includes data about the items purchased within each order. <br>
# 
# * An order might have multiple items.
# * order_id : order unique identifier
# * order_item_id : sequential number identifying number of items included in the same order.
# * shipping_limit_date : Shows the seller shipping limit date for handling the order over to the logistic partner.
# * seller_id: seller unique identifier
# * product_id : product unique identifier
# * price : item price
# * freight_value : item freight value item (if an order has more than one item the freight value is splitted between items)
# * Example:
# * The order_id = 00143d0f86d6fbd9f9b38ab440ac16f5 has 3 items (same product). Each item has the freight calculated accordingly to its measures and weight. To get the total freight value for each order you just have to sum.
# 
# * The total order_item value is: 21.33 * 3 = 63.99
# 
# * The total freight value is: 15.10 * 3 = 45.30
# 
# * The total order value (product + freight) is: 45.30 + 63.99 = 109.29
# 
# ### <span style='color:gray '> Payments Dataset </span> <br> 
# #### This dataset includes data about the orders payment options. <br>
# 
# * order_id : unique identifier of an order.
# * payment_sequential : a customer may pay an order with more than one payment method. If he does so, a sequence will be created to
# * payment_type : method of payment chosen by the customer.
# * payment_installments: number of installments chosen by the customer.
# * payment_value: transaction value.
# 
# ### <span style='color:gray '> Order Reviews Dataset  </span> <br>
# #### This dataset includes data about the reviews made by the customers. <br>
# 
# * After a customer purchases the product from Olist Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.
# 
# * review_id : unique review identifier
# * order_id : unique order identifier
# * review_score : Note ranging from 1 to 5 given by the customer on a satisfaction survey.
# * review_creation_date : Shows the date in which the satisfaction survey was sent to the customer.
# * review_answer_timestamp : Shows satisfaction survey answer timestamp.
# 
# ### <span style='color:gray '> Order Dataset  </span> <br>
# #### This is the core dataset. From each order you might find all other information. <br>
# 
# * order_id : unique identifier of the order.
# * customer_id : key to the customer dataset. Each order has a unique customer_id.
# * order_status : Reference to the order status (delivered, shipped, etc).
# * order_purchase_timestamp : Shows the purchase timestamp.
# * order_approved_at : Shows the payment approval timestamp.
# * order_delivered_carrier_date : Shows the order posting timestamp. When it was handled to the logistic partner.
# * order_delivered_customer_date : Shows the actual order delivery date to the customer.
# * order_estimated_delivery_date : Shows the estimated delivery date that was informed to customer at the purchase moment.
# 
# ### <span style='color:gray '> Products Dataset  </span> <br>
# #### This dataset includes data about the products sold by Olist. <br>
# 
# * product_id : unique product identifier
# * product_category_name : root category of product, in Portuguese.
# * product_name_length : number of characters extracted from the product name.
# * product_description : number of characters extracted from the product description.
# * product_photos_qty : number of product published photos
# * product_weights_g : product weight measured in grams.
# * product_length: product length measured in centimeters.
# * product_height_cm : product height measured in centimeters.
# * product_with_cm : product width measured in centimeters.
# 
# ### <span style='color:gray '>Sellers Dataset  </span> <br>
# #### This dataset includes data about the sellers that fulfilled orders made at Olist. Use it to find the seller location and to identify which seller fulfilled each product. <br>
# 
# * seller_id : seller unique identifier
# * seller_zip_code_prefix : first 5 digits of seller zip code
# * seller_city : seller city name
# * seller_state : seller state
# 
# ### <span style='color:gray '>Category Name Translation  </span> <br>
# #### Translates the product category name to english. <br>
# * product_category_name : category name in Portuguese
# * product_category_name_english :category name in English
# 
# 
# ** NOTE :  All currencies are in Brazilian Reais (BRL). As I write 1 BRL = 0.26 USD.

# In[1]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
img = mpimg.imread('Data Schema.png')
plt.imshow(img)
plt.show()
# This schema is important to be able to see the connections between the data and connect the data together.


# ### <span style='color:gray '>AIM  </span>
# 
# * The objective of this notebook is to propose an analytical view of e-commerce relationship in Brazil.
# * For this I will first go trough an exploratory data analysis using graphical tools to create self explanatory plots for better understanding what is behind braziian online purchasing.
# * By making analyzes on time series, I looked at which periods there was an increase in orders.
# * I tried to make sense out of the data and find patterns that were not visible at first glance.
# * I tried to get business insight.

# In[2]:


# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import calendar
import datetime
from datetime import timedelta
from pandas.api.types import CategoricalDtype
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import scipy
from scipy import stats 

import warnings
warnings.filterwarnings('ignore')


# In[3]:


get_ipython().system('pip install autoviz  # to look auto visualizations')


# * There is more than one data here, and each of these data contains different information about e-commerce sales
# * There are 9 datasets in total, I combined 8 of them and examined the columns I deem necessary. I used the 9th data alone.
# * At the very beginning of the notebook, you can read the explanations one by one and see what the columns mean.
# 
# ## <span style='color:gray '>Getting the Datasets </span>

# In[4]:


# Firstly getting all data -- 9 datasets
customers = pd.read_csv('olist_customers_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
order_payments = pd.read_csv('olist_order_payments_dataset.csv')
order_reviews = pd.read_csv('olist_order_reviews_dataset.csv') 
orders = pd.read_csv('olist_orders_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv') 
product_category = pd.read_csv('product_category_name_translation.csv')


# ## <span style='color:gray '>Overview of the Data</span>
# 
# * Before we build a dataset that contains all the useful information, let's look at the shape of each dataset so that we can be more confident about how to use the join expressions and see the variables in the data as a whole.

# In[5]:


# datasets from one row above, I choosed titles of the data same
datasets = [customers, geolocation, orders, order_items, order_payments, order_reviews, products, sellers, product_category]
titles = ["customers", "geolocation","orders","order_items", "order_payments", "order_reviews", "products","sellers", "product_category"]
plt.figure(figsize=(15,12))

info_data = pd.DataFrame({},)
info_data['datasets']= titles
info_data['columns'] = [', '.join([col for col, null in df.isnull().sum().items() ]) for df in datasets]
info_data['cols_no']= [df.shape[1] for df in datasets]
info_data['null_no']= [df.isnull().sum().sum() for df in datasets]
info_data['null_cols_no']= [len([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]
info_data['null_columns'] = [', '.join([col for col, null in df.isnull().sum().items() if null > 0]) for df in datasets]


info_data.style.background_gradient(cmap='cividis')
# Below image show all datasets, column numbers, columns, null numbers, null columns number  and null columns
#To gather practical information about all datasets 


# In[6]:


# renamed the columns to join two data but I will use geolocation separetly
sellers = sellers.rename(columns={'seller_zip_code_prefix': 'zip_code_prefix'})
geolocation = geolocation.rename(columns={'geolocation_zip_code_prefix': 'zip_code_prefix'})


# In[7]:


# There were 9 datasets in brazil e-commerce data, I combined 8 of them and kept as data, examined geolocation separately.
data = pd.merge(orders,order_payments, on="order_id")
data = data.merge(customers, on="customer_id")
data = data.merge(order_items, on="order_id")
data = data.merge(sellers, on='seller_id')
# data = data.merge(geolocation, on='zip_code_prefix') # I didn't use this data on merge, I will use it separately
data = data.merge(products, on="product_id")
data = data.merge(product_category, on="product_category_name")
data = data.merge(order_reviews , on='order_id') 


# In[8]:


data.shape # at first shape of the data


# In[9]:


data.describe().T


# * Freight value: range of the freight value is high, and mean value is gretaer than median value.Right skewed.
# * Price: range of the price value is so high, and distribution of the price also righr skewed.
# 

# In[10]:


# select the columns to be plotted
cols = ['price', 'freight_value']

# create the figure and axes

fig, axes = plt.subplots(1, 2,figsize=(16,6))
axes = axes.ravel()  # flattening the array makes indexing easier

for col, ax in zip(cols, axes):
    sns.histplot(data=data[col], kde=True, stat='density', ax=ax, bins=300, color='red')

fig.tight_layout()
plt.show()


# In[11]:


# these are the all datasets, I didn't use all columns above, just select some of them and analyzed it.
# above you can see null values


# In[12]:


data.info() # to see dtypes and null values 


# ### <span style='color:gray '>Missing Value Analysis  </span> 

# In[13]:


data = data.drop_duplicates()


# * I will not use product_width_cm, product_height_cm, product_length_cm, product_weight_g columns with whole data.
# * These columns have 1 null values in it, It's actually clean data, but I dropped it because there weren't any columns I was interested in.
# * The product categories in the data were in both Portuguese and English, I dropped the Portuguese ones as well.
# * In addition, there was a yield where I combined the product names in the data in order to get them in English.
# * There were a few more columns, such as the product name length, in that data, so I dropped them because I didn't care about them.

# In[14]:


# In the data review comment title and review comment message contains a lot of null value, I will not use these columns.
# But I kept them in order to use nlp maybe later.
reviews = data[['review_comment_title','review_comment_message','review_score','review_id']]


# In[15]:


# Firstly, I dropped the columns that I don't want to use it
data.drop(['product_category_name'], axis = 1, inplace = True) # dropped it because its english available in tha data, it was portuguese
# I joined product to get product_id and to join product_category with data, I will not use below attributes, I dropped it.
data.drop(['product_name_lenght', 'product_photos_qty', 'product_length_cm'], axis=1, inplace=True)
data.drop(['product_weight_g','product_width_cm','product_description_lenght','product_height_cm'], axis = 1, inplace = True)


# In[16]:


# In the datasets description section, it was stated that the review title and review_comment_message values are in Portuguese.
# I dropped these two columns because their null value rate are too high %88 and %57 and they are in portuguese.
data.drop(['review_comment_title','review_comment_message'], axis = 1, inplace = True)


# In[17]:


# in the below codes I used these but since I didn' find any insight from these I dropped them,
# And I will comment these codes below to see them
data.drop(['review_answer_timestamp','review_creation_date'], axis = 1, inplace = True)


# In[18]:


#data.dropna(inplace=True)


# In[19]:


#let's change the column name product_category_name_english' to product_category.
data.rename(columns={'product_category_name_english': 'product_category'},inplace=True)


# In[20]:


data.shape # to see its first shape before dropping any columns


# In[21]:


# let's begin with null values # Now we see that three columns have null values.
data.isnull().sum().sort_values(ascending=False)


# In[22]:


def missing_value_rate(data):
    print (round((data.isnull().sum() * 100/ len(data)),3)) # I rounded to 3.

missing_value_rate(data)


# In[23]:


data.sample(5) # to look randomly to data


# * Although the null rates of the remaining data are very low rate , I will fill order_approved_at and order_delivered_carrier_date 

# In[24]:


# Let' s first look at the order_approved_at columns, here as you can see the order has been confirmed, the parts are nat, but all the orders have been delivered.
#I will fill in the null values here by looking at the average value of the time between order and confirmation.
data[data['order_approved_at'].isnull()]


# ### <span style='color:gray '>Converting date values from object type to date  </span> 

# * Before filling na values, I changed object type to date type
# * I will use datetime for time series analysis

# In[25]:


data[['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']]= data[['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']].apply(pd.to_datetime)


# In[26]:


data[['shipping_limit_date']]=data[['shipping_limit_date']].apply(pd.to_datetime)
# data[['review_creation_date','review_answer_timestamp']]=data[[review_creation_date','review_answer_timestamp']].apply(pd.to_datetime)


# ### <span style='color:gray '> New Date columns using order purchase timestamp  </span> 

# In[27]:


# I created Year, Month and Day data based on the order date of the customers.
data['purchased_Year'] = data['order_purchase_timestamp'].dt.year
data['purchased_Month'] = pd.Series(pd.Categorical(data['order_purchase_timestamp'].dt.month_name(), categories=list(calendar.month_name)))
data['purchased_Day'] = pd.Series(pd.Categorical(data['order_purchase_timestamp'].dt.day_name(), categories=list(calendar.day_name)))


# In[28]:


data['Day'] = data['order_purchase_timestamp'].dt.day # to use day I created day column


# In[29]:


# I created seasons to look order sales in different seasons
data['Season']=(data['order_purchase_timestamp'].dt.month%12+3)//3
Seasons={1: 'Summer', 2: 'Autumn', 3:'Winter', 4: 'Spring'}
data['SeasonName']= data['Season'].map(Seasons)


# ### <span style='color:gray '>Null Values  </span> 

# In[30]:


# In the vast majority of data, orders are confirmed within 1 day.
(data['order_approved_at']- data['order_purchase_timestamp']).describe()


# In[31]:


# since mean of the time that approved after purchase was hours, I will fill with null values with the time after 1 day
# from the purchase time


# In[32]:


# I created new_approved time to fill nat values in order approved at column, 1 hours after the order purchase timestamp
new_approved_at= data[data['order_approved_at'].isnull()]['order_purchase_timestamp'] + timedelta(hours=1)
new_approved_at


# In[33]:


data['order_approved_at'] = data['order_approved_at'].fillna(value=new_approved_at)


# In[34]:


data[data['order_approved_at'].isnull()]


# In[35]:


# Now I will deal with the null values of delivered_carrier date and delivered_customer_date
len(data[data['order_delivered_customer_date'].isnull()])
# data[data['order_delivered_customer_date'].isnull()]


# In[36]:


# to looking order status I realized that when the status shipped, invoiced or processing their delivered_customer_date null
data.order_status.unique()


# ##### Shipped Orders

# * Status : Shipped
# * Since status is shippped carrier date is available for each shipped ones, but customer delivered times are null because they aren't reached the customers.

# In[37]:


len(data[data['order_status']=='shipped']) 
# Since it appears in cargo, the date of reaching the customer is null in all shipped values


# In[38]:


len(data[(data['order_status']=='shipped') & (data['order_delivered_carrier_date'].isnull())])


# In[39]:


len(data[(data['order_status']=='shipped') & (data['order_delivered_customer_date'].isnull())])


# ##### Invoiced Orders

# In[40]:


len(data[data['order_status']=='invoiced'])


# In[41]:


len(data[(data['order_status']=='invoiced') & (data['order_delivered_carrier_date'].isnull())])


# In[42]:


len(data[(data['order_status']=='invoiced') & (data['order_delivered_customer_date'].isnull())])


# ##### Processing Orders

# In[43]:


len(data[data['order_status']=='processing']) 


# In[44]:


len(data[(data['order_status']=='processing') & (data['order_delivered_carrier_date'].isnull())])


# In[45]:


len(data[(data['order_status']=='processing') & (data['order_delivered_carrier_date'].isnull()) & (data['order_delivered_customer_date'].isnull())])


# In[46]:


data[(data['order_status']=='processing') & (data['order_delivered_carrier_date'].isnull()) & (data['order_delivered_customer_date'].isnull())]


# In[47]:


len(data[(data['order_status']=='approved') & (data['order_delivered_carrier_date'].isnull())])


# ##### Canceled Orders

# In[48]:


# First of all canceled items' delivered_carrier data and delivered_customer_date is null.
# There are multiple scenarios here,A customer can cancel the product after purchasing it,
# cancel it after it has been shipped, or cancel it after it reaches the customer's hand.

canceled = data[data['order_status']=='canceled'] #I gathered the canceled items together
canceled # there were 536 items in the canceled status


# In[49]:


len(data[data['order_status']=='canceled']) # There are 536 products whose status appears to be canceled.


# In[149]:


data[data['order_status']=='canceled'].groupby('review_score')['review_score'].count().plot(kind='pie')
plt.title(" canceled order's review scores" )
# 


# In[50]:


len(data[(data['order_delivered_customer_date'].isnull()) & (data['order_status']=='canceled')])
# 7 f them delivred customers


# In[51]:


data[(~data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull()) & (data['order_status']=='canceled')]


# In[52]:


# Although some products have been canceled, it seems to have reached the customer
data[(data['order_status'] == 'canceled') & (~data['order_delivered_customer_date'].isnull())]


# In[53]:


len(data[(data['order_delivered_carrier_date'].isnull()) & (data['order_status']=='canceled')])
# 68 of them It may have been canceled before shipping.


# In[54]:


len(data[(~data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull())&(data['order_status']=='canceled')])


# In[54]:


# iptal edilen ürünlerin ürün categorilerine bak


# In[55]:


# There are 1194 rows of data with blank delivery dates to the customer and cargo.
len(data[(data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull())])


# In[56]:


# Let's look at the orders which are unavaliable
len(data[data['order_status']=='unavailable'])


# In[57]:


# I will fill these 2 with estimated delivery time
len(data[(data['order_status']=='delivered') & (data['order_delivered_carrier_date'].isnull())])


# In[58]:


(data['order_delivered_carrier_date']- data['order_purchase_timestamp']).describe()


# In[59]:


# I created new_carrier_at time to fill nat values in order approved at column, 1 hours after the order purchase timestamp
new_carrier_at= data[(data['order_status']=='delivered') & (data['order_delivered_carrier_date'].isnull())]['order_purchase_timestamp'] + timedelta(days=2)
data['order_delivered_carrier_date'] = data['order_delivered_carrier_date'].fillna(value=new_carrier_at)


# In[60]:


len(data[(data['order_status']=='delivered') & (data['order_delivered_carrier_date'].isnull())])


# In[61]:


data[(data['order_delivered_carrier_date'].isnull()) & (data['order_status']=='delivered')]


# In[62]:


data['order_delivered_carrier_date'].isnull().sum()


# ### <span style='color:gray '>I looked at the scores for the canceled products and the dates with null values, I did not do any filling, I commented on my attempts.  </span> 

# In[63]:


#data.loc[(data['order_status'] == 'unavailable') & (data['order_delivered_carrier_date'].isnull()), 'order_delivered_carrier_date']=-1
#data.loc[(data['order_status'] == 'unavailable') & (data['order_delivered_customer_date'].isnull()), 'order_delivered_customer_date']=-1


# In[64]:


#data.loc[(data['order_delivered_carrier_date'].isnull()) & (data['order_delivered_customer_date'].isnull()) & ((data['review_score']==4) | (data['review_score']==5)), 'order_delivered_carrier_date']=0
#data.loc[(data['order_delivered_carrier_date'].isnull()) & (data['order_delivered_customer_date'].isnull()) & (data['review_score']==), 'order_delivered_customer_date']=0


# In[65]:


#mask = data[(data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull()) & (data['review_score']==1)]
#mask[['order_estimated_delivery_date', 'review_creation_date']]


# In[66]:


# (mask['review_creation_date']-mask['order_estimated_delivery_date']).describe()


# In[67]:


#data[data['order_delivered_carrier_date'].isnull()]


# In[68]:


# I tried to understand why order_delivered_carrier_data and order_delivered_customer_Date is null
# I thought that the products were not delivered because the reviews of those with a review score of 1
# whose shipping dates and delivery dates are empty, were sent 2 days after the estimated delivery date


#data.loc[(data['order_delivered_carrier_date'].isnull()) & (data['order_delivered_customer_date'].isnull()) & (data['review_score']==1), 'order_delivered_carrier_date']=0
#data.loc[(data['order_delivered_carrier_date'].isnull()) & (data['order_delivered_customer_date'].isnull()) & (data['review_score']==1), 'order_delivered_customer_date']=0


# In[69]:


#time = data[data['order_delivered_customer_date'].isnull()]['order_purchase_timestamp'] + timedelta(days=12)
#data[(data['order_delivered_customer_date'].isnull()) & (data['order_status']=='delivered')] = data['order_delivered_customer_date'].fillna(value=time)


# In[70]:


# Considering the general scores in the data, although 1 score is low, a maximum of 1 score is seen here in null values.
data[data['order_delivered_customer_date'].isnull()].groupby('review_score')['review_score'].count()


# In[71]:


# Considering the general scores in the data, although 1 score is low, a maximum of 1 score is seen here in null values.
data[(data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull())].groupby('review_score')['review_score'].count()


# In[151]:


data[~(data['order_status']=='delivered')].groupby('review_score')['review_score'].count().plot(kind='pie')
plt.title('Scores of products not delivered to customers such as shipped,invoiced,approved,canceled...')


# In[72]:


len(data[(data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull()) & ((data['review_score']==4) | (data['review_score']==5))])


# In[73]:


# I will look order status in details but before that Ilooked them to get insight.
data[(data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull())].groupby('order_status')['order_status'].count()


# In[74]:


#new_time = data[(data['order_delivered_customer_date'].isnull()) & (data['order_status']=='delivered')]['order_purchase_timestamp'] + timedelta(days=12)
#data['order_delivered_customer_date'] = data['order_delivered_customer_date'].fillna(value=new_time)


# In[75]:


#time = data[(data['order_delivered_carrier_date'].isnull()) & (data['order_status']=='delivered')]['order_purchase_timestamp'] + timedelta(days=9)
#data['order_delivered_carrier_date'] = data['order_delivered_carrier_date'].fillna(value=time)


# In[76]:


#skore4_5 = data[(data['order_delivered_carrier_date'].isnull()) & ( data['order_delivered_customer_date'].isnull()) & ((data['review_score']==4) | (data['review_score']==5))]


# In[77]:


#(skore4_5['order_estimated_delivery_date']- skore4_5['review_creation_date']).mean()


# * I only filled order_approved_at and delivered order's null values.
# * Since I decided that the orders contain null values due to their current status, 
# * I decided to deal only with the delivered ones instead of filling in the null values.
# * But I added my analyzes according to different statuses. 
# * I named only the data that was delivered as final data. 
# * I continued the analysis of order statuses through what I named Data.

# In[78]:


# I focused only delivered items which are 97% of the data
# From now on I will use final_data mostly, except looking order status distribution.
final_data = data[data['order_status']=='delivered']


# In[79]:


final_data.isnull().sum()


# In[80]:


final_data.shape


# In[81]:


# Above I changed object types to datetime
# Here is my final data, I will analysis this data from now on.
final_data.info()


# # <span style='color:gray '> Exploratory Data Analysis </span>

# In[82]:


final_data.sample(5) # let's look at the sample of the data from different indexes.


# In[85]:


final_data.shape # before creating some new columns using order_purchase_timestamp


# ## <span style='color:gray '> To look order purchase timestamps </span> 

# In[86]:


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)


# In[87]:


final_data['purchased_Year'].unique() # three years available in data 


# In[88]:


Purchased_Year = final_data.groupby('purchased_Year').size().sort_values(ascending=False)
Purchased_Year


# In[89]:


final_data.groupby('purchased_Year').size().plot(kind='bar', color='green')


# In[90]:


sns.set()
plt.yscale('log')
final_data.groupby('purchased_Year').size().plot(kind='bar', color='darkorange')
plt.ylabel('order counts using --> log')
plt.title('Orders by years', size=(20))


# In[385]:


# Price contribution over the years
total_purchase = final_data.pivot_table('price', index='purchased_Year', columns='purchased_Month', aggfunc='sum')
total_purchase.head()


# In[388]:


# Order counts over the years
total_purchase = final_data.pivot_table('order_id', index='purchased_Year', columns='purchased_Month', aggfunc='count')
total_purchase.head()


# In[389]:


final_data.pivot_table('price', index='Year_Month', columns='product_category', aggfunc='sum', fill_value=0)


# ####  <span style='color:gray '> Year Month </span> 

# In[91]:


# Let's create year month values splitting order purchase timestamp
final_data['Year_Month'] = final_data['order_purchase_timestamp'].dt.strftime('%Y%m')
final_data['Year_Month'].sort_values() # values from september 2016 to september 2018


# In[92]:


sns.set(rc={'figure.figsize':(21.7,6.0)})
sns.lineplot(data=final_data, x=final_data['Year_Month'].sort_values(), y=final_data["price"].sum())

plt.title('Price sum according to year-month dates')


# In[93]:


final_data['Year_Month'].sort_values().unique()


# In[94]:


year_month = final_data.groupby('Year_Month')['Year_Month'].size()
year_month


# In[95]:


Year_Month_Significant = final_data.groupby('Year_Month')['Year_Month'].size()[2:23]
Year_Month_Significant


# In[96]:


plt.figure(figsize=(15,8))
sns.lineplot(data=Year_Month_Significant, x=Year_Month_Significant.index, y=Year_Month_Significant.values,color="red")
sns.set(rc={'figure.figsize':(10.7,8.27)})
plt.ylabel('Order Counts')
plt.title('Orders between 2017 and 2018')


# * Data for October and December 2016 are available. No data for November
# * There is data for all months of 2017.
# * There is data for the first 8 months of 2018.
# * When I make a comparison between the years, I will mostly compare the first 8 months of 2017 and 2018.
# * In order to compare the months in a year, I will examine the year 2017.

# ## <span style='color:gray '> Purchase Months </span> 

# In[97]:


final_data['purchased_Month'].unique() # all months are available in the data


# In[98]:


Purchased_Month = final_data.groupby('purchased_Month').size().sort_values(ascending=False)
Purchased_Month


# In[99]:


plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=Purchased_Month.index, x=Purchased_Month.values)
plt.ylabel(' ', fontsize=18 )
plt.title('Purchased Months', fontsize=20)


# * Caution : Here, I looked at the values by month, but the 11th month of 2016 and the 11th month of 2018 are not available, so looking at the monthly values in all the data will not give the correct results.
# * In that respect, comparing the first 8 months of 2017 and 2018 and comparing 2017 on a monthly basis will yield more accurate results.
# * That's why I created months of 2017 below.

# In[100]:


Monthsof2017 = final_data[final_data['purchased_Year']==2017]['purchased_Month']
Monthsof2017.nunique() # here 12 months are available


# In[101]:


Monthsof2017 = pd.DataFrame(Monthsof2017)
Month_values2017 = Monthsof2017.groupby('purchased_Month').size() #.sort_values(ascending=False)
Month_values2017


# In[102]:


plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=Month_values2017.index, x=Month_values2017.values)
plt.xlabel('Order Counts ', fontsize=10 )
plt.title('Purchased Months of 2017', fontsize=20)


# In[103]:


Monthsof2018 = final_data[final_data['purchased_Year']==2018]['purchased_Month']
Monthsof2018.nunique() # here 8 months are available


# In[106]:


Monthsof2018 = pd.DataFrame(Monthsof2018)
Month_values2018 = Monthsof2018.groupby('purchased_Month').size().sort_values(ascending=False)
Month_values2018 = Month_values2018[:8]
Month_values2018


# In[107]:


plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=Month_values2018.index, x=Month_values2018.values)
plt.ylabel(' ', fontsize=18 )
plt.title('Purchased Months of 2018', fontsize=20)


# In[108]:


plt.figure(figsize=(22,6))
Data2018 = final_data[final_data['purchased_Year']==2018]['purchased_Month'].sort_values() # only october data were in 2016
sns.set_style("darkgrid")
sns.barplot(y=Data2018.index, x=Data2018.values)
plt.ylabel(' ', fontsize=18 )
plt.title('Purchased Months', fontsize=20)


# #### <span style='color:gray '> Purchase Seasons </span> 

# In[109]:


season2017  = final_data[final_data['purchased_Year']==2017].groupby('SeasonName')['order_id'].size().reset_index()
#season2017['SeasonName'] = season2017['SeasonName'].str.capitalize()
season2017.loc[0:]


# In[111]:


# importing libraries for the subplots
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.gridspec import GridSpec
pd.set_option('display.max_columns', 100)
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#for the millions format function
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as tkr
import matplotlib as mpl
from datetime import datetime, timedelta
from pandas import DataFrame
from PIL import Image



fig = make_subplots(rows=1, cols=1, specs=[[{"type": "pie"}]],vertical_spacing=0, horizontal_spacing=0.02)

fig.add_trace(go.Pie(values= season2017['order_id'], labels= season2017['SeasonName'], name='Store type',
                     marker=dict(colors=['purple','bj','gray','yellow']), hole=0.7,
                    hoverinfo='label+value', textinfo='label + value'), 
                    row=1, col=1)

fig.update_yaxes(showgrid=False, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_yaxes(visible=False, row=1, col=1)
fig.update_layout(height=400, bargap=0.4,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  title_text="Season versus Order Counts in 2017",
                  template="plotly_white",
                  title_font=dict(size=22, color='black', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'), 
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.update_traces(rotation=90)
fig.update_traces(textposition="auto", selector=dict(type='pie'))
# I saw tthis graphic drawing while reading notebook on kaggle, so I started using it.
# I saw here : https://www.kaggle.com/luisenrique18/olist-ecommerce-2017-2018-eda/notebook


# In[116]:


final_data['purchased_Day'].unique()


# In[117]:


Purchased_Day = final_data.groupby('purchased_Day').size().sort_values(ascending=False)
Purchased_Day 


# In[118]:


sns.set()
plt.figure(figsize=(12,5))
sns.set_style("whitegrid")
sns.barplot(y=Purchased_Day.index, x=Purchased_Day.values,color='slateblue')
plt.ylabel(' ', fontsize=18 )
plt.title('Purchased Days', fontsize=20)


# In[119]:


final_data['purchased_Hour'] = pd.to_datetime(final_data['order_purchase_timestamp']).dt.hour


# In[120]:


plt.figure(figsize=(8,6))
final_data.groupby('purchased_Hour')['purchased_Hour'].size().plot(kind='bar',color='lightblue')
plt.title('Purchased Hour Periods ')
plt.ylabel('Order Counts')


# In[121]:


plt.figure(figsize=(8,6))
final_data.groupby('purchased_Hour')['purchased_Hour'].size().sort_values(ascending=False).plot(kind='bar', color='green')
plt.title('Purchase by Hour Periods Descending Orders')


# In[122]:


final_data['day_periods'] = (pd.to_datetime(data['order_purchase_timestamp']).dt.hour % 24 + 4) // 4
final_data['day_periods'].replace({1: 'Late Night',
                      2: 'Early Morning',
                      3: 'Morning',
                      4: 'Afternoon',
                      5: 'Evening',
                      6: 'Night'}, inplace=True)


# HOUR -->  SESSION       
# * 1, 2, 3 and 4 --> Late Night     
#    
# * 5, 6, 7 and 8 -->  Early Morning  
#   
# * 9, 10, 11 and 12 --> Morning       
#      
# * 13,14 ,15 and 16 --> Afternoon       
#         
# * 17, 18, 19 and 20 -->  Eve         
# 
# * 21, 22, 23 and 24 --> Night

# In[123]:


final_data.groupby('day_periods')['day_periods'].size().sort_values(ascending=False).plot(kind='bar', color='darkorange')
plt.title('Purchase Day Periods')


# In[131]:


day_periods = final_data.groupby('day_periods').size().reset_index().sort_values(by=0,ascending=False)
day_periods["day_periods"] = day_periods["day_periods"].str.capitalize()
day_periods[0]


# In[125]:


sns.set()
plt.figure(figsize=(8,6))
day_periods = final_data.groupby('day_periods')['day_periods'].size()
day_periods.plot(kind='pie')
plt.title('Purchase Day Periods')
day_periods


# In[132]:



fig = make_subplots(rows=1, cols=1, specs=[[{"type": "pie"}]],vertical_spacing=0, horizontal_spacing=0.02)

fig.add_trace(go.Pie(values= day_periods[0], labels=day_periods["day_periods"], name='Store type',
                     marker=dict(colors=['slateblue','darkblue','lightblue','#91A2BF','#C8D0DF']), hole=0.7,
                    hoverinfo='label+value', textinfo='label + value'), 
                    row=1, col=1)

fig.update_yaxes(showgrid=False, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_yaxes(visible=False, row=1, col=1)
fig.update_layout(height=400, bargap=0.4,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  title_text="Order Counts According to Day Periods in the Data",
                  template="plotly_white",
                  title_font=dict(size=22, color='black', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'), 
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.update_traces(rotation=90)
fig.update_traces(textposition="auto", selector=dict(type='pie'))


# * The Year data is available for 2016, 2017 and 2018
# * The Month data is available for each month
# * The day data is available for each day.
# * Most sales were made in the afternoon and evening hours.
# * Order density on Monday and Tuesday is higher than other days of the week. 
# * Spring is the season with the highest sales.
# * November is the month with the highest sales in a year.
# * When looking at the sales by days in November, the highest number of sales was made on Friday, Friday, November 24, 2017 is the time of the black week.

# ## <span style='color:gray '> Order Status Statistics   </span> 

# In[133]:


# By the time this dataset was created, the highest amount of orders went from delivered ones


# In[134]:


# here the data was used because in the final_data only delivered itesm are available
data.order_status.value_counts()


# In[136]:


# here it was used data because in the final_data only delivered items are available
plt.figure(figsize=(10,5))
plt.yscale('log')
data.order_status.value_counts().plot(kind='bar', color='darkorange')
plt.title('Order Status in the Data')


# In[137]:


order_status = data.groupby('order_status').size().reset_index().sort_values(by=0,ascending=False)
order_status["order_status"] = order_status["order_status"].str.capitalize()
order_status


# In[138]:


fig = make_subplots(rows=1, cols=1, 
                    specs=[[{"type": "pie"}]],
                    vertical_spacing=0, horizontal_spacing=0.02)

fig.add_trace(go.Pie(values= order_status[0], labels=order_status["order_status"], name='Store type',
                     marker=dict(colors=['#6D83AA','#91A2BF','#334668','#496595','#C8D0DF']), hole=0.7,
                    hoverinfo='label+value', textinfo='label + value'), 
                    row=1, col=1)

fig.update_yaxes(showgrid=False, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
fig.update_xaxes(visible=False, row=1, col=1)
fig.update_yaxes(visible=False, row=1, col=1)
fig.update_layout(height=400, bargap=0.4,
                  margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                  title_text="Order Status Distribution in the Data",
                  template="plotly_white",
                  title_font=dict(size=22, color='black', family="Lato, sans-serif"),
                  font=dict(color='#8a8d93'), 
                  hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                  showlegend=False)
fig.update_traces(rotation=90)
fig.update_traces(textposition="auto", selector=dict(type='pie'))


#  I only choosed delivered items in my analysis.
#  97% of the delivered ones.
#  I will continue my analysis using final_data (only delivered items)

# ## <span style='color:gray '> Order Delivery Time Statistics   </span> 

# In[143]:


# Sellers mostly approved the sale after customers' purchases.
diff_app_purc = (final_data['order_approved_at'] - final_data['order_purchase_timestamp']).describe()
diff_app_purc


# In[144]:


# Products given to cargo in 2 to 3 days.
(final_data['order_delivered_carrier_date'] - final_data['order_purchase_timestamp']).describe()


# In[145]:


# Orders were delivered in an average of 10 days.
(final_data['order_delivered_customer_date'] - final_data['order_purchase_timestamp']).describe()


# In[146]:


# The estimated delivery date is 23 days as the median value, which is a bit of a long time.
(final_data['order_estimated_delivery_date'] - final_data['order_purchase_timestamp']).describe()


# In[147]:


# Most of the time, the products were delivered to the customers before the estimated delivery date.
(final_data['order_estimated_delivery_date'] - final_data['order_delivered_customer_date']).describe()


# ## <span style='color:gray '> Product Categories  </span> 

# In[148]:


# There are 71 unique categories
final_data.product_category.nunique()


# In[149]:


# Distribution of product sales by categories
ProductCategories = final_data.groupby('product_category')['product_category'].count().sort_values(ascending=False)
ProductCategories


# In[150]:


plt.figure(figsize=(20,15))
sns.set_style("darkgrid")
sns.barplot(y=ProductCategories.index, x=ProductCategories.values)
plt.ylabel(' Product Categories', fontsize=18 )
plt.title('Product Categories by Orders', fontsize=20)


# In[151]:


# Let's choose top 10 highest categories
Top10highest = ProductCategories[:10]
Top10highest


# In[152]:


sns.set_style()
plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=ProductCategories[:10].index, x=ProductCategories[:10].values)
plt.ylabel(' ', fontsize=18 )
plt.title('Distribution of product sales by categories - Top 10', fontsize=20)
# It is important that we know which products are most demanded by the customers, 
# and the availability of stock in these products can prevent customers from shopping on other sites.
# It can be tried to gain more sellers for these products, to create choices for customers.


# In[153]:


# Categories of least sold products
lowest10=ProductCategories[-10:]
lowest10


# In[154]:


plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=ProductCategories[-10:].index, x=ProductCategories[-10:].values)
plt.ylabel(' ', fontsize=18 )
plt.title('Distribution of product sales by categories - Least 10', fontsize=20)


# * Categories versus Scores

# In[155]:


Category_by_score = final_data.groupby('product_category')['review_score'].mean().sort_values(ascending=False)
Category_by_score


# In[156]:


plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=Category_by_score[:10].index, x=Category_by_score[:10].values)
plt.ylabel(' ', fontsize=18 )
plt.title('Product Ratings of the Least 10 product categories' , fontsize=20)
# The 10 products with the highest scores are not the 10 categories with the highest demand. 


# In[157]:


plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=Category_by_score[-10:].index, x=Category_by_score[-10:].values)
plt.ylabel(' ', fontsize=18 )
plt.title('Lowest 10 Ratings ', fontsize=20)


# * The 10 products with the highest scores are not the 10 categories with the highest demand. 
# * A few of the least sold categories are the products that sell the lowest rated.
# 

# * Total Price for each categories

# In[158]:


Cat_vs_Price = final_data.groupby('product_category')['price'].sum().sort_values(ascending=False)
Cat_vs_Price


# In[159]:


plt.figure(figsize=(12,5))
sns.set_style("darkgrid")
sns.barplot(y=Cat_vs_Price[:10].index, x=Cat_vs_Price[:10].values)
plt.ylabel(' ', fontsize=18 )
plt.title('Categories versus total price - Top 10', fontsize=20)


# In[160]:


# Pivot table for the price sum for each categories
price_pivot = final_data.pivot_table(index =['product_category'], 
                       values =['price'], 
                       aggfunc ='sum')
price_pivot.sort_values(by='price', ascending=False)


# In[161]:


# top 10 price contribution
price_pivot.sort_values(by='price', ascending=False)[:10].plot.barh(figsize=(15,7),title='Price sum for each categories')


# In[162]:


# In 2017 sales for each categories - top 10
final_data[final_data['purchased_Year']==2017].groupby('product_category')['product_category'].size().sort_values(ascending=False)[:10]


# In[164]:


# Now for 2018, top 10 categories
data[data['purchased_Year']==2018].groupby('product_category')['product_category'].size().sort_values(ascending=False)[:10]


# In[163]:


# In 2017 product categories sales least 15 categories
final_data[final_data['purchased_Year']==2017].groupby('product_category')['product_category'].size().sort_values(ascending=False)[-15:]


# In[165]:


# Least 10 categories in 2018
final_data[final_data['purchased_Year']==2018].groupby('product_category')['product_category'].size().sort_values(ascending=False)[-15:]


# In[167]:


# via this pivot table, you can see each categorie's order counts for ach month
# hence we can se which categories increases or decreases monthly
# or in some months no orders available for some categories, it give us insgihts about orders
pd.pivot_table(final_data[(final_data['purchased_Year']==2017) ], 
                   values='order_id', index='product_category', columns=['purchased_Month'], aggfunc='count', fill_value=0)


# In[169]:


# same logic but this time in 2018
pd.pivot_table(final_data[(final_data['purchased_Year']==2018) ], 
                   values='order_id', index='product_category', columns=['purchased_Month'], aggfunc='count', fill_value=0)


# In[170]:


final_data.groupby(['purchased_Year','product_category'])['order_id'].count()


# In[171]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.gridspec import GridSpec
pd.set_option('display.max_columns', 100)
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#for the millions format function
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as tkr
import matplotlib as mpl
from datetime import datetime, timedelta
from pandas import DataFrame
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
import re


def thousand_count_y(y, pos):
    return '{:.0f} K'.format(y*1e-3)
formatter_thousand_count_y = FuncFormatter(thousand_count_y)
#ax2.yaxis.set_major_formatter(formatter_thousand_count_y)

def millons_count_y(y, pos):
    return '{:.0f} M'.format(y*1e-6)
formatter_millons_count_y = FuncFormatter(millons_count_y)
#ax2.yaxis.set_major_formatter(formatter_millons_count_y)

def thousand_count_x(x, pos):
    return '{:.0f} K'.format(x*1e-3)
formatter_thousand_count_x = FuncFormatter(thousand_count_x)
#ax2.yaxis.set_major_formatter(formatter_thousand_count_x)

def millons_count_x(x, pos):
    return '{:.0f} M'.format(x*1e-6)
formatter_millons_count_x = FuncFormatter(millons_count_x)
#ax2.yaxis.set_major_formatter(formatter_millons_count_x)
#__________________________________________________________________________
def thousand_real_y(y, pos):
    return 'R${:.0f} K'.format(y*1e-3)
formatter_thousand_real_y = FuncFormatter(thousand_real_y)
#ax2.yaxis.set_major_formatter(formatter_thousand_real_y)

def millons_real_y(y, pos):
    return 'R${:.1f} M'.format(y*1e-6)
formatter_millons_real_y = FuncFormatter(millons_real_y)
#ax2.yaxis.set_major_formatter(formatter_millons_real_y)

def thousand_real_x(x, pos):
    return 'R${:.0f} K'.format(x*1e-3)
formatter_thousand_real_x = FuncFormatter(thousand_real_x)
#ax2.yaxis.set_major_formatter(formatter_thousand_real_x)

def millons_real_x(x, pos):
    return 'R${:.1f} M'.format(x*1e-6)
formatter_millons_real_x = FuncFormatter(millons_real_x)
#ax2.yaxis.set_major_formatter(formatter_millons_real_x)


# In[173]:


sellers_x_date = pd.merge(order_items,orders, on = "order_id")
sellers_x_date = sellers_x_date.drop_duplicates(subset=["seller_id"])
sellers_x_date.head()


# In[174]:


#The 10 cities with the most clients
clients_by_city = customers.groupby("customer_city").count()["customer_unique_id"].reset_index().sort_values(by="customer_unique_id",ascending=False).head(10)
clients_by_city.rename(columns = {"customer_unique_id":"total"}, inplace=True)
clients_by_city


# In[175]:


#The 10 states with the most clients
clients_by_state = customers.groupby(["customer_state"]).count()["customer_unique_id"].reset_index().sort_values(by="customer_unique_id",ascending=False).head(10)
clients_by_state.rename(columns = {"customer_unique_id":"total"}, inplace=True)
clients_by_state


# In[176]:


#The 10 cities with the most sellers
sellers_by_city = sellers.groupby("seller_city").count()["seller_id"].reset_index().sort_values(by="seller_id",ascending=False).head(10)
sellers_by_city.rename(columns = {"seller_id":"total"}, inplace=True)
sellers_by_city


# In[177]:


clients_x_date = pd.merge(customers, orders, on = "customer_id")
clients_x_date.head()


# In[178]:


clients_x_date[['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']]= clients_x_date[['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']].apply(pd.to_datetime)


# In[179]:


clients_x_date['Year_Month'] = clients_x_date['order_purchase_timestamp'].dt.strftime('%Y%m')
#clients_x_date['Year_Month'].sort_values() # values from september 2016 to september 2018


# In[180]:


sellers_x_date[['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']]= sellers_x_date[['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']].apply(pd.to_datetime)


# In[181]:


sellers_x_date['Year_Month'] = clients_x_date['order_purchase_timestamp'].dt.strftime('%Y%m')


# In[182]:


#The 10 states with the most sellers
sellers_by_states = sellers.groupby("seller_state").count()["seller_id"].reset_index().sort_values(by="seller_id",ascending=False).head(10)
sellers_by_states.rename(columns = {"seller_id":"total"}, inplace=True)
sellers_by_states


# In[183]:


sellers["seller_city"] = sellers["seller_city"].str.capitalize()
sellers.head()


# In[184]:


fig = plt.figure(constrained_layout=True, figsize=(20, 15))

# Axis definition
gs = GridSpec(5, 2, figure= fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
#ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])
#ax7 = fig.add_subplot(gs[3, 1])
ax8 = fig.add_subplot(gs[3, :])

#Customer city
sns.barplot(x="total", y="customer_city", data=clients_by_city, ax=ax1, palette='viridis')
ax1.set_title("The 10 cities with the most clients", size=14, color='black')
ax1.set_xlabel("")
ax1.set_ylabel("")
for rect in ax1.patches:
    ax1.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.axes.get_xaxis().set_visible(False) 

#Customer states
sns.barplot(x="total", y='customer_state', data=clients_by_state, ax=ax2, palette="YlGnBu")
ax2.set_title("The 10 states with the most clients", size=14, color='black')
ax2.set_xlabel("")
ax2.set_ylabel("")
for rect in ax2.patches:
    ax2.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.axes.get_xaxis().set_visible(False)

#Customer per year
sns.lineplot(x="Year_Month", y="order_id", data=clients_x_date.groupby("Year_Month").agg({"order_id" : "count"}).reset_index(),ax=ax3, alpha=0.8,
             color='darkslateblue', linewidth=1, marker='o', markersize=4)
sns.barplot(x="Year_Month", y="order_id", data=clients_x_date.groupby("Year_Month").agg({"order_id" : "count"}).reset_index(),ax=ax3, alpha=0.1)
ax3.set_title("Customer Evolution", size=14, color="black")
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_ylim(0,9000)
#plt.setp(ax3.get_xticklabels(), rotation=45)
for p in ax3.patches:
        ax3.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="top", xytext=(0, 15), textcoords="offset points", 
                    color= "black", size=12)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.set_yticklabels([])
ax3.set_yticks([])

#Total de Customers
ax3.text(-1.5, 8000, "93,396", fontsize=20, ha='center', color="navy")
ax3.text(-1.5, 7200, "Total Customers", fontsize=10, ha='center')
ax3.text(-1.5, 5000, "42,122", fontsize=18, ha='center', color="navy")
ax3.text(-1.5, 4200, "Customers 2017", fontsize=8, ha='center')
ax3.text(-1.5, 2000, "51,619", fontsize=18, ha='center', color="navy")
ax3.text(-1.5, 1200, "Customers 2018", fontsize=8, ha='center')

# Sellers city
sns.barplot(x="total", y="seller_city", data=sellers_by_city, ax=ax5, palette='viridis')
ax5.set_title("The 10 cities with the most sellers", size=14, color='black')
ax5.set_xlabel("")
ax5.set_ylabel("")
for rect in ax5.patches:
    ax5.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=12)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.axes.get_xaxis().set_visible(False)

# Sellers states
sns.barplot(x="total", y="seller_state", data=sellers_by_states, ax=ax6, palette="YlGnBu")
ax6.set_title("The 10 states with the most sellers", size=14, color='black')
ax6.set_xlabel("")
ax6.set_ylabel("")
for rect in ax6.patches:
    ax6.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=12)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.axes.get_xaxis().set_visible(False)

#Sellers per year
sns.lineplot(x="Year_Month", y="order_id", data=sellers_x_date.groupby("Year_Month").agg({"order_id" : "count"}).reset_index(),ax=ax8,
             color='darkslateblue', linewidth=1, marker='o', markersize=5)
sns.barplot(x="Year_Month", y="order_id", data=sellers_x_date.groupby("Year_Month").agg({"order_id" : "count"}).reset_index(),ax=ax8, alpha=0.1)
ax8.set_title("Seller Evolution", size=14, color="black")
ax8.set_xlabel("")
ax8.set_ylabel("")
ax8.set_ylim(0, 500)
plt.setp(ax8.get_xticklabels(), rotation=45)
for p in ax8.patches:
        ax8.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="top", xytext=(0, 15), textcoords="offset points", 
                    color= "black", size=12)
ax8.spines["top"].set_visible(False)
ax8.spines["right"].set_visible(False)
ax8.spines["left"].set_visible(False)
ax8.set_yticklabels([])
ax8.set_yticks([])
        
#Total de Sellers
ax8.text(-1.5, 460, "3028", fontsize=22, ha='center', color="navy")
ax8.text(-1.5, 420, "Total Sellers", fontsize=10, ha='center')
ax8.text(-1.5, 300, "1,236", fontsize=18, ha='center', color="navy")
ax8.text(-1.5, 260, "Sellers 2017", fontsize=8, ha='center')
ax8.text(-1.5, 140, "1,832", fontsize=18, ha='center', color="navy")
ax8.text(-1.5, 100, "Sellers 2018", fontsize=8, ha='center')


plt.suptitle("Customers and Sellers (2017-2018)", size=18)

# plt.tight_layout()


# ## <span style='color:gray '>Payment Methods </span> 

# In[185]:


final_data['payment_type'].unique() # 4 payment method type is available


# In[186]:


payment_types = final_data['payment_type'].value_counts(ascending=False)
payment_types
# Most used payment type credit card


# In[187]:


plt.figure(figsize=(8,5))
x = payment_types.index
y = payment_types.values
sns.barplot(x=x,y=y)
plt.title('Order Counts versus Payment Methods')
plt.ylabel('Order Counts')


# In[188]:


plt.figure(figsize=(10,5))
data['payment_type'].value_counts().plot(kind='pie')
plt.title('The Most Frequent Payment Type'.title() , fontsize=20);


# In[189]:


sns.set()
plt.figure(figsize=(10,8))
sns.barplot(x="purchased_Year", y='price', hue="payment_type", data=final_data)
plt.title('Payment type over the years')
# There was an increase in the use of debit cards from 2017 to 2018.
# It almost reached credit card usage. 


# In[190]:


order_count = final_data.groupby(['Year_Month','payment_type'])['order_id'].count()
order_count


# In[191]:


plt.figure(figsize=(15,8))
sns.barplot(x="purchased_Month", y='price', hue="payment_type", data=data)
plt.title('Payment methods of prices by month')
# The amounts used in some months have increased compared to the average prices, 
# and generally large payments from bank cards have not been made.


# In[192]:


plt.figure(figsize=(10,8))
sns.barplot(x="purchased_Day", y='price', hue="payment_type", data=data)
# Interestingly, on a daily basis, the use of debit cards on Thursday showed values close to the use of credit cards.


# In[193]:


price_vs_paymenttype = final_data.groupby('payment_type')['price'].sum()
price_vs_paymenttype


# In[194]:


plt.figure(figsize=(8,5))
x = price_vs_paymenttype.index
y = price_vs_paymenttype.values
sns.barplot(x=x,y=y)
plt.title('Price Sum using different Payment Methods')
plt.ylabel('Price Sum ')
# Here, while there is a 4-fold difference between boleto and credit card usage according to the number of orders,
# a 5-fold difference is observed in terms of payment amount.


# In[195]:


ordercount_vs_paymenttype = data.groupby('payment_type')['order_id'].count()
ordercount_vs_paymenttype


# In[196]:


plt.figure(figsize=(8,5))
x = ordercount_vs_paymenttype.index
y = ordercount_vs_paymenttype.values
sns.barplot(x=x,y=y)


# In[203]:


pd.pivot_table(final_data, values='order_id', index=['Year_Month'],
                    columns=['payment_type'], aggfunc='count').plot(kind='bar',figsize=(13,5) )

# Credit card usage stands out in the biggest increase in November.


# In[205]:


pd.pivot_table(final_data, values='order_id', index=['Year_Month'],
                    columns=['payment_type'], aggfunc='count').sort_values(by='Year_Month').plot(kind='line', figsize=(10,5))
plt.title('Distribution of orders according to payment methods based on month and year.')


# ## <span style='color:gray '> Payment Sequential  </span> 

# In[206]:


final_data.payment_sequential.unique() 


# In[207]:


len(final_data.payment_sequential.unique() ) # 26 payment sequentil type is avaliable


# In[208]:


# Here we can see when payment type is vocuher sequential takes a lot of values.
pivot = final_data.pivot_table(index ='payment_type',
                       values =['order_id'], columns=['payment_sequential'],
                       aggfunc ='count',fill_value=0)

pivot
# it can be seen that when payment type voucher, payment sequential can take bigger values.


# In[494]:


palette = sns.color_palette("mako_r", 6)
payment=pd.crosstab(index=final_data['payment_type'], columns=final_data['payment_sequential'], values=final_data['order_id'],  aggfunc='count').round(0)
sns.heatmap(payment, cmap=palette, annot=True, fmt='g');
plt.figure(figsize=(25,10))


# In[209]:


len(final_data[final_data['payment_type']=='voucher'].groupby('product_category')['product_category'].unique())
#.count()


# In[210]:


final_data[final_data['payment_type']=='voucher'].groupby('product_category')['product_category'].count().sort_values(ascending=False)
# voucher can be used a lot of categories


# In[211]:


len(final_data[final_data['payment_type']=='debit_card'].groupby('product_category')['product_category'].unique())


# In[212]:


len(final_data[final_data['payment_type']=='boleto'].groupby('product_category')['product_category'].unique())


# Different payment types can be used a lot of categories that is no category-specific payment types

# ## <span style='color:gray '> Payment Installments  </span> 

# In[213]:


final_data.payment_installments.nunique()


# In[214]:


install_order =final_data[final_data['payment_type']=='credit_card'].groupby('payment_installments')['order_id'].count()
install_order


# In[215]:


plt.figure(figsize=(8,5))
plt.yscale('log')
x = install_order.index
y = install_order.values
sns.barplot(x=x,y=y)
plt.ylabel('Using log transformation order counts')
plt.title('Order Counts versus Payment Installments')


# In[216]:


data[data['payment_installments']==0]
# I didn't understand what does 0 payment installments mean


# In[217]:


data.payment_installments.unique()


# In[218]:


data['payment_installments'].describe()


# In[219]:


installment = final_data.groupby('payment_installments')['price'].sum().sort_values(ascending=False)
installment


# In[220]:


plt.figure(figsize=(8,5))
x = installment.index
plt.yscale('log')
y = installment.values
sns.barplot(x=x,y=y)
plt.ylabel('Using log transformation order counts')
plt.title('Price Sum versus Payment Installments')


# In[221]:


sns.boxplot(x=data['payment_installments'])
plt.title('Boxplot of the payment installments')


# In[222]:


data.groupby('payment_installments')['payment_installments'].count().sort_values(ascending=False)


# In[224]:


pivot = data.pivot_table(index =None,
                       values =['order_id'], columns=['payment_installments'],
                       aggfunc ='count',fill_value=0)

pivot


# In[225]:


data.groupby('payment_installments')['payment_installments'].count().sort_values(ascending=False).plot(kind='bar', color='green')


# In[226]:


plt.yscale('log')
data.groupby('payment_installments')['payment_installments'].count().plot(kind='bar', color='green')


# In[227]:


len(final_data[final_data['payment_installments']<5])


# In[228]:


len(final_data[(final_data['payment_installments']>=10) & (data['payment_installments']<=15)])


# In[229]:


len(final_data[final_data['payment_installments']>15])


# In[230]:


sns.barplot(x='payment_installments', y='price', data=final_data)


# Brazilians mostly shopped with one shot, followed by 2, 3 and 4 installments.
# It seems to have been seen in the case of an increase in the amount of 10 installments.

# In[232]:


np.corrcoef(final_data['payment_installments'], final_data['price']) # it has 0.27 correlated, not that much.


# ## <span style='color:gray '> Geolocation </span> 

# In[236]:


import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap


# In[237]:


geolocation = pd.read_csv('olist_geolocation_dataset.csv')
geolocation.head()


# In[238]:


geolocation.groupby('geolocation_city')['geolocation_city'].count().sort_values(ascending=False)


# In[239]:


lat = geolocation['geolocation_lat']
lon = geolocation['geolocation_lng']

plt.figure(figsize=(8,8))

m = Basemap(llcrnrlat=-55.401805,llcrnrlon=-92.269176,urcrnrlat=13.884615,urcrnrlon=-27.581676)
m.bluemarble()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') 
#m.drawcoastlines()
m.drawcountries()
m.scatter(lon, lat,zorder=8,alpha=0.1,color='red')
plt.title('Distribution of orders by States')


# In[240]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8))
img = mpimg.imread('Brazilian_States_by_Population_density.svg.png')
plt.imshow(img)
plt.title('Population Denstiy Map')
plt.show()
#


# Places with high population density and high number of orders in Brazil are similar cities

# ## <span style='color:gray '> Customer States & Seller States </span> 

# In[241]:


data['seller_state'].describe()


# In[242]:


seller_states = final_data.groupby('seller_state')['order_id'].count().sort_values(ascending=False)
seller_states


# In[243]:


plt.figure(figsize=(8,5))
y = seller_states.index
plt.xscale('log')
x = seller_states.values
sns.barplot(x=x,y=y)
plt.ylabel('Order Counts')
plt.title('Seller States')


# In[253]:


data['customer_state'].describe()


# In[244]:


final_data['customer_state'].unique()


# In[245]:


# Here I looked at the order counts in non-seller states
final_data[(final_data['seller_state']=='TO') |(final_data['seller_state']=='AL')| (final_data['seller_state']=='PE') | (final_data['seller_state']=='AP') ]['order_id'].count()


# In[247]:


final_data[(final_data['seller_state']=='TO') |(final_data['seller_state']=='AL')| (final_data['seller_state']=='PE') | (final_data['seller_state']=='AP') & (final_data['seller_state']!='RR') ]['product_category'].unique()


# In[246]:


# Here I looked at the order counts in seller states
final_data[(final_data['seller_state']=='TO') |(final_data['seller_state']!='AL') & (final_data['seller_state']!='PE') & (final_data['seller_state']!='AP') ]['order_id'].count()


# In[248]:


final_data[(final_data['seller_state']!='TO') |(final_data['seller_state']!='AL') & (final_data['seller_state']!='PE') & (final_data['seller_state']!='AP') & (final_data['seller_state']!='RR')]['freight_value'].mean()


# In[249]:


final_data[((final_data['product_category']=='housewares')| (final_data['product_category']=='cool_stuff') | (final_data['product_category']=='sports_leisure')| (final_data['product_category']=='toys')| (final_data['product_category']=='furniture_decor')|(final_data['product_category']=='watches_gifts')| (final_data['product_category']=='computers_accessories')|(final_data['product_category']=='consoles_games')| (final_data['product_category']=='health_beauty') | (final_data['product_category']=='auto')| (final_data['product_category']=='home_confort')) & (final_data['seller_state']=='TO') |(final_data['seller_state']!='AL') & (final_data['seller_state']!='PE') & (final_data['seller_state']!='AP') ]['freight_value'].mean()


# In[250]:


final_data[(final_data['seller_state']=='TO') |(final_data['seller_state']=='AL')| (final_data['seller_state']=='PE') | (final_data['seller_state']=='AP') | (final_data['seller_state']=='RR')]['freight_value'].mean()


# In[251]:


final_data[final_data['seller_state']=='AM']['freight_value']


# In[254]:


Customer_State = final_data.groupby('customer_state')['order_id'].size().sort_values(ascending=False)
Customer_State


# In[255]:


len(Customer_State)


# In[256]:


len(seller_states)


# In[257]:


plt.figure(figsize=(8,6))
y = Customer_State.index
plt.xscale('log')
x = Customer_State.values
sns.barplot(x=x,y=y)
plt.ylabel('Order Counts')
plt.title('Customer States')


# In[258]:


plt.figure(figsize=(12,8))
sns.set_style("darkgrid")
sns.barplot(y=Customer_State.index, x=Customer_State.values)
plt.ylabel('Customer States', fontsize=15 )
plt.title("Customer States Sales", fontsize=20)


# In[259]:


data.groupby('customer_state')['price'].mean().sort_values(ascending=False)


# In[260]:


# States of sale and average sales, maximum values, etc.
data.groupby(by='customer_state')[['payment_value']].agg(['sum','mean','median']).sort_values(by=('payment_value','sum'),ascending=False)[0:15]


# In[261]:


Pivot= pd.pivot_table(data, values='price', index=['product_category'],
                    columns=['customer_state'], aggfunc=np.mean, fill_value=0)


# In[262]:


# function to indicate some values with colors
def add_color(val):
  if val <10 :
    color = 'black'
  elif val < 100:
    color = 'gray'
  elif val < 500:
    color = 'green'
  elif val < 1000:
    color='purple'
  else:
    color='red'
    
  return 'color: %s' % color


# In[263]:


Pivot.style.highlight_null(null_color='gray')


# In[264]:


Pivot.style.applymap(add_color, 
                    subset=["AC","AL","AM","AP","BA", "CE", "DF","ES","GO","MA", "MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"])


# In[265]:


data['seller_city'].describe()


# In[266]:


data['customer_city'].describe()


# * São Paulo, Brazil’s vibrant financial center, is among the world's most populous cities, with numerous cultural institutions and a rich architectural tradition.

# ## <span style='color:gray '> Review Scores </span> 

# In[267]:


data.review_score.unique() # from 1 to 5


# In[268]:


data.groupby('review_score')['review_score'].count().sort_values(ascending=False)  # High rate of 5 score


# In[269]:


plt.figure(figsize=(10,6))
sns.barplot(x=data['review_score'].value_counts().index,y=data['review_score'].value_counts().values)
plt.xlabel('Ratings')
plt.ylabel('Counts')
plt.title('Review Scores')


# In[271]:


# I looked here if there is a correlation between review score and delivery time but I dind't find higher correlations
plt.figure(figsize=(15,6))
corr = final_data.corr()
sns.heatmap(corr, annot=True)
# I looked for correlations on the data


# In[272]:


import scipy


# In[274]:


# correlations between price and 
np.corrcoef(final_data['freight_value'], final_data['price']) # there is a high correlation here, when price is increase freight value also increases


# In[275]:


scipy.stats.pearsonr(final_data['freight_value'], final_data['price'])


# * Review Scores versus Product Categories

# In[276]:


product_reviews_mean = final_data.groupby('product_category').mean()['review_score'].reset_index()
product_reviews_mean.sort_values(by='review_score').reset_index()


# In[280]:


df = final_data.copy() # not to have new column in the final data

df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_approved_at']).dt.total_seconds() / 86400
ax = sns.catplot(x="review_score", y="delivery_time", kind="box",
                 data= df, height=4, aspect=1.5)
plt.title('Delivery Time versus review scores')
# I looked if there is correlation between review score and delivery time 
# here I thought that when delivery time decreases score increase but in the heatmap I didn't find a valuable correlation


# In[283]:


final_data[final_data['review_score']==1]['product_category'].nunique() # when score is 1 it can be seen almost all categories


# In[284]:


final_data[final_data['review_score']==2]['product_category'].nunique()


# In[285]:


final_data[final_data['review_score']==3]['product_category'].nunique()


# In[286]:


final_data[final_data['review_score']==4]['product_category'].nunique()


# In[287]:


final_data[final_data['review_score']==5]['product_category'].nunique()


# Here I looked to see if the scores are specific to some categories or are seen in every category, almost every category is seen in the score values.

# In[ ]:


#scores[scores>4.5].count()


# In[288]:


(final_data['order_delivered_customer_date']-final_data['order_purchase_timestamp']).describe()


# In[289]:


#While it reaches the customer in an average of 12 days in normal times, it reaches the customer in 18 days in those with score 1 
score_1_2 = final_data[(final_data['review_score']==1) | (final_data['review_score']==2)]
(score_1_2['order_delivered_customer_date'] - score_1_2['order_purchase_timestamp']).mean()


# In[290]:


score_3= final_data[(final_data['review_score']==3)]
(score_3['order_delivered_customer_date'] - score_3['order_purchase_timestamp']).mean()


# In[291]:


score_4_5 = final_data[(final_data['review_score']==4) | (final_data['review_score']==5)]
(score_4_5['order_delivered_customer_date'] - score_4_5['order_purchase_timestamp']).mean()


# Here, as the score increases, there is a decrease in the average delivery date.

# In[292]:


seller_scores = pd.pivot_table(final_data, values='review_score', index='seller_id', columns=None, aggfunc='mean', fill_value=0, margins=False, dropna=False, margins_name='All', observed=False).sort_values(by='review_score', ascending=False)
seller_scores


# In[293]:


final_data[final_data['seller_id']=='48efc9d94a9834137efd9ea76b065a38']['order_id'].count()


# In[295]:


# seller who has score 5 sells in different categories
final_data[final_data['seller_id']=='48efc9d94a9834137efd9ea76b065a38']['product_category'].unique()


# In[296]:


# The number of products with a score of 5 in the most sold products is also quite high.
score_5 = final_data[final_data['review_score']==5]
score_5.groupby('product_category')['order_id'].count().sort_values(ascending=False)[:10]


# ## <span style='color:gray '> Seller Analysis </span> 

# In[298]:


final_data.groupby('seller_id')['product_category'].count().sort_values(ascending=False)


# In[307]:


final_data.groupby('seller_id')['price'].sum().sort_values(ascending=False)


# In[299]:


final_data['seller_id'].describe() # there are 3028 sellers


# In[300]:


final_data[final_data['seller_id']=='4a3ca9315b744ce9f8e9374361493884']['product_category'].unique()


# In[301]:


# top one seller sells the most demanded category
top_one_seller =final_data[final_data['seller_id']=='4a3ca9315b744ce9f8e9374361493884']
top_one_seller.groupby('product_category')['product_category'].count().sort_values(ascending=False)


# In[302]:


top_one_seller.groupby('product_category')['price'].sum().sort_values(ascending=False)


# In[304]:


final_data[final_data['seller_id']=='4a3ca9315b744ce9f8e9374361493884']['review_score'].mean() # top1 seller review score


# In[305]:


# most seller according to price contribution
final_data[final_data['seller_id']=='53243585a1d6dc2643021fd1853d8905']['seller_city'].describe()


# In[306]:


# second most seller's product categories telephony and computers
final_data[final_data['seller_id']=='53243585a1d6dc2643021fd1853d8905']['product_category'].unique()


# In[362]:


# second most seller
final_data[final_data['seller_id']=='53243585a1d6dc2643021fd1853d8905']['price'].sum()


# In[309]:


final_data[final_data['seller_id']=='53243585a1d6dc2643021fd1853d8905']['seller_city'].unique()


# * The sellers with the highest sales and those with the highest prices are different, and their contribution to the price varies considerably according to the categories. Since phones and computers are expensive categories, they are valuable in monetary terms even if their sales are low.

# In[308]:


final_data.groupby('seller_id')['order_id'].count().sort_values(ascending=False)[:10]


# In[310]:


top20sellers_ordercounts = final_data.groupby('seller_id')['order_id'].count().sort_values(ascending=False).head(20)
top20sellers_ordercounts


# In[313]:


top20sellerdic = {'4a3ca9315b744ce9f8e9374361493884' : 2093,
'6560211a19b47992c3666cc44a7e94c0' :  2076,
'1f50f920176fa81dab994f9023523100'   : 2003,
'cc419e0650a3c5ba77189a1882b7556a'   : 1828,
'da8622b14eb17ae2831f4ac5b9dab84a'   : 1650,
'955fee9216a65b617aa5c0531780ce60'   : 1492,
'1025f0e2d44d7041d6cf58b6550e0bfa'   : 1456,
'7c67e1448b00f6e969d365cea6b010ab'   : 1445,
'7a67c85e85bb2ce8582c35f2203ad736'   : 1221,
'ea8482cd71df3c1969d7b9473ff13abc'   : 1217,
'3d871de0142ce09b7081e2b9d1733cb1'   : 1176,
'4869f7a5dfa277a7dca6462dcf3b52b2'   : 1170,
'8b321bb669392f5163d04c59e235e066'   : 1014,
'cca3071e3e9bb7d12640c9fbe2301306'   :  872,
'620c87c171fb2a6dd6e8bb4dec959fc6'   :  797,
'a1043bafd471dff536d0c462352beb48'  : 788,
'e9779976487b77c6d4ac45f75ec7afe9'  : 749,
'f8db351d8c4c4c22c6835c19a46f01b0'   : 747,
'd2374cbcbb3ca4ab1086534108cc3ab7'   : 680,
'391fc6631aebcf3004804e51b40bcf1e'   :  630}

plt.bar(range(len(top20sellerdic)), list(top20sellerdic.values()), align='center')
#plt.xticks(range(len(top20sellerdic)), list(top20sellerdic.keys()))
plt.title('Top 20 sellers')


# In[314]:


final_data.groupby('seller_id')['price'].sum().sort_values(ascending=False)[:10]
# The sellers who sell the most and those who sell the most may differ from each other.


# In[315]:


# sellers sale their products from 1 city.
final_data.groupby('seller_id')['seller_city'].nunique().sort_values() # bütün satıcılar bir şehir üzerinden satış yapıyor.


# In[316]:


# a seller where expensive categories are sold, the 2nd highest selling seller in terms of price
final_data[final_data['seller_id']=='4869f7a5dfa277a7dca6462dcf3b52b2']['product_category'].unique()


# In[374]:


#data.pivot_table('price', index='seller_id', columns='product_category', aggfunc='sum', fill_value=0)


# * Sellers who has review score 1

# In[317]:


# Sellers who has review score 1
seller_score_1 = final_data[final_data['review_score']==1]
seller_score_1.groupby('seller_id')['order_id'].count().sort_values(ascending=False)


# In[377]:


seller_score_1.groupby('seller_id')['order_id'].count().sort_values(ascending=False)[:10]


# In[319]:


final_data[final_data['seller_id']=='7c67e1448b00f6e969d365cea6b010ab']['product_category'].unique()


# In[322]:


final_data[final_data['seller_id']=='7c67e1448b00f6e969d365cea6b010ab']['review_score'].value_counts()
# Sellers with always low scores can be warned to ensure that their customers' experience on the site is not bad.


# ## <span style='color:gray '> Customer Analysis </span> 

# In[325]:


final_data['customer_unique_id'].describe()


# In[326]:


final_data.customer_unique_id.value_counts()[:10]


# In[327]:


final_data['customer_id'].describe()


# In[328]:


final_data.customer_id.value_counts()


# In[329]:


topcustomers = final_data.pivot_table(index =['customer_unique_id'], 
                       values =['product_category'], 
                       aggfunc ='count')
topcustomers.sort_values(by='product_category', ascending=False)[:20]


# #### customer with customer unique id : '9a736b248f67d166d2fbb006bcb877c3' 
# 
# ##### I think this customer is an opportunistic customer.

# In[ ]:


#data[data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3']c# to see its orders


# In[330]:


# Let's start with top purchased customer
top_purchaser = final_data[final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3']
top_purchaser.groupby('product_id').size()


# In[333]:


# The customer who bought the most products in the data
final_data[final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3']['product_category'].unique()


# In[332]:


# 12 furniture decor
len(final_data[(final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (data['product_category']=='furniture_decor')])


# In[335]:


final_data[(final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (data['product_id']=='4eb99b5f0d7e411f246a5c9c0ae27a5e')]['order_item_id']


# In[334]:


# These 12 items were also purchased on the same day.
final_data[(final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (data['product_category']=='furniture_decor')]['order_purchase_timestamp'].unique()


# In[336]:


final_data[(final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (final_data['product_id']=='ebf9bc6cd600eadd681384e3116fda85')]['order_item_id'].sum()


# In[339]:


#All 21 products received on the same day by the same customer
final_data[(final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (final_data['product_category']=='housewares')]['order_purchase_timestamp'].unique()


# In[337]:


final_data[(final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (data['product_id']=='5ddab10d5e0a23acb99acf56b62b3276')]['order_item_id'].sum()


# In[387]:


len(data[(data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (data['product_category']=='bed_bath_table')])


# In[340]:


# These 42 products were also sold on the same date.
final_data[(final_data['customer_unique_id']=='9a736b248f67d166d2fbb006bcb877c3') & (final_data['product_category']=='bed_bath_table')]['order_purchase_timestamp'].unique()


# ##### customer_unique_id : ' 6fbc7cdadbb522125f4b27ae9dee4060'  - second most buyer

# In[341]:


len(final_data[final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060'])
# this customer buy 38 same product in a same day


# In[343]:


# Let's look at the categories of products bought by the second most shopping customer.
final_data[final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060']['product_category'].unique()


# In[342]:


final_data[final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060']['product_id'].unique()


# In[404]:


len(final_data[final_data['product_id']=='0554911df28fda9fd668ce5ba5949695'])


# In[344]:


len(final_data[(final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060') & (final_data['product_category']=='office_furniture')])


# In[347]:


# this customer also bought 38 of the same product and they were all bought on the same day
final_data[(final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060') & (final_data['product_category']=='office_furniture')]['product_id'].unique()


# In[345]:


final_data[(final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060') & (final_data['product_category']=='office_furniture')]['order_item_id'].sum()


# In[348]:


# this customer also bought 38 of the same product and they were all bought on the same day
final_data[(final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060') & (final_data['product_category']=='office_furniture')]['order_purchase_timestamp'].unique()


# In[346]:


final_data[(final_data['customer_unique_id']=='6fbc7cdadbb522125f4b27ae9dee4060') & (final_data['product_category']=='office_furniture')][['product_id','order_purchase_timestamp','order_item_id','seller_id','payment_installments','payment_type','payment_sequential']]


#  Receiving all products in the same category as the same product on the same day made us think that this customer may also be a seller.

# ##### customer_unique_id : ' f9ae226291893fda10af7965268fb7f6'

# In[349]:


# Let's look at the categories of products bought by the third most shopping customer.
final_data[final_data['customer_unique_id']=='f9ae226291893fda10af7965268fb7f6']['product_category'].unique()


# In[350]:


final_data[(final_data['customer_unique_id']=='f9ae226291893fda10af7965268fb7f6') & (final_data['product_category']=='garden_tools')]['product_id'].unique()


# In[351]:


final_data[(final_data['customer_unique_id']=='f9ae226291893fda10af7965268fb7f6') & (data['product_category']=='garden_tools')][['order_purchase_timestamp','order_item_id','seller_id','product_id','payment_type','payment_sequential']]


# In[352]:


third_purchaser =final_data[final_data['customer_unique_id']=='f9ae226291893fda10af7965268fb7f6']
third_purchaser.groupby('product_id').size()


# In[353]:


final_data[(final_data['customer_unique_id']=='f9ae226291893fda10af7965268fb7f6') & (final_data['product_category']=='garden_tools')]['order_purchase_timestamp'].unique()


# In[354]:


final_data[final_data['customer_unique_id']=='f9ae226291893fda10af7965268fb7f6']['order_item_id'].sum()


# I could not decide whether this customer is an opportunistic customer, he may have made various campaigns in the garden tools sale in March, the products he bought could be flower pots, but I can see that he bought a lot of products from the same product.

# In[ ]:


#data[(data['customer_unique_id']=='8af7ac63b2efbcbd88e5b11505e8098a')]


# In[ ]:


#index = pivot.sort_values(by='product_category', ascending=False)[:20].index
#index


# In[ ]:


#category = []
#count_different_product = []
#date = []
#for i in index:
#    i=str(i)
#    count_different_product.append(data[data['customer_unique_id']==i]['product_id'].unique())
#    date.append(data[data['customer_unique_id']==i]['order_purchase_timestamp'][0:9].unique())
#    print('Customer unique id :', i,'\nproduct id count :',count_different_product)
#    print('                                           ')
#    print('Unique dates to purchase:\n', date)
#    print('                                           ')
#    print('*******************************************')


# ## <span style='color:gray '> Order Items </span> 

# In[355]:


final_data['order_item_id'].describe()


# In[356]:


final_data['order_item_id'].unique()


# In[357]:


final_data.groupby('order_id')['order_item_id'].sum().sort_values(ascending=False)


# In[192]:


# Health and beauty products can be makeup products with the highest number of items in an order.
final_data[final_data['order_id']=='8272b63d03f5f79c56e9e4120aec44ef'] 


# ## <span style='color:gray '> Order ID </span> 

# In[358]:


final_data['order_id'].describe()


# In[359]:


len(final_data[final_data['order_id']=='895ab968e7bb0d5659d16cd74cd1650c'])


# In[360]:


final_data[final_data['order_id']=='895ab968e7bb0d5659d16cd74cd1650c']


# ##### some orders 
# 895ab968e7bb0d5659d16cd74cd1650c	63	 0.1%
# 
# fedcd9f7ccdc8cba3a18defedd1a5547	38	 < 0.1%
# 
# fa65dad1b0e818e3ccc5cb0e39231352	29	 < 0.1%
# 
# ccf804e764ed5650cd8759557269dc13	26	 < 0.1%
# 
# a3725dfe487d359b5be08cac48b64ec5	24	 < 0.1%
# 
# 6d58638e32674bebee793a47ac4cbadc	24	 < 0.1%
# 
# 68986e4324f6a21481df4e6e89abcf01	24	 < 0.1%
# 
# c6492b842ac190db807c15aff21a7dd6	24	 < 0.1%
# 
# 465c2e1bee4561cb39e0db8c5993aafc	24	 < 0.1%
# 
# 285c2e15bebd4ac83635ccc563dc71f4

# ## <span style='color:gray '> Product ID </span> 

# In[361]:


final_data['product_id'].describe() # most purchased product


# In[362]:


final_data.groupby('product_id')['product_id'].count().sort_values(ascending=False)


# In[368]:


len(final_data[final_data['product_id']=='ebf9bc6cd600eadd681384e3116fda85']) 
# bu ürün 44 defa satılmış, 42 tanesi aynı müşteri tarafından alınmış. Diğer iki ürün 12.  ayda alınmış


# In[369]:


final_data[final_data['product_id']=='ebf9bc6cd600eadd681384e3116fda85']['customer_unique_id'].unique()


# In[370]:


final_data[final_data['product_id']=='ebf9bc6cd600eadd681384e3116fda85']['customer_unique_id'].value_counts()


# An opportunistic customer may be, this product was sold 44 times, 42 of which were bought by the same customer. It is important to identify these customers as it prevents new customers from benefiting from the campaign.

# In[371]:


final_data[final_data['product_id']=='ebf9bc6cd600eadd681384e3116fda85'][['product_category','price']]


# In[364]:


len(final_data[final_data['product_id']=='5ddab10d5e0a23acb99acf56b62b3276']) # burdaki ürün de aynı müşteri tarafından alınmış tamamı


# In[365]:


len(final_data[final_data['product_id']=='4eb99b5f0d7e411f246a5c9c0ae27a5e']) # burda da 12 tanesi aynı müşteri tarfından alınmış.


# In[367]:


len(final_data[final_data['product_id']=='0449db5eede617c5fd413071d582f038'])


# In[366]:


len(final_data[final_data['product_id']=='5dddb31154cbd968caa4706ef0f4e0f0']) # 1 tanesi başka müşteri tarafından alınmış.


# ## <span style='color:gray '> Price or Payment Values </span> 

# In[372]:


final_data['price'].describe()


# In[375]:


plt.figure(figsize=(10,5))
sns.boxplot(x=final_data['price'])


# In[376]:


plt.figure(figsize=(10,5))
sns.violinplot(x=final_data['price'])


# In[377]:


plt.figure(figsize=(10,5))
sns.histplot(data=final_data, x='price',binwidth=10)


# In[379]:


plt.figure(figsize=(10,5))
sns.distplot(final_data['price'], bins=100)


# In[419]:


data['payment_value'].describe()


# In[381]:


plt.figure(figsize=(10,5))
sns.boxplot(x=final_data['payment_value'])


# In[382]:


plt.figure(figsize=(10,5))
sns.violinplot(x=final_data['payment_value'])


# In[384]:


plt.figure(figsize=(10,5))
sns.boxplot(x="purchased_Day", y="price", data=final_data)
sns.set(rc = {'figure.figsize':(20,8)})


# In[387]:


sns.lineplot(data=final_data, x="purchased_Month", y="price")
sns.set(rc={'figure.figsize':(10.7,8.27)})


# ## <span style='color:gray '>Freight Values </span> 

# In[390]:


final_data['freight_value'].describe()


# In[391]:


final_data['freight_value'].sort_values() # some of the order's freight values are 0


# In[392]:


plt.figure(figsize=(10,5))
sns.boxplot(x=final_data['freight_value'])


# In[ ]:


#sns.boxplot(x=data['freight_value'])
#sns.set(rc = {'figure.figsize':(20,8)})


# In[394]:


plt.figure(figsize=(10,5))
sns.regplot(x="payment_value", y="freight_value", data=final_data, color='green')


# In[441]:


sns.regplot(x="price", y="freight_value", data=final_data, color='green')


# In[395]:


plt.figure(figsize=(10,5))
sns.scatterplot(data=final_data, x="price", y="freight_value")


# In[396]:


# I looked for each category to see prices and freight values
sns.relplot(
    data=final_data, x="price", y="freight_value",
    col="product_category",
    kind="scatter"
)


# In[501]:


plt.figure(figsize=(12,6))
sns.violinplot(x=final_data['freight_value']) # distribution of the freight value in the data


# In[397]:


plt.figure(figsize=(12,6))
sns.boxplot(x= final_data['price']) 


# In[398]:


final_data[final_data['price']>2000]['product_category'].nunique()


# In[399]:


final_data[final_data['price']>2000]['product_category'].unique()


# In[400]:


final_data[final_data['freight_value']>200]['product_category'].nunique()


# In[401]:


final_data[final_data['freight_value']>200]['product_category'].unique()


# In[402]:


sns.boxplot(x="purchased_Year", y="freight_value", data=final_data)
sns.set(rc = {'figure.figsize':(20,8)})


# In[403]:


sns.boxplot(x="purchased_Day", y="freight_value", data=final_data)
sns.set(rc = {'figure.figsize':(20,8)})


# In[404]:


with sns.axes_style('white'):
    sns.jointplot("price", "freight_value", data=final_data)


# In[405]:


sns.jointplot("price", "freight_value", data=final_data, kind='reg');


# In[435]:


len(final_data[(final_data['price'] + data['freight_value'])==data['payment_value']])


# In[406]:


len(final_data[(final_data['price'] + data['freight_value'])!=data['payment_value']])


# ## <span style='color:gray '> Outlier Detection </span> 

# In[407]:


# Method 1 — Standard Deviation:

def finding_anomalies(data):
  # I've created empty list to gather anomalies
  anomalies = []

  data_std = np.std(data['price'])  # price std values
  data_mean = np.mean(data['price']) # price mean values
  anomaly_cut_off = data_std * 3

  # Here above the 3 times std plus mean values are outliers
  upper_limit = data_mean + anomaly_cut_off
  # Similarly below the mean minus 3 times std values are also outliers.
  lower_limit  = data_mean - anomaly_cut_off 

  print('Lower limit is:',lower_limit)
  print('Upper limit is:',upper_limit)

      # Generate outliers
  for outlier in data['price']:
      if ((outlier > upper_limit) or (outlier < lower_limit)):
          anomalies.append(outlier)
  return anomalies


# In[408]:


finding_anomalies(final_data)


# ## <span style='color:gray '> Special Day Analysis </span> 

# In[409]:


november = final_data[final_data['Year_Month']=='201711']
november.groupby('purchased_Day')['price'].sum()


# In[410]:


november.groupby('purchased_Day')['price'].sum().plot(kind='bar')
plt.title('Total price by days in November 2017')


# In[411]:


november = final_data[final_data['Year_Month']=='201711']
november.groupby('purchased_Day')['order_id'].count()


# In[412]:


november.groupby('purchased_Day')['order_id'].count().plot(kind='bar', color='orange')
plt.title('Total sales by days in November 2017')


# In[413]:


november.groupby('Day')['order_id'].count().plot(kind='bar', color='green')
plt.title('Novermber Sales by days')


# In[458]:


november['product_category'].nunique()


# In[464]:


november.groupby('product_category')['price'].sum().sort_values(ascending=False)


# In[463]:


november.groupby('product_category')['order_id'].count().sort_values(ascending=False)


# In[466]:


#data[(data['Year_Month']=='201712')]


# In[415]:


feb2017 = final_data[(final_data['purchased_Year']==2017) & (final_data['purchased_Month']=='February')]
feb2017.groupby('Day')['order_id'].count().plot(kind='bar')


# In[416]:


plt.figure(figsize=(12,5))
feb2018 = final_data[(final_data['purchased_Year']==2018) & (data['purchased_Month']=='February')]
feb2018.groupby('Day')['order_id'].count().plot(kind='bar', color='red')
plt.title('Orders in February 2018')


# In[417]:


plt.figure(figsize=(12,5))
rio_feb_2018 = final_data[(final_data['purchased_Year']==2018) & (final_data['seller_state']=='RJ') & (data['purchased_Month']=='February')]
rio_feb_2018.groupby('Day')['order_id'].count().plot(kind='bar', color='red')
plt.title('Orders in February 2018')


# In[418]:


plt.figure(figsize=(12,5))
june2018 = final_data[(final_data['purchased_Year']==2018) & (data['purchased_Month']=='June')]
june2018.groupby('Day')['order_id'].count().plot(kind='bar', color='green')
plt.title('Orders in June 2018')
plt.ylabel('Order Counts')


# In[419]:


plt.figure(figsize=(12,5))
may2018 = final_data[(final_data['purchased_Year']==2018) &  (data['purchased_Month']=='May')]
may2018.groupby('Day')['order_id'].count().plot(kind='bar', color='red')
plt.title('Orders in May 2018') # to see mother's day difference


# In[420]:


plt.figure(figsize=(12,5))
august2018 = final_data[(final_data['purchased_Year']==2018) &  (data['purchased_Month']=='August')]
august2018.groupby('Day')['order_id'].count().plot(kind='bar', color='orange')
plt.title('Orders in May 2018') # to see father's day difference


# In[421]:


plt.figure(figsize=(10,7))
jan2018 = final_data[(final_data['Year_Month']=='201801') & (data['purchased_Month']=='January')]
jan2018.groupby('Day')['order_id'].count().plot(kind='bar')


# In[423]:


# Let' s look at the december 2017 
december2017 = final_data[(final_data['Year_Month']=='201712') & (data['purchased_Month']=='December')]
december2017.groupby('Day')['order_id'].count().plot(kind='bar')


# In[424]:


final_data[(final_data['Year_Month']=='201712') & (final_data['purchased_Month']=='December') & (final_data['Day']==4)]['product_category'].unique()


# In[426]:


final_data[(final_data['purchased_Year']==2017) & (final_data['purchased_Month']=='February') & (final_data['Day']==14)]['product_category'].nunique()


# In[428]:


final_data[(final_data['Year_Month']=='201702') & (data['Day']>20)] # shrove tuesday


# In[429]:


# I didn't find any insights
plt.figure(figsize=(10,6))
april2017 = final_data[(final_data['Year_Month']=='201704')] # Good Friday	Fri, Apr 14, 2017
april2017.groupby('Day')['order_id'].count().plot(kind='bar',color='black')


# In[430]:


# I didnt't get any insights from the 17-10-2017
plt.figure(figsize=(10,6))
october2018 = final_data[(final_data['Year_Month']=='201710')] # teacher's day --> october 15
october2018.groupby('Day')['order_id'].count().plot(kind='bar',color='black')


# In[431]:


# I looked for Brazilian's valentine day,but didn'find any insight in 2017
plt.figure(figsize=(10,6))
june2017 = final_data[(final_data['Year_Month']=='201706')] #
june2017.groupby('Day')['order_id'].count().plot(kind='bar',color='black')


# In[433]:


february = final_data[final_data['Year_Month']=='201702']
february.groupby('product_category')['price'].sum().sort_values(ascending=False)


# In[434]:


february.groupby('product_category')['order_id'].count().sort_values(ascending=False)


# In[435]:


garden_tool =final_data[((final_data['product_category']=='garden_tools')|(final_data['product_category']=='costruction_tools_garden')) & (final_data['purchased_Year']==2017)]
garden_tool.groupby('SeasonName')['order_id'].count().sort_values(ascending=False)


# In[436]:


garden_tool.groupby('SeasonName')['order_id'].count().sort_values(ascending=False).plot(kind='bar', color='slateblue')
plt.title('Garden Tools Sales by Season')


# In[437]:


health_beauty =final_data[(final_data['product_category']=='health_beauty') & (final_data['purchased_Year']==2017)]
health_beauty.groupby('SeasonName')['order_id'].count().sort_values(ascending=False)


# In[438]:


sports_leisure =final_data[(final_data['product_category']=='sports_leisure') & (final_data['purchased_Year']==2017)]
sports_leisure.groupby('SeasonName')['order_id'].count().sort_values(ascending=False)


# In[439]:


toys =final_data[(final_data['product_category']=='toys') & (final_data['purchased_Year']==2017)]
toys.groupby('SeasonName')['order_id'].count().sort_values(ascending=False)


# In[440]:


watches_gifts =final_data[(final_data['product_category']=='watches_gifts') & (final_data['purchased_Year']==2017)]
watches_gifts.groupby('SeasonName')['order_id'].count().sort_values(ascending=False)


# In[441]:


watches_gifts.groupby('SeasonName')['order_id'].count().sort_values(ascending=False).plot(kind='bar', color='darkblue')
plt.title(' Watch Sales by Season')


# In[442]:


perfumery =final_data[(final_data['product_category']=='perfumery') & (final_data['purchased_Year']==2017)]
perfumery.groupby('SeasonName')['order_id'].count().sort_values(ascending=False)


# In[443]:


perfumery.groupby('SeasonName')['order_id'].count().sort_values(ascending=False).plot(kind='bar',color='lightblue')
plt.title('Perfume sales by season')


# In[444]:


day_periods = final_data.groupby('day_periods').size().reset_index().sort_values(by=0,ascending=False)
day_periods["day_periods"] = day_periods["day_periods"].str.capitalize()
day_periods


# In[445]:



Pivot= pd.pivot_table(final_data[final_data['purchased_Year']==2017], values='order_id', index=['product_category'],
                    columns=['SeasonName'], aggfunc='count', fill_value=0)
Pivot


# In[446]:


garden_tools =final_data[((final_data['product_category']=='garden_tools')| (final_data['product_category']=='costruction_tools_garden')) & (final_data['purchased_Year']==2018)]
garden_tools.groupby('purchased_Month')['order_id'].count().sort_values(ascending=False)


# In[447]:


garden_tools =final_data[((final_data['product_category']=='garden_tools')| (final_data['product_category']=='costruction_tools_garden')) & (final_data['purchased_Year']==2017)]
garden_tools.groupby('purchased_Month')['order_id'].count().sort_values(ascending=False)


# ## <span style='color:gray '> Categories and Their Average Delivery Time </span> 

# In[448]:


(final_data['shipping_limit_date']- final_data['order_purchase_timestamp']).describe()


# In[449]:


bed_bath_table = final_data[final_data["product_category"]=='bed_bath_table']
(bed_bath_table['order_estimated_delivery_date'] - bed_bath_table['order_delivered_customer_date']).describe()


# In[450]:


health_beauty = final_data[final_data["product_category"]=='health_beauty']
(health_beauty['order_estimated_delivery_date'] - health_beauty['order_delivered_customer_date']).describe()


# In[451]:


sports_leisure = final_data[final_data["product_category"]=='sports_leisure']
(sports_leisure['order_estimated_delivery_date'] - sports_leisure['order_delivered_customer_date']).describe()


# In[452]:


furniture_decor = final_data[final_data["product_category"]=='furniture_decor']
(furniture_decor['order_estimated_delivery_date'] - furniture_decor['order_delivered_customer_date']).describe()


# In[453]:


computers_accessories = final_data[final_data["product_category"]=='computers_accessories']
(computers_accessories['order_estimated_delivery_date'] - computers_accessories['order_delivered_customer_date']).describe()


# In[454]:


housewares = data[data["product_category"]=='housewares']
(housewares['order_estimated_delivery_date'] - housewares['order_delivered_customer_date']).describe()


# In[455]:


#data.groupby(['product_category','payment_sequential'])['product_category'].count()


# ## <font color='gray'> Libraries Like Sweetviz, pandas profiling </font> 

# In[37]:


pip install sweetviz 


# In[457]:


import sweetviz as sv
analyze_report = sv.analyze(final_data)
analyze_report.show_html('analyze.html', open_browser=False) 


# In[49]:


import  IPython
IPython.display.HTML('analyze.html')


# In[50]:


from autoviz.AutoViz_Class import AutoViz_Class 

AV = AutoViz_Class() # object cretaed


# In[204]:


pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip 


# In[205]:


pip install pandas-profiling[notebook,html]


# In[458]:


import pandas_profiling as pp
from pandas_profiling import ProfileReport


# In[50]:


pp.ProfileReport(data)


# ## <span style='color:gray '>Some Visualizations </span> 

# In[160]:



sellers_x_date['Year_Month'] = sellers_x_date['order_purchase_timestamp'].dt.strftime('%Y%m')
#clients_x_date['Year_Month'].sort_values() # values from september 2016 to september 2018


# In[459]:


final_data['customer_unique_id'].describe()


# In[460]:


final_data[final_data['purchased_Year']==2017]['customer_unique_id'].describe()


# In[461]:


final_data[final_data['purchased_Year']==2018]['customer_unique_id'].describe()


# ## <font color='gray'> Visualizations </font> 

# In[57]:


sns.pairplot(data)


# In[462]:


# Correlations 
corr = final_data.corr()
corr


# In[479]:


# here it is seen that payment value and prices are correlated, I used price values in my visualizations
# I dropped the payment value
sns.heatmap(corr, annot=True)


# In[398]:


#Category Performance


# In[463]:


customer_state = final_data.groupby(['customer_state'])['price','order_id','customer_unique_id'].agg({'price':'sum','order_id':'nunique','customer_unique_id':'nunique'}).reset_index()

category = final_data.groupby(['product_category'])['price','order_id','customer_unique_id'].agg({'price':'sum','order_id':'nunique','customer_unique_id':'nunique'}).reset_index()

yearmonth = final_data.groupby(['Year_Month'])['price','order_id','customer_unique_id'].agg({'price':'sum','order_id':'nunique','customer_unique_id':'nunique'}).reset_index()
seller_state = final_data.groupby(['seller_state'])['price','order_id','customer_unique_id'].agg({'price':'sum','order_id':'nunique','customer_unique_id':'nunique'}).reset_index()


# In[464]:


customer_state.head()


# In[465]:


seller_state.head()


# In[466]:


yearmonth


# In[467]:


category.head()


# In[468]:


category['contribution'] = category['price']/category['price'].sum()
# above I created category data, in the data price column is available
# if I divide price column to total price, we will see for price contribution for each product category
category.head()


# In[469]:


category['category'] = category.apply(lambda x: x['product_category'] if x['contribution']>=0.0350 else 'Many_kinds_of_category',axis=1)

category_contribution = category.groupby('category')['price','order_id','customer_unique_id'].sum().reset_index()


# In[470]:


category_contribution


# In[471]:


sns.set_style()
fig, (ax1,ax2,ax3) = plt.subplots(3,figsize=(20,20))
fig.suptitle('Category Performance ')
x = category_contribution['category']
ax1.pie(category_contribution['price'],labels=x,autopct='%1.2f%%')
ax1.set_title('Price contribution for each category', size=18, color='darkblue')
ax2.pie(category_contribution['order_id'],labels=x,autopct='%1.2f%%')
ax2.set_title('Order Count for each Category', size=18, color='darkblue')
ax3.pie(category_contribution['customer_unique_id'],labels=x,autopct='%1.2f%%')
ax3.set_title('Customer count for each category', size=18, color='darkblue')


# In[394]:


product = final_data.groupby('product_category')['price'].sum().sort_values(ascending=False)[:10]
product


# In[493]:


palette = sns.color_palette("mako_r", 6)
seller_freight=pd.crosstab(index=final_data['customer_state'], columns=final_data['seller_state'], values=final_data['freight_value'],  aggfunc=np.mean).round(0)
sns.heatmap(seller_freight, cmap=palette, annot=True, fmt='g');
plt.figure(figsize=(25,10))


# In[472]:


final_data.drop(['payment_value'], axis = 1, inplace = True)


# In[495]:


#sns.catplot(x="day_periods", y="price",
 #                data=final_data)               


# In[496]:


#telephony = final_data[final_data['product_category']=='telephony']
#sns.relplot(
#    data=telephony, x="price", y="freight_value",
#    col="purchased_Month", hue='purchased_Day',
#    kind="scatter"
#)


# In[497]:


#sns.relplot(
#    data=telephony, x="price", y="freight_value",
#    col="product_category", hue='purchased_Year',
#    kind="scatter"
#)


# In[498]:


#sns.relplot(
#    data=final_data, x="price", y="freight_value",
#    col="product_category", hue='purchased_Year',
#    kind="scatter"
#)


# ##### <font color='gray'> REFERENCES </font> 
# * https://www.kaggle.com/olistbr/brazilian-ecommerce
# * https://www.kaggle.com/luisenrique18/olist-ecommerce-2017-2018-eda - I used two graphs from here

# # <font color='gray'> Thank you </font> 
# ### <font color='gray'> Gözde Gözütok </font> 
