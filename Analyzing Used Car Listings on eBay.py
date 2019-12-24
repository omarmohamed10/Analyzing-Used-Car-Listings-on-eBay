#!/usr/bin/env python
# coding: utf-8

# 
# # Analyzing Used Car Listings on eBay Kleinanzeigen
# 
# We will be working on a dataset of used cars from eBay Kleinanzeigen, a classifieds section of the German eBay website.
# 
# The dataset was originally scraped and uploaded to Kaggle. The version of the dataset we are working with is a sample of 50,000 data points that was prepared by Dataquest including simulating a less-cleaned version of the data.
# 
# The data dictionary provided with data is as follows:
# 
# - dateCrawled - When this ad was first crawled. All field-values are taken from this date.
# - name - Name of the car.
# - seller - Whether the seller is private or a dealer.
# - offerType - The type of listing
# - price - The price on the ad to sell the car.
# - abtest - Whether the listing is included in an A/B test.
# - vehicleType - The vehicle Type.
# - yearOfRegistration - The year in which year the car was -first registered.
# - gearbox - The transmission type.
# - powerPS - The power of the car in PS.
# - model - The car model name.
# - kilometer - How many kilometers the car has driven.
# - monthOfRegistration - The month in which year the car was first registered.
# - fuelType - What type of fuel the car uses.
# - brand - The brand of the car.
# - notRepairedDamage - If the car has a damage which is not yet repaired.
# - dateCreated - The date on which the eBay listing was created.
# - nrOfPictures - The number of pictures in the ad.
# - postalCode - The postal code for the location of the vehicle.
# - lastSeenOnline - When the crawler saw this ad last online.
#  
#  The aim of this project is to clean the data and analyze the included used car listings.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


autos = pd.read_csv('autos.csv',encoding='Latin-1')


# In[3]:


autos.head()


# In[4]:


autos.info()


# 
# Our dataset contains 20 columns, most of which are stored as strings. There are a few columns with null values, but no columns have more than ~20% null values. There are some columns that contain dates stored as strings.
# 
# We'll start by cleaning the column names to make the data easier to work with.

# ## Clean Columns

# In[5]:


autos.columns


# 
# We'll make a few changes here:
# 
# - Change the columns from camelcase to snakecase.
# - Change a few wordings to more accurately describe the columns.

# In[6]:


autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'ab_test',
       'vehicle_type', 'registration_year', 'gearbox', 'power_ps', 'model',
       'odometer', 'registration_month', 'fuel_type', 'brand',
       'unrepaired_damage', 'ad_created', 'num_photos', 'postal_code',
       'last_seen']

autos.head()


# ## Initial Data Exploration and Cleaning

# In[7]:


autos.describe(include='all')


# 
# Our initial observations:
# 
#  There are a number of text columns where all (or nearly all) of the values are the same:
# - **seller**
# - **offer_type**
# 
# The **num_photos** column looks odd, we'll need to investigate this further contain **0** value for all rows
# 
# so, will drop this columns

# In[8]:


autos.drop(['seller','offer_type','num_photos'],axis=1,inplace=True)


# and convert **price** and **odometer** columns to numeric values

# In[9]:


autos['price'] = autos['price'].str.replace('$','').str.replace(',','')
autos['odometer'] = autos['odometer'].str.replace('km','').str.replace(',','')
autos['price'] = autos['price'].astype(int)
autos['odometer'] = autos['odometer'].astype(int)


# In[10]:


autos[['price','odometer']].head()


# convert **odometer** to **odometer_km** to illustrate it

# In[11]:


autos.rename({'odometer':'odometer_km'},axis = 1 , inplace=True)


# In[12]:


autos[['price','odometer_km']].head()


# ## Exploring Odometer and Price

# In[13]:


print(autos['price'].unique().shape)
print(autos['price'].describe())


# In[14]:


print(autos['price'].value_counts().sort_index(ascending=False).head(25))


# In[15]:


print(autos['price'].value_counts().sort_index(ascending=True).head(25))


# Given that eBay is an auction site, there could legitimately be items where the opening bid is \$1. We will keep the \$1 items, but remove anything above \$350,000, since it seems that prices increase steadily to that number and then jump up to less realistic numbers.

# In[16]:


autos = autos[autos['price'].between(1,351000)]
autos['price'].describe()


# In[17]:


print(autos['odometer_km'].unique().shape)
print(autos['odometer_km'].shape)


# In[18]:


print(autos['odometer_km'].describe())


# ## Exploring the date columns

# 
# There are a number of columns with date information:
# 
# - date_crawled
# - registration_month
# - registration_year
# - ad_created
# - last_seen
# 
# These are a combination of dates that were crawled, and dates with meta-information from the crawler. The non-registration dates are stored as strings.
# 
# We'll explore each of these columns to learn more about the listings.

# In[19]:


autos[['date_crawled','ad_created','last_seen']][:5]


# In[27]:


autos['date_crawled'].str[:10].value_counts(normalize=True,dropna=False).sort_index()


# In[32]:


(autos['date_crawled']
.str[:10]
.value_counts(normalize=True,dropna=False)
.sort_values())


# Looks like the site was crawled daily over roughly a one month period in March and April 2016. The distribution of listings crawled on each day is roughly uniform.

# In[31]:


(autos["last_seen"].str[:10]
.value_counts(normalize=True, dropna=False)
.sort_index())


# The last three days contain a disproportionate amount of 'last seen' values. Given that these are 6-10x the values from the previous days, it's unlikely that there was a massive spike in sales, and more likely that these values are to do with the crawling period ending and don't indicate car sales

# In[33]:


(autos["ad_created"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        )


# There is a large variety of ad created dates. Most fall within 1-2 months of the listing date, but a few are quite old, with the oldest at around 9 months.

# In[35]:


print(autos["registration_year"].head())
autos["registration_year"].describe()


# 
# The year that the car was first registered will likely indicate the age of the car. Looking at this column, we note some odd values. The minimum value is 1000, long before cars were invented and the maximum is 9999, many years into the future.

# ## Dealing with Incorrect Registration Year Data

# One thing that stands out from the exploration we did in the last screen is that the registration_year column contains some odd values:
# 
# - The minimum value is 1000, before cars were invented
# - The maximum value is 9999, many years into the future
# 
# Because a car can't be first registered after the listing was seen, any vehicle with a registration year above 2016 is definitely inaccurate. Determining the earliest valid year is more difficult. Realistically, it could be somewhere in the first few decades of the 1900s.

# In[48]:


((~autos['registration_year'].between(1900,2016))
.sum()/autos.shape[0]*100)


# Given that this is less than 4% of our data, we will remove these rows.

# In[49]:


autos = autos[autos["registration_year"].between(1900,2016)]
autos["registration_year"].value_counts(normalize=True).head(10)


# ## Exploring price by brand

# In[54]:


freq_brands = autos['brand'].value_counts().sort_values(ascending = False).head(20)
freq_brands


# German manufacturers represent four out of the top five brands, almost 50% of the overall listings. Volkswagen is by far the most popular brand, with approximately double the cars for sale of the next two brands combined.

# In[57]:


common_brands = freq_brands[:5].index
common_brands


# In[58]:


brand_mean_price = {}
for brand in common_brands:
    brand_only = autos[autos['brand'] == brand]
    mean_price = brand_only['price'].mean()
    brand_mean_price[brand] = mean_price

brand_mean_price


# ## Exploring Mileage

# In[60]:


bmp_series = pd.Series(brand_mean_price)
pd.DataFrame(bmp_series, columns=["mean_price"])


# In[65]:


brand_mean_mileage = {}

for brand in common_brands:
    brand_only = autos[autos["brand"] == brand]
    mean_mileage = brand_only["odometer_km"].mean()
    brand_mean_mileage[brand] = mean_mileage

mean_mileage = pd.Series(brand_mean_mileage).sort_values(ascending=False)
mean_prices = pd.Series(brand_mean_price).sort_values(ascending=False)


# In[66]:


brand_info = pd.DataFrame(mean_mileage,columns=['mean_mileage'])
brand_info


# In[67]:


brand_info["mean_price"] = mean_prices
brand_info


# The range of car mileages does not vary as much as the prices do by brand, instead all falling within 10% for the top brands. There is a slight trend to the more expensive vehicles having higher mileage, with the less expensive vehicles having lower mileage.

# In[ ]:




