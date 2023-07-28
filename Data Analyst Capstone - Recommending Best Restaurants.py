#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as ply
import statistics as stc


# In[2]:


data = pd.read_excel("Restaurant_data.xlsx")


# In[3]:


data.head(3)


# In[4]:


data.tail(2)


# In[5]:


cc = pd.read_excel("Country-Code.xlsx")


# In[6]:


cc.head(2)


# In[7]:


## Merging the data on basis of thecountry code in the main data :-


# In[8]:


merged = pd.merge(data,cc,on = 'Country Code', how='left')
merged.head()


# In[9]:


merged.info()


# In[10]:


merged.isna().sum()


# In[11]:


## Droping the values :-


# In[12]:


merged.dropna(axis=0,subset=['Restaurant Name'],inplace=True)


# In[13]:


merged[merged['Cuisines'].isnull()]                           ## Providing the cusines value to the null values as shown.


# In[14]:


merged.isna().sum()


# In[15]:


## from above we are getting the 9 null values of cuisines so we hae to replace with others :-
merged['Cuisines'].fillna('Others',inplace = True)


# In[16]:


merged.isna().sum()


# In[17]:


merged.duplicated()


# In[18]:


##  For Duplicate Data Findings :-

duplicateRowsDF = merged[merged.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)


# ## EDA-1
# #### Explore the geographical distribution of the restaurants
# #### Finding out the cities with maximum / minimum number of restaurants

# In[19]:


merged.columns


# In[20]:


country_distribution = merged.groupby(['Country Code','Country']).agg( Count = ('Restaurant ID','count'))
country_distribution.sort_values(by = 'Count', ascending=False)


# ### Seeing the barplot graph :-

# In[21]:


country_distribution.plot(kind = 'barh')


# In[22]:


merged.columns


# In[23]:


city_dist = merged.groupby(['Country','City']).agg(Count = ('Restaurant ID','count'))
city_dist.sort_values(by = "Count", ascending = True)


# In[24]:


country_distribution.describe()


# In[25]:


min_cnt_rest = city_dist[city_dist['Count']==1]
min_cnt_rest.info()
min_cnt_rest


# In[26]:


merged.columns


# In[27]:


merged.head(2)


# ####  Converting the dummy values to 0 and 1 as shown :-

# In[28]:


merged1 = merged.copy()
merged1.head(2)


# In[29]:


merged1.columns


# In[30]:


dummy = ['Has Table booking','Has Online delivery']
merged1 = pd.get_dummies(merged1,columns = dummy, drop_first = True)
merged1.head(3)


# In[31]:


#Ratios between restaurants allowing table booking and those which dont
tbl_book_y = merged1[merged1['Has Table booking_Yes']==1]['Restaurant ID'].count()
tbl_book_n = merged1[merged1['Has Table booking_Yes']==0]['Restaurant ID'].count()
print('Ratio between restaurants that allow table booking vs. those that do not allow table booking: ',
      round((tbl_book_y/tbl_book_n),2))


# ### Pie chart to show percentage of restaurants which allow table booking and those which don't

# In[32]:


import matplotlib.pyplot as plt
labels = 'Table Booking', 'No Table Booking'
sizes = [tbl_book_y,tbl_book_n]
explode = (0.3, 0)                                                # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(20,5))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=180)
ax1.set_title("Table Booking vs No Table Booking")
ax1.axis('equal')                                                 # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[33]:


merged1.columns


# ### Find out the percentage of restaurants providing online delivery:--

# In[34]:


order_on = merged1[merged1['Has Online delivery_Yes'] == 1]['Restaurant ID'].count()
order_off = merged1[merged1['Has Online delivery_Yes'] == 0]['Restaurant ID'].count()
print('Percentage of restaurants providing online delivery : {} %'.format((round(order_on/len(merged1),3)*100)))
  


# In[35]:


labels = 'Online Delivery','No Online Delivery'
size = [order_on,order_off]
explode = (0.5,0)
fig1,ax1 = plt.subplots(figsize = (20,5))
ax1.pie(size, explode = explode, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 360)
ax1.set_title("Online Delivery  VS No Online Delivery")
ax1.axis('equal')
plt.show()


# In[36]:


### Seeing the Box plot as follows :-


# In[37]:


merged.columns


# In[38]:


sns.boxplot(merged['Price range'])


# In[39]:


sns.boxplot(merged['Votes'])


# In[40]:


merged.head(2)


# In[41]:


# Calculate the difference in number of votes for the restaurants that deliver and the restaurants that do not deliver
# first detect and remove (replace it with mean/closest possible value) outlier for VOTE
import sklearn
import pandas as pd


# ### Detection of the Outliers as shown:-

# In[42]:


Q1 = np.percentile(merged['Votes'],25, interpolation = 'midpoint')
Q3 = np.percentile(merged['Votes'],75, interpolation = 'midpoint')

IQR = Q3 - Q1
print("Old shape:", merged.shape)


# In[43]:


## Then finding the new shapes of the data :-
upper = np.where(merged['Votes'] >= (Q3+1.5 *IQR))
lower = np.where(merged['Votes'] >=(Q3 - 1.5 * IQR))


# In[46]:


## Removing the Outliers :-

merged.drop(upper[0], inplace = True)
merged.drop(lower[0], inplace = True)
print("New shape :", merged.shape)


# In[47]:


merged.columns


# In[48]:


import seaborn as sns
sns.boxplot(merged['Average Cost for two'])


# In[ ]:


import sklearn
import pandas as pd

''' Detection '''
# IQR
Q1 = np.percentile(merged['Average_Cost_for_two'], 25,
                   interpolation = 'midpoint')
 
Q3 = np.percentile(merged['Average_Cost_for_two'], 75,
                   interpolation = 'midpoint')
IQR = Q3 - Q1
print("Old Shape: ", merged.shape)
 
# Upper bound
upper = np.where(merged['Average_Cost_for_two'] >= (Q3+1.5*IQR))
print("Upper bound:",upper)
print(np.where(upper))
# Lower bound
lower = np.where(merged['Average_Cost_for_two'] <= (Q1-1.5*IQR))
 
''' Removing the Outliers '''
merged.drop(upper[0], inplace = True)
merged.drop(lower[0], inplace = True)
 
print("New Shape: ", merged.shape)


# In[ ]:


merged.columns


# In[49]:


merged.hist(['Votes', 'Average Cost for two'], figsize=(15,5))          ##Aafter removing the outliers :-


# In[50]:


dimen=(18,5)
fig, ax = plt.subplots(figsize=dimen)
sns.boxplot(x='Votes',y='Has Online delivery',data=merged,ax=ax)


# In[ ]:


merged.head(2)


# ### To see the votes distribution in the online and offline delivery system :-

# In[51]:


rest_deliever = merged1[merged1['Has Table booking_Yes'] ==1 ]['Votes'].sum()
rest_ndeliever = merged[merged1['Has Table booking_Yes'] == 0]['Votes'].sum()
print("Difference in the votes distribution :", abs((rest_deliever - rest_ndeliever)))


# In[52]:


labels = 'Online Delivery','No Online Delivery'
explode = (0.2,0)
size = [rest_deliever,rest_ndeliever]
dimen = (15,4)
fig,ax1 = plt.subplots(figsize = dimen)
ax1.pie(size,explode=explode,labels=labels,autopct='%1.1f%%',shadow = True, startangle = 180)
ax1.set_title("Votes Distribution for Online Delivery  VS No Online Delivery")
plt.show()


# In[53]:


merged.head(2)


# In[54]:


merged.columns


# ### What are the top 10 cuisines served across cities?

# In[55]:


# What are the top 10 cuisines served across cities?
top_10_couisines = merged.groupby(['City','Cuisines']).agg( Count = ('Cuisines','count'))
df=top_10_couisines.sort_values(by='Count',ascending=False)
#top_10_couisines = merged['Cuisines'].value_counts()
#top_10_couisines.head(10)
#top_10_couisines.plot(kind='barh')
df.head(10).plot(kind='bar')


# In[56]:


merged.columns


# #### What is the maximum and minimum number of cuisines that a restaurant serves? 

# In[57]:


cuis_count = merged.groupby([ 'Restaurant Name', 'Cuisines']).agg(Count = ('Cuisines','count'))
cuis_count.sort_values(by = 'Count', ascending = False)


# In[118]:


cuis_count_ct = merged.groupby(['City','Cuisines']).agg( Count = ('Cuisines','count'))
cuis_count_ct.sort_values(by='Count',ascending=False)


# In[142]:


merged.head(2)


# In[141]:


## creating or spliting the cuisines columns even if containing NAN values :-
cuisines = merged['Cuisines'].apply(lambda x: pd.Series(x.split(',')))
cuisines


# In[143]:


## Giving those cuisines column with the names :-
cuisines.columns = ['Cuisine_1','Cuisine_2','Cuisine_3','Cuisine_4','Cuisine_5','Cuisine_6','Cuisine_7','Cuisine_8']
cuisines.tail()


# In[122]:


cuisines.head(2)


# In[144]:


## Merging the cuisines :-

df_cuisines = pd.concat([merged, cuisines], axis = 1)
df_cuisines.head(2)


# ### Or creating the dummies would be like this :-

# In[ ]:





# In[ ]:





# In[145]:


merged.columns


# In[146]:


## Creating the dataframe having desired columns :-

cuisine_loc = pd.DataFrame(df_cuisines[['Country','City','Locality Verbose','Cuisine_1','Cuisine_2','Cuisine_3',
                                        'Cuisine_4','Cuisine_5','Cuisine_6','Cuisine_7','Cuisine_8']])


# In[147]:


## making the columns all together as in same frame :-

cuisine_loc_stack = pd.DataFrame(cuisine_loc.stack())
cuisine_loc.head(2)


# In[148]:


cuisine_loc_stack.head(20)


# In[128]:


merged.head(2)


# In[149]:


keys = [c for c in cuisine_loc  if c.startswith('Cuisine')]
a=pd.melt(cuisine_loc, id_vars='Locality Verbose', value_vars=keys, value_name='Cuisines') 
#melting the stack into one row
a


# In[130]:


max_rate = pd.DataFrame(a.groupby(by = ['Locality Verbose','variable','Cuisines']).size().reset_index())
max_rate
del max_rate['variable']


# In[153]:


## Giving the count to the cuisines for different Locaality as well :-
max_rate.columns = ['Locality Verbose','Cuisines',"Count"]
max_rate


# In[154]:


#find the highest restuarant in the city:-

loc = max_rate.sort_values("Count", ascending = True).groupby(by = ['Locality Verbose'],as_index = False).first()
loc


# In[156]:


rating_res=loc.merge(merged1,left_on='Locality Verbose',right_on='Locality Verbose',how='inner') 
                                                                                          #inner join to merge the two dataframe
rating_res.head(4)


# In[157]:


merged.head(2)


# In[159]:


## Creating the new rating  for the restaurants for frther process:-

df=pd.DataFrame(rating_res[['Country','City','Locality Verbose','Cuisines_x','Count']]) 
#making a dataframe of rating restaurant
df


# In[160]:


country=rating_res.sort_values('Count', ascending=False).groupby(by=['Country'],as_index=False).first()
#grouping the data by country code
country


# In[162]:


con=pd.DataFrame(country[['Country','City','Locality','Cuisines_x','Count']])
con.columns=['Country','City','Locality','Cuisines','Number of restaurants in the country']
#renaming the columns
con


# In[163]:


con.head(20)


# In[165]:


con1=con.sort_values('Number of restaurants in the country', ascending=False) 
#sorting the restaurants on the basis of the number of restaurants in the country
con1[:10]


# In[166]:


import matplotlib.pyplot as plt
plt.bar(con1['Cuisines'],con1['Number of restaurants in the country'])

plt.xlabel("Cuisines")
plt.ylabel("Number of restaurants in the country")
plt.xticks(rotation=90)

#con1.plot(kind='bar')


# In[167]:


rest_cuisine = pd.DataFrame(df_cuisines[['Restaurant Name','City','Cuisine_1','Cuisine_2','Cuisine_3','Cuisine_4',
                                         'Cuisine_5','Cuisine_6','Cuisine_7','Cuisine_8']])
rest_cuisine_stack=pd.DataFrame(rest_cuisine.stack()) #stacking the columns 
rest_cuisine.head()


# In[173]:


keys1 = [c for c in rest_cuisine  if c.startswith('Cuisine')]
b=pd.melt(rest_cuisine, id_vars='Restaurant Name', value_vars=keys, value_name='Cuisines') 
#melting the stack into one row
max_rate1=pd.DataFrame(b.groupby(by=['Restaurant Name','variable','Cuisines']).size().reset_index()) 
#find the highest restuarant in the city
max_rate1
del max_rate1['variable']
max_rate1.columns=['Restaurant Name','Cuisines','Count']
max_rate1.head(10)


# In[177]:


max_rate1 = max_rate1.apply(lambda x: x.replace('#','')) 
max_rate1 = max_rate1.apply(lambda x: x.replace('©',''))                    ##Cleaning the Restaurant values as shpwn:-
max_rate1 = max_rate1.apply(lambda x: x.replace('€±','')) 


# In[178]:


max_rate1


# In[176]:


max_rate1.sort_values('Count',ascending=False)


# In[170]:


#Cafe Coffee Day has the max number of cuisines, counts 82 and The least number of cuisines in a resaurant is 1 of Ìàukura€Ùa Sofras€±.


# In[180]:


merged1.head(3)


# In[183]:


rating = merged1[['Restaurant ID','Restaurant Name','Country','City','Aggregate rating'
                  ,'Average Cost for two','Votes','Price range','Has Table booking_Yes','Has Online delivery_Yes']]


# In[184]:


rating = rating.merge(max_rate1,left_on = 'Restaurant Name', right_on = 'Restaurant Name',how = 'left')
rating


# In[185]:


merged1.corr()


# In[186]:


merged.corr()


# In[197]:


fig, ax = plt.subplots(figsize=(20,8))
dataplot = sns.heatmap(merged1.corr(), cmap="YlGnBu", annot=True,linewidth=0.5,ax=ax)


# In[199]:


fig, ax = plt.subplots(figsize = (18,8))
dataplot = sns.heatmap(merged1.corr(), annot = True, linewidth = 0.5, ax = ax)


# #### From the above we have drastically concluded that not a single factor is responsible for the rating of the Restaurants., 
# ## There are several factors such as shown above depending upon the mind set of spectator or any preferrable objects which is 
# # Perfectly  analysed in this prediction and with uncertainty.
# 

# In[ ]:




