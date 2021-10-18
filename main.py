#!/usr/bin/env python
# coding: utf-8

# The data set includes title, text, url and human labeled category. All 3 -- title, text and url can inform us about category. But I belive the most rubust one is the *text*. The intuition behind that is:
# * Some times titles tries to be creatives and missleading, like "il pastore tedesco" that was the headline of the a left italian newspaper in April 2005, or "The importance of being erroneous" which is about mutations in virouses.
# * url is informative, and it can help to construct a nice descision tree, but it is not generalizable, it would not work when a new source is intruduced, and when there is a change in url pattern of a website. 
# 
# So here I prefere to stick to "text".
# 
# The data here are not enough to construct my own word embedding, so I take a pretrained model and use it here. I go to TensorFlow Hub and choose a model that is close to our problem, runs on my laptop (no GPU) and doesn't take ages to download, and it also works with TF 2.
# 
# That leaves me with **nnlm-en-dim50-with-normalization** which is trained on google news, very similar data set :) 

# In[1]:


import pandas as pd
df = pd.read_table('data/data_redacted.tsv')


# In[2]:


df.sample(3)


# In[3]:


df.category.value_counts()


# * I would also consider adding the element of time to the data set. For example, around US election, there will be a spike in political articles. Or perhaps there might be some syncronus behaviur among sources. 
# * 

# In[4]:


n = df.category.nunique()  # number of categories. 
# Make a sparse matrix out of categories. 
target = pd.get_dummies(df.category, columns=['category'])  # not the best way to do this :) , I am aware of that. 


# In[5]:


# Split data into train and test sets. 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.text, target, test_size=0.2)


# ## The Baseline
# The logloss of the mean field approximation:

# In[6]:


# First let see what is the performance of the mean field solution.
import numpy as np
-(np.log(y_train.mean()) * y_train).mean().sum()


# ## The model

# In[7]:


import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras

# Just an embedding that is trained on google news.
nnlm = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2",
                           input_shape=[], dtype=tf.string)

# Construct a simple neural net that output probabilities of each category for a given text. 
model = keras.Sequential()
model.add(nnlm)
model.add(keras.layers.Dense(25, activation='relu'))  # no parameter tuning for now. 
model.add(keras.layers.Dense(n, activation='sigmoid'))
model.add(keras.layers.Activation('softmax'))
#opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', metrics=keras.metrics.AUC())
model.summary()


# In[8]:


# Train the model.
model.fit(X_train, y_train, epochs=10, validation_split=0.1, shuffle=True)


# In[9]:


# Make prediction
model.predict(X_test).argmax(axis=1)


# In[10]:


# The original labels
y_test.values.argmax(axis=1)


# In[11]:


# Evaluate model on the test set. 
model.evaluate(X_test, y_test)


# In[12]:


_ = pd.DataFrame({'category': df.category.unique()}, index=range(-n,0))
_['text'] = _.category.str.replace('_', ' ')
df = df.append(_)


# In[13]:


embedding = nnlm.call(df.text).numpy()


# In[14]:


from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1)
z = tsne.fit_transform(embedding) 


# In[15]:


import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})

df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]
df["size"] = 1 + 1 * (df.index < 0)
sns.scatterplot(x="comp-1", y="comp-2", 
                hue="category",
                palette=sns.color_palette("hls", n),
                data=df, size="size", style="size")


# ## Test

# In[16]:


test = ["This dress is beutifull", '', 'iphone vs android', 'kills', 'Tesla', 'Oil price is high.', 'earth is a planet', 
        'interview', 'Google', 'footbal', 'procastination', 'depression']
cat = y_train.columns[model.predict(test).argmax(axis=1)]

list(zip(test, cat))


# In[17]:


text = '''
he turning point came just after 6:30 on a Tuesday morning.

It was the 9 December 2020 and 91-year-old Margaret Keenan and 81-year-old William Shakespeare – who delighted the world by hailing from Warwickshire, like the poet – had just become the first people to ever receive an initial dose of a Covid-19 vaccine outside clinical trials.

The entire room burst into applause. The day was named "V-day", and the atmosphere was giddy. One British newspaper celebrated with the whimsical headline "The taming of the flu" – while footage of a particularly charismatic early vaccine recipient, who was more concerned about his "rather nasty lunch" than the needle went viral on Twitter. The pandemic was far from over, but this was the first step on the way out. 

Nine months later, around 5.7 billion doses of various vaccines have now been administered worldwide – with 41.8% of the global population at least partially protected. But the list of unknowns is growing by the day.

"One ghastly thing about this pandemic is that people get cross with us [scientists]," says Danny Altmann, professor of immunology at University College London, "because we change our minds, because it's such a moving target."

Now that it's clear the world is likely to be riddled with Covid-19 – and its many variant successors – for years to come, the next big question is whether two doses of each vaccine is enough.

Altmann explains that not that long ago – in April and May – he was writing articles and giving interviews saying that most vaccinated people had immunity that was so stupendous, there was no need to worry about booster doses.

"The expression I used was 'protective headroom'," says Altmann. "That you've got lots of protective headroom and even if variants come along that drop the effectiveness of your vaccine 10 times, say, because you've got a 1,000 times excess of antibodies, it wouldn't do any harm." The strong antibody response was initially reflected in their efficacy, too – while the World Health Organization (WHO) recommended approving any over 50%, in reality it was orders of magnitude higher.

Then the Delta variant came rampaging – and though most people still have high levels of antibodies, breakthrough infections are not rare events. "We're seeing breakthrough infections in the face of quite decent levels of neutralising antibodies," he says.
'''


# In[18]:


y_train.columns[model.predict([text]).argmax(axis=1)]


# It works :)
# 
# Now we can use TensorFlow Serving to deploy ... 

# In[ ]:




