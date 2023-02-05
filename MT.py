#!/usr/bin/env python
# coding: utf-8

# In[2]:



import string
import numpy as np
import pandas as pd
s="C:/Users/HOME/Downloads/rr.txt"
lines=pd.read_table(s,names=['eng','fr'])


# In[3]:


lines=lines[0:50000]


# In[4]:


lines.eng=lines.eng.apply(lambda x:x.lower())


# In[5]:


lines.fr=lines.fr.apply(lambda x:x.lower())


# In[6]:


exclude=set(string.punctuation)


# In[7]:


lines.eng=lines.eng.apply(lambda x:"".join(cc for cc in x if cc not in exclude))


# In[8]:


lines.fr=lines.fr.apply(lambda x:"".join(cc for cc in x if cc not in exclude))


# In[9]:


lines.fr


# In[10]:


lines.fr=lines.fr.apply(lambda x:"strt "+ x +" end")


# In[11]:


lines.fr


# In[12]:


# fit a tokenizer
from keras.preprocessing.text import Tokenizer
import json
from collections import OrderedDict
def create_tokenizer(data):
     tokenizer = Tokenizer()
     tokenizer.fit_on_texts(data)
     return tokenizer


# In[13]:


eng_tokenizer = create_tokenizer(lines.eng)
output_dict = json.loads(json.dumps(eng_tokenizer.word_counts))
df =pd.DataFrame([output_dict.keys(), output_dict.values()]).T
df.columns = ['word','count']
df = df.sort_values(by='count',ascending = False)
df['cum_count']=df['count'].cumsum()
df['cum_perc'] = df['cum_count']/df['cum_count'].max()
final_eng_words = df[df['cum_perc']<0.8]['word'].values


# In[16]:


fr_tokenizer = create_tokenizer(lines.fr)
output_dict = json.loads(json.dumps(fr_tokenizer.word_counts))
df =pd.DataFrame([output_dict.keys(), output_dict.values()]).T
df.columns = ['word','count']
df = df.sort_values(by='count',ascending = False)
df['cum_count']=df['count'].cumsum()
df['cum_perc'] = df['cum_count']/df['cum_count'].max()
final_fr_words = df[df['cum_perc']<0.8]['word'].values


# In[17]:


def filter_eng_words(x):
     t = []
     x = x.split()
     for i in range(len(x)):
         if x[i] in final_eng_words:
             t.append(x[i])
         else:
             t.append('unk')
     x3 = ''
     for i in range(len(t)):
         x3 = x3+t[i]+' '
     return x3


# In[18]:


a='hi'
filter_eng_words(a)


# In[19]:


def filter_fr_words(x):
     t = []
     x = x.split()
     for i in range(len(x)):
         if x[i] in final_fr_words:
             t.append(x[i])
         else:
             t.append('unk')
     x3 = ''
     for i in range(len(t)):
         x3 = x3+t[i]+' '
     return x3


# In[20]:


lines['fr']=lines['fr'].apply(filter_fr_words)
lines['eng']=lines['eng'].apply(filter_eng_words)


# In[21]:


lines.eng


# In[22]:


all_eng_words=set()
for eng in lines.eng:
     for word in eng.split():
         if word not in all_eng_words:
             all_eng_words.add(word)

all_french_words=set()
for fr in lines.fr:
     for word in fr.split():
         if word not in all_french_words:
             all_french_words.add(word)


# In[23]:


input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_french_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_french_words)


# In[24]:


input_token_index = dict( [(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict( [(word, i+1) for i, word in enumerate(target_words)])


# In[25]:


length_list=[]
for l in lines.fr:
     length_list.append(len(l.split(' ')))
fr_max_length = np.max(length_list)


# In[26]:


length_list=[]
for l in lines.eng:
     length_list.append(len(l.split(' ')))
eng_max_length = np.max(length_list)


# In[39]:


encoder_input_data = np.zeros((len(lines.eng), eng_max_length),dtype='float32')
decoder_input_data = np.zeros((len(lines.fr), fr_max_length),dtype='float32')
decoder_target_data = np.zeros((len(lines.fr), fr_max_length, num_decoder_tokens+1),dtype='float32')


# In[40]:


for i, (input_text, target_text) in enumerate(zip(lines.eng, lines.fr)):
     for t, word in enumerate(input_text.split()):
         encoder_input_data[i, t] = input_token_index[word]
     for t, word in enumerate(target_text.split()):
 # decoder_target_data is ahead of decoder_input_data by one timestep
         decoder_input_data[i, t] = target_token_index[word]
         if t>0: 
 # decoder_target_data will be ahead by one timestep
 # and will not include the start character.
             decoder_target_data[i, t - 1, target_token_index[word]] = 1.
         if t== len(target_text.split())-1:
             decoder_target_data[i, t:, 89] = 1


# In[41]:


for i in range(decoder_input_data.shape[0]):
     for j in range(decoder_input_data.shape[1]):
         if(decoder_input_data[i][j]==0):
             decoder_input_data[i][j] = 89


# In[42]:


print(decoder_input_data.shape,encoder_input_data.shape,decoder_target_data.shape)


# In[43]:


from keras import Sequential
from keras.layers import Embedding,Input,Dense,Bidirectional,LSTM


# In[44]:


model = Sequential()
model.add(Embedding(len(input_words)+1, 128, input_length=fr_max_length, mask_zero=True))
model.add((Bidirectional(LSTM(256, return_sequences = True))))
model.add((LSTM(256, return_sequences=True)))
model.add((Dense(len(target_token_index)+1, activation='softmax')))


# In[45]:


decoder_target_data[45000]


# In[52]:


b = np.zeros((5,2,3),dtype='float32')


# In[56]:


c=np.argmax(b[4],axis=1)
c


# In[55]:


target_token_index['end']


# In[45]:


embedding_size=128


# In[46]:


encoder_inputs = Input(shape=(None,))
en_x= Embedding(num_encoder_tokens+1, embedding_size)(encoder_inputs)
encoder = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# In[47]:


decoder_inputs = Input(shape=(None,))
dex= Embedding(num_decoder_tokens+1, embedding_size)
final_dex= dex(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex, initial_state=encoder_states)
decoder_outputs = Dense(2000,activation='tanh')(decoder_outputs)
decoder_dense = Dense(num_decoder_tokens+1, activation='softmax')#(decoder_outputs)
decoder_outputs = decoder_dense(decoder_outputs)


# In[48]:


from keras.models import Model
model3 = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


# In[49]:


model.summary()


# In[50]:


history3 = model3.fit([encoder_input_data, decoder_input_data], decoder_target_data,
 batch_size=32,epochs=5,validation_split=0.05)


# In[1]:


a='he is a good man'
#a=pd.read_table("sample.txt")
#a=indices[a]
a.lower();
exclude = set(string.punctuation)
a= ''.join(ch for ch in a if ch not in exclude)

#a=filter_eng_words(a)
#b=1
inpp=np.zeros((1,eng_max_length),dtype='float32')


# In[ ]:



for j in a.split():
    inpp[j] = input_token_index[j]
    


# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:



t = model3.predict([inpp,decoder_input_data]).reshape(decoder_input_data.shape[1], num_decoder_tokens+1)


# In[242]:


t2=np.argmax(t,axis=1) 
    for i in range(len(t2)): 
        if int(t2[i]!=0): 
               print(list(target_token_index.keys())[int(t2[i]-1)])


# In[ ]:




