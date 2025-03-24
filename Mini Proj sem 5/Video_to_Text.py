#!/usr/bin/env python
# coding: utf-8

# ## 0. Install and Import Dependencies

# In[1]:


get_ipython().system('pip install ibm_watson')
get_ipython().system('brew install ffmpeg')


# In[3]:


import subprocess
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


# ## 1. Extract Audio

# In[4]:


command = 'ffmpeg -i resume.mp4 -ab 160k -ar 44100 -vn audio.wav'
subprocess.call(command, shell=True)


# ## 2. Setup STT Service

# In[ ]:


apikey = ''
url = ''


# In[ ]:


# Setup service
authenticator = IAMAuthenticator(apikey)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url)


# ## 3. Open Audio Source and Convert

# In[ ]:


with open('audio.wav', 'rb') as f:
    res = stt.recognize(audio=f, content_type='audio/wav', model='en-AU_NarrowbandModel', continuous=True).get_result()


# In[ ]:


res


# ## 4. Process Results and Output to Text

# In[ ]:


len(res['results'])


# In[ ]:


text = [result['alternatives'][0]['transcript'].rstrip() + '.\n' for result in res['results']]


# In[ ]:


text = [para[0].title() + para[1:] for para in text]
transcript = ''.join(text)
with open('output.txt', 'w') as out:
    out.writelines(transcript)


# In[ ]:


transcript


# In[ ]:





# In[ ]:




