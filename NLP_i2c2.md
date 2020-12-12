```python
import pandas as pd
import re
import string
import numpy as np
from sklearn.metrics import confusion_matrix as cm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
```


```python
hospital_data = pd.read_excel("/Users/vishalpattanshetty/Downloads/smokers_surrogate_train_all_version2.xlsx", header=0)
```


```python
hospital_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>STATUS</th>
      <th>TEXT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>641</td>
      <td>CURRENT SMOKER</td>
      <td>977146916\nHLGMC\n2878891\n022690\n01/27/1997 ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>643</td>
      <td>CURRENT SMOKER</td>
      <td>026738007\nCMC\n15319689\n3/25/1998 12:00:00 A...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>681</td>
      <td>CURRENT SMOKER</td>
      <td>071962960\nBH\n4236518\n417454\n12/10/2001 12:...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>704</td>
      <td>CURRENT SMOKER</td>
      <td>418520250\nNVH\n61562872\n3/11/1995 12:00:00 A...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>757</td>
      <td>CURRENT SMOKER</td>
      <td>301443520\nCTMC\n49020928\n448922\n1/11/1990 1...</td>
    </tr>
  </tbody>
</table>
</div>




```python
hospital_data.dtypes
```




    ID         int64
    STATUS    object
    TEXT      object
    dtype: object




```python
li = []
ly = []
```


```python
# def clean_text_round1(text):
#     '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
#     text = re.sub('[%s]' % re.escape(string.punctuation),'', text) #toremove punctuations
#     #text = re.findall('((?<=HISTORY OF PRESENT ILLNESS :\n)(.\n?)*)*(?=PAST MEDICAL HISTORY)',text) #standaloneuse
#     text = re.findall('((?<=HISTORY OF PRESENT ILLNESS \n)(.\n?)*)*(?=PAST MEDICAL HISTORY)',text) #use with punct
#     if text:
#         li.append(text[0][0].replace('\n', '').replace('  ', ' '))
#         # maintain a list of y
#     # remove the entire row if text is not there
#     return text[0][0].replace('\n', '') if text else None
for i in range(len(hospital_data)):
    text = re.sub('[%s]' % re.escape(string.punctuation),'', hospital_data.loc[i, 'TEXT']) #remove punctuation
    text = re.findall('((?<=HISTORY OF PRESENT ILLNESS \n)(.\n?)*)*(?=PAST MEDICAL HISTORY)',text) #use with punct
    if text and hospital_data.loc[i, 'STATUS']:
        q = text[0][0].replace('\n', '').replace('  ', ' ').lower()
        if q != '':
            li.append(q)
            ly.append(hospital_data.loc[i, 'STATUS'])
#round1 = lambda x: clean_text_round1(x)
```


```python
import gensim.models.keyedvectors as word2vec 
path='/Users/vishalpattanshetty/Downloads/PubMed-and-PMC-w2v.bin'
model = word2vec.KeyedVectors.load_word2vec_format(path, binary = True)
```


```python
def getVector(w):
    global model
    if w in model:
        return model[w]
    else:
        return np.zeros(200)
    
sent_vectors = []
for s in li:
    vec = np.zeros(200)
    for w in s.split():
        vec = np.add(vec, getVector(w))
#    vec /= len(s.split())
    sent_vectors.append(vec)
```


```python
X_train, X_test, y_train, y_test = train_test_split(sent_vectors, ly, test_size = 0.1)
```


```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
```


```python
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

    Accuracy: 0.42105263157894735

