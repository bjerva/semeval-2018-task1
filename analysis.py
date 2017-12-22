import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = {'0.0': pd.Series([0.4, 0.4, 0.5, 0.2, 0.45], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '0.1': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average'])}

df = pd.DataFrame(d)
#df.plot(kind='bar', title="Pearson scores of different dropout values")
#plt.show()

e = {'0.0': pd.Series([0.4, 0.2, 0.1, 0.4, 0.7, 0.9, 0.1, 0.2, 0.4, 0.1, 0.2, 0.05, 0.33, 0.5], index=['Anger', 'Anticipation', 'disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust','No emotions', 'Average f-micro', 'Accuracy']),
    '0.1':  pd.Series([0.4, 0.2, 0.1, 0.4, 0.7, 0.9, 0.1, 0.2, 0.4, 0.1, 0.2, 0.05, 0.33, 0.5], index=['Anger', 'Anticipation', 'disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust','No emotions', 'Average f-micro', 'Accuracy']),
    '0.2':  pd.Series([0.4, 0.2, 0.1, 0.4, 0.7, 0.9, 0.1, 0.2, 0.4, 0.1, 0.2, 0.05, 0.33, 0.5], index=['Anger', 'Anticipation', 'disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust','No emotions', 'Average f-micro', 'Accuracy']),
    '0.3':  pd.Series([0.4, 0.2, 0.1, 0.4, 0.7, 0.9, 0.1, 0.2, 0.4, 0.1, 0.2, 0.05, 0.33, 0.5], index=['Anger', 'Anticipation', 'disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust','No emotions', 'Average f-micro', 'Accuracy']),
    '0.4':  pd.Series([0.4, 0.2, 0.1, 0.4, 0.7, 0.9, 0.1, 0.2, 0.4, 0.1, 0.2, 0.05, 0.33, 0.5], index=['Anger', 'Anticipation', 'disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust','No emotions', 'Average f-micro', 'Accuracy']),
    '0.5':  pd.Series([0.4, 0.2, 0.1, 0.4, 0.7, 0.9, 0.1, 0.2, 0.4, 0.1, 0.2, 0.05, 0.33, 0.5], index=['Anger', 'Anticipation', 'disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust','No emotions', 'Average f-micro', 'Accuracy']),
    '0.6':  pd.Series([0.4, 0.2, 0.1, 0.4, 0.7, 0.9, 0.1, 0.2, 0.4, 0.1, 0.2, 0.05, 0.33, 0.5], index=['Anger', 'Anticipation', 'disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust','No emotions', 'Average f-micro', 'Accuracy'])}

ef = pd.DataFrame(e)
ef.plot(kind='bar',title='F micro', subplots=True)
plt.show()