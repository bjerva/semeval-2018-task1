import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = {'500': pd.Series([0.4, 0.4, 0.5, 0.2, 0.45], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '1000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '1500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '2000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '2500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '3000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '3500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '4000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '4500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '5000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '5500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '6000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '6500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '7000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '7500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '8000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '8500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '9000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '9500': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average']),
    '10000': pd.Series([0.2, 0.8, 0.3, 0.1, 0.55], index=['Anger', 'Fear', 'Joy', 'Sadness','Average'])}

#df = pd.DataFrame(d)
#df.plot(kind='bar', title="Pearson scores of different max_features with n-gram range 1-5 and custom hashtag feature added")
#plt.show()

max_f = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
y = [0.224, 0.371, 0.338, 0.361, 0.374, 0.411, 0.355, 0.212]

plt.plot(max_f, y)
plt.grid()
plt.xticks(np.arange(0.1, 0.95, 0.1))
plt.xlim(min(max_f), max(max_f))
plt.title('Pearson score as a function of differing dropout values')
plt.ylabel('Pearson score')
plt.xlabel('Dropout values')
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.savefig('../report/pictures/DropoutPlot.png', dpi=200)

#ef = pd.DataFrame(e)
#ef.plot(kind='bar',title='F micro', subplots=True)
#plt.show()

f = {'Anger':   pd.Series([0.306, 0.379, 0.246, 0.423, 0.413, 0.353, -0.097, 0.395, 0.321], index=['0.2', '0.3', '0.4', '0.45', '0.5','0.6','0.7','0.8','0.9']),
    'Fear':     pd.Series([0.545, 0.435, 0.426, 0.523, 0.447, 0.517, 0.014, 0.439, 0.509], index=['0.2', '0.3', '0.4', '0.45', '0.5','0.6','0.7','0.8','0.9']),
    'Joy':      pd.Series([-0.021, 0.045, 0.134, 0.118, 0.134, 0.174, -0.037, 0.218, 0.230], index=['0.2', '0.3', '0.4', '0.45', '0.5','0.6','0.7','0.8','0.9']),
    'Sadness':  pd.Series([0.442, 0.462, 0.339, 0.436, 0.411, 0.460, -0.086, 0.452, 0.438], index=['0.2', '0.3', '0.4', '0.45', '0.5','0.6','0.7','0.8','0.9']),
    'Avg.':     pd.Series([0.318, 0.330, 0.286, 0.375, 0.351, 0.376, -0.051, 0.376, 0.374], index=['0.2', '0.3', '0.4', '0.45', '0.5','0.6','0.7','0.8','0.9'])}


g = {'Anger':   pd.Series([0.282, 0.405], index=['NADAM', 'ADAM']),
    'Fear':     pd.Series([0.480, 0.457], index=['NADAM', 'ADAM']),
    'Joy':      pd.Series([0.205, 0.150], index=['NADAM', 'ADAM']),
    'Sadness':  pd.Series([0.373, 0.429], index=['NADAM', 'ADAM']),
    'Avg.':     pd.Series([0.335, 0.360], index=['NADAM', 'ADAM'])}

'''gf = pd.DataFrame(g)
gf = gf[['Anger', 'Fear', 'Joy', 'Sadness', 'Avg.']]
print(gf)'''
'''test_lengths = [388, 777, 1067, 1464]

data = np.loadtxt('./preds/statpreds.txt')

anger_data = data[:test_lengths[0]][:,12]
fear_data = data[test_lengths[0]:test_lengths[1]][:,12]
joy_data = data[test_lengths[1]:test_lengths[2]][:,12]
sadness_data = data[test_lengths[2]:test_lengths[3]][:,12]

anger_data_gold = data[:test_lengths[0]][:,0]
fear_data_gold = data[test_lengths[0]:test_lengths[1]][:,0]
joy_data_gold = data[test_lengths[1]:test_lengths[2]][:,0]
sadness_data_gold = data[test_lengths[2]:test_lengths[3]][:,0]

data_general = [anger_data,anger_data_gold,fear_data, fear_data_gold,joy_data, joy_data_gold,sadness_data, sadness_data_gold]

fig = plt.figure()
fig.set_size_inches(12,6)
ax = fig.add_subplot(111)
bpl = plt.boxplot(data_general)
ax.set_xticklabels(['Anger', 'Anger, Gold', 'Fear', 'Fear, Gold', 'Joy', 'Joy, Gold', 'Sadness', 'Sadness, Gold'])
plt.title('Boxplots over predicted regression values, deep learning')
plt.savefig('../report/pictures/boxplotdeep.png', dpi=250)'''



'''ff = pd.DataFrame(f)
ff = ff[['Anger', 'Fear', 'Joy', 'Sadness', 'Avg.']]
ax = ff.plot(kind='bar', title='Pearson score as a result of differing Loss Weights')
plt.legend(loc='best')
plt.xlabel('Weighted BCE values')
plt.ylabel('Pearson score')
plt.grid(axis='y', linestyle='dashed')
ax.set_axisbelow(True)
fig = plt.gcf()
fig.set_size_inches(18, 6)
plt.savefig('../report/pictures/LossWeightsvalues.png', dpi=250)'''