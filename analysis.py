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

'''anger_training = np.loadtxt('./val_losses/anger.csv', delimiter=',', skiprows=1)
anticipation_training = np.loadtxt('./val_losses/anticipation.csv', delimiter=',', skiprows=1)
disgust_training = np.loadtxt('./val_losses/disgust.csv', delimiter=',', skiprows=1)
fear_training = np.loadtxt('./val_losses/fear.csv', delimiter=',', skiprows=1)
joy_training = np.loadtxt('./val_losses/joy.csv', delimiter=',', skiprows=1)
love_training = np.loadtxt('./val_losses/love.csv', delimiter=',', skiprows=1)
optimism_training = np.loadtxt('./val_losses/optimism.csv', delimiter=',', skiprows=1)
pessimism_training = np.loadtxt('./val_losses/pessimism.csv', delimiter=',', skiprows=1)
sadness_training = np.loadtxt('./val_losses/sadness.csv', delimiter=',', skiprows=1)
surprise_training = np.loadtxt('./val_losses/surprise.csv', delimiter=',', skiprows=1)
trust_training = np.loadtxt('./val_losses/trust.csv', delimiter=',', skiprows=1)

values = np.c_[anger_training[:,2], anticipation_training[:,2],disgust_training[:,2], fear_training[:,2],
            joy_training[:,2], love_training[:,2], optimism_training[:,2], pessimism_training[:,2],sadness_training[:,2],
            surprise_training[:,2], trust_training[:,2]]
avg = np.mean(values, axis=1)
epochs = anger_training[:,1]
reg = np.loadtxt('./val_losses/main_output.csv', delimiter=',', skiprows=1)[:,2]

plt.plot(epochs, avg, label='Average classification')
plt.plot(epochs, reg, label='Regression')
plt.grid()
plt.legend(loc='best')
plt.xticks(np.arange(0, 12, 1))
plt.ylim(0, 0.3)
plt.title('Validation loss as a function of number of epochs')
plt.ylabel('loss')
plt.xlabel('Epochs')
fig = plt.gcf()
fig.set_size_inches(12,5)
plt.savefig('../report/pictures/regclassvalidation.png', dpi=200)'''

'''classrun = np.loadtxt('../kode/testkode/preds.txt', dtype={ 'names': ('max', 'bla', 'feats', 'blahh', 'accuracy', 'f-micro', 'blahhhh', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'none', 'hardbrack'),
                                                            'formats': ['S4', 'S4', 'i4', 'S4', 'f4', 'S4', 'S4', 'f4', 'f4', 'f4', 'f4', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'S4']})

y = classrun['accuracy']
x = classrun['feats']
plt.plot(x, y)
plt.grid()
plt.xticks(np.arange(0, 31000, 1000))
plt.title('Accuracy as a function of max features')
plt.ylabel('Accuracy')
plt.xlabel('Max features')
fig = plt.gcf()
fig.set_size_inches(18,5)
plt.savefig('../report/pictures/max_f_accuracy.png', dpi=200)'''



'''
LW = [0.2,    0.3,   0.4,   0.45,  0.5,   0.6,   0.7,    0.8,   0.9]
reg = [0.318, 0.330, 0.286, 0.411, 0.351, 0.376, -0.051, 0.376, 0.374]
#Ec = [0.432, 0.431, 0.434, 0.451, 0.434, 0.452, 0.438, 0.416, 0.432]

plt.plot(LW, reg)
#plt.plot(LW, Ec, label='Classification as accuracy')
plt.grid()
#plt.legend(loc='best')
plt.xticks(np.arange(0.1, 0.95, 0.1))
plt.xlim(min(LW), max(LW))
plt.title('Pearson score as a function of differing loss weight')
plt.ylabel('Pearson score')
plt.xlabel('Loss weights')
fig.set_size_inches(12,5)
plt.savefig('../report/pictures/LossWeightsPlot.png', dpi=200)'''

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
test_lengths = [388, 777, 1067, 1464]

deep_data = np.loadtxt('./preds/statpreds.txt')
feat_data = np.loadtxt('../kode/testkode/statpreds.txt')

anger_feat = feat_data[:test_lengths[0]][:,1]
fear_feat = feat_data[test_lengths[0]:test_lengths[1]][:,1]
joy_feat = feat_data[test_lengths[1]:test_lengths[2]][:,1]
sadness_feat = feat_data[test_lengths[2]:test_lengths[3]][:,1]

anger_data = deep_data[:test_lengths[0]][:,12]
fear_data = deep_data[test_lengths[0]:test_lengths[1]][:,12]
joy_data = deep_data[test_lengths[1]:test_lengths[2]][:,12]
sadness_data = deep_data[test_lengths[2]:test_lengths[3]][:,12]

anger_data_gold = deep_data[:test_lengths[0]][:,0]
fear_data_gold = deep_data[test_lengths[0]:test_lengths[1]][:,0]
joy_data_gold = deep_data[test_lengths[1]:test_lengths[2]][:,0]
sadness_data_gold = deep_data[test_lengths[2]:test_lengths[3]][:,0]

data_general = [anger_feat,anger_data,anger_data_gold,fear_feat,fear_data,fear_data_gold,joy_feat,joy_data,joy_data_gold,sadness_feat,sadness_data,sadness_data_gold]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
fig.set_size_inches(18,6)
ax.boxplot(data_general, zorder=2.6)
minor_ticks = np.arange(3.5, 12, 3)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(np.arange(0,1,0.1),minor=True)
ax.xaxis.grid(which='minor', linewidth=5)
ax.yaxis.grid(which='minor', linewidth=0.5)
ax.set_xticklabels(['Anger, Feat','Anger, Deep', 'Anger, Gold', 'Fear, Feat','Fear, Deep', 'Fear, Gold', 'Joy, Feat', 'Joy, Deep', 'Joy, Gold', 'Sadness, Feat', 'Sadness, Deep', 'Sadness, Gold'])
plt.title('Boxplots over predicted regression values')
#plt.show()
plt.savefig('../report/pictures/boxplotmix.png', dpi=250)



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