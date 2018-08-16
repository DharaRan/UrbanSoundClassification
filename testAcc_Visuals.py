"""
Model Accuracy visualization
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


modelAcc=pd.read_csv('allModelResults.csv',float_precision='round_trip')
ax=sns.boxplot(x='Models',y='Classification Accuracy',data=modelAcc, order=['Dense_CNN','LinearSVC','Random_Forest'])
means = modelAcc.groupby(['Models'])['Classification Accuracy'].mean().values

mean_labels = [str(np.round(s, 2)) for s in means]
pos = range(len(means))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], means[tick] + 0.5, mean_labels[tick], 
            horizontalalignment='center', size='medium', color='w',weight='semibold')

ax.set_title("Classification Accuracies Across Models")  
ax.set_ylabel('Classification Accuracy (%)')




modelAcc=pd.read_csv('randomForestBaseline_Corr.csv',float_precision='round_trip')
ax=sns.boxplot(x='Feature Type',y='Classification Accuracy (%)',data=modelAcc)
means = modelAcc.groupby(['Feature Type'])['Classification Accuracy (%)'].mean().values

mean_labels = [str(np.round(s, 2)) for s in means]
pos = range(len(means))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], means[tick]-0.4, mean_labels[tick], 
            horizontalalignment='center', size='small', color='w',weight='bold')
ax.set_title("Random Forest: Features Based Accuracies")  





modelAcc=pd.read_csv('SVM_FeatureBoxplot.csv',float_precision='round_trip')

ax=sns.boxplot(x='Feature Type',y='Classification Accuracy (%)',data=modelAcc)
means = modelAcc.groupby(['Feature Type'])['Classification Accuracy (%)'].mean().values

mean_labels = [str(np.round(s, 2)) for s in means]
pos = range(len(means))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], means[tick]-0.1, mean_labels[tick], 
            horizontalalignment='center', size='small', color='w',weight='bold')
ax.set_title("Linear SVC: Features Based Accuracies")  



modelAcc=pd.read_csv('CNNBoxplot.csv',float_precision='round_trip')

ax=sns.boxplot(x='Model',y='Classification Accuracy (%)',data=modelAcc)
means = modelAcc.groupby(['Model'])['Classification Accuracy (%)'].mean().values

mean_labels = [str(np.round(s, 2)) for s in means]
pos = range(len(means))
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], means[tick]-0.1, mean_labels[tick], 
            horizontalalignment='center', size='small', color='w',weight='bold')
ax.set_title("CNN Model Accuracies")  
















