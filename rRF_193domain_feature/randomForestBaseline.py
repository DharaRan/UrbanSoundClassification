from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


"""
Random Forest Code
"""


def load_data(filename):
    print('loading %s...' % filename)
    return pickle.load(open(filename, "rb"), encoding="latin1")  # specifying encoding for python3 compat

def inv_one_hot_encode(labels):  # inverse (index of 1 for each row)
    return np.argmax(labels, axis=1)

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
"""
#################################################################################
######################### Read in Data ##########################################
#################################################################################
"""


VERSION=1

TRAIN_FOLD = [1,2,3,4,5,6,7,8,9]
TESTING_FOLD = [10] 

folderDir='data139features/features_mean_fold'
folderDirLabel='data139features/labels_mean_fold'


"""
# READ Train and label data
"""


train_filename=folderDir+str(TRAIN_FOLD[0])+'.pkl'
train=load_data(train_filename)

trainLabel_filename=folderDirLabel+str(TRAIN_FOLD[0])+'.pkl'
label=load_data(trainLabel_filename)

for i in range(1,len(TRAIN_FOLD)):

    train_filename=folderDir+str(TRAIN_FOLD[i])+'.pkl'
    a=load_data(train_filename)
    train=np.append(train,a, axis=0)
    
    trainLabel_filename=folderDirLabel+str(TRAIN_FOLD[i])+'.pkl'
    a1=load_data(trainLabel_filename)
    label=np.append(label,a1, axis=0)
    
    
print('Train shape: ',train.shape)
print('Train Label: ',label.shape)

#flatten data
#train=train.flatten()
label=inv_one_hot_encode(label)

"""
# READ Test and label data
"""

test_filename=folderDir+str(TESTING_FOLD[0])+'.pkl'
test=load_data(test_filename)

testLabel_filename=folderDirLabel+str(TESTING_FOLD[0])+'.pkl'
test_label=load_data(testLabel_filename)

for i in range(1,len(TESTING_FOLD)):
    test_filename=folderDir+str(TESTING_FOLD[1])+'.pkl'
    a=load_data(test_filename)
    test=np.append(test,a, axis=0)
  
    testLabel_filename=folderDirLabel+str(TESTING_FOLD[0])+'.pkl'
    a1=load_data(testLabel_filename)
    test_label=np.append(test_label,a1, axis=0)

print('test shape: ',test.shape)
print('test Label: ',test_label.shape)
test_label=inv_one_hot_encode(test_label)


"""
#################################################################################
######################### Random Forest model ##########################################
#################################################################################
"""

# this training on data 193 featues

start_time = time.time()
print('Starting Model fitting',time.strftime("%H:%M:%S", time.gmtime(start_time)))
forest = OneVsRestClassifier(RandomForestClassifier(n_estimators = 500, max_depth=20, min_samples_leaf=30,verbose=1))
forestmodel = forest.fit(train, label)
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('End of RF Fitting (Elapsed time)',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))



"""
Prediction and Evaluation 
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion Matrix')

    print("confusion matrix:\n%s" % cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=85)
    plt.yticks(tick_marks, classes)
    plt.grid(False)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



###########################################################################
# Compute confusion matrix
#########################################################################

# labels

label=['air_conditioner','car_horn','children_playing','dog_bark', 'drilling',
       'engine_idling','gun_shot','jackhammer','siren','street_music']



prediction = forestmodel.predict(test)
cm = confusion_matrix(test_label, prediction, labels=None)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=label,
                      title='Confusion Matrix: '+'Test: '+str(TESTING_FOLD[0]))
plt.show()
#print("confusion matrix:\n%s" % cm)

##########################################################
### Per class accuracy and other metrics
##########################################################


print(classification_report(test_label, prediction))

accuracy =np.sum(prediction == test_label)/test_label.shape[0]
print("Accuracy of test: ", accuracy)



###################################################
########## Test Distribution ######################
###################################################
#Total count per True class
list_classes=[0,1,2,3,4,5,6,7,8,9]
Truecount_list = []
sumTot=0
for c in list_classes:
    sumTot=0
    for i in test_label:
        #print(i)
        if i==c:
            sumTot=sumTot+1
    Truecount_list.append(sumTot)

list_classes=['AI','CA','CH','DO','DR','EN','GU','JA','SI','ST']

import seaborn as sns   
ax=sns.barplot(x=list_classes, y=Truecount_list)   
titlename=' Test Folder: '+str(TESTING_FOLD[0])
ax.set(xlabel='Classes', ylabel='Count',title=titlename)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center") 








