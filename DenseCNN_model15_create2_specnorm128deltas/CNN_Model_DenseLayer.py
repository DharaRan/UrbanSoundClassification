import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import SGD, Adamax
from keras import regularizers
import pickle
from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
from sklearn import svm

import time



def load_data(filename):
    print('loading %s...' % filename)
    return pickle.load(open(filename, "rb"), encoding="latin1")  


"""
#################################################################################
######################### Read in Data ##########################################
#################################################################################
"""

#dataset = pd.read_pickle('dataset.pkl') # Dataframe shape: (54058,2464)

VERSION=15

VALIDATION_FOLD=10
TRAIN_FOLD = [2,3,4,5,6,7,8,9]
TESTING_FOLD = [1] 

folderDir='data128norm_deltas/features_specsNorm_fold'
folderDirLabel='data128norm_deltas/labels_specsNorm_fold'

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



"""
# READ Validation and label data
"""


validation_filename=folderDir+str(VALIDATION_FOLD)+'.pkl'
validation=load_data(validation_filename)

validationLabel_filename=folderDirLabel+str(VALIDATION_FOLD)+'.pkl'
validation_label=load_data(validationLabel_filename)

print('validation shape: ',validation.shape)
print('validation Label shape: ',validation_label.shape)


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

"""
#######################################################################################
############################## CNN Model ##############################################
#######################################################################################
"""


def create_model2():
    # Model similar to the Salmon and Bello CNN model 

    model = Sequential()
    
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=(128, 128, 2)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid'))
    model.add(Activation('relu'))   

    model.add(Flatten())
    
    model.add(Dense(units=64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(units=10, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))

    model.summary()
    return model

def create_model3():
    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=(128, 128, 2)))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))   
	
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 2)))	
    model.add(Dropout(0.50))

    model.add(Flatten())
    
    model.add(Dense(units=128, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(units=10, activation='softmax',kernel_regularizer=regularizers.l2(0.001)))
    model.summary()
    return model
	

print('Creating Model')
model=create_model2()
model.compile(loss='categorical_crossentropy',
              optimizer=Adamax(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])


epochs = 40 #25 #
batch_size = 30 #15#

start_time = time.time()
print('Starting Model fitting',time.strftime("%H:%M:%S", time.gmtime(start_time)))

his=model.fit(train, label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(validation, validation_label))

elapsed_time = time.time() - start_time
print('Time elapsed: ',elapsed_time)
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print('End of CNN Model Fitting (Elapsed time)',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


#SAVE Model
modelName='model_delta_'+str(VERSION)+'validate_'+str(VALIDATION_FOLD)+'128'+'.h5'
model.save(modelName)  # creates a HDF5 file 'my_model.h5'
print('Model Saved! as '+modelName)
print('Completed running '+'Validate: '+str(VALIDATION_FOLD)+' Test: '+str(TESTING_FOLD[0]))

#del model  # deletes the existing model
# returns a compiled model
# identical to the previous one
#model = load_model('my_model.h5')

"""
#########################################################################################
################################ Show the Epochs vs ACC #################################
#########################################################################################
"""
# summarize history for accuracy
plt.plot(his.history['acc'])
plt.plot( his.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot( his.history['loss'])
plt.plot( his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


"""
###############################################################################
################ Accuracy and Confusion Matrix ################################
###############################################################################
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

#prediction= model.predict_proba(test,batch_size=batch_size, verbose=1)
y_prob = model.predict_proba(test, verbose=1)
#y_pred = np_utils.probas_to_classes(y_prob)
y_pred = y_prob.argmax(axis=-1)
y_true = np.argmax(test_label, 1)
cm = confusion_matrix(y_true, y_pred, labels=None)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=label,
                      title='Confusion Matrix: '+'Validate: '+str(VALIDATION_FOLD)+' Test: '+str(TESTING_FOLD[0]))
plt.show()
#print("confusion matrix:\n%s" % cm)

##########################################################
### Per class accuracy and other metrics
##########################################################


print(classification_report(y_true, y_pred))

#Total count per predicted class
list_classes=[0,1,2,3,4,5,6,7,8,9]
count_list = []
for c in range(0,len(list_classes)):
    for i in range(0,len(cm)):
        if c==i:
            dia=cm[c][i]
            count_list.append(dia)

#Total count per True class
Truecount_list = []
sumTot=0
for c in list_classes:
    sumTot=0
    for i in y_true:
        #print(i)
        if i==c:
            sumTot=sumTot+1
    Truecount_list.append(sumTot)

# per class accuracy
classAcc=[]
for i in range(0,len(Truecount_list)):
    c_Acc=count_list[i]/Truecount_list[i]
    classAcc.append(c_Acc)
print(classAcc)

##################################################################
################### True Test class distribution #################
##################################################################

# Following Finds the distribution of the slices frame
list_classes=['AI','CA','CH','DO','DR','EN','GU','JA','SI','ST']

import seaborn as sns   
ax=sns.barplot(x=list_classes, y=Truecount_list)   
titlename='Validate: '+str(VALIDATION_FOLD)+' Test: '+str(TESTING_FOLD[0])
ax.set(xlabel='Classes', ylabel='Count',title=titlename)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.0f}'.format(height),
            ha="center")    
    
    

###########################################################################
##################3 Evaluate the model ####################################
###########################################################################
scores = model.evaluate(test, test_label,batch_size=batch_size)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
















    
    
    