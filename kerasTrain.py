#Dependencies
import numpy as np
import pandas as pd
#dataset import
train_dataset = pd.read_csv('wildfires_dataset.csv', index_col=0)
test_dataset = pd.read_csv("wildfires_test_dataset.csv", index_col=0)

#Changing pandas dataframe to numpy array
X = train_dataset.drop(columns='fire_type').values
y = train_dataset['fire_type'].values

#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.preprocessing import LabelBinarizer
labelbin = LabelBinarizer()
y = labelbin.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.01)



#Dependencies
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(100, input_dim=9, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)

model.save('model.h5')

y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)