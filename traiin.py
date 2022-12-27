

import numpy as np
import os
from PIL import Image

from tensorflow.keras import layers
from tensorflow.keras import models


TRAIN_DATA = 'dataset/train_data'
TEST_DATA = 'dataset/test_data'

Xtrain = []
Ytrain = []

#Xtrain[0][0], Xtrain[0][1]

#Xtrain = [x for i, x in enumerate(Xtrain) ]

Xtest = []
Ytest = []

dict = {'mrchinh': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'mrphuc': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'mrsam': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'mrsngan': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'mrsthao': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'mrtrong': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'mrtulong': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'mrvddam': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],'mrvuong': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],'mrxuanbac': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'mrchinht': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'mrphuct': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'mrsamt': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'mrsngant': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'mrsthaot': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'mrtrongt': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'mrtulongt': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'mrvddamt': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],'mrvuongt': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],'mrxuanbact': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

def getData(dirData, listData):
    for whatever in os.listdir(dirData):
        whatever_path = os.path.join(dirData, whatever)
        list_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path, filename)
            label = filename_path.split('\\')[1]
            
            img = np.array(Image.open(filename_path))
            list_filename_path.append((img, dict[label]))
            
        listData.extend(list_filename_path)
    return listData

Xtrain = getData(TRAIN_DATA, Xtrain)
Xtest = getData(TEST_DATA, Xtest)
        
#print(Xtrain[4000])


model_training_first = models.Sequential([
    layers.Conv2D(32, (3,3), input_shape = (100, 100, 3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax'),    
])

model_training_first.summary()

model_training_first.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_training_first.fit(np.array([x[0] for _, x in enumerate(Xtrain)]), np.array([y[1] for _, y in enumerate(Xtrain)]), epochs = 10)

model_training_first.save('model-ducc10people_10epochs.h5')

models = models.load_model('model-ducc10people_10epochs.h5')