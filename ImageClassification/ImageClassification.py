import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks
from keras import metrics

def runImageClassification(getModel=None,fitModel=None,seed=1):
    print("Preparing data...")
    data=CIFAR(seed)
        
    # Create model 
    print("Creating model...")
    model=getModel(data)

    # Fit model
    print("Fitting model...")
    hist, model=fitModel(model,data)

    # Evaluate on test data
    print("Evaluating model...")
    score = model.evaluate(data.x_test, data.y_test, verbose=0)
    print('Test accuracy:', score[1])
    
    return hist

def myGetModel(c_data):
    model = models.Sequential()
    # 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=c_data.x_train[0].shape, padding = 'same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    # 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    # 3
    model.add(layers.Conv2D(128, (1, 1), activation='relu', padding = 'same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    # Flatten the output tensor and pass to the densely connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(c_data.y_train.shape[1], activation='softmax'))
    
    optim = optimizers.RMSprop(lr = 0.001)
    
    model.compile(loss = "categorical_crossentropy",
                optimizer = optim,
                metrics = ['acc'])
    
    return model

def myFitModel(c_model, c_data):
    es = callbacks.EarlyStopping(monitor='val_loss',
                                patience=10,
                                restore_best_weights=True,
                                min_delta = 0.01, verbose=1)
    
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001, verbose=1,
                                           min_delta = 0.01, cooldown=2)
    
    history = c_model.fit(c_data.x_train, c_data.y_train,
                batch_size = 128,
                epochs = 100,
                validation_data = (c_data.x_valid, c_data.y_valid),
                shuffle = True,
                callbacks = [es, reduce_lr])

    c_model.save('weights-best.h5')

    return history, c_model

class CIFAR:
    def __init__(self,seed=0):
        # Get and split data
        data = self.__getData(seed)
        self.x_train_raw=data[0][0]
        self.y_train_raw=data[0][1]
        self.x_valid_raw=data[1][0]
        self.y_valid_raw=data[1][1]
        self.x_test_raw=data[2][0]
        self.y_test_raw=data[2][1]
        # Record input/output dimensions
        self.num_classes=10
        self.input_dim=self.x_train_raw.shape[1:]
         # Convert data
        self.y_train = np_utils.to_categorical(self.y_train_raw, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid_raw, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test_raw, self.num_classes)
        self.x_train = self.x_train_raw.astype('float32')
        self.x_valid = self.x_valid_raw.astype('float32')
        self.x_test = self.x_test_raw.astype('float32')
        self.x_train  /= 255
        self.x_valid  /= 255
        self.x_test /= 255
        # Class names
        self.class_names=['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

    def __getData (self,seed=0):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return self.__shuffleData(x_train,y_train,x_test,y_test,seed)
    
    def __shuffleData (self,x_train,y_train,x_test,y_test,seed=0):
        tr_perc=.75
        va_perc=.15
        x=np.concatenate((x_train,x_test))
        y=np.concatenate((y_train,y_test))
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)
        indices = np.random.permutation(len(x))
        tr=round(len(x)*tr_perc)
        va=round(len(x)*va_perc)
        self.tr_indices=indices[0:tr]
        self.va_indices=indices[tr:(tr+va)]
        self.te_indices=indices[(tr+va):len(x)]
        x_tr=x[self.tr_indices,]
        x_va=x[self.va_indices,]
        x_te=x[self.te_indices,]
        y_tr=y[self.tr_indices,]
        y_va=y[self.va_indices,]
        y_te=y[self.te_indices,]
        return ((x_tr,y_tr),(x_va,y_va),(x_te,y_te))

    # Print 25 random figures from the validation data
    def showImages(self):
        images=self.x_valid_raw
        labels=self.y_valid_raw
        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        plt.figure(figsize=(10,10))
        indices=np.random.randint(0,images.shape[0],25)
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[indices[i]], cmap=plt.cm.binary)
            # The CIFAR labels happen to be arrays, 
            # which is why we need the extra index
            plt.xlabel(class_names[labels[indices[i]][0]])
        plt.show()