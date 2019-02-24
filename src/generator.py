import keras
#from skimage.transform import resize
from keras.utils import to_categorical
import cv2
from cv2 import resize
import numpy as np

class SUNGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, image_dir, label_dir, batch_size=1, dim=(512,512), n_channels=3,
                 n_classes=2, shuffle=True, mode='training'):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        crop_shape = dim
        resize_shape = dim
        
        # Preallocate memory
        if mode == 'training' and crop_shape:
            self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.Y1 = np.zeros((batch_size, crop_shape[1]//4, crop_shape[0]//4, self.n_classes), dtype='float32')
            self.Y2 = np.zeros((batch_size, crop_shape[1]//8, crop_shape[0]//8, self.n_classes), dtype='float32')
            self.Y3 = np.zeros((batch_size, crop_shape[1]//16, crop_shape[0]//16, self.n_classes), dtype='float32')
        elif resize_shape:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y1 = np.zeros((batch_size, resize_shape[1]//4, resize_shape[0]//4, self.n_classes), dtype='float32')
            self.Y2 = np.zeros((batch_size, resize_shape[1]//8, resize_shape[0]//8, self.n_classes), dtype='float32')
            self.Y3 = np.zeros((batch_size, resize_shape[1]//16, resize_shape[0]//16, self.n_classes), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')
        
    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))
        
    def __getitem__(self, index):
        # Generate one batch of data
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data        
        #X, y = self.__data_generation(list_IDs_temp)
        self.__data_generation(list_IDs_temp)

        return self.X, [self.Y1, self.Y2, self.Y3]
    
    def get_single(self, ID):
        image_np = np.load(image_dir + ID + '.npy')
        label_np = np.load(label_dir + ID + '.npy')
        #image_resized, label_resized = random_crop_or_pad(image_np, label_np, self.dim)
        image_resized, label_resized = crop_and_scale(image_np, label_np, self.dim)
        label = label_resized # multidimenstional       
        X = image_resized
        Y1 = to_categorical(cv2.resize(label, (label.shape[1]//4, label.shape[0]//4)), self.n_classes).reshape((label.shape[0]//4, label.shape[1]//4, -1))   
        Y2 = to_categorical(cv2.resize(label, (label.shape[1]//8, label.shape[0]//8)), self.n_classes).reshape((label.shape[0]//8, label.shape[1]//8, -1))
        Y3 = to_categorical(cv2.resize(label, (label.shape[1]//16, label.shape[0]//16)), self.n_classes).reshape((label.shape[0]//16, label.shape[1]//16, -1))
        return X, [Y1, Y2, Y3]
        
    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image_np = np.load(image_dir + ID + '.npy')
            #image_resized = resize(image_np, (*self.dim, self.n_channels), mode='constant')
            #image_resized = cv2.resize(image_np, *self.dim, None, 0, 0, cv2.INTER_LINEAR)
            
            # Store class
            label_np = np.load(label_dir + ID + '.npy')
            #label_resized = resize(image_np, (*self.dim, 1), mode='constant')
            #label_resized = cv2.resize(label_np, *self.dim, None, 0, 0, cv2.INTER_LINEAR)
            #image_resized, label_resized = random_crop_or_pad(image_np, label_np, self.dim)
            image_resized, label_resized = crop_and_scale(image_np, label_np, self.dim)
            label = label_resized # multidimenstional                  
            
            # input: float between 0 and 255
            
            self.X[i] = image_resized
            self.Y1[i] = to_categorical(cv2.resize(label, (label.shape[1]//4, label.shape[0]//4)), self.n_classes).reshape((label.shape[0]//4, label.shape[1]//4, -1))   
            self.Y2[i] = to_categorical(cv2.resize(label, (label.shape[1]//8, label.shape[0]//8)), self.n_classes).reshape((label.shape[0]//8, label.shape[1]//8, -1))
            self.Y3[i] = to_categorical(cv2.resize(label, (label.shape[1]//16, label.shape[0]//16)), self.n_classes).reshape((label.shape[0]//16, label.shape[1]//16, -1))