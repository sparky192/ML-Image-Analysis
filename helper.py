# Helper

def load_train_data(path='./data'):
    '''
    FUNCTION TO LOAD TRAIN IMAGES AND LABELS 
    
    :param path: path to the data location (string)
    
    '''
    import os
    import gzip
    import numpy as np
    
    labels_path = os.path.join(path,'train-labels-idx1-ubyte.gz')
    images_path = os.path.join(path,'train-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path,'rb') as lp:
        labels = np.frombuffer(lp.read(), dtype=np.uint8, offset=8)
        
    with gzip.open(images_path,'rb') as ip:
        images = np.frombuffer(ip.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28*28)
        
    return images, labels



def load_test_data(path='./data'):
    
    '''
    FUNCTION TO LOAD TEST IMAGES AND LABELS
    
    :param path: path to the data location (string)
    
    '''
    
    import os
    import gzip
    import numpy as np
    
    labels_path = os.path.join(path,'t10k-labels-idx1-ubyte.gz')
    images_path = os.path.join(path,'t10k-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path,'rb') as lp:
        labels = np.frombuffer(lp.read(), dtype=np.uint8, offset=8)
        
    with gzip.open(images_path,'rb') as ip:
        images = np.frombuffer(ip.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28*28)
        
    return images, labels



def show_image(image, label=''):
    
    '''
    Function to show image for the given data
    
    :param image: row vector of the test/train image
    :param label: label for the image (optional)
   
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    get_ipython().magic('matplotlib inline')
    
    arr = image.reshape(28,28)
    ax = plt.imshow(arr, cmap='gray')
    
    plt.title(label)
    plt.show()

    
    
def show_image_for_label(i):
    import numpy as np
    import warnings
    import random

    '''
    Shows a random image from test set with label 'i'
    
    :param i: label to show image (int 0-9)
    
    '''
    
    if(i>9):
        warnings.warn('Invalid Label')
        return
    
    data,label = load_test_data()
    index = np.where(label == i)
    
    show_image(data[random.choice(index[0])])


def show_random(selection):
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    import warnings
    get_ipython().magic('matplotlib inline')
    
    '''
    Shows random images from specified set
    :param selection: dataset to load "train/test"
    '''
    if(selection == 'train'):
        data,label = load_train_data()
    elif(selection == 'test'):
        data,label = load_test_data()
    else:
        warnings.warn('Invalid argument passed: choose"train" or "test"')
        return
    
    hst = data[random.choice(range(len(data)))].reshape(28,28)
    arr = data[random.choice(range(len(data)))].reshape(28,28)
    np.append(hst,arr, axis=1)
    for i in range(5):
        arr = data[random.choice(range(len(data)))].reshape(28,28)
        hst = np.append(hst,arr, axis=1)
    vst = hst
    for i in range(5):
        hst= data[random.choice(range(len(data)))].reshape(28,28)
        for i in range(5):
            arr = data[random.choice(range(len(data)))].reshape(28,28)
            hst = np.append(hst,arr, axis=1)
        vst = np.append(vst,hst, axis=0)

    plt.imshow(vst, cmap='gray')
   


