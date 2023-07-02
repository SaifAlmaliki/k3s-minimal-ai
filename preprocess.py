import numpy as np

def preprocess(img):
    '''
    Preprocessing required on the images for inference with mxnet gluon
    The function takes loaded image and returns processed tensor
    '''
    img= np.array(img.resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img