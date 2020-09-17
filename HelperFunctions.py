# Loading datasets: MNIST, CIFAR-10, SVHN

def load_data(choice, split_val):

    if choice=='mnist':
    
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784')
        x = mnist.data
        y = mnist.target
        
    elif choice =='svhm':
        # loading data in the for of mat file
        from scipy.io import loadmat
        data_train = loadmat('SVHMData/train_32x32.mat')
        data_test = loadmat('SVHMData/test_32x32.mat')
        
        # getting train data
        Ntrain = np.shape(data_train['X'])[3]
        Ntest = np.shape(data_test['X'])[3]
        
        X_train = np.transpose(data_train['X'].reshape(32*32*3,Ntrain)/255)
        y_train = data_train['y']
        
        X_test = np.transpose(data_test['X'].reshape(32*32*3,Ntest)/255)
        y_test = data_test['y']
        
        return X_train, X_test, y_train, y_test
        
    # Cifar-10 uses unpickle function
    else:
        c1=unpickle('cifarData/data_batch_1')
        c2=unpickle('cifarData/data_batch_2')
        c3=unpickle('cifarData/data_batch_3')
        c4=unpickle('cifarData/data_batch_4')
        c5=unpickle('cifarData/data_batch_5')

        x = np.concatenate((c1[b'data'], c2[b'data'], c3[b'data'], c4[b'data'], c5[b'data']))
        y = np.concatenate((c1[b'labels'], c2[b'labels'], c3[b'labels'], c4[b'labels'], c5[b'labels']))
    
    # Normalizing data
    Data = x/255
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Data, y, test_size=split_val, random_state=42)    
    
    return X_train, X_test, y_train, y_test
    
 #Helper function for loading Cifar-10 data   
 
 def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
 
 

