from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from os.path import join
from sklearn.cluster import KMeans
import pickle

class K_means:
    def __init__(self):
        self.prePath = "D:\\Working Project\\Python\\CS332.J11.KHTN\\Dataset\\Caltech256\\"
        self.sz = 257
        self.vgg16_feature_train_list = []
        self.vgg16_feature_test_list = []
    def ExtractFeature(self):
        
        model = VGG16(weights='imagenet', include_top=True)
        
        
        feature_path = join(self.prePath,"Features\\Vgg16_fc7\\")
        
        folders = os.listdir(join(self.prePath, "Images"))
        
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
            for i in range(0, self.sz): 
                #get name image in each class
                for imgdir in os.listdir(join(self.prePath, "Images" , folders[i])):
                    #join path to open image
                    p = join(self.prePath, "Images" , folders[i], imgdir)
                    #create feature image in each folder
                    if not os.path.exists(feature_path + folders[i]):
                        os.makedirs(feature_path + folders[i])
                    f = open(join(feature_path, folders[i], imgdir + ".npy"), "wb")
        
                    img = image.load_img(p, target_size=(224, 224))
                    img_data = image.img_to_array(img)
                    img_data = np.expand_dims(img_data, axis=0)
                    img_data = preprocess_input(img_data)        
                    vgg16_feature = model.predict(img_data)
                    vgg16_feature_np = np.array(vgg16_feature)
                    #savefeature to file
                    np.save(f, vgg16_feature_np.flatten())
                    f.close()
        else:
            print("Existing folder")
    def CreateDataBase(self, typedb = 1):
        if typedb == 1:
            db_path = join(self.prePath, "Database\\db1\\")
        if typedb == 2:
            db_path = join(self.prePath, "Database\\db2\\")
        if typedb == 3:
            db_path = join(self.prePath, "Database\\db3\\")
        
        #Place to save test and train data
        train_path = join(db_path, "train")
        test_path = join(db_path, "test")
        
        np.random.seed(10)
        
        img_path = join(self.prePath, "Images")
        
        img_subfolders = []
        for subfolder in os.listdir(img_path):
            img_subfolders.append(join(img_path, subfolder))
        
        #init list, train and test are temporary list
        train70 = []
        test30 = []
        train = []
        test = []
        #if db is not existe then create
        if not os.path.exists(db_path):
            os.makedirs(db_path)                     
            os.makedirs(train_path)
            os.makedirs(test_path)
            #get random sample
            for subfolder in img_subfolders:
                images = []
                images = os.listdir(subfolder)
                len_images = len(images)
                train = np.random.permutation(images)[:int(len_images * 70 / 100)]
                test = list(set(images) - set(train))
                
                train70.append(train)
                test30.append(test)
                
            train70 = np.array(train70)
            test30 = np.array(test30)
            #save train and test data to files
            f = open(join(train_path, "train70.npy"), "wb")
            np.save(f, train70)
            f.close()
            
            f = open(join(test_path, "test30.npy"), "wb")
            np.save(f, test30)
            f.close()
        else:
            print("Exist directory")
    def LoadFeature(self, typefeature = 1, typedb = 1):
        if typefeature == 1:
            feature_path = join(self.prePath, "Features\\vgg16_fc7")
        if typefeature == 2:
            feature_path = join(self.prePath, "Features\\ABC")
        if typedb == 1:
            train_path = join(self.prePath, "Database\\db1\\train\\train70.npy")
            test_path = join(self.prePath, "Database\\db1\\test\\test30.npy")
        if typedb == 2:
            train_path = join(self.prePath, "Database\\db2\\train\\train70.npy")
            test_path = join(self.prePath, "Database\\db2\\test\\test30.npy")
        if typedb == 3:
            train_path = join(self.prePath, "Database\\db3\\train\\train70.npy")
            test_path = join(self.prePath, "Database\\db3\\test\\test30.npy")
        
        #Load train and test file
        f = open(train_path, "rb")
        train = np.load(f)
        f.close()
        
        f = open(test_path, "rb")
        test = np.load(f)
        f.close()
        
        folders = os.listdir(feature_path)
        
        #Assign data to array
        
        #read train data
        for i in range(0, self.sz):
            for j in range(0, len(train[i])):
                try:
                    f = open(join(join(feature_path, folders[i]), train[i][j]) + ".npy", "rb")
                    vgg16_feature = np.load(f)
                    self.vgg16_feature_train_list.append(vgg16_feature)
                    f.close()
                except:
                    pass
        #read test data
        for i in range(0, self.sz):
            for j in range(0, len(test[i])):
                try:
                    f = open(join(join(feature_path, folders[i]), test[i][j]) + ".npy", "rb")
                    vgg16_feature = np.load(f)
                    self.vgg16_feature_test_list.append(vgg16_feature)
                    f.close()
                except:
                    pass
        
        self.vgg16_feature_train_list = np.array(self.vgg16_feature_train_list)
        self.vgg16_feature_test_list = np.array(self.vgg16_feature_test_list)
    
        kmeans = KMeans(n_clusters=257, random_state=0).fit(self.vgg16_feature_train_list)
        #save model in oder not to rebuild
        f = open("model", "wb")
        pickle.dump(kmeans, f)
        f.close()
        
    def LoadModel(self):
        f = open("model", "rb")
        self.model = pickle.load(f)
        f.close()
    def Predict(self):
        result = []
        for sample in self.vgg16_feature_test_list:
            result.append(self.model.predict(sample.reshape(1, 4096)))
        
if __name__ == "__main__":
    kmeans = K_means()
    kmeans.ExtractFeature()
    #kmeans.CreateDataBase()
    #kmeans.LoadFeature()
    #kmeans.LoadModel()
    #kmeans.Predict()