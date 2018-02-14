from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

#import pandas as pd

# def compute_mean_image(images):
#     # print("len(images) {}".format(len(images)))
#     count = len(images)

#     sum_of_images = np.zeros(3072)
#     for image in images:
#         sum_of_images +=image
#     mean_image = (sum_of_images/count).astype(int)
#     # return mean_image
#     return mean_image

def compute_princ_comps(images):
    pass

def compute_distance(vector1, vector2):
    pass

def create_2D_map(vector):
    pass

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_class_images_from_files(files, class_id):
    images = []
    for file in files:
        # print("### Loading images of class {} from file {}".format(class_id, file))
        content = unpickle(file)
        labels= content[b'labels']
        img_data= content[b'data']
        
        #filter out the non-class_ids 
        mask = (np.array(labels) == class_id)
        img_data = img_data[mask]

        images = images + list(img_data)
    #mean_image = compute_mean_image(images)
    return (images)

def show_image(image_vector):
    arr = np.array(image_vector)
    redarr = arr[0:1024]
    greenarr = arr[1024:2048]
    bluearr = arr[2048:]
    rgbArray = np.zeros((32,32,3), 'uint8')
    rgbArray[..., 0] = redarr.reshape((32,-1))
    rgbArray[..., 1] = greenarr.reshape((32,-1))
    rgbArray[..., 2] = bluearr.reshape((32,-1))
    image = Image.fromarray(rgbArray)
    image.show()


#main entry
if __name__ == "__main__":
    print(" ##### AML HW2 SVM Classifier  ##### ")
    datafiles = ["./cifar-10-batches-py/data_batch_1","./cifar-10-batches-py/data_batch_2","./cifar-10-batches-py/data_batch_3","./cifar-10-batches-py/data_batch_4","./cifar-10-batches-py/data_batch_5","./cifar-10-batches-py/test_batch"]
    image_means = []
    image_files_by_category = []

    pca = PCA(n_components=20, svd_solver='auto')
    for i in range(0,1):
        #load all images of this class
        images = get_class_images_from_files(datafiles, i)
        print("Found {} images of class {}".format(len(images), i))
        # show_image(mean_image)
        image_files_by_category.append(images)
        print("")

        pca.fit(np.array(images))
        print("variance ratios {}".format(pca.explained_variance_ratio_)) 
        print("singular values {}".format(pca.singular_values_))   
        print("components {}".format(pca.components_))
        print("mean_ {}".format(pca.mean_))
        mean_image = pca.mean_

        print("###transforming images")
        reduced_images = pca.fit_transform(images)
        print("we got back {} images".format(reduced_images.shape))
        print("reduced_image[0] {}".format(reduced_images[0]))
        image_means.append(mean_image)



    # data1 = unpickle("./cifar-10-batches-py/data_batch_1")
    # # print(data1.keys())
    # filenames =data1[b'filenames']
    # labels= data1[b'labels']
    # print("First image label {}".format(labels[0]))
    # first_image = data1[b'data'][0]

    # show_image(first_image)
    