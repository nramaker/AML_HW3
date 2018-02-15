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

def compute_eigens(pca):
    for eigenvalue, eigenvector in zip(pca.explained_variance_, pca.components_):    
        print("")
        print("Eigenvector: {}".format(eigenvector))
        print("EigenValue: {}".format(eigenvalue))
        print("Computed EigenValue: {}".format(np.dot(eigenvector.T, np.dot(pca.get_covariance(), eigenvector))))

# def compute_error_method1(pca):
#     accuracy = 0.0
#     # print("explained variances {}".format(pca.explained_variance_))
#     # print("explained variance ratios {}".format(pca.explained_variance_ratio_))
#     for val in pca.explained_variance_ratio_:
#         accuracy+=val
#     print("Error of {} using Method 1".format(1-accuracy))

def compute_error_method2(images, reduced_images, class_id):
    print("Calculating Errors using Method 2...")

    error_vector = np.array(images) - np.array(reduced_images)
    N =len(error_vector)
    squares = []
    print("Computing squares of errors...")
    for i in range(0, len(error_vector)):
        squares.append(list(map(lambda x: x*x, error_vector[i])))
    print("Computing sums of squares...")
    sums = list(map(lambda x: sum(x), squares))
    error = sum(sums)/N
    print("Found mean error of {} for class_id {} using Method 2".format(error, class_id))
    return error

def calculate_reduced_images(images, n_components=20):
    pca = PCA(n_components)
    pca.fit(np.array(images))
    mean_image = pca.mean_
    eigenvectors = pca.components_
    diffs = images - mean_image
    
    # reduced_images = []
    # for i in range(0, len(images)):
    #     adjustment = 0
    #     for j in range(0, n_components):
    #         u = eigenvectors[j]
    #         adjustment += u.T*diffs[i]*u 
    #     reduced_images.append(mean_image + adjustment)

    transformed = pca.transform(images)
    print("transformed.shape {}".format(transformed.shape))
    inverse_transformed = pca.inverse_transform(transformed)
    print("inverse_transformed.shape {}".format(inverse_transformed.shape))

    return inverse_transformed, mean_image

#main entry
if __name__ == "__main__":
    print(" ##### AML HW2 SVM Classifier  ##### ")
    datafiles = ["./cifar-10-batches-py/data_batch_1","./cifar-10-batches-py/data_batch_2","./cifar-10-batches-py/data_batch_3","./cifar-10-batches-py/data_batch_4","./cifar-10-batches-py/data_batch_5","./cifar-10-batches-py/test_batch"]
    image_means = []
    image_files_by_category = []

    # pca = PCA(n_components=20)
    for i in range(0,1):
        #load all images of this class
        print(" ")
        print("### Processing class {}".format(i))
        images = get_class_images_from_files(datafiles, i)
        print("Found {} images of class {}".format(len(images), i))
        # show_image(mean_image)
        image_files_by_category.append(images)
        # print("")

        print("Generating Reduced Images...")
        reduced_images, mean_image = calculate_reduced_images(images, 3072)
        print("Mean image[:10] {}".format(mean_image[:10]))

        # man_mean_image = compute_mean_image(images)
        # print("Manual Mean image[:10] {}".format(man_mean_image[:10]))
        # print("Generated {} Reduced Images of class {}".format(len(reduced_images), i))
        # print("Calculating Errors...")
        # error_images = np.array(images) - np.array(reduced_images)
        compute_error_method2(images, reduced_images, i)


        show_image(images[0])
        show_image(reduced_images[0])
        # show_image(error_images[10])
        # show_image(mean_image)



    