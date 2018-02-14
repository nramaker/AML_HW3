
from PIL import Image
import numpy as np
#import pandas as pd

def compute_mean_image(images):
    # print("len(images) {}".format(len(images)))
    count = len(images)


    sum_of_images = np.zeros(3072)
    # sum_of_images = [sum(x.T) for x in images]
    for image in images:
        sum_of_images +=image
    # print("images[0] {}".format(images[0]))
    # print("images[1] {}".format(images[1]))
    # print("images[2] {}".format(images[2]))
    # print("sum_of_images {}".format(sum_of_images))
    mean_image = (sum_of_images/count).astype(int)
    # return mean_image
    return mean_image

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
        print("### Loading images of class {} from file {}".format(class_id, file))
        content = unpickle(file)
        labels= content[b'labels']
        img_data= content[b'data']
        
        #filter out the non-class_ids 
        mask = (np.array(labels) == class_id)
        img_data = img_data[mask]

        images = images + list(img_data)
    mean_image = compute_mean_image(images)
    return (images, mean_image)

def show_image(image_vector):
    arr = np.array(image_vector)
    redarr = arr[0:1024]
    greenarr = arr[1024:2048]
    bluearr = arr[2048:]
    # print("red array {}".format(redarr.shape))
    # print("green array {}".format(greenarr.shape))
    # print("blue array {}".format(bluearr.shape))
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
    for i in range(0,10):
        #load all images of this class
        images, mean_image = get_class_images_from_files(datafiles, i)
        print("mean images for class {} is {}".format(i, mean_image))
        show_image(mean_image)
        print("Found {} images of class {}".format(len(images), i))
        print("")

    data1 = unpickle("./cifar-10-batches-py/data_batch_1")
    # print(data1.keys())
    filenames =data1[b'filenames']
    labels= data1[b'labels']
    print("First image label {}".format(labels[0]))
    first_image = data1[b'data'][0]

    show_image(first_image)
    