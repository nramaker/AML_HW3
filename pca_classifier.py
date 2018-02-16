from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import euclidean_distances

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

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

    transformed = pca.transform(images)
    inverse_transformed = pca.inverse_transform(transformed)

    return inverse_transformed, mean_image

def plot_category_errors(category_errors):
    plt.figure(1)
    
    x = np.arange(len(category_errors))
    # print("x {}".format(x))
    # print("errors {}".format(category_errors))
    plt.bar(x, category_errors)
    plt.xticks(x, (x))
        
    plt.title('Error for each image category.')
    plt.legend()
    plt.grid(True)
    plt.show()

def calc_and_plot_mds(mean_images, n_components=20):
    seed = 42
    similarities = euclidean_distances(mean_images)

    mds = manifold.MDS(n_components=n_components, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_

    labels = []
    for i in range(0, len(mean_images)):
        labels.append("Category {}".format(i))

    x = pos[:, 0]
    y = pos[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='cyan', s=100, lw=0)
    ax.set_title('Mean Images Similarities')
    ax.legend(loc='best')
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i],y[i]))
    plt.show()

#main entry
if __name__ == "__main__":
    print(" ##### AML HW2 SVM Classifier  ##### ")
    datafiles = ["./cifar-10-batches-py/data_batch_1","./cifar-10-batches-py/data_batch_2","./cifar-10-batches-py/data_batch_3","./cifar-10-batches-py/data_batch_4","./cifar-10-batches-py/data_batch_5","./cifar-10-batches-py/test_batch"]
    mean_images= []
    image_files_by_category = []
    category_errors = []

    for i in range(0,10):
        #load all images of this class
        print(" ")
        print("### Processing class {}".format(i))
        images = get_class_images_from_files(datafiles, i)
        print("Found {} images of class {}".format(len(images), i))
        image_files_by_category.append(images)

        print("Generating Reduced Images...")
        reduced_images, mean_image = calculate_reduced_images(images, 20)
        error = compute_error_method2(images, reduced_images, i)
        category_errors.append(error)
        mean_images.append(mean_image)

        # show_image(images[0])
        # show_image(reduced_images[0])
        # show_image(error_images[10])
        # show_image(mean_image)

    plot_category_errors(category_errors)
    calc_and_plot_mds(mean_images)


    