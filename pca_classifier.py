def compute_mean_image(images, class_id):
    pass

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
    pass

#main entry
if __name__ == "__main__":
    print(" ##### AML HW2 SVM Classifier  ##### ")
    datafiles = ["./cifar-10-batches-py/data_batch_1","./cifar-10-batches-py/data_batch_2","./cifar-10-batches-py/data_batch_3","./cifar-10-batches-py/data_batch_4","./cifar-10-batches-py/data_batch_5"]
    data1 = unpickle("./cifar-10-batches-py/data_batch_1")
    # print(data1.keys())
    filenames =data1[b'filenames']
    labels= data1[b'labels']
    first_image = data1[b'data'][0]
    print(first_image)
    
    # train_and_predict()
    # show_plots()