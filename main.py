from neuron_netowrk import make_model
#from RBF_model import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    #for neuron_network
    dataset_without_noise = "dataset/dataset_100000.csv"
    make_model(dataset_without_noise)


    #
    # X, y = load_data()
    #
    # # test simple RBF Network with random  setup of centers
    # test(X, y, InitCentersRandom(X))
    #
    # # test simple RBF Network with centers set up by k-means
    # test(X, y, InitCentersKMeans(X))
    #
    # # test simple RBF Networks with centers loaded from previous
    # # computation
    # test(X, y, InitFromFile("centers.npy"))
    #
    # # test InitFromFile initializer
    # test_init_from_file(X, y)



