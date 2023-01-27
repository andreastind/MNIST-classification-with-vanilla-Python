from source import (plot_images, 
                    plot_graphs,
                    plot_weights,
                    learn, 
                    read_images, 
                    read_labels, 
                    image_to_vector,
                    vectors_to_images,
                    evaluate, 
                    linear_save, 
                    linear_load)


##### LOAD DATA #####

# load training data
labels_60k = read_labels('data/train-labels-idx1-ubyte.gz')
images_60k = read_images('data/train-images-idx3-ubyte.gz')

# load test data
images_10k = read_images('data/t10k-images-idx3-ubyte.gz')
labels_10k = read_labels('data/t10k-labels-idx1-ubyte.gz')

# convert images to vectors
images_60k_vec = [image_to_vector(image) for image in images_60k]
images_10k_vec = [image_to_vector(image) for image in images_10k]


##### TRAIN AND SAVE MODEL #####

# train network (and evaluate for accuracy graph) 
best_network, best_cost, best_accuracy, accuracy_cost_vector = learn(
    images=images_60k_vec,
    labels=labels_60k, 
    epochs=5, 
    batch_size=100, 
    img_test=images_10k_vec, 
    lab_test=labels_10k
)

# save network to file
linear_save("model/mnist_linear.weights", best_network)

# load network from file
#best_network = linear_load('model/mnist_linear.weights')


##### INSPECTION OF RESULTS #####

# choose number of images for visual inspection
n_img_inspect = 20


# load images and labels for inspection
images_inspect = images_10k_vec[0:n_img_inspect]
labels_inspect = labels_10k[0:n_img_inspect]


# show and save accuracy and cost graphs during model training
plot_graphs(*(zip(*accuracy_cost_vector)))


# evaluate inspection images using network
pred_test, cost_test, acc_test = evaluate(best_network, 
                                          images_inspect, 
                                          labels_inspect)


# show and save inspection plot
plot_images(images_10k[:n_img_inspect], 
            labels_10k[:n_img_inspect], 
            prediction=pred_test, 
            rows=4)


# show and save model weight plot
plot_weights(images=vectors_to_images(best_network[0]), 
             labels=range(0, 10), 
             rows=2)

