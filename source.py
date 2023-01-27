import json
import gzip
import matplotlib.pyplot as plt
from random import shuffle
from random import uniform


def read_labels(filename):
    """Loads a file containing labels.

    Arguments:
        filename (str): The name of the file containing the labels.

    Returns:
        list: A list containing the labels (ints) from the loaded file.

    Asserts:
        AssertionError: The given filename is not a legal MNIST label file.
    """
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big', signed=False)
        assert magic == 2049, filename + ' is not an MNIST label file'
        size = int.from_bytes(f.read(4), byteorder='big', signed=False)
        return list(f.read(size))


def read_images(filename):
    """Loads a file containing images.

    Arguments:
        filename (str): The name of the file containing the images.

    Returns:
        list: A list containing the images (list-of-lists) from the loaded file.

    Asserts:
        AssertionError: The given filename is not a legal MNIST label file.
    """
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big', signed=False)
        assert magic == 2051, filename + ' is not an MNIST label file'
        images, rows, columns = [
            int.from_bytes(f.read(4), byteorder='big', signed=False) for _ in
            range(3)]
        return [[list(f.read(columns)) for _ in range(rows)] for _ in
                range(images)]


def plot_images(images, labels, rows=1, columns=None, prediction=None):
    """Shows a plot of images and corresponding labels. If given 
    predictions, the subplots are colored according to whether or not the 
    predictions were correct.

    Arguments:
        images (list): A list containing images.
        labels (list): A list containing labels.
        rows (int): The number of rows.
        columns (int): The number of columns.
        prediction (list): A list containing predicted labels.
    """
    if not columns:
        columns = len(labels) // rows
    for i in range(columns * rows):
        correct = not prediction or prediction[i] == labels[i]
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        if correct:
            plt.title(labels[i])
            plt.imshow(images[i], cmap='Greys')
        else:
            plt.title(f'{prediction[i]}, correct {labels[i]}', color='red')
            plt.imshow(images[i], cmap='Reds')
    plt.tight_layout(0.5)
    plt.savefig('plots/ImagesPredictionsPlot.png', dpi=800)
    plt.show()


def linear_load(file_name):
    """Loads a linear classifier network = (A, b) using JSON. 

    Arguments:
        file_name (str): The name of the file containing the network.

    Returns:
        network (tuple):  The first element in the tuple is a weight matrix, 
        and the second element is a bias vector.
    """
    with open(file_name) as f:
        return json.load(f)


def linear_save(file_name, network):
    """Saves a linear classifier network = (A, b) using JSON. Overwrites file,
    if the filename already exists.

    Arguments:
        file_name (str): The name of the file the network is to be saved as.
        network (tuple): A tuple where the first element is a weight matrix, 
        and the second element is a bias vector.
    """
    with open(file_name, 'w') as f:
        json.dump(network, f)


def image_to_vector(image):
    """Converts an image (list-of-lists) with integer pixel values in the range 
    [0, 255] to an image vector (list) with pixel values being floats in the 
    range [0, 1].

    Arguments:
        image (list-of-lists): An image.

    Returns:
        list: A list with pixel values being floats in the range [0, 1].
    """
    image_vector = []
    for row in image:
        image_vector += row
    for pixel_index in range(len(image_vector)):
        image_vector[pixel_index] /= 255.0
    return image_vector


def vector_to_image(image_vector):
    image_vector_copy = image_vector[:]
    image = []
    for i in range(28):
        image.append([None] * 28)
    count = 0
    for i in range(28):
        for j in range(28):
            image[i][j] = image_vector_copy[count]
            count += 1
    return image


def vectors_to_images(image_vectors):
    images = []
    for image in transpose(image_vectors):
        images.append(vector_to_image(image))
    return images


def add(U, V):
    """Adds two lists element-wise.

    Arguments:
        U (list): A vector.
        V (list): A vector.

    Returns:
        list: U + V.
        
    Asserts:
        AssertionError: The lengths of the input lists are not the same.
    """
    assert len(U) == len(V), 'Vectors\' lengths are of different sizes'
    return [sum(x) for x in zip(U, V)]


def sub(U, V):
    """Subtracts one list (V) from another list (U) element-wise.

    Arguments:
        U (list): A vector.
        V (list): A vector.

    Returns:
        list: U - V.
        
    Asserts:
        AssertionError: The lengths of the input lists are not the same.
    """
    assert len(U) == len(V), 'Vectors\' lengths are of different sizes'
    return [U[i] - V[i] for i in range(len(U))]


def scalar_multiplication(scalar, V):
    """Multiplies a vector with a scalar.

    Arguments:
        scalar (int or float): A scalar.
        V (list or tuple): A vector.

    Returns:
        list: The vector multiplied by the scalar.
        
    Asserts:
        AssertionError: The scalar is neither an int nor a float.
        AssertionError: The vector V is neither a list nor a tuple.
    """
    assert isinstance(scalar, (int, float)), 'Scalar is neither int nor float'
    assert isinstance(V, (list, tuple)), 'V is neither list nor tuple'
    return [scalar * i for i in V]


def multiply(V, M):
    """Multiplies a vector with a matrix.

    Arguments:
        V (list): A vector.
        M (list-of-lists): A matrix.

    Returns:
        list: A vector containing the vector/matrix product.

    Asserts:
        AssertionError: Dimensions of vector and matrix are not compatible.
        AssertionError: V and/or M is not a list.
    """
    assert len(V) == len(M), 'Dimensions of V and M are not compatible'
    assert all(isinstance(i, list) for i in [V, M]), 'V and/or M is not a list'
    zip_M = list(zip(*M))
    return [sum(ele_V * ele_M for ele_V, ele_M in zip(V, col_M)) for col_M in
            zip_M]


def transpose(M):
    """Transposes a matrix M.

    Arguments:
        M (list-of-lists): A list containing equal length lists.

    Returns:
        list: M transposed.
    
    Asserts:
        AssertionError: M must be a list.
        AssertionError: The lengths of the lists in M must be equal. 
    """
    assert isinstance(M, list), 'M is not a list'
    assert len({len(row) for row in M}) == 1, 'List lengths in M must be equal'
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def mean_square_error(U, V):
    """Computes the mean squared error between two vectors U and V.

    Arguments:
        U (list): A vector.
        V (list): A vector.

    Returns:
        float: The mean squared error between two vectors.

    Asserts:
        AssertionError: The two vectors do not have the same length.
    """
    assert len(U) == len(V), "U and V must have the same length"
    return sum([x ** 2 for x in sub(U, V)]) / len(U)


def argmax(V):
    """Returns the index containing the largest value of a list V.

    Arguments:
        V (list): A list of ints or floats.

    Returns:
        int: The index of the largest value in V.
    """
    return V.index(max(V))


def categorical(label, classes=10):
    """Takes a label (int from 0 to (classes-1)) and returns a vector of length 
    classes, with all elements being zero, except the element with index label, 
    which is set to one.

    Arguments:
        label (int): An integer from 0 to one less than classes.
        classes (int): The length of the output vector.

    Returns:
        list: A list with all elements set to zero, except element at index 
        label, which is set to one.

    Asserts:
        AssertionError: The amount of classes must be a positive int.
        AssertionError: The label is not within range(0, classes).
    """
    assert isinstance(classes,
                      int) and classes > 0, 'classes must be a positive int'
    assert label in range(classes), 'The label must be within range(classes)'
    categorical_vector = [0] * classes
    categorical_vector[label] = 1
    return categorical_vector


def predict(network, image):
    """Returns the predicted value for one given image using a given network.
    It is calculated as xA + b, where x is an image vector, A is a weight 
    matrix, and b is a bias vector.

    Arguments: network (tuple): Tuple (A,b) containing a weight matrix and a
    bias vector.
    image (list): An image vector.

    Returns:
        list: prediction for image, xA + b
    """
    A, b = network
    return add(multiply(image, A), b)


def evaluate(network, image_vectors, labels):
    """Evalutes a network's prediction accuracy and cost for a given set of 
    images and corresponding labels.

    Arguments:
        network (tuple): A tuple of a weight matrix A and a bias vector b.
        image_vectors (list): a list of n lists containing 784 pixel values each
        labels (list): a list of n labels (ints).

    Returns:
        tuple:
            predictions: A list of predicted values.
            cost: The average of mean square errors over all input-output pairs.
            accuracy: Fraction of correctly predicted values.
    """
    prediction_vectors = [(predict(network, image)) for image in image_vectors]
    predictions = [argmax(prediction_vector) for prediction_vector in
                   prediction_vectors]

    label_categorical_vectors = [categorical(label) for label in labels]

    cost_list = list(
        map(mean_square_error, prediction_vectors, label_categorical_vectors))
    cost = sum(cost_list) / len(cost_list)

    accuracy = sum(
        (predictions[i] == labels[i] for i in range(len(predictions)))) / len(
        predictions)

    return predictions, round(cost, 3), accuracy


def plot_weights(images, labels, rows=2, columns=None, cmap='plasma'):
    """Shows a plot of the given weights with corresponding labels. The plot 
    will at most show (rows * (length of labels) // columns) images and no more 
    than the amount of images in the input. The plot is saved as 
    'WeightsPlot.png'. If the filename already exists, it is overwritten.

    Arguments:
        images (list): A list of n lists each containing 28 lists of 28 pixel
        values.
        labels (list): A list of n labels (the different categories).
        rows (int): The number of rows in the plot.
        columns (int): The number of columns in the plot.
        cmap (str): The colormap for the plot.

    Returns:
        list: A list containing the labels (ints) from the loaded file.
    """
    if columns is None:
        columns = len(labels) // rows
    assert all(isinstance(i, int) and i > 0 for i in
               [rows, columns]), 'rows and columns must be positive ints'

    for i, (image, label) in enumerate(
            zip(images[:columns * rows], labels[:columns * rows])):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(label)
        im = plt.imshow(image, cmap=cmap)

    plt.tight_layout(pad=0.5, rect=(0.1, 0, 1, 1))
    colorbar_ax = plt.axes(
        [0.05, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    plt.colorbar(im, cax=colorbar_ax, ticks=[])
    plt.savefig('plots/WeightsPlot.png', dpi=800)
    plt.show()


def create_batches(values, batch_size):
    """Randomly shuffles a list of values and then partions them into batches 
    of size batch_size. The last element in the returned list might have a 
    smaller batch size, depending on the length of values and batch_size.

    Arguments:
        values (list): A list of values.
        batch_size (int): The size of the batches.

    Returns:
        list: A list of the generated batches.
    """
    shuffle(values)
    l1 = len(values) // batch_size
    l2 = len(values) % batch_size
    l3 = [values[batch_size * i:batch_size * (i + 1)] for i in range(l1)]
    if l2 != 0:
        l3 += [values[-l2:]]
    return l3


def update(network, images, labels):
    """Updates the given network, optimizing for the cost function (MSE) going 
    one step towards the gradient descent based on the given images and labels.

    Arguments:
        network (tuple): A weight matrix A and a bias vector b.
        images (list): A list of n image vectors.
        labels (list): A list of n labels (int).
    """
    A, b = network
    b_sum = [0] * 10
    A_sum = [[0] * 10 for _ in range(784)]
    n = len(images)
    constant = 0.1 * (1 / n) * 2 / 10

    for x, label in zip(images, labels):
        a = predict(network, x)
        y = categorical(label)

        for category in range(len(y)):
            b_sum[category] += (a[category] - y[category])
            for pixel in range(len(x)):
                A_sum[pixel][category] += x[pixel] * (a[category] - y[category])

    for category in range(len(y)):
        b[category] -= constant * b_sum[category]
        for pixel in range(len(x)):
            A[pixel][category] -= constant * A_sum[pixel][category]


def dim(a):
    """Given a list of possibly multiple lists, it returns the dimensions of 
    said list.
    
    Arguments:
        a (list): A list possibly containing other lists (and so on).
        
    Returns:
        list: A list where each element is the amount of elements at the level 
        corresponding to the index in a.
    """
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def learn(images, labels, epochs, batch_size, img_test, lab_test):
    """Generates a random network and updates it 
    (epochs*(amount of images)/batch_size) times. 

    Arguments:
        images (list): A list of n image vectors.
        labels (list): A list of n labels (ints).
        epochs (int): The number of epochs.
        batch_size (int): The size of each partitioning in each epoch.
        img_test (list): The images the network is evaluated upon
        (the first 100).
        lab_test (list): The labels the network is evaluated upon
        (the first 100).

    Returns:
        tuple:
            best_network: The network with the lowest cost evaluated upon the
            first 100 test images.
            best_cost: The lowest cost evaluated upon the first 100 test images.
            best_accuracy: The accuracy corresponding to the network with the
            lowest cost.
    """
    A = [[uniform(0, 1 / 784) for _ in range(10)] for _ in range(784)]
    b = [uniform(0, 1) for _ in range(10)]
    best_cost = None

    # uncomment line below, if plot_graphs function is to be used
    accuracy_cost_vector = []

    for epoch in range(epochs):
        print(f"Epoch number {epoch + 1} initiated.")
        batches = create_batches(list(zip(images, labels)), batch_size)

        for batch in batches:
            img, lab = zip(*batch)
            update((A, b), img, lab)
            prediction, cost, accuracy = evaluate((A, b), 
                                                  img_test[:100],
                                                  lab_test[:100])

            if not best_cost or best_cost > cost:
                best_cost = cost
                best_accuracy = accuracy  # best based on cost
                best_network = (A, b)

        # uncomment line below and in return statement, if plot_graphs function
        # is to be used
        accuracy_cost_vector.append((accuracy, cost))

        print(f"Epoch number {epoch + 1} finished.")

    return best_network, best_cost, best_accuracy, accuracy_cost_vector


def plot_graphs(accuracy, cost):
    """Shows a plot containing the subplots accuracy/update and cost/update. A
    dotted line is added for the best accuracy and cost, respectively. The plot
    is saved with the filename 'AccuracyCostPlot.png'. If the filename already
    exists, it is overwritten.

    Arguments:
        accuracy (list): A list containing accuracy values.
        cost (list): A list containing cost values.
    """
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(range(len(accuracy)), accuracy)
    axs[0].set_ylabel('Accuracy')
    axs[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axs[0].axhline(y=max(accuracy), linestyle='--', color='r')
    axs[0].text((len(accuracy) * 0.6), max(accuracy) * 0.6,
                f'Best accuracy: {max(accuracy)}', color="red")

    axs[1].plot(range(len(accuracy)), cost)
    axs[1].set_ylabel('Cost')
    axs[1].set_xlabel('Network updates')
    axs[1].axhline(y=min(cost), linestyle='--', color='r')
    axs[1].text((len(accuracy) * 0.6), max(cost) * 0.6,
                f'Best cost: {min(cost)}', color="red")
    axs[1].set_yticks([0, 0.05, 0.1, 0.15])

    plt.margins(x=0)
    fig.suptitle('Accuracy and cost evaluated on the first 100 test images',
                 fontsize=12)
    plt.savefig('plots/AccuracyCostPlot.png', dpi=800)
    plt.show()
