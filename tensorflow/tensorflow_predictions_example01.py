import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# simple RNN example in tensorflow without using build-in RNN utilities
#hyperparameters

num_epochs = 100
total_series_length = 150
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

# generates a random series of 0 and 1: x = 010011101011
# the RNN will be trained to predict the next digits which are echo_steps in the future
# hence y = 000010011101 (011) dropped
def generateData():
    # 0,1, 150 samples, 50% chance each chosen
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    # shift 3 steps to the left
    y = np.roll(x, echo_step)
    # padding first 3 values with 0
    y[0:echo_step] = 0
    # Gives a new shape to an array without changing its data.
    # The reshaping takes the whole dataset and puts it into a matrix,
    # which later will be sliced up into these mini-batches of length "truncated_backprop_length".
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    #print(x.shape, y.shape)
    return (x, y)

#data = generateData()
#print(data)
#raise SystemExit(0)
x, y = generateData()

# TensorFlow works by first building up a computational graph, that specifies what operations will be done.
# The input and output of this graph are multidimensional arrays, also known as tensors.
# The graph then be executed iteratively in a session, this can either be done on the CPU or GPU

# The two basic TensorFlow data-structures that will be used in this example are placeholders and variables.
# On each run the batch data is fed to the placeholders, which are “starting nodes” of the
# computational graph. Also the RNN-state is supplied in a placeholder, which is saved from the output of the
# previous run.

#Step 2 - Build the Model

#datatype, shape (5, 15) 2D array or matrix, batch size shape for later
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

#and one for the RNN state, 5,4
init_state = tf.placeholder(tf.float32, [batch_size, state_size])


#The weights and biases of the network are declared as TensorFlow variables,
#which makes them persistent across runs and enables them to be updated
#incrementally for each batch.

#3 layer recurrent net, one hidden state

#randomly initialize weights
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
#anchor, improves convergance, matrix of 0s
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)




#Now it’s time to build the part of the graph that resembles the actual RNN computation,
#first we want to split the batch data into adjacent time-steps.

# Unpack columns
# Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)


# Forward pass
# state placeholder
current_state = init_state
# series of states through time
states_series = []


# for each set of inputs
# forward pass through the network to get new state value
# store all states in memory
for current_input in inputs_series:
    # format input
    current_input = tf.reshape(current_input, [batch_size, 1])
    # mix both state and input data
    input_and_state_concatenated = tf.concat([current_input, current_state],1)
    # perform matrix multiplication between weights and input, add bias
    # squash with a nonlinearity, for probabiolity value
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    # store the state in memory
    states_series.append(next_state)
    # set current state to next one
    current_state = next_state


# calculate loss
# second part of forward pass
# logits short for logistic transform
logits_series = [tf.matmul(state, W2) + b2 for state in states_series]
# apply softmax nonlinearity for output probability
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

# measure loss, calculate softmax again on logits, then compute cross entropy
# measures the difference between two probability distributions
# this will return a tensor of the same shape as labels and of the same type as logits
# with the softmax cross entropy loss.
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
# computes average, one value
total_loss = tf.reduce_mean(losses)
# use adagrad to minimize with .3 learning rate

# minimize it with adagrad, not SGD
# great paper http://seed.ucsd.edu/mediawiki/images/6/6a/Adagrad.pdf
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


#visualizer
def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(1, 1, 1)
    plt.cla()
    plt.plot(loss_list)

    plt.draw()
    plt.pause(0.0001)


# Step 3 Training the network
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # interactive mode
    plt.ion()
    # initialize the figure
    plt.figure()
    # show the graph
    plt.show()
    # to show the loss decrease
    loss_list = []

    for epoch_idx in range(num_epochs):
        # generate data at every epoch, batches run in epochs
        # x, y = generateData()
        # initialize an empty hidden state
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)
        # each batch
        for batch_idx in range(num_batches):
            # starting and ending point per batch
            # since weights reoccur at every layer through time
            # These layers will not be unrolled to the beginning of time,
            # that would be too computationally expensive, and are therefore truncated
            # at a limited number of time-steps
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            # run the computation graph, give it the values
            # we calculated earlier
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()

