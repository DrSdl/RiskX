import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # implement sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # Hidden layer
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)  # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

            # Output layer
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
            final_outputs = final_inputs * 1  # signals from final output layer

            ### Backward pass ###

            # Output error
            error = y - final_outputs  # Output layer error is the difference between desired target and actual output.

            # Backpropagated error terms
            output_error_term = error * 1

            # Calculate the hidden layer's contribution to the error
            # hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
            hidden_error = output_error_term.dot(self.weights_hidden_to_output.T)

            # hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

            # Weight step (input to hidden)
            X = X[np.newaxis]
            hidden_error_term = hidden_error_term[np.newaxis]
            delta_weights_i_h += X.T.dot(hidden_error_term)
            # Weight step (hidden to output)
            hidden_outputs = hidden_outputs[np.newaxis]
            output_error_term = output_error_term[np.newaxis]
            delta_weights_h_o += hidden_outputs.T.dot(output_error_term)

        # Update the weights
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs * 1  # signals from final output layer

        return final_outputs


def MSE(y, Y):
    return np.mean((y-Y)**2)
