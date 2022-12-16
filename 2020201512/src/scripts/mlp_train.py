# coding = <utf-8>

"""
implement of MLP based on tensorflow

Model Structure:
    Input Layer[0] : X [3*21]
    Hidden Layer[1] : 28 units, Relu
    Output Layer[2] : Y [14], Softmax
"""
##======== packages ========##
from package import *
from utils import load_dataset, random_mini_batches, convert_to_one_hot
##======== functions ========##
def initialize_parameters_MLP(seed=1):
    """
    initialization parameters of MLP model (3-layers)
    
    (Input)  
        Layer[0] u[0] = x with shape(x) = [63,]
    (Hidden) 
        Layer[1] z[1] = w[1]u[0]+b[1] with shape(w[1]) = [28,63], shape(b[1]) = [28,1]
                 u[1] = activate(z[1])
    (Output)
        Layer[2] z[2] = w[2]u[1]+b[2] with shape(w[2]) = [14,28], shape(b[2]) = [14,1]
                 u[2] = activate(z[2])
                 y[2] = u[2] with shape(y) = [14,]
    
    @param seed: random seed, default(1)
    @return: initialized parameters list
    """
    
    tf.random.set_seed(seed)                   # so that your "random" numbers match ours
        
    W1 = tf.compat.v1.get_variable("W1", [28,63], initializer = tf.initializers.GlorotUniform(seed = 1))
    b1 = tf.compat.v1.get_variable("b1", [28,1], initializer = tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable("W2", [14,28], initializer = tf.initializers.GlorotUniform(seed = 1))
    b2 = tf.compat.v1.get_variable("b2", [14,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def forward_propagation_MLP(X, parameters):
    """
    Implements the forward propagation for the model: 
    
    LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    @param X: input dataset placeholder, of shape (input size, number of examples)
    @param parameters: python dictionary containing your parameters "W1", "b1", "W2", "b2"....
    
    @return Z2: the output of the last lINEAR unit (not SOFTMAX)
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = tf.add(tf.matmul(W1,X),b1)                        # Z1 = W1*X + b1
    U1 = tf.nn.relu(Z1)                                    # U1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,U1),b2)                       # Z2 = W2*U1 + b2
    
    return Z2

def compute_cost_MLP(Z2, Y):
    """
    Computes the cost
    
    @param Z2: the output of forward propagation
    @param Y: labels vector placeholder, same shape as U3
    
    @return cost: Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def model_train_MLP(X_train, Y_train, X_test, Y_test, 
                    learning_rate=0.00001, epochs=1500, minibatch_size=32,
                    seed=1, verbose=0):
    """
    training a three-layer tensorflow neural network: 
        LINEAR->RELU->LINEAR->SOFTMAX
    
    @param X_train: training set of shape [63, m], m is the number of training sample size
    @param Y_train: training label of shape [14, m]
    @param X_test: testing set while training with shape [84, m_tst], m_tst is the number of test sample size
    @param Y_test: testing label of shape [14, m_tst]
    
    @param learning_rate: learning rate of the optimization
    @param epochs: number of epochs of training loop
    @param minibatch_size: size of a minibatch
    @param seed: random seed
    @param verbose: 0 to keep silence
                    1 to print the cost of training for 10 epoch
                    2 to print the cost of training & testing for 10 epoch
    
    @return: parameters - after model training 
    @return: cost of train and test if need
    """
    
    tf.random.set_seed(seed)
    ops.reset_default_graph()  # re-run of model without overwriting variables

    n_x, m = X_train.shape
    n_y, _ = Y_train.shape
    train_costs = []  # keep track of the train cost
    test_costs = []  # keep track of the test cost
    
    #---- create the computation graph ----# 
    # create placeholder of shape [n_x, n_y] in computation graph
    tf.compat.v1.disable_eager_execution()
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[n_x,None], name='X')
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[n_y,None], name='Y')
    # parameters initialization
    parameters = initialize_parameters_MLP(seed)
    # forward propagation
    Z2 = forward_propagation_MLP(X, parameters)
    # calculation of cost
    cost = compute_cost_MLP(Z2, Y)
    # construction of optimizer
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
                
    init = tf.compat.v1.global_variables_initializer()  # initial all the variables of tf

    #---- Start the session to compute the tensorflow graph ----#
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        
        # training loop
        for epoch in range(epochs):
            epoch_cost = 0.0  # this turn cost
            num_minibatches = int(m/minibatch_size)
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)  # subsample
            
            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                        
                # run the graph on a minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost/num_minibatches
                
            # printing the cost every epoch
            if verbose == 1 and epoch % 10 == 0:
                print("epoch %i: train cost %f" % (epoch, epoch_cost))
                train_costs.append(epoch_cost)   
                   
            if verbose == 2 and epoch % 10 == 0:
                # calculating test cost
                Z3_2 = forward_propagation_MLP(X, sess.run(parameters))  # computation graph
                cost_2 = compute_cost_MLP(Z3_2, Y)
                with tf.compat.v1.Session() as sess_2:  # session
                    test_cost = sess_2.run(cost_2, feed_dict={X:X_test, Y:Y_test})  
                               
                print("epoch %i: train cost%f, test cost%f" % (epoch, epoch_cost, test_cost))
                train_costs.append(epoch_cost)  
                test_costs.append(test_cost)
                
        # save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")        

        # get accuracy result
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))    

        return parameters, train_costs, test_costs
    

if __name__ == '__main__':
    ## data loading and pre_processing (scale transform)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()
    # Flatten the training and test images
    X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    # Convert training and test labels to one hot matrices
    Y_train_orig = Y_train_orig[0]
    Y_test_orig = Y_test_orig[0]
    Y_train_orig_int = []
    for a in Y_train_orig:
        Y_train_orig_int.append(int(a))
    Y_train_orig_int = np.array(Y_train_orig_int)
    Y_test_orig_int = []
    for a in Y_test_orig:
        Y_test_orig_int.append(int(a))
    Y_test_orig_int = np.array(Y_test_orig_int)
    Y_train = convert_to_one_hot(Y_train_orig_int, 14)
    Y_test = convert_to_one_hot(Y_test_orig_int, 14)
    
    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    
    ## construct model, training & testing
    lr = 0.005

    parameters, train_costs, test_costs = model_train_MLP(X_train, Y_train, X_test, Y_test, 
                                                          learning_rate=lr, epochs=1000,
                                                          verbose=2)
    
    pickle.dump(parameters, open("parameters",'wb')) 

    ## plot the cost during training and testing
    plt.plot(np.squeeze(train_costs), label='train cost')
    plt.plot(np.squeeze(test_costs), label='test cost')
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("MLP train/test cost \n learning rate =" + str(lr))
    plt.grid(True)
    plt.legend()
    plt.show()