# result visualizing

from package import *
from mlp_train import forward_propagation_MLP
def predict_MLP(X, parameters):
    """
    prediction on test sample X

    @param X: sample of shape [63, m] as m is the samples number
    @param parameters: MLP's parameters

    @return: predict labels Y of shape [1, m]
    """
    # load parameter
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    # construct computation graph
    x = tf.compat.v1.placeholder("float", [63, X.shape[1]])
    z2 = forward_propagation_MLP(x, params)
    Y_pred = tf.nn.softmax(logits=z2, axis=0)
    p = tf.argmax(Y_pred, axis=0)
    # create session and run
    with tf.compat.v1.Session() as sess:
        y_pred = sess.run(p, feed_dict={x: X})
    return y_pred