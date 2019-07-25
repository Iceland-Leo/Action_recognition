import tensorflow as tf
layer_num = 5
hidden_num = 1024
fc1_node = 512


def variable_summaries(var, name):
    with tf.name_scope("summaries"):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean/" + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev" + name, stddev)


'''
def inference(input_tensor, class_num, train=True, reuse=False):
    with tf.variable_scope("lstm", reuse=reuse):
        stacked_lstm_cell = []
        for i in range(layer_num):
            lstm_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
            stacked_lstm_cell.append(lstm_cell)

        multi_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm_cell)

        outputs, states = tf.nn.dynamic_rnn(multi_cell, input_tensor, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]

    with tf.variable_scope("full-connect1", reuse=reuse):
        fc1_weights = tf.get_variable("weights", [hidden_num, class_num],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc1_weights, "full-connect1/fc1_weights")
        fc1_biases = tf.get_variable("biases", [class_num], initializer=tf.constant_initializer(0.0))
        variable_summaries(fc1_biases, "full-connect1/fc1_biases")

        logits = tf.matmul(outputs, fc1_weights) + fc1_biases
        softmax_logits = tf.nn.softmax(logits)

    return logits, softmax_logits
'''


'''
def inference(input_tensor, class_num, train=True, reuse=False):
    with tf.variable_scope("lstm", reuse=reuse):
        stacked_lstm_cell = []
        for i in range(layer_num):
            lstm_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
            stacked_lstm_cell.append(lstm_cell)

        multi_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm_cell)

        outputs, states = tf.nn.dynamic_rnn(multi_cell, input_tensor, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]

    with tf.variable_scope("full-connect1", reuse=reuse):
        fc1_weights = tf.get_variable("weights", [hidden_num, fc1_node],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc1_weights, "full-connect1/fc1_weights")
        fc1_biases = tf.get_variable("biases", [fc1_node], initializer=tf.constant_initializer(0.0))
        variable_summaries(fc1_biases, "full-connect1/fc1_biases")

        fc1 = tf.nn.relu(tf.matmul(outputs, fc1_weights) + fc1_biases)

        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("full-connect2", reuse=reuse):
        fc2_weights = tf.get_variable("weights", [fc1_node, class_num],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc2_weights, "full-connect2/fc2_weights")
        fc2_biases = tf.get_variable("biases", [class_num], initializer=tf.constant_initializer(0.0))
        variable_summaries(fc2_biases, "full-connect2/fc2_biases")

        logits = tf.matmul(fc1, fc2_weights) + fc2_biases
        softmax_logits = tf.nn.softmax(logits)
    
        
    return logits, softmax_logits
'''


def inference(input_tensor, class_num, regularizer, train=True, reuse=False):
    with tf.variable_scope("lstm", reuse=reuse):
        stacked_lstm_cell = []
        for i in range(layer_num):
            lstm_cell = tf.contrib.rnn.LSTMCell(hidden_num, use_peepholes=True)
            if train:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.6)
            stacked_lstm_cell.append(lstm_cell)

        multi_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm_cell)

        outputs, states = tf.nn.dynamic_rnn(multi_cell, input_tensor, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]

    with tf.variable_scope("full-connect1", reuse=reuse):
        fc1_weights = tf.get_variable("weights", [hidden_num, fc1_node],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc1_weights, "full-connect1/fc1_weights")
        fc1_biases = tf.get_variable("biases", [fc1_node], initializer=tf.constant_initializer(0.0))
        variable_summaries(fc1_biases, "full-connect1/fc1_biases")

        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weights))

        fc1 = tf.nn.relu(tf.matmul(outputs, fc1_weights) + fc1_biases)

        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope("full-connect2", reuse=reuse):
        fc2_weights = tf.get_variable("weights", [fc1_node, class_num],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        variable_summaries(fc2_weights, "full-connect2/fc2_weights")
        fc2_biases = tf.get_variable("biases", [class_num], initializer=tf.constant_initializer(0.0))
        variable_summaries(fc2_biases, "full-connect2/fc2_biases")

        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        logits = tf.matmul(fc1, fc2_weights) + fc2_biases
        softmax_logits = tf.nn.softmax(logits)

    return logits, softmax_logits

