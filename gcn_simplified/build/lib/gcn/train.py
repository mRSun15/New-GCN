from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.python import debug as tf_debug

from gcn.utils import *
from gcn.models import GCN, MLP, LanczosGCN

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_lanczos', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
# Some preprocessing
features = preprocess_features(features)
# test_acc_dict = {}
# accuracy_list = []
# duration_list = []

# for degree in range(1, 10):
#     FLAGS.max_degree = degree
    # print(FLAGS.max_degree)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
elif FLAGS.model == 'gcn1':
    support = improved_adj_first(adj, state=True)
    num_supports = 2
    model_func = GCN
elif FLAGS.model == 'gcn2':
    num_supports = 1
    model_func = GCN
    support = [improved_adj_second(adj)]
elif FLAGS.model == 'gcn3':
    support = improved_adj_third(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'gcn_lanczos':
    support = lanczos_adj(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = LanczosGCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    model_func = GCN


# Define placeholders
placeholders = {
'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
'labels_mask': tf.placeholder(tf.int32),
'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
'dropout': tf.placeholder_with_default(0., shape=()),
'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}


# test_parameter = np.linspace(0.1, 10, 100)

# for lambda_ in test_parameter:

# Create model

model = model_func(placeholders, input_dim=features[2][1], node_num = features[2][0],logging=False)
print('input dim is: ',features[2][1])
# Initialize session
sess = tf.Session()


merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('summary/',
                                      sess.graph)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
cost_val = []

# Train model
for epoch in range(FLAGS.epochs):


    t = time.time()
    # Construct feed dictionary
    
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    summary ,outs = sess.run([merged,[model.opt_op, model.loss, model.accuracy]], feed_dict=feed_dict)

    train_writer.add_summary(summary, epoch)
    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
# print("Current Degree is: ", degree)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    # test_acc_dict[degree] = [test_acc,test_duration]
    # accuracy_list.append(test_acc)
    # duration_list.append(test_duration)

# print(test_acc_dict)


# plt.figure()
# x = test_parameter
# x = range(1,10)
# y1 =  accuracy_list
# y2 = duration_list
# plt.subplot(211)
# plt.plot(x,y1)
# plt.grid(True, linestyle = "-.", color = "y", linewidth = "1")
# plt.subplot(212)
# plt.plot(x,y2)
# plt.grid(True, linestyle = "-.", color = "y", linewidth = "1")
# plt.show()