import tensorflow as tf
sess = tf.InteractiveSession()

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

# with tf.name_scope(self.name):
operator = tf.SparseTensor(indices=[[0, 0], [1, 1], [2,2], [0,2], [2,0], [1,2], [2,1]], values=[0.1, 0.2, 0.5, -0.1, -0.1, 0.3,0.3], dense_shape=[3, 3])
print(operator.eval())
starting_vector = tf.constant([[2],[-0.4], [0.7]])
Vm = []
beta = []
alpha = []
sparse_inputs=False
if sparse_inputs:
    v0 = tf.sparse_tensor_to_dense(starting_vector)
else:
    v0 = starting_vector
v0 = tf.nn.l2_normalize(tf.reshape(v0,[-1,1]))

Vm.append(v0)
k = 2


for j in range(1, k):
#     # print(j)
    w = dot(operator, Vm[j-1],sparse = True)
    alpha_j = dot(tf.transpose(w),Vm[j-1], sparse = False)
    print(alpha_j.eval())
    print(Vm[j-1].eval())
    new = tf.multiply(alpha_j, Vm[j-1])
    new_v = w - tf.multiply(alpha_j, Vm[j-1])
    print(new_v.eval())
    alpha.append(alpha_j)
    if j > 1:
        new_v -= tf.multiply(beta[j-2],Vm[j-2]) 
    
    beta_j = tf.norm(new_v)
    print(beta_j.eval())
    
    beta.append(beta_j)
    new_v = tf.multiply(1.0/beta_j,new_v)
    print(new_v.eval())
    Vm.append(new_v)
#     # print("finish Lanczos")
# return Vm
VV = tf.concat(Vm, 1)
print(VV.eval())