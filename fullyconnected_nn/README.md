# fully connected nueral networks
with same manner to <a href=https://github.com/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/tree/main/Onelayer_nn>here</a>
we use fully connected layer for mnist dataset but here fully connected have been written in classes in order to make it easier extending it to more than one layer for example
the below code is the one layer network class
```
    class nn1:
  def __init__(self, sizes):
    self.sizes = sizes
    self.weights = [np.random.uniform(-2, 2,size=(sizes[i-1], sizes[i]))
     for i in range(1, len(self.sizes))]
    self.biases = [np.random.uniform(-2, 2,size=(sizes[i],1))
     for i in range(1, len(self.sizes))]

  def feed_forward(self,input):
     a = input
     pre_activations = []
     activations = [a]
     z = np.dot(self.weights[0].T, input) + self.biases[0]
     a = Softmax(z)
     pre_activations.append(z)
     activations.append(a)
     return a ,pre_activations,activations

  def compute_deltas(self, pre_activations, y_true, y_pred):
      delta_L = (y_pred-y_true)
      deltas = [0] * (len(self.sizes) - 1)
      deltas[-1] = delta_L
      return deltas
  def backpropagate(self, deltas, pre_activations, activations):
        dW = []
        db = []
        dW0_l = np.dot(deltas[0], activations[0].T).T
        db0_l = deltas[0]
        dW.append(dW0_l)
        db.append(db0_l)
        return dW, db
  def train(self,alpha,dW,db):
      for i in range(len(dW)):
          self.weights[i] -= alpha*dW[i]
      for i in range(len(db)):
          self.biases[i] -= alpha*db[i]
```
and this is its training process
```
    yte = []
ytr = []
nuenet1 = nn1([784,10])
alpha = 0.01
for _ in range(100):
   decay_rate = 0.95
   alpha *= decay_rate
   for i in range(len(train_x)):
       A0 = train_x[i].flatten().reshape((-1, 1))
       A0 = A0 / 255
       arr = nuenet1.feed_forward(A0)
       yhat =  np.zeros((10, 1))
       yhat[train_y[i]][0] = 1
       deltas = nuenet1.compute_deltas(arr[1],yhat,arr[0])
       dW, db = nuenet1.backpropagate(deltas,arr[1],arr[2])
       nuenet1.train(alpha,dW,db)
   yte.append(accuracy(test_x,test_y,nuenet1))
   ytr.append(accuracy(train_x,train_y,nuenet1))
   print(accuracy(test_x,test_y,nuenet1))
   print('=============================')
```
