# handwritten_digit_recognition_from_scratch
in this repository i tried to solve handwritten digits recognition problem with different nueral network architecture.

- without using common NNs libraries like tensorflow or pytorch just with pure code
- mnist dataset has been used
- gradient descent and backpropagation method have been used

## dumbest network
probably the dumbest network for such a task is fully connecting the input and the output layer without any hidden layer.<br />
by flattening the $`28 \times 28`$ grey scale image we would have a $`784 \times 1`$ vector as an input,
so input layer $`A_0`$ would have 784 nodes and the output layer $`A_1`$ would have 10 nodes each one correspond to probablity of being a digit,
notice that the predicted label would be equal to $`argmax(A_1)`$.<br />
### training
as there is no hidden layer in this network only two vector would determine the predicted label, a $`784 \times 10`$ weight vector(W) and a $`10 \times 1`$ bias vector(B),
the output would be $`A_1 = \sigma{(W^T \times A_0 + B)}`$ <br />
according to gradient descent algorithm first of all W and B should be initialized randomly and get updated to minimize the cost function,
where $`Cost = (A_1 - \hat{A_1})^2`$ and update rules are $`W^T = W^T + r \times \frac{dCost}{dW}`$ and $`B = B + r \times \frac{dCost}{dB}`$, so we would have: <br />
$` Cost = (A_1 - \hat{A_1})^2 `$ <br />
$`\frac{dCost}{dW} = 2 \times (A_1 - \hat{A_1}) \times \frac{dA_1}{dW}`$ <br />
$`\frac{dCost}{dW} = 2 \times (A_1 - \hat{A_1}) \times A_0 \times \sigma ^ \prime{(W^T \times A_0 + B)}`$ <br />
$`\frac{dCost}{dB} = 2 \times (A_1 - \hat{A_1}) \times \frac{dA_1}{dB}`$ <br />
$`\frac{dCost}{dB} = 2 \times (A_1 - \hat{A_1}) \times \sigma ^ \prime{(W^T \times A_0 + B)}`$ <br />
so the update rules would change to $`W^T = W^T + r \times (A_1 - \hat{A_1}) \times A_0`$ and $`B = B + r \times (A_1 - \hat{A_1})`$, <br />
notice that like $`{A_1}`$ Cost and $`\hat{A_1}`$ also are $`10 \times 1`$ vectors,
and all elements of $`\hat{A_1}`$ are zero except the element with label index which is one.
so we can write the train our model by iterating over training set,
here it is the code:
```
def train(train_X,train_y,WT,B):
    for i in range(len(train_X)):
        # print("episode number: "+ str(i))
        A0 = train_X[i].flatten().reshape((-1, 1))
        A0 = A0 / 255
        A1 = np.vectorize(sigmoid)(np.matmul(WT,A0) + B)
        A1hat =  np.zeros((10, 1))
        A1hat[train_y[i]][0] = 1
        cost = A1-A1hat
        activation_func_deriv = A1 - A1**2
        r = 0.05
        for j in range(10):
            for k in range(784):
                WT[j][k] = WT[j][k] - r * (A0[k]) * (activation_func_deriv[j]) * (cost[j])
            B[j] = B[j] - r * (activation_func_deriv [j]) * (cost[j])
    return WT,B
```
### testing
this model have been tested over 10000 images and it predicted the right label in more than 5000 cases,
which means accuracy is equal to 80% (not bad for such a network) <br />
![Screenshot from 2023-08-08 00-35-01](https://github.com/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/assets/53050138/345c590e-ed1e-4358-becd-d4c8f2b245ed)


### Run it yourself
- run it yourself but note that you might get different accuracy due to local optima problem </br>


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/blob/main/dumbest_network.ipynb)


![diag](https://github.com/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/assets/53050138/fb1743a7-9bf5-49c9-b61e-896486f696b2)
