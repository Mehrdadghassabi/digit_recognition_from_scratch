# handwritten_digit_recognition_from_scratch
in this repository i tried to solve handwritten digits recognition problem with different nueral network architecture.

- without using common NNs libraries like tensorflow or pytorch just with pure code
- mnist dataset has been used
- gradient descent and backpropagation method have been used

## dumbest network
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/blob/main/dumbest%D9%80network.ipynb) <br />
probably the dumbest network for such a task is fully connecting the input and the output layer without any hidden layer.<br />
by flattening the $`28 \times 28`$ grey scale image we would have a $`784 \times 1`$ vector as an input,
so input layer A0 would have 784 nodes and the output layer A1 would have 10 nodes each one correspond to probablity of being a digit,
notice that the predicted label would be equal to argmax(A1).<br />
### training
as there is no hidden layer in this network only two vector would determine the predicted label, a $`784 \times 10`$ weight vector(W) and a $`10 \times 1`$ bias vector(B),
the output would be $`A1 = \sigma{(W^T \times A0 + B)}`$ <br />
according to gradient descent algorithm first of all W and B should be initialized randomly and get updated to minimize the cost function,
where $`Cost = (A1 - \hat{A1})^2`$ and update rules are $`W^T = W^T + r \times \frac{d}{dW}Cost`$ and $`B = B + r \times \frac{d}{dB}Cost`$ <br />
$` Cost = (A1 - \hat{A1})^2 \\
\frac{dCost}{dW} = 2 \times (A1 - \hat{A1}) \times \frac{dA1}{dW} \\
\frac{dCost}{dW} = 2 \times (A1 - \hat{A1}) \times A0`$ <br />
$` Cost = (A1 - \hat{A1})^2 \\
\frac{dCost}{dB} = 2 \times (A1 - \hat{A1}) \times \frac{dA1}{dB} \\
\frac{dCost}{dB} = 2 \times (A1 - \hat{A1}) `$
so the update rules would change to $`W^T = W^T + r \times (A1 - \hat{A1}) \times A0`$ and $`B = B + r \times (A1 - \hat{A1})`$ <br />
notice that like A1,Cost and $`\hat{A1}`$ also are $`10 \times 1`$ vectors, <br />
and all elements of $`\hat{A1}`$ are zero except the element with label index which is one. <br />
so we can write the train our model by iterating over training set,
here it is the code:
```
    def train(training_size,WT,B):
    assert training_size <= 6000
    for i in range(training_size):
        # print("episode number: "+ str(i))
        A0 = train_X[i].flatten().reshape((-1, 1))
        A0 = A0 / 255
        A1 = np.vectorize(sigmoid)(np.matmul(WT,A0) + B)
        yhat =  np.zeros((10, 1))
        yhat[train_y[i]][0] = 1
        cost = A1-yhat
        activation_func_deriv = A1 - A1**2
        r = 0.05
        for j in range(10):
            for k in range(784):
                WT[j][k] = WT[j][k] - r * (A0[k]) * (activation_func_deriv [j]) * (cost[j])
            B[j] = B[j] - r * (activation_func_deriv [j]) * (cost[j])
    return WT,B
```
### testing
![diag](https://github.com/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/assets/53050138/fb1743a7-9bf5-49c9-b61e-896486f696b2)
