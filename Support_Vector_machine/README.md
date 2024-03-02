# Support Vector machine
we need to divide feature space into 10 part for solving hand written digit recognition problem
but here we used a trick by using 10 plane each one divide feature space into two parts,
each of these 10 plane is corresponding to a digit and answers the question "How much is it likely that the input belongs to the digit class?"
and finaly we choose the most likely one.
but how to find these planes?

## problem
using support Vector machine approach we should solve the below problem which is a convex quadratic optimization problem

$`minimize \frac{1}{2} \|w\|^{2}`$ <br />
$`s.t.:y_{i}(w x_{i}+b)-1 \geq 0, i=1, \ldots, m`$

after doing some math (which have been explained in <a href=https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Support%20Vector%20Machine/2.%20SVM%20with%20hard%20margin%20(from%20scratch).ipynb>this</a>
notebook) we would reach to Wolfe dual problem, reforming it to the standard form we use a python QP solver to find the planes.

Here it is its code:
```
def SVM(X,y):
    m = X.shape[0]
    K = np.array([np.dot(X[i].T, X[j])
              for i in range(m)
              for j in range(m)]).reshape((m, m))
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(m))
    A = cvxopt.matrix(y, (1, m))
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.diag(-1 * np.ones(m)))
    h = cvxopt.matrix(np.zeros(m))
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    multipliers = np.ravel(solution['x'])
    has_positive_multiplier = multipliers > 1e-07
    sv_multipliers = multipliers[has_positive_multiplier]
    support_vectors = X[has_positive_multiplier]
    support_vectors_y = y[has_positive_multiplier]
    w = compute_w(multipliers, X, y)
    w_from_sv = compute_w(sv_multipliers, support_vectors, support_vectors_y)
    b = compute_b(w, support_vectors, support_vectors_y)
    return w_from_sv,b
```
## result
finaly we got 80 percent accuracy in both test and training sets here it is its confusion matrix:

![Screenshot from 2024-03-02 22-15-14](https://github.com/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/assets/53050138/e5c6116c-2e6e-4b1d-9947-5017d11ed2c4)

![Screenshot from 2024-03-02 22-15-31](https://github.com/Mehrdadghassabi/handwritten_digit_recognition_from_scratch/assets/53050138/21900674-207c-42d1-9226-574628c5f657)



## thanks
special thnaks to <a href=https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Support%20Vector%20Machine/2.%20SVM%20with%20hard%20margin%20(from%20scratch).ipynb>this</a>.
