# Support Vector machine
we need to divide feature space into 10 part for solving hand written digit recognition problem
but here we used a trick by using 10 plane each one divide feature space into two parts,
each of these 10 plane is corresponding to a digit and answers the question "How much is it likely that the input belongs to the digit class?"
and finaly we choose the most likely one.
but how to find these planes?

## problem
it is a convex quadratic optimization problem

$`minimize \frac{1}{2} \|w\|^{2}`$ <br />
$`s.t.:y_{i}(w x_{i}+b)-1 \geq 0, i=1, \ldots, m`$

after doing some math (which have been explained in the notebook) we would reach to Wolfe dual problem,
then we use a QP solver to find the answer.
special thnaks to <a href=https://github.com/Girrajjangid/Machine-Learning-from-Scratch/blob/master/Support%20Vector%20Machine/2.%20SVM%20with%20hard%20margin%20(from%20scratch).ipynb>this</a>.
