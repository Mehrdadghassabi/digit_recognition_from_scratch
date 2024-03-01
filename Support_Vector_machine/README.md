# Support Vector machine
we need to divide feature space into 10 part for solving hand written digit recognition problem
but here we used a trick by using 10 plane each one divide feature space into two parts,
each of these 10 plane is corresponding to a digit and answers the question "How much is it likely that the input belongs to the digit class?"
and finaly we choose the most likely one.
but how to find these planes?

## problem
it is a convex quadratic optimization problem
$minimize \frac{1}{2} \|w\|^{2}$
