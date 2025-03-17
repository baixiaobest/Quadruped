import cvxpy as cp
import numpy as np
from sympy.abc import lamda


def foot_tip_IK_LM(x_des, theta_min, theta_max, theta0, Jac, FK, max_iter=50, epsilon=1e-5):
    '''
    Compute the foot tip inverse kinematics using the Levenberg-Marquardt method with box constraints.

    :param x_des: Desired foot tip position (target) in R^3.
    :param theta_min: Lower bounds for joint angles in R^3.
    :param theta_max: Upper bounds for joint angles in R^3.
    :param theta0: Initial guess for joint angles in R^3.
    :param Jac: Function handle returning the Jacobian given theta.
    :param FK: Function handle for forward kinematics, returning foot tip position.
    :param max_iter: Maximum number of iterations.
    :param epsilon: Convergence tolerance.
    :return: Estimated joint angles theta that satisfy the inverse kinematics.
    '''

    theta_i = theta0
    lambda_param = 1
    for i in range(max_iter):
        # Define the optimization variable
        theta = cp.Variable(3)
        J = Jac(theta_i)
        x_i = FK(theta_i)
        err = x_des - x_i
        # Define the objective function
        objective = cp.Minimize(cp.sum_squares(err - J @ (theta - theta_i)) + lambda_param * cp.sum_squares(theta - theta_i))
        # Define the constraints: elementwise inequality for box constraints
        constraints = [theta >= theta_min, theta <= theta_max]

        # Formulate and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("Solver did not converge to an optimal solution.")
            break

        new_err = x_des - FK(theta.value)
        if np.linalg.norm(new_err) < epsilon:
            return theta.value

        if np.linalg.norm(err) > np.linalg.norm(new_err):
            theta_i = theta.value
            lambda_param *= 0.8
        else:
            lambda_param *= 2

    print("Max iteration reached")
    return theta_i