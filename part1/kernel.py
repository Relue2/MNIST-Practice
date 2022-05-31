import numpy as np



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    #from the definition of the polynomial kernel, using simple matrix operations
    kernel_matrix = (X @ Y.transpose() + c)**p
    return kernel_matrix



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    #entries on diagonal of each matrix multiplied by its transpose is precisely x_i dot x_i and y_j dot j_j respectively
    A = np.reshape(np.diagonal(X @ X.transpose()), (X.shape[0], 1))
    B = np.reshape(np.diagonal(Y @ Y.transpose()), (Y.shape[0], 1)).transpose()

    C = 2 * X @ Y.transpose()
    #addition between A and B reshapes then to nxm, rows are the respective X vector and columns are the respective Y vector
    #subtract from this 2 * x dot y as follows from expanding norm(x - y)^2

    return np.exp(-gamma * (A + B - 2 * C))