import numpy as np

def singular_value_decomposition():
    # generator samples
    epsilon = 6
    X = np.random.rand(10, 3)
    X = X * epsilon * 2 - epsilon
    # singular value decomposition
    u, sigma, vt = np.linalg.svd(X)
    print(f'X is \n{X}\nU is \n{u}\nsigma is \n{sigma}\nV.T is \n{vt}\n')
    # compress
    X_ = (u[:, 0, np.newaxis] * sigma[0]) @ vt[0, :, np.newaxis].T
    # difference
    X_diff = X - X_
    print(f'difference between X and X^ is \n{X_diff}')
    print(f'max error is {X_diff.max()}')


if __name__ == "__main__":
    singular_value_decomposition()