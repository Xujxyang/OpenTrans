import math
import numpy as np
import torch

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    # import pdb; pdb.set_trace()
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))
    if var1 == 0.0 or var2 == 0.0:
        return 1
    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


if __name__=='__main__':
    # X = np.random.randn(100, 64)
    # Y = np.random.randn(100, 64)
    f1 = torch.load("han-subnetwork.pt")
    f2 = torch.load("fc-clip.pt")
    # 256, 512, 1024, 2048
    import pdb; pdb.set_trace()
    X1 = f1['res5']
    X1 = X1.cpu().numpy()
    # X1 = X1[:, :, ::2, ::2]
    Y1 = f2['res5']
    Y1 = Y1.cpu().numpy()
    L1 = 0

    # import pdb; pdb.set_trace()
    # matrix1 = X1[0, 17, :, :]
    # matrix2 = Y1[0, 17, :, :]
    # a = linear_CKA(matrix1, matrix2)

    for i in range(2048):
        # if i == 170 or i == 227 or i == 728 or i == 766 or i == 798 or i == 872 or i == 1156 or i == 1285 or i == 1367 or i == 1428 or i == 1492 or i == 1997: #han fc-clip
        #     continue
        # if i == 17 or i == 20 or i == 64 or i == 88 or i == 227 or i == 728 or i == 766 or i == 798 or i == 872 or i == 1156 or i == 1285 or i == 1367 or i == 1428 or i == 1492 or i == 1997: #han deeplab
        #     continue
        matrix1 = X1[0, i, :, :]
        matrix2 = Y1[0, i, :, :]
        L1 += linear_CKA(matrix1, matrix2)
        print(i)
        print(L1)
        # import pdb; pdb.set_trace()
        # matrices.append(matrix)layer
    print("average")
    print(L1/2048)

    # import pdb; pdb.set_trace()
    # print('Linear CKA, between X and Y: {}'.format(linear_CKA(X1, Y1)))
    # print('Linear CKA, between X and X: {}'.format(linear_CKA(X1, X1)))

    # print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    # print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))

