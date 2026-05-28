import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, a):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + a * d))

    def grad_directional(self, x, d, a):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + a * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, mv_Ax, mv_ATx, mm_ATsA, b, rc):
        self.mv_Ax = mv_Ax
        self.mv_ATx = mv_ATx
        self.mm_ATsA = mm_ATsA
        self.b = b
        self.rc = rc

    def func(self, x):
        Ax = self.mv_Ax(x)
        z = self.b * Ax
        loss = np.mean(np.log(1 + np.exp(-z)))
        reg = (self.rc * np.sum(x**2)) / 2
        return loss + reg
        

    def grad(self, x):
        Ax = self.mv_Ax(x)
        z = self.b * Ax
        s = expit(z) - 1
        sb = s * self.b
        ATsb = self.mv_ATx(sb)
        m = len(self.b)
        return (1 / m) * ATsb + self.rc * x

    def hess(self, x):
        Ax = self.mv_Ax(x)
        z = self.b * Ax
        sn = expit(z) * (1 - expit(z))
        mp = self.mm_ATsA(sn)
        m = len(self.b)
        return (1 / m) * mp + self.rc * np.eye(len(x))


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, mv_Ax, mv_ATx, mm_ATsA, b, rc):
        super().__init__(mv_Ax, mv_ATx, mm_ATsA, b, rc)
        self._cx = None
        self._cAx = None
        self._cd = None
        self._cAd = None

    def _get_Ax(self, x):
        # Check if x = cached_x + alpha * cached_d for some alpha
        if self._cx is not None and self._cAd is not None:
            diff = x - self._cx
            # Check if diff is a scalar multiple of cached_d
            nrm = np.linalg.norm(self._cd)
            if nrm > 0:
                a = np.dot(diff, self._cd) / (nrm ** 2)
                if np.allclose(diff, a * self._cd):
                    return self._cAx + a * self._cAd

        if self._cx is None or not np.array_equal(self._cx, x):
            self._cx = x.copy()
            self._cAx = self.mv_Ax(x)
            self._cd = None
            self._cAd = None
        return self._cAx

    def _get_Ad(self, x, d):
        self._get_Ax(x)
        if self._cd is None or not np.array_equal(self._cd, d):
            self._cd = d.copy()
            self._cAd = self.mv_Ax(d)
        return self._cAd

    def func(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        loss = np.mean(np.log(1 + np.exp(-z)))
        reg = (self.rc * np.sum(x**2)) / 2
        return loss + reg

    def grad(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        s = expit(z) - 1
        ATsb = self.mv_ATx(s * self.b)
        m = len(self.b)
        return (1 / m) * ATsb + self.rc * x

    def hess(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        sn = expit(z) * (1 - expit(z))
        mp = self.mm_ATsA(sn)
        m = len(self.b)
        return (1 / m) * mp + self.rc * np.eye(len(x))

    def func_directional(self, x, d, a):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(x, d)
        Axn = Ax + a * Ad
        xn = x + a * d
        z = self.b * Axn
        loss = np.mean(np.log(1 + np.exp(-z)))
        reg = (self.rc * np.sum(xn**2)) / 2
        return np.squeeze(loss + reg)

    def grad_directional(self, x, d, a):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(x, d)
        Axn = Ax + a * Ad
        xn = x + a * d
        z = self.b * Axn
        s = expit(z) - 1
        return np.squeeze(np.dot(s * self.b, Ad) / len(self.b) + self.rc * np.dot(xn, d))


def create_log_reg_oracle(A, b, rc, ot='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    mv_Ax = lambda x: A.dot(x)
    mv_ATx = lambda x: (A.T).dot(x)

    def mm_ATsA(s):
        if scipy.sparse.issparse(A):
            return (A.T).dot(A.multiply(s[:, np.newaxis]))
        return (A.T).dot(np.diag(s).dot(A))

    if ot == 'usual':
        oracle = LogRegL2Oracle
    elif ot == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % ot
    return oracle(mv_Ax, mv_ATx, mm_ATsA, b, rc)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    g = np.zeros_like(x)          
    fx = func(x)                 
    
    for i in range(len(x)):       
        xp = x.copy()    
        xp[i] += eps     
        
        fp = func(xp) 
        g[i] = (fp - fx) / eps
        
    return g
    


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    H = np.zeros((n, n))
    fx = func(x)
    for i in range(n):
      for j in range(n):
        xi = x.copy()
        xi[i]+= eps
        xj = x.copy()
        xj[j]+= eps
        xij = x.copy()
        xij[i] +=eps
        xij[j] +=eps
        fi = func(xi)
        fj = func(xj)
        fij = func(xij)
        H[i, j] = (fij - fi - fj + fx) / (eps**2)

    return H
