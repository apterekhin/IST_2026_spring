import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.a0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.a0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, pa=None):
        if self._method == 'Constant':
            return self.c

        elif self._method == 'Armijo':
            a = pa if pa is not None else self.a0
            p0 = oracle.func_directional(x_k, d_k, 0)
            dp0 = oracle.grad_directional(x_k, d_k, 0)
            while oracle.func_directional(x_k, d_k, a) > p0 + self.c1 * a * dp0:
                a /= 2.0
                if a < 1e-15:
                    return None
            return a

        elif self._method == 'Wolfe':
            a, *_ = scipy.optimize._linesearch.line_search_wolfe2(oracle.func, oracle.grad, x_k, d_k, c1=self.c1, c2=self.c2)
            if a is None:
                a = self.a0 if pa is None else pa
                p0 = oracle.func_directional(x_k, d_k, 0)
                dp0 = oracle.grad_directional(x_k, d_k, 0)
                while oracle.func_directional(x_k, d_k, a) > p0 + self.c1 * a * dp0:
                    a /= 2.0
                    if a < 1e-15:
                        return None
            return a

        return None


def get_line_search_tool(lso=None):
    if lso:
        if type(lso) is LineSearchTool:
            return lso
        else:
            return LineSearchTool.from_dict(lso)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tol=1e-5, mi=10000,
                     lso=None, tr=False, disp=False):
    hist = defaultdict(list) if tr else None
    lst = get_line_search_tool(lso)
    x_k = np.copy(x_0)
    st = datetime.now()
    g0 = oracle.grad(x_0)
    g0ns = np.dot(g0, g0)
    a = None

    for it in range(mi + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        if not np.isfinite(f_k) or not np.all(np.isfinite(g_k)):
            return x_k, 'computational_error', hist
        gn = np.linalg.norm(g_k)
        if tr:
            el = (datetime.now() - st).total_seconds()
            hist['time'].append(el)
            hist['func'].append(f_k)
            hist['grad_norm'].append(gn)
            if x_k.size <= 2:
                hist['x'].append(np.copy(x_k))
        if disp:
            print(f'Iter {it}: f={f_k:.6f}, ||grad||={gn:.6f}')
        if gn ** 2 <= tol * g0ns:
            return x_k, 'success', hist
        if it == mi:
            break
        d_k = -g_k
        a = lst.line_search(oracle, x_k, d_k, pa=a)
        if a is None:
            return x_k, 'computational_error', hist
        x_k = x_k + a * d_k
        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', hist

    return x_k, 'iterations_exceeded', hist


def newton(oracle, x_0, tol=1e-5, mi=100,
           lso=None, tr=False, disp=False):
    hist = defaultdict(list) if tr else None
    lst = get_line_search_tool(lso)
    x_k = np.copy(x_0)
    st = datetime.now()
    g0 = oracle.grad(x_0)
    g0ns = np.dot(g0, g0)

    for it in range(mi + 1):
        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)
        H_k = oracle.hess(x_k)
        if not np.isfinite(f_k) or not np.all(np.isfinite(g_k)) or not np.all(np.isfinite(H_k)):
            return x_k, 'computational_error', hist
        gn = np.linalg.norm(g_k)
        if tr:
            el = (datetime.now() - st).total_seconds()
            hist['time'].append(el)
            hist['func'].append(f_k)
            hist['grad_norm'].append(gn)
            if x_k.size <= 2:
                hist['x'].append(np.copy(x_k))
        if disp:
            print(f'Iter {it}: f={f_k:.6f}, ||grad||={gn:.6f}')
        if gn ** 2 <= tol * g0ns:
            return x_k, 'success', hist
        if it == mi:
            break
        try:
            d_k = np.linalg.solve(H_k, -g_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', hist
        if np.dot(g_k, d_k) >= 0:
            return x_k, 'newton_direction_error', hist
        a = lst.line_search(oracle, x_k, d_k)
        if a is None:
            return x_k, 'computational_error', hist
        x_k = x_k + a * d_k
        if not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', hist

    return x_k, 'iterations_exceeded', hist
