from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import linalg
import matplotlib as mpl
import ipywidgets
from ipywidgets import interact, FloatSlider


def lsq_min(x, yn):
    def lsq(alpha, beta):
        out = np.zeros((alpha.size, beta.size))
        for i in xrange(alpha.size):
            for j in xrange(beta.size):
                out[i, j] = linalg.norm(yn - alpha[i]*x - beta[j])
        return out
    return lsq

x = np.arange(10)
A = np.c_[x, np.ones(x.size)]
alpha = 2.0
beta = -1.0
sigma = 1.5
y = alpha*x + beta
np.random.seed(102944)
yn = y + sigma*np.random.randn(x.size)
ab = np.polyfit(x, yn, 1)
alph = np.linspace(-10, 10, 200) + ab[0]
bet = np.linspace(-20, 20, 200) + ab[1]
ma, mb = np.meshgrid(alph, bet)

lsqf = lambda alpha, beta: linalg.norm(yn - alpha*x - beta)
lsq = lsq_min(x, yn)
obj = lsq(alph, bet)
objf = lambda alpha, beta: linalg.norm(yn - alpha*xt2 - beta*0.1)
percents = [0.1, 1, 3, 5, 7, 9, 11, 13, 15]

def kern2mat(H, size):
    """Create a matrix corresponding to the application of the convolution kernel H.
    
    The size argument should be a tuple (output_size, input_size).
    """
    N = H.size
    half = int((N - 1) / 2)
    Nout, Mout = size
    if Nout == Mout:
        return linalg.toeplitz(np.r_[H[half:], np.zeros(Nout - half)], np.r_[H[half:], np.zeros(Mout-half)])
    else:
        return linalg.toeplitz(np.r_[H, np.zeros(Nout - N)], np.r_[H[-1], np.zeros(Mout-1)])

def manipulate_line(alpha, beta,):
    labelsize=14
    texsize=16
    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    
    ax[0].plot(x, yn, 'go')
    a, b = np.polyfit(x, yn, 1)
    data_line = ax[0].plot(x, x * alpha + beta, 'g')
    A = np.c_[x, np.ones_like(x)]
    u, d, v = linalg.svd(A)
    ax[0].set_xlim(0, 9)
    ax[0].set_ylim(-5, 20)
    ax[0].set_xlabel("t", fontsize=labelsize)
    ax[0].set_ylabel("y", fontsize=labelsize)
    ax[0].text(1, 15, r"$y \, \sim \,  \alpha t + \beta$", fontsize=texsize)
    ax[1].contour(alph, bet, obj.T,
                [np.percentile(obj, i) for i in percents],
                colors=[plt.cm.Blues((100-i)/100.0) for i in percents])
    scale = 0.1
    d = d * scale
    ax[1].plot([a, a+v[0,0]*d[0]], [b, b+v[0, 1] * d[0]], 'k')
    ax[1].plot([a, a+v[1,0]*d[1]], [b, b+v[1, 1]*d[1]], 'k')
    data_pt = ax[1].plot(np.array([alpha]), np.array([beta]), 'bo', markeredgecolor='b')
    r = yn - (alpha*x + beta)
    grad = np.dot(A.T, r) 
    grad = grad / linalg.norm(grad)
    data_grad = ax[1].plot([alpha, alpha+grad[0]], [beta, beta+grad[1]], 'b-')
    ax[1].set_ylim(-3, 3.0)
    ax[1].set_xlim(-1, 5.0)
    ax[1].set_xlabel(r"$\alpha$", fontsize=texsize)
    ax[1].set_ylabel(r"$\beta$", fontsize=texsize)
    fig.tight_layout()
    return fig, ax, data_line, data_pt, data_grad


def plot_landweber(alpha, beta, eps_norm, N):
    labelsize=14
    texsize=16
    a, b = np.polyfit(x, yn, 1)
    A = np.c_[x, np.ones_like(x)]
    u, s, v = linalg.svd(A)
    fig, (ax0, ax) = plt.subplots(ncols=2, figsize=(8, 4))
    ax.contour(alph, bet, obj.T,
                [np.percentile(obj, i) for i in percents],
                colors=[plt.cm.Blues((100-i)/100.0) for i in percents])
    scale = 0.1
    d = s * scale
    ax.plot([a, a+v[0,0]*d[0]], [b, b+v[0, 1] * d[0]], 'k')
    ax.plot([a, a+v[1,0]*d[1]], [b, b+v[1, 1]*d[1]], 'k')
    data_pt = ax.plot(np.array([alpha]), np.array([beta]), 'bo', markeredgecolor='b')
    
    many_alpha = [np.array([[alpha, beta]]).T]
    for i in xrange(N):
        landweber_interate(many_alpha, eps_norm / s[0]**2)
    xy = np.array(many_alpha).squeeze().T
    ax.plot(xy[0, :], xy[1, :], 'b-')
    ax0.plot(xy[0, :], xy[1, :], 'b-')
    ax0.set_xlabel(r"$\alpha$", fontsize=texsize)
    ax0.set_ylabel(r"$\beta$", fontsize=texsize)
    ax.set_ylim(-4, 2.0)
    ax.set_xlim(-1, 5.0)
    ax.set_xlabel(r"$\alpha$", fontsize=texsize)
    ax.set_ylabel(r"$\beta$", fontsize=texsize)
    return fig


def update_plot(state, alpha, beta):
    fig, ax, data_line, data_pt, data_grad = state
    line = data_line[0]
    pt = data_pt[0]
    grad_line = data_grad[0]
    line.set_ydata(line.get_xdata() * alpha + beta)
    pt.set_xdata(np.array([alpha]))
    pt.set_ydata(np.array([beta]))
    A = np.c_[x, np.ones_like(x)]
    r = yn - (alpha*x + beta)
    grad = np.dot(A.T, r)
    grad = grad / linalg.norm(grad)
    grad_line.set_xdata(np.array([alpha, alpha+grad[0]]))
    grad_line.set_ydata(np.array([beta, beta+grad[1]]))
    return fig


def landweber_interate(xs, epsilon):
    """Perform a landweber iteration with step size epsilon, appending the result to xs."""
    x = xs[-1]
    resid =  yn.reshape((-1, 1)) - np.dot(A, x)
    x1 = x + np.dot(A.T, resid)*epsilon
    xs.append(x1)




