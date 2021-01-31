#!python

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# Hyperparameters
window_size = 7
order = 3

scatter_av = 0.1
scatter_sigma = 0.05
window = 15
order = 4

xx = np.arange(0.0, 10.0, 0.01)
y_true = np.sin(xx)

x = np.arange(0.0, 10.0, 0.2)
y = np.sin(x)
np.random.seed(250)

q_err = (np.abs(np.random.normal(scatter_av, scatter_sigma,
        (len(x)))))

cov = np.diag(np.ones(q_err.size))

# A block diagonal covariance matrix where the correlation coefficient
# falls as a function of offset from the diagonal
for i in range(q_err.size):
    for offset in range(1, q_err.size):
        if i+offset >= q_err.size:
            continue
        cov[i, i+offset] = 0.3/offset**2
        cov[i+offset, i] = 0.3/offset**2

for i in range(q_err.size):
    for j in range(i, q_err.size):
        cov[i, j] = cov[i, j] * q_err[i] * q_err[j]
        cov[j, i] = cov[i, j]

y = np.random.multivariate_normal(y, cov)

print (y)

sg = savitzky_golay(y, window_size, order, deriv = 0)

print (sg)

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.set_ylim([-2, 2])
#import palettable
#ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)

ax.errorbar(x, y, yerr = q_err, fmt = '.', marker = 'o', ms = 3.0,
        label = 'Noisy data with covariant errors')
ax.plot(x, sg, '-', label = 'Traditional SG')
ax.plot(xx, y_true, '-', label = 'Noiseless')
ax.legend(loc = 3, frameon=0, ncol=2, fontsize=13)
ax.set_ylabel("y(x)")
ax.set_xticklabels([])
plt.savefig('Test_Sine_wcov.png')
'''
# Let us compute chisquare from the truth
# Chi-squared values compared to the underlying truth
print ("Chi-squared values compared to truth")
chisq_trad_truth = (np.dot(np.dot((sg-np.sin(x)).T, inv(cov)), (sg-np.sin(x))))
print ("Traditional: %.2f " % chisq_trad_truth )
chisq_new_truth = (np.dot(np.dot((sg_err-np.sin(x)).T, inv(cov)), (sg_err-np.sin(x))))
print ("This work: %.2f " % chisq_new_truth )
print ("Chi-squared values compared to data")
chisq_trad_data = (np.dot(np.dot((sg-y).T, inv(cov)), (sg-y)) )
print ("Traditional: %.2f " % chisq_trad_data)
chisq_new_data = (np.dot(np.dot((sg_err-y).T, inv(cov)), (sg_err-y)))
print ("This work: %.2f " % chisq_new_data)


ax = fig.add_subplot(2, 1, 2)
#ax.set_color_cycle(palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
ax.errorbar(x, (y-np.sin(x))/q_err, q_err/q_err, fmt = '.', marker = 'o', ms = 3.0)
ax.plot(x, (sg_err-np.sin(x))/q_err, '-', label = r'This work $\chi^2=%.2f$' % chisq_new_truth)
ax.plot(x, (sg-np.sin(x))/q_err, '-', label= "Traditional SG $\chi^2=%.2f$" % chisq_trad_truth)
ax.axhline(0.0, color='grey')
ax.set_xlabel("x")
ax.set_ylabel(r"[y-sin(x)]/$\sigma_y$")
ax.legend(loc = 3, frameon=0, ncol=2, fontsize=13)
ax.set_ylim([-3, 3])

plt.tight_layout()
plt.savefig('Test_Sine_wcov.pdf')'''
