#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:28:47 2019

@author: dejan
"""
from inspect import signature
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from matplotlib.widgets import Button
from scipy import sparse
from scipy.sparse.linalg import spsolve, inv


def baseline_als3(y, lam=1e5, p=5e-5, niter=12):
    '''Found on stackoverflow.
    Schematic explanaton of the params to
    get the "feel" of how the algo works:
    Params:
        y: your spectrum on which to find the baseline
        lam: can be viewed as the radius of the ball
        p: can be viewed as the measure of how much the ball
            can penetrate into the spectra from below
        niter: number of iterations
            (the resulting baseline should stabilize after
            some number of iterations)
    Returns:
        z: the baseline
    ---------------------------------------------
    For more info, see the discussion on:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library'''
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def pV(x, h=30, x0=0, w=10, factor=0.5):
    '''Manualy created pseudo-Voigt profile
    Parameters:
    ------------
    x: Independent variable
    h: height
    x0: The position of the peak on the x-axis
    w: FWHM
    factor: the ratio of lorentz vs gauss in the peak
    Returns:
    y-array of the same shape as the input x-array
    '''

    def Gauss(x, w):
        return((2/w) * np.sqrt(np.log(2)/np.pi) * np.exp(
                -(4*np.log(2)/w**2) * (x - x0)**2))

    def Lorentz(x, w):
        return((1/np.pi)*(w/2) / (
                (x - x0)**2 + (w/2)**2))

    intensity = h * np.pi * (w/2) / (
                    1 + factor * (np.sqrt(np.pi*np.log(2)) - 1))

    return(intensity * (factor * Gauss(x, w)
                        + (1-factor) * Lorentz(x, w)))


def multi_pV(x, *params):
    '''
    The function giving the sum of the pseudo-Voigt peaks.
    Parameters:
    x: independent variable
    *params: is a list of parameters.
    Its length is = 4 * "number of peaks",
    where 4 is the number of parameters in the "pV" function.
    Look in the docstring of pV function for more info on theese.
    '''
    # The following presupposes that the first argument of the function
    # is the independent variable and that the subsequent parameters are
    # the function parameters:
    n = 4
    nn = len(params)
    if nn % n != 0:
        raise Exception(f"You gave {nn} parameters and your basic function"
                        f"takes {n} parameters (The first one should be x"
                        "and the remainign ones the parameters of the funct.")
    result = np.zeros_like(x, dtype=np.float)
    for i in range(0, nn, n):
        result += pV(x, *params[i:i+n])  # h, x0, w, r)
    return result


def create_map_spectra(x=np.arange(150, 250, 0.34), initial_peak_params=[171, 200, 8, 0.7], N=2000, ponderation=None):
    '''Creates simulated spectra
    Params:
        x: independent variable
        initial_peak_params: list of peak parameters
            the number of parameters must be a multiple of number of parameters
            demanded by peak_function (let's say that number is N)
            So, then you can set-up M peaks, just by supplying M x N parameters
        peak_function: default is pseudo-Voigt
        N: the number of spectra to create
        ponderation: How much you want the spectra to differ between them
    '''
    if not ponderation:
        ponderation=np.asarray(initial_peak_params)/5 + 1
    else:
        ponderation = np.asarray(ponderation)
    nn = len(initial_peak_params)
    peaks_params = (1 + (np.random.rand(N, nn) - 0.5) / ponderation) \
                    * np.asarray(initial_peak_params)

    spectra = np.asarray([(multi_pV(x, *peaks_params[i]) + (np.random.random(len(x))-0.5)*5) * (1 + (np.random.random(len(x))-0.5)/20) for i in range(N)])

    return spectra
# %%



class NavigationButtons(object):
    '''This class allows you to visualize multispectral data and
    navigate trough your spectra simply by clicking on the
    navigation buttons on the graph.
    -------------------
    Parameters:
        sigma: 1D numpy array of your x-values (raman shifts, par ex.)
        spectra: 3D ndarray of shape (n_spectra, len(sigma), n_curves)
        autoscale: bool determining if you want to adjust the scale to each spectrum
        title: The initial title describing where the spectra comes from
    Output:
        matplotlib graph with navigation buttons to cycle trought spectra
    Example:
    # Let's say you have a ndarray containing 10 spectra, each 500 points long
    # base_spectras.shape should give (10, 500)
    # your sigma.shape should be (500, )
    # Then let's say you fitted each of your spectra with 3 gaussian peaks
    # and you want to plot these as well. For each of your ten spectra,
    # you will have something like:
    >>>spectra_fitted[i] = multiple_gaussian_function(sigma, *params[i])
    # your spectra_fitted should have the same shape as your spectra.
    # Now, let's say you want also to plot each of the gaussian peaks as well
    # for "i"th spectra you will have 3 gaussians
    >>>for k in range(3):
    >>>G[i][k] = single_gaussian_function(sigma, *params[i][k])
    # At the end, you stack all of this in one ndarray :
    >>>multiple_curves_to_plot = np.stack((
            base_spectras, spectra_fitted, G1, G2, G3), axis=-1)
    '''
    ind = 0

    def __init__(self, sigma, spectra, autoscale_y=False, title='',
                 **kwargs):
        self.y_autoscale = autoscale_y
        self.n_points = spectra.shape[0]
        if len(spectra.shape) == 2:
            self.s = spectra[:,:, np.newaxis]
        elif len(spectra.shape) == 3:
            self.s = spectra
        else:
            raise ValueError('Check the shape of your spactra. It should be (n_spectra, n_points, n_curves)')
        self.title = title
        self.sigma = sigma
        if "Temp" in kwargs:
            self.temp = kwargs.pop("Temp")
            self.temp_ar = np.resize(self.temp, self.n_points)
            self.title_ar = np.fromiter((self.title + ' : N°'+str(i)+
                                         ', T='+
                                         str(self.temp_ar[i]) for i in range(self.n_points)),
                                        dtype='U50', count=self.n_points)
        else:
            self.Temp = 'unknown'
            self.title_ar = np.char.add(self.title+' : N°',
                                        np.arange(self.n_points).astype(str))

        self.figr, self.axr = plt.subplots(**kwargs)
        self.axr.set_title(f'{title} : spectrum number 0')
        self.figr.subplots_adjust(bottom=0.2)
        self.l = self.axr.plot(self.sigma, self.s[0], lw=2) # l potentially contains multiple lines
        self.axprev1000 = plt.axes([0.097, 0.05, 0.1, 0.04])
        self.axprev100 = plt.axes([0.198, 0.05, 0.1, 0.04])
        self.axprev10 = plt.axes([0.299, 0.05, 0.1, 0.04])
        self.axprev1 = plt.axes([0.4, 0.05, 0.1, 0.04])
        self.axnext1 = plt.axes([0.501, 0.05, 0.1, 0.04])
        self.axnext10 = plt.axes([0.602, 0.05, 0.1, 0.04])
        self.axnext100 = plt.axes([0.703, 0.05, 0.1, 0.04])
        self.axnext1000 = plt.axes([0.804, 0.05, 0.1, 0.04])

        self.bprev1000 = Button(self.axprev1000, 'Prev.1000')
        self.bprev1000.on_clicked(self.prev1000)
        self.bprev100 = Button(self.axprev100, 'Prev.100')
        self.bprev100.on_clicked(self.prev100)
        self.bprev10 = Button(self.axprev10, 'Prev.10')
        self.bprev10.on_clicked(self.prev10)
        self.bprev = Button(self.axprev1, 'Prev.1')
        self.bprev.on_clicked(self.prev1)
        self.bnext = Button(self.axnext1, 'Next1')
        self.bnext.on_clicked(self.next1)
        self.bnext10 = Button(self.axnext10, 'Next10')
        self.bnext10.on_clicked(self.next10)
        self.bnext100 = Button(self.axnext100, 'Next100')
        self.bnext100.on_clicked(self.next100)
        self.bnext1000 = Button(self.axnext1000, 'Next1000')
        self.bnext1000.on_clicked(self.next1000)

    def next1(self, event):
        self.ind += 1
        _i = self.ind % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next10(self, event):
        self.ind += 10
        _i = self.ind % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next100(self, event):
        self.ind += 100
        _i = self.ind % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next1000(self, event):
        self.ind += 1000
        _i = self.ind % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev1(self, event):
        self.ind -= 1
        _i = self.ind % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev10(self, event):
        self.ind -= 10
        _i = self.ind % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev100(self, event):
        self.ind -= 100
        _i = self.ind % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev1000(self, event):
        self.ind -= 1000
        _i = (self.ind) % self.n_points
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()


# %%


def fitonclick(event):
    global it, peaks_present, scroll_count, clicked_indice
    global x_size, y_size, x, block
    # this sets up the color palette to be used for plotting lines:
    plt.rcParams["axes.prop_cycle"] =\
        cycler('color',
               ['#332288', '#CC6677', '#DDCC77', '#117733', '#88CCEE', '#AA4499',
                '#44AA99', '#999933', '#882255', '#661100', '#6699CC', '#AA4466'])
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Initiating variables to which we will atribute peak caractéristics:
    pic = {}
    pic['line'] = []  # List containing matplotlib.Line2D object for each peak
    pic['h'] = []  # List that will contain heights of each peak
    pic['x0'] = []  # List that will contain central positions of each peak
    pic['w'] = []  # List containing widths
    pic['fill'] = []

    # Iterator used normally for counting right clicks
    # (each right click launches the plot of the cumulative curbe)
    it = 0

    # List of cumulated graphs
    # (used later for updating while removing previous one)
    sum_peak = []

    peaks_present = 0
    cid3 = []
    scroll_count = 0  # counter to store the cumulative values of scrolling
    artists = []
    clicked_indice = -1
    indice = 0
    if event.inaxes == ax:  # if you click inside the plot
        if event.button == 1:  # left click
            # Create list of all elipes and check if the click was inside:
            click_in_artist = [artist.contains(event)[0] for artist in artists]
            if not any(click_in_artist):  # if click was not on any elipsis
                peaks_present += 1
                h = event.ydata
                x0 = event.xdata
                artists.append(ax.add_artist(
                        Ellipse((x0, h),
                                x_size, y_size, alpha=0.5,
                                picker=max(x_size, y_size),
                                gid=peaks_present)))

                yy = pV(x=x, h=h, x0=x0, w=x_size)
                pic['line'].append(ax.plot(x, yy, alpha=0.75, lw=2.5,
                                   picker=5))
                # ax.set_ylim(auto=True)
                pic['h'].append(h)
                pic['x0'].append(x0)
                pic['w'].append(x_size)
# ax.fill_between(x, yy.min(), yy, alpha=0.3, color=cycle[peaks_present])
                fig.canvas.draw_idle()

            elif any(click_in_artist):  # if the click was on one of the elipses
                clicked_indice = click_in_artist.index(True)
                artists[clicked_indice].remove()
                artists.pop(clicked_indice)
#                ax.lines[clicked_indice].remove()
                ax.lines.remove(pic['line'][clicked_indice][0])
                pic['line'].pop(clicked_indice)
                pic['x0'].pop(clicked_indice)
                pic['h'].pop(clicked_indice)
                pic['w'].pop(clicked_indice)
                fig.canvas.draw_idle()
                peaks_present -= 1

        elif event.button == 3 and not event.step:
            # On my laptop middle click and right click have the same values
            if it > 0:  # Checks if there is already a cumulated graph plotted
                # remove the last cumulated graph from the figure:
                ax.lines.remove(sum_peak[-1][0])
                sum_peak.pop()
                it -= 1
            # Sum all the y values from all the peaks:
            sumy = np.sum(np.asarray(
                    [pic['line'][i][0].get_ydata() for i in range(peaks_present)]),
                    axis=0)
            # Added this condition for the case where you removed all peaks,
            # but the cumulated graph is left
            # then right-clicking need to remove that one as well:
            if sumy.shape == x.shape:
                # plot the cumulated graph:
                sum_peak.append(ax.plot(x, sumy, '--', color='lightgreen',
                                        lw=3, alpha=0.6))
                it+=1 # One cumulated graph added
            else:
                # if you right clicked on the graph with no peaks,
                # you removed the cumulated graph as well
                it-=1
            fig.canvas.draw_idle()

        if event.step != 0:
            if peaks_present:
                # -1 means that scrolling will only affect the last plotted peak
                peak_identifier = -1
                '''(this may change in the future so to permit the user
                to modify whatewer peak's widht he wishes to adjust)
                This however turns out to be a bit too difficult to acheive.
                For now, I'll settle with this version, where,
                if you want to readjust some previously placed peak,
                you need in fact to repace it with a new one
                (you can first add the new one on the position that you think
                 is better, adjust it's width, and then remove the one
                 you didn't like by clicking on it's top)'''

                # This adjust the "speed" of width change with scrolling:
                scroll_count += x_size*event.step/15

                if scroll_count > -x_size+0.01:
                    w2 = x_size + scroll_count
                else:
                    w2 = 0.01
                    # This doesn't allow you to sroll to negative values
                    # (basic width is x_size)
                    scroll_count = -x_size+0.01

                center2 = pic['x0'][peak_identifier]
                h2 = pic['h'][peak_identifier]
                pic['w'][peak_identifier] = w2
                yy = pV(x=x, x0=center2, h=h2, w=w2)
                active_line = pic['line'][peak_identifier][0]
                # This updates the values on the peak identified by
                # "peak_identifier" (last peak if = -1).
                active_line.set_ydata(yy)
                ax.draw_artist(active_line)
#                if peak_identifier > -1:
#                    cycle_indice = peak_identifier
#                else:
#                    cycle_indice = peaks_present
#                pic['fill'].append(ax.fill_between(x, 0, yy, alpha=0.3, color=cycle[cycle_indice]))
                fig.canvas.draw_idle()

        if event.button != 1 and event.dblclick:
            block = True
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.mpl_disconnect(cid2)
            plt.close()
            return


# Williams' functions for Raman spectra:
def long_correction(sigma, lambda_laser, T=30):
    """
    Function computing the Long correction factor according to Long
    1977. This function can operate on numpy.ndarrays as well as on
    simple numbers.

    Parameters
    ----------
    sigma : numpy.ndarray
        Wavenumber in cm-1
    lambda_inc : float
        Laser wavelength in nm.

    Examples
    --------
    >>> sigma, i = deconvolution.acquire_data('my_raman_file.CSV')
    >>> corrected_i = i * long_correction(sigma)
    """
    c = 2.998e10                          # cm/s
    lambda_inc = lambda_laser * 1e-7      # cm
    sigma_inc = 1. / lambda_inc           # cm-1
    h = 6.63e-34                          # J.s
    TK = 273.0 + T                        # K
    kB = 1.38e-23                         # J/K
    return (sigma_inc**3 * sigma / (sigma_inc - sigma)**4
            * (1 - np.exp(-h*c*sigma/kB/TK)))

# %%
def clean(sigma, raw_spectra, mode='area'):
    """
    Cleans the spectra to remove abnormal ones, remove the baseline offset,
    correct temperature & frequency effects, and make them comparable
    by normalizing them according to their area or their maximum.

    Parameters
    ----------
    sigma : numpy.ndarray
        Wavenumber in cm-1
    raw_spectra : numpy.ndarray, n_spectra * n_features
        Input spectra
    mode : {'area', 'max'}
        Controls how spectra are normalized
    delete : list of int, default None
        Spectra that should be removed, eg outliers
    long_cor : float, optional
        Laser wavelength in nm. If given, then temperature-frequence correction
        will be applied. If None or False, no correction is applied.
    """
    clean_spectra = np.copy(raw_spectra)
    # Remove the offset
    clean_spectra -= clean_spectra.min(axis=1)[:, np.newaxis]
    # Normalize the spectra
    if mode == 'max':
        clean_spectra /= clean_spectra.max(axis=1)[:, np.newaxis]
    elif mode == 'area':
        clean_spectra /= np.trapz(clean_spectra)[:, np.newaxis]
    else:
        print('Normalization mode not understood; No normalization applied')
    return clean_spectra
