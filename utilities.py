#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:28:47 2019

@author: dejan
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button
from scipy import sparse
from scipy.sparse.linalg import spsolve
# %%

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

    def __init__(self, sigma, spectra, autoscale_y=True, title='My spectra',
                 **kwargs):
        self.y_autoscale = autoscale_y
        self.n_points = spectra.shape[0]
        if len(spectra.shape) == 2:
            self.s = spectra[:, :, np.newaxis]
        elif len(spectra.shape) == 3:
            self.s = spectra
        else:
            raise ValueError('Check the shape of your spectra.'
                             'It should be (n_spectra, n_points, n_curves)')
        self.title = title
        self.sigma = sigma
        self.figr, self.axr = plt.subplots(**kwargs)
        self.axr.set_title(f'{title} : spectrum number 0')
        self.figr.subplots_adjust(bottom=0.2)
        # here "l" potentially contains multiple lines:
        self.line = self.axr.plot(self.sigma, self.s[0], lw=2)
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
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next10(self, event):
        self.ind += 10
        _i = self.ind % self.n_points
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next100(self, event):
        self.ind += 100
        _i = self.ind % self.n_points
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next1000(self, event):
        self.ind += 1000
        _i = self.ind % self.n_points
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev1(self, event):
        self.ind -= 1
        _i = self.ind % self.n_points
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev10(self, event):
        self.ind -= 10
        _i = self.ind % self.n_points
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev100(self, event):
        self.ind -= 100
        _i = self.ind % self.n_points
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev1000(self, event):
        self.ind -= 1000
        _i = (self.ind) % self.n_points
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title} : spectrum number {_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()


# %%

class fitonclick(object):
    '''This class is used to interactively draw pseudo-voigt (or other type)
    peaks, on top of your data.
    It was originaly created to help defining initial fit parameters to
    pass on to SciPy CurveFit.
    IMPORTANT! See the Example below, to see how to use the class
    Parameters:
        x: independent variable
        y: your data
        initial_GaussToLorentz_ratio:float between 0 and 1, default=0.5
            Pseudo-Voigt peak is composed of a Gaussian and of a Laurentzian
            part. This ratio defines the proportion of those parts.
        scrolling_speed: float>0, default=1
            defines how quickly your scroling widens peaks
        initial_width: float>0, default=5
            defines initial width of peaks
        **kwargs: dictionary, for exemple {'figsize':(9,9)}
            whatever you want to pass to plt.subplots(**kwargs)
    Returns:
        Nothing, but you can access the atributes using class instance, like
        fitonclick.pic: dictionnary containing the parameters of each peak added
        fitonclick.sum_peak: list containing cumulated graph line
            to get the y-values, use sum_peak[-1][0].get_ydata()
        fitonclick.peak_counter: int giving the number of peaks present
        etc.

    Example:
        >>>my_class_instance = fitonclick(x, y)
        >>>while my_class_instance.block:
        >>>    plt.waitforbuttonpress(timeout=-1)


    '''
    # Initiating variables to which we will atribute peak caractéristics:
    pic = {}
    pic['line'] = []  # List containing matplotlib.Line2D object for each peak
    pic['h'] = []  # List that will contain heights of each peak
    pic['x0'] = []  # List that will contain central positions of each peak
    pic['w'] = []  # List containing widths
    pic['GL'] = []
    # List of cumulated graphs
    # (used later for updating while removing previous one)
    sum_peak = []
    peak_counter: int = 0  # number of peaks on the graph
    cum_graph_present: int = 0  # only 0 or 1
    scroll_count = 0  # counter to store the cumulative values of scrolling
    artists = []  # will be used to store the elipses on tops of the peaks

    block = True

    def __init__(self, x, y,
                 initial_GaussToLoretnz_ratio=0.5,
                 scrolling_speed=1,
                 initial_width=5,
                 **kwargs):
        plt.ioff()
        self.x = x
        self.y = y
        self.GL = initial_GaussToLoretnz_ratio
        self.scrolling_speed = scrolling_speed
        self.initial_width = initial_width
        # Setting up the plot:
        self.fig, self.ax = plt.subplots(**kwargs)
        self.ax.plot(self.x, self.y,
                     linestyle='none', marker='o', c='k', ms=4, alpha=0.5)
        self.ax.set_title('Left-click to add/remove peaks,'
                          'Scroll to adjust width, \nRight-click to draw sum,'
                          ' Double-Middle-Click when done')
        self.x_size = self.set_size(self.x)
        self.y_size = 2*self.set_size(self.y)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid2 = self.fig.canvas.mpl_connect('scroll_event', self.onclick)

    def set_size(self, variable, rapport=70):
        return (variable.max() - variable.min())/rapport

    def _add_peak(self, event):
        self.peak_counter += 1
        h = event.ydata
        x0 = event.xdata
        yy = pV(x=self.x, h=h,
                x0=x0, w=self.x_size*self.initial_width, factor=self.GL)
        one_elipsis = self.ax.add_artist(
                        Ellipse((x0, h),
                                self.x_size, self.y_size, alpha=0.5,
                                gid=str(self.peak_counter)))
        self.artists.append(one_elipsis)
        self.pic['line'].append(self.ax.plot(self.x, yy,
                                alpha=0.75, lw=2.5,
                                picker=5))
        # ax.set_ylim(auto=True)
        self.pic['h'].append(h)
        self.pic['x0'].append(x0)
        self.pic['w'].append(self.x_size*self.initial_width)
        self.fig.canvas.draw_idle()
#        return(self.artists, self.pic)

    def _adjust_peak_width(self, event, peak_identifier=-1):
        self.scroll_count += self.x_size * np.sign(event.step) *\
                             self.scrolling_speed/10

        if self.scroll_count > -self.x_size*self.initial_width*0.999:
            w2 = self.x_size*self.initial_width + self.scroll_count
        else:
            w2 = self.x_size * self.initial_width / 1000
            # This doesn't allow you to sroll to negative values
            # (basic width is x_size)
            self.scroll_count = -self.x_size * self.initial_width * 0.999

        center2 = self.pic['x0'][peak_identifier]
        h2 = self.pic['h'][peak_identifier]
        self.pic['w'][peak_identifier] = w2
        yy = pV(x=self.x, x0=center2, h=h2, w=w2, factor=self.GL)
        active_line = self.pic['line'][peak_identifier][0]
        # This updates the values on the peak identified
        active_line.set_ydata(yy)
        self.ax.draw_artist(active_line)
        self.fig.canvas.draw_idle()
#        return(scroll_count, pic)

    def _remove_peak(self, clicked_indice):
        self.artists[clicked_indice].remove()
        self.artists.pop(clicked_indice)
        self.ax.lines.remove(self.pic['line'][clicked_indice][0])
        self.pic['line'].pop(clicked_indice)
        self.pic['x0'].pop(clicked_indice)
        self.pic['h'].pop(clicked_indice)
        self.pic['w'].pop(clicked_indice)
        self.fig.canvas.draw_idle()
        self.peak_counter -= 1
#        return(artists, pic)

    def _draw_peak_sum(self):
        if self.peak_counter < 1:
            return

        def _remove_sum(self):
            assert self.cum_graph_present == 1, "no sum drawn, nothing to remove"
            self.ax.lines.remove(self.sum_peak[-1][0])
            self.sum_peak.pop()
            self.cum_graph_present -= 1
#            return sum_peak

        def _add_sum(self, sumy):
            assert sumy.shape == self.x.shape, "something's wrong with your data"
            self.sum_peak.append(self.ax.plot(self.x, sumy, '--',
                                              color='lightgreen',
                                              lw=3, alpha=0.6))
            self.cum_graph_present += 1
#            return sum_peak

        # Sum all the y values from all the peaks:
        sumy = np.sum(np.asarray(
                [self.pic['line'][i][0].get_ydata() for i in range(self.peak_counter)]),
                axis=0)
        # Check if there is already a cumulated graph plotted:
        if self.cum_graph_present == 1:
            # Check if the sum of present peaks correponds to the cumulated graph
            if np.array_equal(self.sum_peak[-1][0].get_ydata(), sumy):
                pass
            else:  # if not, remove the last cumulated graph from the figure:
                _remove_sum(self)
                # and then plot the new cumulated graph:
                _add_sum(self, sumy=sumy)
        # No cumulated graph present:
        elif self.cum_graph_present == 0:
            # plot the new cumulated graph
            _add_sum(self, sumy=sumy)

        else:
            raise("WTF?")
        self.fig.canvas.draw_idle()
#        return(cum_graph_present, sum_peak)

    def onclick(self, event):
        if event.inaxes == self.ax:  # if you click inside the plot
            if event.button == 1:  # left click
                # Create list of all elipes and check if the click was inside:
                click_in_artist = [art.contains(event)[0] for art in self.artists]
                if any(click_in_artist):  # if click was on any of the elipsis
                    clicked_indice = click_in_artist.index(True) # identify the one
                    self._remove_peak(clicked_indice=clicked_indice)

                else:  # if click was not on any of the already drawn elipsis
                    self._add_peak(event)

            elif event.button == 3 and not event.step:
                # On some computers middle and right click have both the value 3
                self._draw_peak_sum()

            elif event.step != 0:
                if self.peak_counter:
                    self._adjust_peak_width(event, peak_identifier=-1)
                    # -1 means that scrolling will only affect the last plotted peak

            elif event.button !=1 and event.dblclick:
                # Double Middle (or Right?) click ends the show
                assert len(self.pic['line']) == self.peak_counter
                assert self.cum_graph_present == len(self.sum_peak)
                self.fig.canvas.mpl_disconnect(self.cid)
                self.fig.canvas.mpl_disconnect(self.cid2)
                self.pic['GL'] = [self.GL] * self.peak_counter
                self.block = False
