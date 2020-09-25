#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Jun 11 15:28:47 2019

@author: dejan
"""
import numpy as np
from joblib import Parallel, delayed
from warnings import warn
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from cycler import cycler
from scipy import sparse
from scipy.ndimage import median_filter
from scipy.optimize import minimize_scalar


def find_barycentre(x, y, method="trapz_minimize"):
    '''Calculates the index of the barycentre value.
        Parameters:
        ----------
        x:1D ndarray: ndarray containing your raman shifts
        y:1D ndarray: Ndarray containing your intensity (counts) values
        method:string: only "trapz_minimize" for now
        Returns:
        ---------
        (x_value, y_value): the coordinates of the barycentre
        '''
    assert(method in ['trapz_minimize'])#, 'sum_minimize', 'trapz_list'])
    razlika = np.asarray(np.diff(x, append=x[-1]+x[-1]-x[-2]), dtype=np.float16)
    #assert(np.all(razlika/razlika[np.random.randint(len(x))] == np.ones_like(x))),\
    #"your points are not equidistant"
    half = np.trapz(y, x=x)/2
    #from scipy.interpolate import interp1d
    #xx=np.linspace(x.min(), x.max(), 2*len(x))
    #f = interp1d(x, y, kind='quadratic')
    #yy = f(xx)
    if method in 'trapz_minimize':
        def find_y(Y0, xx=x, yy=y, method=method):
            '''Internal function to minimize
            depending on the method chosen'''
            # Calculate the area of the curve above the Y0 value:
            part_up = np.trapz(yy[yy>=Y0]-Y0, x=xx[yy>=Y0])
            # Calculate the area below Y0:
            part_down = np.trapz(yy[yy<=Y0], x=xx[yy<=Y0])
            # for the two parts to be the same
            to_minimize_ud = np.abs(part_up - part_down)
            # fto make the other part be close to half
            to_minimize_uh = np.abs(part_up - half)
            # to make the other part be close to half
            to_minimize_dh = np.abs(part_down - half)
            return to_minimize_ud**2+to_minimize_uh+to_minimize_dh

        def find_x(X0, xx=x, yy=y, method=method):
            part_left = np.trapz(yy[xx<=X0], x=xx[xx<=X0])
            part_right = np.trapz(yy[xx>=X0], x=xx[xx>=X0])
            to_minimize_lr = np.abs(part_left - part_right)
            to_minimize_lh = np.abs(part_left - half)
            to_minimize_rh = np.abs(part_right - half)
            return to_minimize_lr**2+to_minimize_lh+to_minimize_rh

        minimized_y = minimize_scalar(find_y, method='Bounded',
                                    bounds=(np.quantile(y, 0.01),
                                            np.quantile(y, 0.99)))
        minimized_x = minimize_scalar(find_x, method='Bounded',
                                    bounds=(np.quantile(x, 0.01),
                                            np.quantile(x, 0.99)))
        y_value = minimized_y.x
        x_value = minimized_x.x

    elif method == "list_minimize":
        ys = np.sort(yy)
        z2 = np.asarray(
            [np.abs(np.trapz(yy[yy<=y_val], x=xx[yy<=y_val]) -\
                    np.trapz(yy[yy>=y_val]-y_val, x=xx[yy>=y_val]))\
             for y_val in ys])
        y_value = ys[np.argmin(z2)]
        x_ind = np.argmin(np.abs(np.cumsum(yy) - np.sum(yy)/2)) + 1
        x_value = xx[x_ind]

    return x_value, y_value


def rolling_median(arr, w_size, ax=0, mode='nearest', *args):
    '''Calculates the rolling median of an array
    along the given axis on the given window size.
    Parameters:
    -------------
        arr:ndarray: input array
        w_size:int: the window size
                    (should be less then the dimension along the given axis)
        ax:int: the axis along which to calculate the rolling median
        mode:str: to choose from ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
        see the docstring of ndimage.median_filter for details
    Returns:
    ------------
        ndarray of same shape as the input array'''
    shape = np.ones(np.ndim(arr), dtype=int)
    shape[ax] = w_size
    return median_filter(arr, size=shape, mode=mode, *args)



def baseline_als(y, lam=1e5, p=5e-5, niter=12):
    '''Adapted from:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    To get the feel on how the algorithm works, you can think of it as
    if the rolling ball which comes from beneath the spectrum and thus sets
    the baseline.

    Then, to follow the image, schematic explanaton of the params would be:

    Params:
    ----------
        y:          1D or 2D ndarray: the spectra on which to find the baseline

        lam:number: Can be viewed as the radius of the ball.
                    As a rule of thumb, this value should be around the
                    twice the width of the broadest feature you want to keep
                    (width is to be measured in number of points, since
                    for the moment no x values are taken into accound
                    in this algorithm)

        p:number:   Can be viewed as the measure of how much the ball
                    can penetrate into the spectra from below

        niter:int:  number of iterations
                   (the resulting baseline should stabilize after
                    some number of iterations)

    Returns:
    -----------
        b_line:ndarray: the baseline (same shape as y)

    Note:
    ----------
        It takes around 2-3 sec per 1000 spectra with 10 iterations
        on i7 4cores(8threads) @1,9GHz

    '''
    def _one_bl(yi, lam=lam, p=p, niter=niter, z=None):
        if z is None:
            L = yi.shape[-1]
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
            w = np.ones(L)
            W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w) # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = sparse.linalg.spsolve(Z, w*yi)
            w = p * (yi > z) + (1-p) * (yi < z)
        return z

    if y.ndim == 1:
        b_line = _one_bl(y)
    elif y.ndim == 2:
        b_line = np.asarray(Parallel(n_jobs=-1)(delayed(_one_bl)(y[i])
                                                for i in range(y.shape[0])))
    else:
        warn("This only works for 1D or 2D arrays")

    return b_line


def slice_lr(spectra, sigma=None, pos_left=None, pos_right=None):
    '''
    Several reasons may make you want to apply the slicing.

    a) Your spectra might have been recorded with the dead pixels included.
    It is normaly a parameter which should had been set at the spectrometer
    configuration (Contact your spectros's manufacturer for assistance)
    b) You might want to isolate only a part of the spectra which
    interests you.
    c) You might have made a poor choice of the spectral range at the
       moment of recording the spectra.

    Parameters:
    ---------------
    spectra: N-D ndarray: your spectra. The last dimension is corresponds
                          to one spectrum recorded at given position
    sigma: 1D ndarray: your Raman shifts. Default is None, meaning
                       that the slicing will be applied based on the
                       indices of spectra, not Raman shift values
    pos_left :int or float: position from which to start the slice. If sigma
                      is given, pos_left is the lower Raman shift value,
                      if not, it's the lower index of the spectra.
    pos-right:int or float: same as for pos_left, but on the right side.
                            It can be negative (means you count from the end)

    Returns:
    ---------------
    spectra_kept: N-D ndarray: your spectra containing only the zone of
                              interest.
                              spectra_kept.shape[:-1] = spectra_shape[:-1]
                              spectra_kept.shape[-1] <= spectra.shape[-1]
    sigma_kept: 1D ndarray: if sigma is given: your Raman shift values for the
                            isolated zone.
                            len(sigma_kept)=spectra_kept.shape[-1] <=
                            len(sigma)=spectra.shape[-1]
                            if sigma is not given: indices of the zone of
                            interest.
    '''

    if sigma is None:
        sigma = np.arange(spectra.shape[-1])

    # If you pass a negative number as the right position:
    if isinstance(pos_right, (int, float)):
        if pos_right < 0:
            pos_right = sigma[pos_right]

    if pos_left is None:
        pos_left = sigma.min()
    if pos_right is None:
        pos_right = sigma.max()

    assert pos_left <= pos_right, "Check your initialization Slices!"
    _condition = (sigma >= pos_left) & (sigma <= pos_right)
    sigma_kept = sigma[_condition]  # add np.copy if needed
    spectra_kept = np.asarray(spectra[..., _condition], order='C')

    return spectra_kept, sigma_kept


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

class AllMaps(object):

    def __init__(self, map_spectra, sigma=None, first_frame=None, last_frame=None, **kwargs):
        self.map_spectra = map_spectra
        if sigma is None:
            self.sigma = np.arange(map_spectra.shape[-1])
        else:
            assert map_spectra.shape[-1] == len(sigma), "Check your Ramans shifts array"
            self.sigma = sigma

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        if first_frame is None:
            self.first_frame = 0
        else:
            assert isinstance(first_frame, int), "first_frame should be int"
            self.first_frame = first_frame
        if last_frame is None:
            self.last_frame = len(self.sigma)-1
        else:
            assert isinstance(last_frame, int), "last_frame should be int"
            self.last_frame = last_frame

        self.l = plt.imshow(self.map_spectra[:,:,0])
        self.l.set_clim(np.percentile(self.map_spectra[:,:,0], [1,99]))
        self.ax.set_title(f"Raman shift = {self.sigma[0]:.1f}cm⁻¹")

        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)


        self.sframe = Slider(self.axframe, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)

        self.cbax,_ = mpl.colorbar.make_axes(self.ax)#plt.axes([0.77, 0.2, 0.03, 0.68])
        self.my_cbar = mpl.colorbar.colorbar_factory(self.cbax, self.l)

        self.sframe.on_changed(self.update) # calls the "update" function when changing the slider position
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def update(self, val):
        '''This function is for using the slider to scroll through frames'''
        frame = int(self.sframe.val)
        img = self.map_spectra[:,:,frame]
        self.l.set_data(img)
        self.l.set_clim(np.percentile(img, [1,99]))
        self.ax.set_title(f"Raman shift = {self.sigma[frame]:.1f}cm⁻¹")
        self.fig.canvas.draw_idle()

    def press(self, event):
        '''This function is to use arrow keys left and right to scroll
        through frames one by one'''
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.sigma)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.map_spectra[:,:,new_frame]
        self.l.set_data(img)
        self.l.set_clim(np.percentile(img, [1,99]))
        self.ax.set_title(f"Raman shift = {self.sigma[new_frame]:.1f}cm⁻¹")
        self.fig.canvas.draw_idle()

# %%

class NavigationButtons(object):
    '''This class allows you to visualize multispectral data and
    navigate trough your spectra simply by clicking on the
    navigation buttons on the graph.
    -------------------
    Parameters:
        sigma: 1D numpy array of your x-values (raman shifts, par ex.)
        spectra: 3D or 2D ndarray of shape (n_spectra, len(sigma), n_curves).
                 The last dimension may be ommited it there is only one curve
                 to be plotted for each spectra),
        autoscale: bool determining if you want to adjust the scale to each spectrum
        title: The initial title describing where the spectra comes from
        label: list: A list explaining each of the curves. len(label) = n_curves
    Output:
        matplotlib graph with navigation buttons to cycle through spectra
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
    >>>NavigationButtons(sigma, multiple_curves_to_plot)
    '''
    ind = 0

    def __init__(self, sigma, spectra, autoscale_y=False, title='Spectrum', label=False,
                 **kwargs):
        self.y_autoscale = autoscale_y

        if len(spectra.shape) == 2:
            self.s = spectra[:,:, np.newaxis]
        elif len(spectra.shape) == 3:
            self.s = spectra
        else:
            raise ValueError("Check the shape of your spectra.\n"
                             "It should be (n_spectra, n_points, n_curves)\n"
                             "(this last dimension might be ommited if it's equal to one)")
        self.n_spectra = self.s.shape[0]
        if isinstance(title, list) or isinstance(title, np.ndarray):
            if len(title) == spectra.shape[0]:
                self.title = title
            else:
                raise ValueError(f"you have {len(title)} titles,\n"
                                f"but you have {len(spectra)} spectra")
        else:
            self.title = [title]*self.n_spectra

        self.sigma = sigma
        if label:
            if len(label)==self.s.shape[2]:
                self.label = label
            else:
                warn("You should check the length of your label list.\nFalling on to default labels...")
                self.label = ["Curve n°"+str(numb) for numb in range(self.s.shape[2])]
        else:
            self.label = ["Curve n°"+str(numb) for numb in range(self.s.shape[2])]

        self.figr, self.axr = plt.subplots(**kwargs)
        self.axr.set_title(f'{title[0]}')
        self.figr.subplots_adjust(bottom=0.2)
        self.l = self.axr.plot(self.sigma, self.s[0], lw=2, alpha=0.7) # l potentially contains multiple lines
        self.axr.legend(self.l, self.label)
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
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next10(self, event):
        self.ind += 10
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next100(self, event):
        self.ind += 100
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next1000(self, event):
        self.ind += 1000
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev1(self, event):
        self.ind -= 1
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev10(self, event):
        self.ind -= 10
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev100(self, event):
        self.ind -= 100
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def prev1000(self, event):
        self.ind -= 1000
        _i = (self.ind) % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
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

# %%
# Williams' functions for Raman spectra:
def long_correction(sigma, lambda_laser, T=30, T0=0):
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
    T : float
        Actual temperature in °C
    T0 : float
        The temperature to which to make the correction in °C
    Returns:
    ----------
    lcorr: numpy.ndarray of the same shape as sigma

    Examples
    --------
    >>> sigma, spectra_i = deconvolution.acquire_data('my_raman_file.CSV')
    >>> corrected_spectra = spectra_i * long_correction(sigma)
    """
    c = 2.998e10                          # cm/s
    lambda_inc = lambda_laser * 1e-7      # cm
    sigma_inc = 1. / lambda_inc           # cm-1
    h = 6.63e-34                          # J.s
    TK = 273.0 + T                        # K
    T0K = 273.0 + T0                      # K
    kB = 1.38e-23                         # J/K
    ss = sigma_inc / sigma
    cc = h*c/kB
    return (ss**3 / (ss - 1)**4
            * (1 - np.exp(cc*sigma*(1/TK-1/T0K))))

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


def rolling_window(trt, window_size, ax=0):
    '''
    NOTE: Due to usage of as_strided function from numpy.stride_tricks,
          the results are sometimes unpredictible.
          You have been warned :)

    Function to create the 1D rolling window of the given size, on the
    given axis. The "window" is added as the new dimension to the input array,
    this new dimension is set as the first (0) axis of the resulting array.
    Parameters:
        trt:ndarray: input array
        window_size:int: the size of the window, must be odd
        ax:int: the axis you want to roll the window on
    Returns:
        ndarray of the shape (window_size,)+trt.shape
    Example:
        test = (np.arange(90)**2).reshape(9,10)
    '''
    assert window_size % 2 != 0, "Window size must be odd integer!"
    ee = window_size//2
    arr_shape = np.asarray(trt.shape)
    # If we want the result to be of the same shape as input array,
    # we have to expand the edges.
    # Here, we just duplicate the edge values ee times
    to_prepend = np.asarray([np.take(trt, 0, axis=ax).tolist()]*ee)
    to_append = np.asarray([np.take(trt, -1, axis=ax).tolist()]*ee)
    # Then we need to reshape so that the concatanation works well:
    concat_shape = arr_shape
    concat_shape[ax] = 1
    to_prepend = to_prepend.reshape(tuple(concat_shape))
    to_append = to_append.reshape(tuple(concat_shape))
    # Concatenate:
    a = np.concatenate((to_prepend, trt, to_append), axis=ax)
    # Final shape (we are adding one new dimension at the beggining)
    shape = (window_size,) + trt.shape
    # that new axis will cycle trough the same values as the axis given with
    # the ax parameter
    strides = (trt.strides[ax],) + trt.strides
    # Return thus created array:
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)
