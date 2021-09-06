#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:13:46 2021

@author: dejan
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io, transform
from matplotlib.widgets import Slider, Button, AxesWidget, RadioButtons, SpanSelector
from matplotlib.patches import Circle
from warnings import warn
import calculate as cc
import preprocessing as pp
try:
    from read_WDF_class import WDF
except:
    pass


class ShowCollection(object):
    """Visualize a collection of images.

    Parameters
    ----------
    image_pattern : str
        Can take asterixes as wildcards. For ex.: "./my_images/*.jpg" to select
        all the .jpg images from the folder "my_images"
    load_func : function
        The function to apply when loading the images
    first_frame : int
        The frame from which you want to stard your slideshow
    load_func_kwargs : dict
        The named arguments of the load function

    Outputs
    -------
    Interactive graph displaying the images one by one, whilst you can
    scroll trough the collection using the slider or the keyboard arrows

    Example
    -------
    >>> import numpy as np
    >>> from skimage import io, transform

    >>> def binarization_load(f, shape=(132,132)):
    >>>     im = io.imread(f, as_gray=True)
    >>>     return transform.resize(im, shape, anti_aliasing=True)

    >>> ss = ShowCollection(images, load_func=binarization_load, shape=(128,128))
    """

    def __init__(self, image_pattern, load_func=io.imread, first_frame=0,
                 **load_func_kwargs):

        self.coll_all = io.ImageCollection(image_pattern, load_func=load_func,
                                           **load_func_kwargs)
        self.first_frame = first_frame
        self.nb_pixels = self.coll_all[0].size
        self.titles = np.arange(len(self.coll_all))
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        self.last_frame = len(self.coll_all)-1
        self.line = plt.imshow(self.coll_all[self.first_frame])
        self.ax.set_title(f"{self.titles[self.first_frame]}")

        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)

        self.sframe = Slider(self.axframe, 'Frame', self.first_frame,
                             self.last_frame, valinit=self.first_frame,
                             valfmt='%d', valstep=1)
        # calls the update function when changing the slider position
        self.sframe.on_changed(self.update)

        # Calling the press function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        plt.show()

    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        img = self.coll_all[frame]
        self.line.set_data(img)
        self.ax.set_title(f"{self.titles[frame]}")
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use the left and right arrow keys to scroll through frames one by one"""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.coll_all)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.coll_all[new_frame]
        self.line.set_data(img)
        self.ax.set_title(f"{self.titles[new_frame]}")
        self.fig.canvas.draw_idle()


# %%

class AllMaps(object):
    """Rapidly visualize maps of Raman spectra.

    You can also choose to visualize the map and plot the
    corresponding component side by side if you set the
    "components" parameter.

    Parameters
    ----------
    map_spectra : 3D ndarray
        the spectra shaped as (n_lines, n_columns, n_wavenumbers)
    sigma : 1D ndarray
        an array of wavenumbers (len(sigma)=n_wavenumbers)
    components: 2D ndarray
        The most evident use-case would be to help visualize the decomposition
        results from PCA or NMF. In this case, the function will plot the
        component with the corresponding map visualization of the given
        components' presence in each of the points in the map.
        So, in this case, your map_spectra would be for example
        the matrix of components' contributions in each spectrum,
        while the "components" array will be your actual components.
        In this case you can ommit your sigma values or set them to
        something like np.arange(n_components)
    components_sigma: 1D ndarray
        in the case explained above, this would be the actual wavenumbers
    **kwargs: dict
        can only take 'title' as a key for the moment

    Returns
    -------
    The interactive visualization.\n
    (you can scroll through sigma values with a slider,
     or using left/right keyboard arrows)
    """

    def __init__(self, map_spectra, sigma=None, components=None,
                 components_sigma=None, **kwargs):
        if isinstance(map_spectra, WDF):
            shape = map_spectra.map_params['NbSteps'][
                    map_spectra.map_params['NbSteps'] > 1][::-1]
            self.map_spectra = map_spectra.spectra.reshape(tuple(shape) + (-1,))
            self.sigma = map_spectra.x_values
        else:
            self.map_spectra = map_spectra
            self.sigma = sigma
            if sigma is None:
                self.sigma = np.arange(map_spectra.shape[-1])
        assert self.map_spectra.shape[-1] == len(
                self.sigma), "Check your Ramans shifts array"

        self.first_frame = 0
        self.last_frame = len(self.sigma)-1
        if components is not None:
            # assert len(components) == map_spectra.shape[-1], "Check your components"
            self.components = components
            if components_sigma is None:
                self.components_sigma = np.arange(components.shape[-1])
            else:
                self.components_sigma = components_sigma
        else:
            self.components = None
        if components is not None:
            self.fig, (self.ax2, self.ax, self.cbax) = plt.subplots(
                ncols=3, gridspec_kw={'width_ratios': [40, 40, 1]})
            self.cbax.set_box_aspect(
                40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
        else:
            self.fig, (self.ax, self.cbax) = plt.subplots(
                ncols=2, gridspec_kw={'width_ratios': [40, 1]})
            self.cbax.set_box_aspect(
                40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
            # self.cbax = self.fig.add_axes([0.92, 0.3, 0.03, 0.48])
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)

        self.im = self.ax.imshow(self.map_spectra[:, :, 0])
        self.im.set_clim(np.percentile(self.map_spectra[:, :, 0], [1, 99]))
        if self.components is not None:
            self.line, = self.ax2.plot(
                self.components_sigma, self.components[0])
            self.ax2.set_box_aspect(
                self.map_spectra.shape[0]/self.map_spectra.shape[1])
            self.ax2.set_title(f"Component {0}")
        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes(
            [0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)

        self.sframe = Slider(self.axframe, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)

        self.my_cbar = mpl.colorbar.Colorbar(self.cbax, self.im)

        # calls the "update" function when changing the slider position
        self.sframe.on_changed(self.update)
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def titled(self, frame):
        if self.components is None:
            if self.title is None:
                self.ax.set_title(f"Raman shift = {self.sigma[frame]:.1f}cm⁻¹")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")
        else:
            self.ax2.set_title(f"Component {frame}")
            if self.title is None:
                self.ax.set_title(f"Component n°{frame} contribution")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")

    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        img = self.map_spectra[:, :, frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, [1, 99]))
        if self.components is not None:
            self.line.set_ydata(self.components[frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use the left and right arrow keys to scroll through frames one by one."""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.sigma)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.map_spectra[:, :, new_frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, [1, 99]))
        self.titled(new_frame)
        if self.components is not None:
            self.line.set_ydata(self.components[new_frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()

# %%


class ShowSpectra(object):
    """Rapidly visualize Raman spectra.

    Imortant: Your spectra can either be a 2D ndarray
    (1st dimension is for counting the spectra, the 2nd dimension is for the intensities)
    And that would be the standard use-case, But:
    Your spectra can also be a 3D ndarray,
    In which case the last dimension is used to store additional spectra
    (for the same pixel)
    Fo example, you can store spectra, the baseline and the corrected spectra all together.

    Returns
    -------
    The interactive visualization.\n
    (you can scroll through the spectra with a slider,
     or using left/right keyboard arrows)
    """

    def __init__(self, my_spectra, sigma=None, **kwargs):

        if isinstance(my_spectra, WDF):
            self.my_spectra = my_spectra.spectra
            self.sigma = my_spectra.x_values
        else:
            self.my_spectra = my_spectra
            if sigma is None:
                self.sigma = np.arange(self.my_spectra.shape[1])
            else:
                self.sigma = sigma
        if self.my_spectra.ndim == 2:
            self.my_spectra = self.my_spectra[:,:,np.newaxis]
        assert self.my_spectra.shape[1] == len(
                self.sigma), "Check your Raman shifts array. The dimensions "+\
                f"of your spectra ({self.my_spectra.shape[1]}) and that of "+\
                f"your Ramans shifts ({len(self.sigma)}) are not the same."

        self.first_frame = 0
        self.last_frame = len(self.my_spectra)-1
        self.fig, self.ax = plt.subplots()
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)

        self.spectrumplot = self.ax.plot(self.sigma, self.my_spectra[0])
        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes(
            [0.15, 0.1, 0.7, 0.03])#, facecolor=self.axcolor)
        # self.axframe.plot(self.sigma, np.median(self.my_spectra, axis=0))
        self.sframe = Slider(self.axframe, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)

        # calls the "update" function when changing the slider position
        self.sframe.on_changed(self.update)
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def titled(self, frame):
        if self.title is None:
            self.ax.set_title(f"Spectrum N° {frame} /{self.last_frame + 1}")
        elif isinstance(self.title, str):
            self.ax.set_title(f"{self.title} n°{frame}")
        elif hasattr(self.title, '__iter__'):
            self.ax.set_title(f"{self.title[frame]}")

    def update(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.sframe.val)
        current_spectrum = self.my_spectra[frame]
        for i,line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:,i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        """Use the left and right arrow keys to scroll through frames one by one."""
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < self.last_frame:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        current_spectrum = self.my_spectra[new_frame]
        for i,line in enumerate(self.spectrumplot):
            line.set_ydata(current_spectrum[:,i])
            self.ax.relim()
            self.ax.autoscale_view()
        self.titled(new_frame)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()

# %%


class NavigationButtons(object):
    """Interactivly visualize multispectral data.

    Navigate trough your spectra by simply clicking on the navigation buttons.

    Parameters
    ----------
        sigma: 1D ndarray
            1D numpy array of your x-values (raman shifts, par ex.)
        spectra: 2D or 3D ndarray
            3D or 2D ndarray of shape (n_spectra, len(sigma), n_curves).
            The last dimension may be ommited it there is only one curve
            to be plotted for each spectra).
        autoscale: bool
            determining if you want to adjust the scale to each spectrum
        title: str
            The initial title describing where the spectra comes from
        label: list
            A list explaining each of the curves. len(label) = n_curves

    Output
    ------
        matplotlib graph with navigation buttons to cycle through spectra

    Example
    -------
        Let's say you have a ndarray containing 10 spectra,
        and let's suppose each of those spectra contains 500 points.

        >>> my_spectra.shape
        (10, 500)
        >>> sigma.shape
        (500, )

        Then let's say you show the results of baseline substraction.

        >>> my_baseline[i] = baseline_function(my_spectra[i])
        your baseline should have the same shape as your initial spectra.
        >>> multiple_curves_to_plot = np.stack(
                (my_spectra, my_baseline, my_spectra - my_baseline), axis=-1)
        >>> NavigationButtons(sigma, multiple_curves_to_plot)
    """
    ind = 0

    def __init__(self, sigma, spectra, autoscale_y=False, title='Spectrum',
                 label=False, **kwargs):
        self.y_autoscale = autoscale_y

        if len(spectra.shape) == 2:
            self.s = spectra[:, :, np.newaxis]
        elif len(spectra.shape) == 3:
            self.s = spectra
        else:
            raise ValueError("Check the shape of your spectra.\n"
                             "It should be (n_spectra, n_points, n_curves)\n"
                             "(this last dimension might be ommited"
                             "if it's equal to one)")
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
            if len(label) == self.s.shape[2]:
                self.label = label
            else:
                warn(
                    "You should check the length of your label list.\n"
                    "Falling on to default labels...")
                self.label = ["Curve n°"+str(numb)
                              for numb in range(self.s.shape[2])]
        else:
            self.label = ["Curve n°"+str(numb)
                          for numb in range(self.s.shape[2])]

        self.figr, self.axr = plt.subplots(**kwargs)
        self.axr.set_title(f'{title[0]}')
        self.figr.subplots_adjust(bottom=0.2)
        # l potentially contains multiple lines
        self.line = self.axr.plot(self.sigma, self.s[0], lw=2, alpha=0.7)
        self.axr.legend(self.line, self.label)
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

    def update_data(self):
        _i = self.ind % self.n_spectra
        for ll in range(len(self.line)):
            yl = self.s[_i][:, ll]
            self.line[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next1(self, event):
        self.ind += 1
        self.update_data()

    def next10(self, event):
        self.ind += 10
        self.update_data()

    def next100(self, event):
        self.ind += 100
        self.update_data()

    def next1000(self, event):
        self.ind += 1000
        self.update_data()

    def prev1(self, event):
        self.ind -= 1
        self.update_data()

    def prev10(self, event):
        self.ind -= 10
        self.update_data()

    def prev100(self, event):
        self.ind -= 100
        self.update_data()

    def prev1000(self, event):
        self.ind -= 1000
        self.update_data()

# %%

class ShowSelected(object):
    """Select a span and plot a map of a chosen function in that span.

    To be used for visual exploration of the maps.
    The lower part of the figure contains the spectra you can scroll through
    using the slider just beneath the spectra.
    You can use your mouse to select a zone in the spectra and a plot should
    appear in the upper part of the figure.
    On the left part of the figure you can select what kind of function
    you want to apply on the selected span."""


    def __init__(self, map_spectra, x=None):

        if isinstance(map_spectra, WDF):
            self.x = map_spectra.x_values
            self.nshifts = map_spectra.npoints
            self.nx, self.ny = map_spectra.map_params['NbSteps'][
                                map_spectra.map_params['NbSteps'] > 1]
            self.spectra = map_spectra.spectra
            self.xlabel, self.ylabel = map_spectra.map_params["StepSizes"][
                                    map_spectra.map_params["StepSizes"] > 0]
        else:
            self.x = x
            self.ny, self.nx, self.nshifts = map_spectra.shape
            self.spectra = map_spectra.reshape(-1, self.nshifts)
        if self.x is None:
            self.x = np.arange(self.nshifts)
        self.spectra, self.x = pp.order(self.spectra, self.x)
        self.map_spectra = self.spectra.reshape(self.ny, self.nx, self.nshifts)

        # Preparing the plot:
        self.fig = plt.figure()
        # Add all the axes:
        self.aximg = self.fig.add_axes([.23, .3, .8, .6])
        self.axspectrum = self.fig.add_axes([.05, .075, .9, .15])
        self.axradio = self.fig.add_axes([.075, .275, .1, .6])
        self.axscroll = self.fig.add_axes([.05, .02, .9, .02])

        self.axradio.axis('off')

        self.first_frame = 0
        self.last_frame = len(self.spectra)-1
        self.sframe = Slider(self.axscroll, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)
        self.sframe.on_changed(self.scroll_spectra)

        self.spectrumplot, = self.axspectrum.plot(self.x, self.spectra[0])
        self.titled(self.axspectrum, 0)
        self.vline = None
        self.func = "max"
        self.xmin = None
        self.xmax = None
        self.reduced_x = None
        self.span = SpanSelector(self.axspectrum, self.onselect, 'horizontal',
                                 useblit=True, span_stays=True,
                                 rectprops=dict(alpha=0.5,
                                                facecolor='tab:blue'))
        self.func_choice = RadioButtons(self.axradio,
                                         ["max",
                                          "reduced max",
                                          "peak position",
                                          "barycenter x",
                                          "reduced barycenter x",
                                          "area",
                                          "reduced area"])
        self.func_choice.on_clicked(self.determine_func)

        # Plot the empty image:
        self.imup = self.aximg.imshow(np.empty_like(self.map_spectra[:,:,0]))
        if isinstance(map_spectra, WDF):
            self.aximg.set_xlabel(f"units :  {self.xlabel:.1g}")
            self.aximg.set_ylabel(f"units :  {self.ylabel:.1g}")
        plt.show()

    def determine_func(self, label):
        self.func = label
        if self.xmin: # if area selected, change img on click
            self.onselect(self.xmin, self.xmax)

    def straightline_koeffs(self):
        y1_arr = self.spectra[:, self.indmin]
        y2_arr = self.spectra[:, self.indmax]
        a_arr = (y2_arr - y1_arr) / (self.x[self.indmax] - self.x[self.indmin])
        b_arr = y1_arr - a_arr * self.x[self.indmin]
        return a_arr, b_arr

    def calc_func(self):
        self.reduced_x = self.x[self.indmin:self.indmax]
        self.aximg.set_title(f"Calculated {self.func} between "
                             f"{self.reduced_x[0]:.2f} and "
                             f"{self.reduced_x[-1]:.2f} cm-1")

        if self.func.split()[0] == "reduced":
            self.func = ' '.join(self.func.split()[1:][:]) # The "reduced" part of the name is only used here
            A, B = self.straightline_koeffs()
            straight_line = np.outer(A , self.reduced_x) + B[:, np.newaxis]
            reduced_spectra = self.spectra[:, self.indmin:self.indmax] - \
                straight_line
        else:
            reduced_spectra = self.spectra[:, self.indmin:self.indmax]

        if self.func == "max":
            return np.max(reduced_spectra,
                          axis=-1).reshape(self.ny, self.nx)
        elif self.func == "area":
            if np.ptp(self.reduced_x) == 0:
                return np.ones((self.ny, self.nx))
            else:
                return np.trapz(reduced_spectra,
                                x=self.reduced_x
                                ).reshape(self.ny, self.nx)
        elif self.func == "peak position":
            return self.reduced_x[np.argmax(reduced_spectra,
                             axis=-1)].reshape(self.ny, self.nx)
        elif self.func == "barycenter x":
            return cc.find_barycentre(self.reduced_x, reduced_spectra,
                                      method="weighted_mean"
                                      )[0].reshape(self.ny, self.nx)

    def onselect(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        if self.vline:
            self.axspectrum.lines.remove(self.vline)
            self.vline = None
        self.indmin, self.indmax = np.searchsorted(self.x, (xmin, xmax))
        self.indmax = min(len(self.x) - 1, self.indmax)
        if self.indmax == self.indmin:
            self.indmax = self.indmin + 1
            self.vline = self.axspectrum.axvline(xmin)
        img = self.calc_func()
        biggest = np.max(img)
        smallest = np.min(img)
        img /= biggest
        self.imup.set_data(img)
        self.imup.set_clim(np.percentile(img, [1, 99]))
        self.aximg.set_title(f"Max/Min value = {biggest:.2f} / {smallest:.2f}")
        self.fig.canvas.draw()

    def scroll_spectra(self, val):
        """Use the slider to scroll through individual spectra"""
        frame = int(self.sframe.val)
        current_spectrum = self.spectra[frame]
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()

    def titled(self, ax, frame):
        ax.set_title(f"Spectrum N° {frame} /{self.last_frame + 1}")

class FindBaseline(object):
    """Visualy adjust parameters for the baseline.

    Parameters
    ----------
    my_spectra: 2D ndarray

    Returns
    -------
    The interactive graph facilitating the parameter search.
    You can later recover the parameters with:
        MyFindBaselineInstance.p_val
        MyFindBaselineInstance.lam_val

    Note that you can use the same function for smoothing
    (by setting the `p_val` to 0.5 and `lam_val` to some "small" value (like 13))
    """

    def __init__(self, my_spectra, sigma=None, **kwargs):
        if my_spectra.ndim == 1:
            self.my_spectra = my_spectra[np.newaxis, :]
        else:
            self.my_spectra = my_spectra
        if sigma is None:
            self.sigma = np.arange(my_spectra.shape[1])
        else:
            assert my_spectra.shape[-1] == len(
                sigma), "Check your Raman shifts array"
            self.sigma = sigma

        self.nb_spectra = len(self.my_spectra)
        self.current_spectrum = self.my_spectra[0]
        self.title = kwargs.get('title', None)
        self.p_val = 5e-5
        self.lam_val = 1e5

        self.fig = plt.figure(figsize=(14, 10))
        # Add all the axes:
        self.ax = self.fig.add_axes([.2, .15, .75, .8]) # [left, bottom, width, height]
        self.axpslider = self.fig.add_axes([.05, .15, .02, .8], yscale='log')
        self.axlamslider = self.fig.add_axes([.1, .15, .02, .8], yscale='log')
        if self.nb_spectra > 1: # scroll through spectra if there are many
            self.axspectrumslider = self.fig.add_axes([.2, .05, .75, .02])
            self.spectrumslider = Slider(self.axspectrumslider, 'Frame',
                                         0, self.nb_spectra-1,
                                         valinit=0, valfmt='%d', valstep=1)
            self.spectrumslider.on_changed(self.spectrumupdate)

        self.pslider = Slider(self.axpslider, 'p-value',
                                     1e-10, 1, valfmt='%.2g',
                                     valinit=self.p_val,
                                     orientation='vertical')
        self.lamslider = Slider(self.axlamslider, 'lam-value',
                                     .1, 1e10, valfmt='%.2g',
                                     valinit=self.lam_val,
                                     orientation='vertical')
        self.pslider.on_changed(self.blupdate)
        self.lamslider.on_changed(self.blupdate)

        self.spectrumplot, = self.ax.plot(self.sigma, self.current_spectrum,
                                          label="original spectrum")
        self.bl = cc.baseline_als(self.current_spectrum, p=self.p_val,
                                  lam=self.lam_val)
        self.blplot, = self.ax.plot(self.sigma, self.bl, label="baseline")
        self.corrplot, = self.ax.plot(self.sigma,
                                      self.current_spectrum - self.bl,
                                      label="corrected_plot")
        self.ax.legend()
        self.titled(0)

        plt.show()

    def titled(self, frame):
        if self.title is None:
            self.ax.set_title(f"Spectrum N° {frame} /{self.nb_spectra}")
        else:
            self.ax.set_title(f"{self.title} n°{frame}")

    def spectrumupdate(self, val):
        """Use the slider to scroll through frames"""
        frame = int(self.spectrumslider.val)
        self.current_spectrum = self.my_spectra[frame]
        self.spectrumplot.set_ydata(self.current_spectrum)
        self.blupdate(val)
        self.ax.relim()
        self.ax.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def blupdate(self, val):
        self.p_val = self.pslider.val
        self.lam_val = self.lamslider.val
        self.bl = cc.baseline_als(self.current_spectrum, p=self.p_val, lam=self.lam_val)
        self.blplot.set_ydata(self.bl)
        self.corrplot.set_ydata(self.current_spectrum - self.bl)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
