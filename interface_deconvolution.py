# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:31:28 2017

@author: M3272834
"""

#!/usr/bin/python


import matplotlib as mpl
import matplotlib.pyplot as plt

import deconvolution
import glob
import ntpath
import numpy as np

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

try: # Python 2
    
    import Tkinter as tk
    from tkFileDialog import askopenfilename
    
except ImportError: # Python 3
    
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['lines.markersize'] = 18
mpl.rcParams['legend.markerscale'] = 2
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.handlelength'] = 2
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.handletextpad'] = 0.2
mpl.rcParams['legend.labelspacing'] = .5
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['figure.figsize'] =[10,6]


#==============================================================================
#               Graphical interface for Raman spectra deconvolution
#==============================================================================


def new_interface(new_file):

    """
    Open a new window for every new file that needs to be analysed.
    """
    
    
    #==========================================================================
    #               Functions to load/save data from/to text files
    #==========================================================================


    def load_raw_file(filename, delimiter = '\t'):
        
        """
        Load a raw text file with Raman spectroscopy measurements.
        """
            
        xtot  = np.genfromtxt(filename, delimiter = delimiter, usecols = 0, skip_header = 1)
        ytot = np.genfromtxt(filename, delimiter = delimiter, usecols = 1, skip_header = 1)
        #ztot = np.genfromtxt(data_files[0], delimiter = delimiter, usecols = 2, skip_header = 1)
        sigma = np.genfromtxt(filename, delimiter = delimiter, usecols = 2, skip_header = 1)
        
        s0,n_sigma = sigma[0],1
        while sigma[n_sigma] != s0:
            n_sigma += 1
        
        n_spectra = xtot.size/n_sigma
        sigma = sigma[:n_sigma]
        spectratot = np.genfromtxt(filename, delimiter = delimiter, usecols = 3, skip_header = 1)
        spectra = np.array(np.split(spectratot, n_spectra))
    
        xsel = np.array(xtot[::n_sigma])
        ysel = np.array(ytot[::n_sigma])
        #zsel = np.array(ztot[::n_sigma])    
        
        i,k,x0,y0 = 0,0,xtot[0],ytot[0]
        while xtot[i] == x0 and i<xtot.size-1:
            i += 1
        while ytot[k] == y0 and k<xtot.size-1:
            k += 1
        dx,dy = xtot[i]-x0,ytot[k]-y0
        
        return xsel,ysel,sigma,spectra,dx,dy,n_sigma,n_spectra
    
    def load_saved_file(filename):
        
        """
        Load a text file that has been saved from a raw file with the function
        save_to_file.
        """
        
        with open(filename,'r') as f:
            f.readline()
            s = f.readline()
            [n_comp,n_sigma,n_spectra,dx,dy] = [float(x) for x in s.split()]
            [n_comp,n_sigma,n_spectra,dx,dy] = [int(n_comp),int(n_sigma),int(n_spectra),dx,dy]
        
        sigma = np.genfromtxt(filename, usecols = 0, skip_header = 2, skip_footer = n_spectra).transpose()
        
        components = np.genfromtxt(filename, usecols = range(1,n_comp+1),
                                   skip_header = 2, skip_footer = n_spectra).transpose()
                                   
        xsel = np.genfromtxt(filename, usecols = 0, skip_header = 3+n_sigma).transpose()
        ysel = np.genfromtxt(filename, usecols = 1, skip_header = 3+n_sigma).transpose()
        mix = np.genfromtxt(filename, usecols = range(2,n_comp+2), skip_header = 3+n_sigma)
        return xsel,ysel,sigma,dx,dy,n_sigma,n_spectra,n_comp,components,mix
        
    def save_to_file(sigma,n_sigma,n_spectra,xsel,ysel,dx,dy,components,mix):
        
        """
        Save n_sigma,n_spectra,sigma,components,mix,xsel,ysel,dx,dy to a text
        file.
        """
        
        n_comp = int(n_components.get())
        file_path = filename.replace('.txt','_save_{}_composantes.txt'.format(n_comp))
        f = open(file_path,'wb')
        np.savetxt(f,[[n_comp,n_sigma,n_spectra,dx,dy]],
                   header='n_components n_sigma n_spectra dx dy')
        A = np.transpose(np.vstack((sigma,components)))
        np.savetxt(f,A,header='\t sigma \t\t\t components')
        B = np.transpose(np.vstack((xsel,ysel,mix.transpose())))
        np.savetxt(f,B,header='\t x \t\t\t y \t\t\t mix')
        f.close()
        return
    
    
    #==========================================================================
    #               Functions for data analysis and plotting
    #==========================================================================
    
    
    def traitement(sigma,spectra,L):
        
        """
        Clean the raw spectra and perform the deconvolution.
        """
        
        L[0] = deconvolution.clean(sigma, spectra)
        n_comp = int(n_components.get())
        components1, mix1, error1 = deconvolution.nmf_step(L[0], n_comp)
        L[1],L[2],L[3] = components1,mix1,error1
        
        return
    
    def composantes_plot(sigma,components):
        
        """
        Plot the spectrum of each component.
        """
        
        cmap = plt.get_cmap('gnuplot')
        n_comp = int(n_components.get())
        colors = [cmap(i) for i in np.linspace(0,1,n_comp)]
        for i, color in enumerate(colors, start = 0):
            fig,ax1 = plt.subplots()
            ax1.plot(sigma, components[i], color = color)
            ax1.set_ylim([0,max(components[i])+0.005])
            plt.title('Composante '+ str(i+1))
            plt.ylabel('AU')
            plt.xlabel('Wavenumber (cm-1)')
        
        plt.show()
        return
    
    def plot_2D(dx,dy,xsel,ysel,mix,bouton):
        
        """
        Plot a 2D map with the proportion of each component
        at differents spatial locations.
        """
        
        x_av = np.average(xsel)
        y_av = np.average(ysel)
        if bouton.get():
            x_fin = xsel-x_av
            y_fin = ysel-y_av
        else:
            x_fin = xsel
            y_fin = ysel
        
        if dx != 0 and dy !=0:
            y, x = np.mgrid[slice(min(y_fin), max(y_fin) + dy, dy),
                        slice(min(x_fin), max(x_fin) + dx, dx)]
            
            if x[0,0]>y[0,0]:
                x_2 = y
                y_2 = x
                x = x_2
                y = y_2
        
            z =[]
            n_comp = int(n_components.get())
            
            for i in np.arange(n_comp):
                z = mix[:,i]
                z = np.reshape(z,(len(y),len(y[0])))
                z = z[:-1, :-1]
                levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
                cmap = plt.get_cmap('hot')
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        
                fig, (ax0, ax1) = plt.subplots(nrows=2)
                im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
                fig.colorbar(im, ax=ax0)
                ax0.set_title('Composante '+str(i+1))
                cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                          y[:-1, :-1] + dy/2., z, levels=levels,
                          cmap=cmap)
                fig.colorbar(cf, ax=ax1)
                fig.tight_layout()
                plt.xlabel(u'X (µm)')
                plt.ylabel(u'Y (µm)')
        elif dy == 0:
            n_comp = int(n_components.get())
            for i in range(n_comp):
                plt.figure()
                plt.plot(x_fin,mix[:,i])
                plt.title('Composante {}'.format(i+1))
                plt.ylabel('Mix')
                plt.xlabel(u'X (µm)')
        elif dx == 0:
            n_comp = int(n_components.get())
            for i in range(n_comp):
                plt.figure()
                plt.plot(y_fin,mix[:,i])
                plt.title('Composante {}'.format(i+1))
                plt.ylabel('Mix')
                plt.xlabel(u'Y (µm)')
                
        plt.show()
        
        return
    
    def plot_3D(xsel,ysel,mix):
        
        x_av = np.average(xsel)
        y_av = np.average(ysel)
    #    z_av = np.average(zsel)
        x_fin = xsel-x_av
        y_fin = ysel-y_av
    #    z_fin = zsel-z_av
        min_test = min(x_fin + y_fin + z_fin)
        max_test = max(x_fin + y_fin + z_fin)
    
        c = []  
        n_comp = int(n_components.get())
        
        for i in np.arange(n_comp): 
            c = mix[:,i]
            im = (mix[:,i],mix[:,i])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            cax = ax.imshow(im, interpolation='nearest')
            ax = plt.axes(projection='3d')
            ax.scatter(x_fin, y_fin, z_fin, c=c)
            ax.set_xlim([min_test,max_test])
            ax.set_ylim([min_test,max_test])
            ax.set_zlim([min_test,max_test])
            ax.set_xlabel(u'X (µm)')
            ax.set_ylabel(u'Y (µm)')
            ax.set_zlabel(u'Z (µm)')
            plt.title('Composante '+ str(i+1))
            plt.colorbar(cax)
        
        plt.show()
            
        return
    
    
    #==========================================================================
    #                     Initialisation of the new window
    #==========================================================================
        

    global fenetre,n_window
    
    new_fenetre = tk.Toplevel(fenetre)
    new_fenetre.update() # Needed to properly close the window created by askopenfilename
                         # and avoid errors with n_components.get
    
    filename = askopenfilename(title="Ouvrir votre document",
                               filetypes=[('txt files','.txt'),('all files','.*')])
    
    #filename = unicode(filename,'utf-8')
    new_fenetre.title('Fenetre {}'.format(n_window))
    n_window += 1

    if new_file: 
        
        L = [0,0,0,0] # = [cleaned_spectra,components,mix,error]
        try:
            xsel,ysel,sigma,spectra,dx,dy,n_sigma,n_spectra = load_raw_file(filename, delimiter ='\t')
        except:
            new_fenetre.destroy()
            return
        frame2 = tk.Frame(new_fenetre)
        frame2.grid(row=1,pady=5,padx=20)
        
        filename_label = tk.Label(frame2,text='Fichier : '+ntpath.basename(filename))
        filename_label.pack(pady=20)
        
        n_components = tk.StringVar()
        components_label = tk.Label(frame2,text='Nombre de composantes')
        components_label.pack()
        entree = tk.Entry(frame2,textvariable=n_components,width=30)
        entree.pack()
        entree.focus_set()
        
        var_recentrage = tk.IntVar()
        case_recentrage = tk.Checkbutton(frame2,text='Recentrage des axes',
                                         variable=var_recentrage)
        
        b1 = tk.Button(frame2, text = "Traiter",
                    command = lambda :traitement(sigma,spectra,L))
        
        b2 = tk.Button(frame2, text = "Tracer composantes",
                    command = lambda :composantes_plot(sigma,L[1]))
        
        b3 = tk.Button(frame2, text ="Tracer graphes 2D",
                    command = lambda :plot_2D(dx,dy,xsel,ysel,L[2],var_recentrage))   
        
        b4 = tk.Button(frame2, text ="Tracer graphes 3d",
                    command = lambda :plot_3D(xsel,ysel,L[2]),
                    state = 'disabled')
        
        b5 = tk.Button(frame2, text ="Sauvegarder les donnees",
                    command = lambda :save_to_file(sigma,n_sigma,n_spectra,xsel,
                                                   ysel,dx,dy,L[1],L[2]))
                    
        b1.pack()
        b2.pack()
        b3.pack()
        case_recentrage.pack()
        b4.pack()
        b5.pack()
        
        champ_label = tk.Label(frame2, text="Prêt")
        champ_label.pack()
        
    else:
        
        try:
            xsel,ysel,sigma,dx,dy,n_sigma,n_spectra,n_comp,components,mix = load_saved_file(filename)
        except:
            new_fenetre.destroy()
            return
        
        frame2 = tk.Frame(new_fenetre)
        frame2.grid(row=1,pady=5,padx=20)
        
        filename_label = tk.Label(frame2,text='Fichier : '+ntpath.basename(filename))
        filename_label.pack(pady=20)

        n_components = tk.StringVar()
        n_components.set(n_comp)
        components_label = tk.Label(frame2,text='Nombre de composantes = {}'.format(n_comp))
        components_label.pack()
        
        var_recentrage = tk.IntVar()
        case_recentrage = tk.Checkbutton(frame2,text='Recentrage des axes',
                                         variable=var_recentrage)
        
        b2 = tk.Button(frame2, text = "Tracer composantes",
                    command = lambda :composantes_plot(sigma,components))
        
        b3 = tk.Button(frame2, text ="Tracer graphes 2D",
                    command = lambda :plot_2D(dx,dy,xsel,ysel,mix,var_recentrage))
        
        b4 = tk.Button(frame2, text ="Tracer graphes 3d",
                    command = lambda :plot_3D(xsel,ysel,mix),state = 'disabled')
        
        b2.pack()
        b3.pack()
        case_recentrage.pack()
        b4.pack()
        
        champ_label = tk.Label(frame2, text="Prêt")
        champ_label.pack()
    
    return
    

#==============================================================================
#           Initialisation of the main window for files loading
#==============================================================================

n_window = 1

fenetre = tk.Tk()

fenetre.title('Interface pour la deconvolution de spectres Raman')
frame1 = tk.Frame(fenetre)
frame1.grid(row=0,pady=5,padx=20)

b1_load = tk.Button(frame1,text='Charger nouveau fichier',command=lambda :new_interface(new_file=True))
b2_load = tk.Button(frame1,text='Charger fichier traite',command=lambda :new_interface(new_file=False))

b1_load.pack(side='left')
b2_load.pack(side='right')

fenetre.mainloop()
