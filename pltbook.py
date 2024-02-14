
import numpy as np
import datetime
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


hscale = 1.0 
debug  = False

#===============================================================================
# 2D xarray container for plotting fields

def container(*field):

    if len(field) == 1:
        x = np.arange(field[0].shape[1])
        y = np.arange(field[0].shape[0])

        return xr.DataArray( field[0], dims=("ny", "nx"), coords={"X": (["nx"], x), 
                                                                  "Y": (["ny"], y)} )
    elif len(field) == 3:

        return xr.DataArray(field[2], dims=("ny", "nx"), coords={"X": (["ny", "nx"], field[0].data), 
                                                                 "Y": (["ny", "nx"], field[1].data)} )
    
#===============================================================================
# 2D generic plotting code using container

def plot_contour_row(fields, levels=0, transpose=False, extra_levels=None, range=None, 
                     title='', var='', xlabel='x', ylabel='y',  cmap='RdBu_r', **kwargs):

    if len(fields) == 1:
         fig, ax = plt.subplots(1,1, constrained_layout=True, figsize=(10,10))
         axes = [ax,]
    else:
        fig, axes = plt.subplots(1,len(fields), constrained_layout=True, figsize=(5*len(fields),5))

    for ax, field in zip(axes, fields):

        fld = field.values
        x   = field.X.values
        y   = field.Y.values
        
        if levels == 0:
            amin, amax, cint, clevels = nice_clevels(fld.min(), fld.max(), **kwargs)
        else:
            clevels = levels
            cint    = levels[1] - levels[0]
                 
        CF = ax.contourf(x, y, fld, levels=clevels, cmap=cmap)

        if extra_levels:

           CC = ax.contour(x, y, fld, levels = extra_levels, colors='k', alpha=0.5)
           ax.clabel(CC, extra_levels[::2], inline=1, fmt='%2.0f', fontsize=14) # label every second level
        
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
        ax.set_title("%s: %s  Max: %6.2f  Min: %6.2f CINT: %6.2f" % (title, var, fld.max(), fld.min(), cint), 
                    fontsize=10)

        if range:
            ax.set_xlim(range)

    # fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([1.,0.175, 0.03, 0.7])
    fig.colorbar(CF, cax=cbar_ax)

    return fig, axes
#===============================================================================
def nice_mxmnintvl(dmin, dmax, outside=True, max_steps=25, cint=None, sym=False):
    """ Description: Given min and max values of a data domain and the maximum
                     number of steps desired, determines "nice" values of 
                     for endpoints and spacing to create a series of steps 
                     through the data domainp. A flag controls whether the max 
                     and min are inside or outside the data range.
  
        In Args: float   dmin 		the minimum value of the domain
                 float   dmax       the maximum value of the domain
                 int     max_steps	the maximum number of steps desired
                 logical outside    controls whether return min/max fall just
                                    outside or just inside the data domainp.
                     if outside: 
                         min_out <= min < min_out + step_size
                                         max_out >= max > max_out - step_size
                     if inside:
                         min_out >= min > min_out - step_size
                                         max_out <= max < max_out + step_size
      
                 float    cint      if specified, the contour interval is set 
                                    to this, and the max/min bounds, based on 
                                    "outside" are returned.

                 logical  sym       if True, set the max/min bounds to be anti-symmetric.
      
      
        Out Args: min_out     a "nice" minimum value
                  max_out     a "nice" maximum value  
                  step_size   a step value such that 
                                     (where n is an integer < max_steps):
                                      min_out + n * step_size == max_out 
                                      with no remainder 
      
        If max==min, or a contour interval cannot be computed, returns "None"
     
        Algorithm mimics the NCAR NCL lib "nice_mxmnintvl"; code adapted from 
        "nicevals.c" however, added the optional "cint" arg to facilitate user 
        specified specific interval.
     
        Lou Wicker, August 2009 """

    table = np.array([1.0,2.0,2.5,4.0,5.0,10.0,20.0,25.0,40.0,50.0,100.0,200.0,
                      250.0,400.0,500.0])

    if nearlyequal(dmax,dmin):
        return None

    # Help people like me who can never remember - flip max/min if inputted reversed
    if dmax < dmin:
        amax = dmin
        amin = dmax
    else:
        amax = dmax
        amin = dmin

    if sym:
        smax = max(amax.max(), amin.min())
        amax = smax
        amin = -smax

    d = 10.0**(np.floor(np.log10(amax - amin)) - 2.0)
    if cint == None or cint == 0.0:
        t = table * d
    else:
        t = cint
    if outside:
        am1 = np.floor(amin/t) * t
        ax1 = np.ceil(amax/t)  * t
        cints = (ax1 - am1) / t 
    else:
        am1 = np.ceil(amin/t) * t
        ax1 = np.floor(amax/t)  * t
        cints = (ax1 - am1) / t
    
    # DEBUG LINE BELOW
    # print t, am1, ax1, cints
    
    if cint == None or cint == 0.0:   
        try:
            index = np.where(cints < max_steps)[0][0]
            return am1[index], ax1[index], cints[index]
        except IndexError:
            return None
    else:
        return am1, ax1, cint

#===============================================================================

def nice_clevels(dmin, dmax, **kwargs):
    """ Extra function to generate the array of contour levels for plotting 
        using "nice_mxmnintvl" code.  Removes an extra step.  Returns 4 args,
        with the 4th the array of contour levels.  The first three values
        are the same as "nice_mxmnintvl". """

# Parse kwargs

    cint      = kwargs.get("cint", None)
    max_steps = kwargs.get("max_steps", 25)
    sym       = kwargs.get("sym", False)
    outside   = kwargs.get("outside", True)
    
    try:
        amin, amax, cint = nice_mxmnintvl(dmin, dmax, cint=cint, max_steps=max_steps, sym=sym, outside=True)
        return amin, amax, cint, N.arange(amin, amax+cint, cint) 
    except:
        return None

#===============================================================================
def kde_plotter(mdata, mlabel, mcolor, ax=None):

    xlim = [-15,55]
    
    if ax == None:
        fig, ax = plt.subplots(1,1, constrained_layout=True,figsize=(7,7))

    for data, label, color in zip(mdata,mlabel,mcolor):
        # print(label, color)
        
        hist, bin_edges = np.histogram(data.flatten())

        data_no_zero = data.flatten()
    
        eval_points = np.linspace(np.min(bin_edges), np.max(bin_edges))
        kde_sp      = gaussian_kde(data_no_zero, bw_method=0.9)
        y_sp        = kde_sp.pdf(eval_points)
        
        ax.plot(eval_points, y_sp, color=color, linewidth=2.0, label='%s  %d' % (label,data_no_zero.shape[0]))
             
    ax.set_xlim(xlim[:])
    ax.set_yscale("log")
#    ax.set_xscale("log", base=2.0)
    plt.grid(axis='y', alpha=0.75)
    plt.grid(axis='x', alpha=0.75)
    ax.set_xlabel('W (m/s)',fontsize=15)
    ax.set_ylabel('Density',fontsize=15)
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
    ax.axvline(x=40.0, color='k', linestyle='--', linewidth=2.0)
    ax.axvline(x=80.0, color='k', linestyle='--', linewidth=2.0)
    ax.set_title('W' , fontsize=15)
