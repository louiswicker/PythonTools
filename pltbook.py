
import numpy as np
import datetime
import netCDF4
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copy
from metpy.plots import ctables
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


import warnings
warnings.filterwarnings("ignore")

hscale = 1.0 
debug  = True

_default_cmap = plt.cm.viridis_r

#===============================================================================
# 2D xarray container for plotting fields

def container(*field):

    if len(field) == 1:
        x1 = np.arange(field[0].shape[1])
        x  = np.broadcast_to(x1[np.newaxis,:], field[0].shape)
        y1 = np.arange(field[0].shape[0])
        y  = np.broadcast_to(y1[:,np.newaxis], field[0].shape)

        return {'field': field[0], 'x': x, 'y': y}

#       return xr.DataArray(field[0], name='field', dims=("ny", "nx"), coords={"X": (["ny", "nx"], x), 
#                                                                              "Y": (["ny", "nx"], y)} )

    elif len(field) == 3:

        if debug:  print('X/Y/2D: ',field[0].shape, field[1].shape, field[2].shape)

        if field[1].ndim == 1:
#           if field[1].shape[0] == field[0].shape[0]:
#           x = np.broadcast_to(field[1][:], field[0].shape)
#               x = np.hstack([field[1]]*field[0].shape[1]).reshape(field[0].shape)

            if field[1].shape[0] == field[0].shape[0]: 
                x = field[2]
            else:
                x = field[1]

            if debug:  print('X: ',x.shape)

        else:
            x = field[1]

        if field[2].ndim == 1:
#           y = np.broadcast_to(field[2][np.newaxis, :], field[0].shape)
#           y = np.hstack([field[2]]*field[0].shape[0]).reshape(field[0].shape)

            if field[2].shape[0] == field[0].shape[1]: 
                y = field[1]
            else:
                y = field[2]

            if debug:  print('Y: ', y.shape)
        else:
            y = field[2]

        if debug:  print('FLD: ', field[0].max(), field[0].min())
        if debug:  print('X: ', x.max(), x.min())
        if debug:  print('Y: ', y.max(), y.min())

        return {'field': field[0], 'x': x, 'y': y}

#       return xr.DataArray(field[0], name='field', dims=("ny", "nx"), coords={"X": (["ny", "nx"], x), 
#                                                                "Y": (["ny", "nx"], y)} )

    else:

        if field[-1].ndim == 2:

            print("\n --->Container Error: the number of items is not 1 or 3")
            print(" --->Container Error: creating fake axis data and returning 2D array\n")

            x = np.arange(field[-1].shape[1])
            y = np.arange(field[-1].shape[0])

            return {'field': None, 'x': x, 'y': y}

#           return xr.DataArray( field[-1], dims=("ny", "nx"), coords={"X": (["nx"], x), 
#                                                                      "Y": (["ny"], y)} )

        else:
            print("\n --->Container Error: data passed is weird!\n")

    
#===============================================================================
# 2D generic plotting code using container

def plot_contour_row(fields, levels=0, cl_levels=None, xlim=None, ylim=None, 
                     ptitle=[], var='', cmap=_default_cmap, **kwargs):

# Parse kwargs

    suptitle  = kwargs.get("suptitle", None)
    xlabel    = kwargs.get("xlabel", 'x')
    ylabel    = kwargs.get("ylabel", 'y')
    fmask     = kwargs.get("fmask", None)
    cbar      = kwargs.get("cbar", False)
    plot_ref  = kwargs.get("plot_ref", False)
    ax_in     = kwargs.get("ax_in", False)

    cbar = []

    if len(fields) > len(ptitle):
        ptitle = ['NAME'] * len(fields)

    if  ax_in == False:
        if len(fields) == 1:
            fig, axes = plt.subplots(1,1, constrained_layout=True, figsize=(10,10))
            axes = list(axes)
        else:
            fig, axes = plt.subplots(1,len(fields), constrained_layout=True, figsize=(5*len(fields),5))
    else:
        axes = ax_in
        print(axes)

    for ax, field, title in zip(axes, fields, ptitle):

        fld = field['field']

        if fmask != None:
            fld = np.ma.masked_array( fld, mask = ( np.abs(fld) < fmask ) )
        
        x   = field['x']
        y   = field['y']

        if debug:
            print('PLOT_ROW_CONTOUR:  ',fld.max(), fld.min())
            print('PLOT_ROW_CONTOUR:  ',x.max(), x.min())
            print('PLOT_ROW_CONTOUR:  ',y.max(), y.min())
        
        if levels == 0:
            amin, amax, cint, clevels = nice_clevels(fld.min(), fld.max(), **kwargs)
        else:
            clevels = levels
            cint    = levels[1] - levels[0]

        if debug > 10:
            print('PLOT_ROW_CONTOUR:  ', clevels )
                 
        if plot_ref:

            mycolors = copy.deepcopy(ctables.colortables['NWSReflectivity'])
            mycolors.insert(0,(1,1,1))
            color_map = mcolors.ListedColormap(mycolors)
            cmin = 0.0
            cmax = 85.0
            cinc = 5.0
            cntlevels = list(np.arange(cmin,cmax,5.0))
            cbar.append( ax.contourf(x, y, fld, levels=cntlevels, cmap=color_map, **kwargs) )

        else:

            if type(clevels) != type(None):

                cbar.append( ax.contourf(x, y, fld, levels=clevels, cmap=cmap, **kwargs) )

                if cl_levels != None:
                    CC = ax.contour(x, y, fld, levels = cl_levels, colors='k', alpha=0.5, **kwargs);
                    ax.clabel(CC, cl_levels[::2], inline=1, fmt='%2.0f', fontsize=14) # label every second level
                else:
                    CC = ax.contour(x, y, fld, levels = clevels[::2], colors='k', alpha=0.5, **kwargs);
                    ax.clabel(CC, clevels[::2], inline=1, fmt='%2.0f', fontsize=14) # label every second level
        
        if ax_in == False:
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
        
        ax.set_title("%s:   %s  Max: %7.4f  Min: %7.4f CINT: %6.4f" % (title, var, fld.max(), fld.min(), cint), 
                    fontsize=10)

        if ax_in == False and xlim:
            ax.set_xlim(xlim)
        if ax_in == False and ylim:
            ax.set_ylim(ylim)

    # fig.subplots_adjust(right=0.9)

    if cbar:
        cb = plt.colorbar(cbar[-1], shrink=0.8, pad=0.05)
        cb.set_label(var.upper())

#   if ax_in.any() == None:
#       cbar_ax = fig.add_axes([1.,0.175, 0.03, 0.7])
#       fig.colorbar(CF, cax=cbar_ax)

    if suptitle != None:
        plt.suptitle(suptitle, fontsize=12)

    if ax_in == False:
        return fig, axes
    else:
        return

#===============================================================================

def nice_clevels(dmin, dmax, **kwargs):
    """ Extra function to generate the array of contour levels for plotting 
        using "nice_mxmnintvl" code.  Removes an extra step.  Returns 4 args,
        with the 4th the array of contour levels.  The first three values
        are the same as "nice_mxmnintvl". """

    amin, amax, cint = nice_mxmnintvl(dmin, dmax, **kwargs)

    if debug > 10:
        print('NICE_CLEVELS:  ',amax, amin, cint )

    if cint == None:

        return amin, amax, 0.0, None
            
    else:

        return amin, amax, cint, np.arange(amin, amax+cint, cint) 

#===============================================================================
def nice_mxmnintvl(dmin, dmax, **kwargs):
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

# Parse kwargs

    climits   = kwargs.get("climits", None)
    cint      = kwargs.get("cint", None)
    max_steps = kwargs.get("max_steps", 25)
    sym       = kwargs.get("sym", False)
    outside   = kwargs.get("outside", True)
    
    table = np.array([1.0,2.0,2.5,4.0,5.0,10.0,20.0,25.0,40.0,50.0,100.0,200.0,
                      250.0,400.0,500.0])

    if nearlyequal(dmax,dmin):
        return 0.0, 0.0, None

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

    if climits:
        amax = climits[1]
        amin = climits[0]

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
    if debug > 10:
        print('NICE_MAXMIN:  ',t, am1, ax1, cints)
    
    if cint == None or cint == 0.0:   
        try:
            index = np.where(cints < max_steps)[0][0]
            return am1[index], ax1[index], cints[index]
        except IndexError:
            return None, None, None
    else:
        return am1, ax1, cint

#===============================================================================
def setup_map(npanels, extent=None, draw_gridlines=False, map_details=False):
    
    def colorize_state(geometry):
        facecolor = (0.93, 0.93, 0.85)
        return {'facecolor': facecolor, 'edgecolor': 'black'}

    shapename  = 'admin_1_states_provinces_lakes'
    
    states_shp = shpreader.natural_earth(resolution='110m',category='cultural', name=shapename)

    theproj = ccrs.PlateCarree() #choose another projection to obtain non-rectangular grid

    fig, ax = plt.subplots(1,npanels, figsize=(5*npanels,5), 
              subplot_kw={'projection': theproj}, constrained_layout=True)  #, 'axisbg': 'w'

#   print(type(ax))
#   if ( type(ax) != type(np.zeros([1,1]))):
#       print('000')
#       ax = np.array([ax,])
#
#   print(type(ax))

#   plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    for n in np.arange(npanels):
        ax[n].add_feature(cfeature.COASTLINE)
        ax[n].add_feature(cfeature.STATES, edgecolor='black')

        if draw_gridlines:
            gl = ax[n].gridlines(draw_labels=True, linewidth=1, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

    if map_details:
        #ax[0].add_feature(cfeature.OCEAN, facecolor='#CCFEFF')
        ax[0].add_feature(cfeature.LAKES, facecolor='#CCFEFF')
        ax[0].add_feature(cfeature.RIVERS, edgecolor='#CCFEFF')
        ax[0].add_feature(cfeature.LAND, facecolor='#FFE9B5')

    if extent != None:
        for n in np.arange(npanels):
            ax[n].set_extent(extent)
            
    return fig, ax

#===============================================================================
def nearlyequal(a, b, sig_digit=None):
    """ Measures the equality (for two floats), in unit of decimal significant
        figures.  If no sigificant digit is specified, default is 7 digits. """

    if sig_digit == None or sig_digit > 7:
        sig_digit = 7
    if a == b:
        return True
    difference = abs(a - b)
    avg = (a + b)/2

    return np.log10(avg / difference) >= sig_digit

