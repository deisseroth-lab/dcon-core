import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib_scalebar.scalebar import ScaleBar

def side_by_side(ch_stack):
    X = ch_stack.shape[1]
    return ch_stack.swapaxes(0, 1).reshape(X, -1)

def faces3d(volume, dxy=1, dz=1, title=None, units="um", **kwargs):
    labelpad = 0.02

    if volume.ndim == 3:
        Z, Y, X = volume.shape
    elif volume.ndim == 4:
        if volume.shape[-1] == 3:
            Z, Y, X, C = volume.shape
        else:
            raise ValueError("Last dim of 4D data must have size 3 (to represent color)")
    else:
        raise ValueError("Unsupported number of dimensions ({:d})".format(volume.ndim))

    yx = volume.max(0)
    zx = volume.max(1)
    yz = volume.max(2).swapaxes(0, 1)

    fig = plt.gcf()
        
    tx = (Z*dz + X*dxy)
    ty = (Z*dz + Y*dxy)
    ax_yx = plt.axes([0, (Z*dz)/tx, 1-(Z*dz)/ty, 1-(Z*dz)/tx], frameon=False)
    ax_zx = plt.axes([0, 0, 1-(Z*dz)/ty, (Z*dz)/tx], frameon=False)
    ax_yz = plt.axes([1-(Z*dz)/tx, (Z*dz)/ty, (Z*dz)/ty, 1-(Z*dz)/ty], frameon=False)
    ax_br = plt.axes([1-(Z*dz)/tx, 0, (Z*dz)/ty, (Z*dz)/tx], frameon=False)

    ax_yx.imshow(yx, extent=[0, X*dxy, Y*dxy, 0], **kwargs)
    ax_yx.xaxis.set_ticks_position('top')
    ax_yx.text(X*labelpad, Y*labelpad, 'X - Y', va='top',
               fontsize='x-large', weight='bold', color='w')

    ax_zx.imshow(zx, extent=[0, X*dxy, Z*dz, 0], **kwargs)
    ax_zx.text(X*labelpad, Z*labelpad, 'X - Z', va='bottom',
               fontsize='x-large', weight='bold', color='w')
    ax_zx.invert_yaxis()

    ax_yz.imshow(yz, extent=[0, Z*dz, Y*dxy, 0], **kwargs)
    ax_yz.xaxis.set_ticks_position('top')
    ax_yz.yaxis.set_ticks_position('right')
    ax_yz.text(Z*labelpad, Y*labelpad, 'Y - Z', va='top', ha='right',
               fontsize='x-large', weight='bold', color='w')
    ax_yz.invert_xaxis()

    #fig.text(0.5, 1.05, '(microns)', ha='center')
    #fig.text(-0.08, 0.5, '(microns)', va='center', rotation='vertical')
    
    ax_br.get_xaxis().set_visible(False)
    ax_br.get_yaxis().set_visible(False)
    if title is not None:
        ax_br.text(0.5, 0.5, title, va='center', ha='center',
                   transform=ax_br.transAxes,
                   fontsize='x-large', weight='bold')

    sb = ScaleBar(1, units, frameon=False, pad=0.5, location='lower right', color='w')
    ax_yx.add_artist(sb)

    
    for ax in [ax_yx, ax_zx, ax_yz]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        sns.despine(ax=ax, top=False, bottom=False, left=False, right=False)
        for spine in ax.spines.values():
            spine.set_color('0.65')
            spine.set_linewidth(1)
            
    return fig

def plane_grid(vol, shape, dxy=1, dz=1, units="um", **kwargs):
    gX, gY = shape
    N_planes = gX * gY
    
    zs = np.linspace(0, vol.shape[0], num=N_planes, endpoint=False).astype(np.int)[::-1]
    
    fig = plt.gcf()
    
    for idx, z in enumerate(zs):
        a = fig.add_subplot(gX, gY, idx+1)
        a.imshow(vol[z], **kwargs)
        a.axis('off')
        a.text(0.02, 0.98, "{:0.1f}µm".format(z * dz), transform=a.transAxes, va='top', color='w')
    
    sb = ScaleBar(dxy, units, frameon=False, pad=-1.2, location='lower right', color='w')
    a.add_artist(sb)
    
    plt.tight_layout()
    return fig
    
def txy_movie(arr, vmin=None, vmax=None, cmap='gray', fps=30):
    import moviepy.editor as mpy
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    normed = (arr - vmin) / (vmax - vmin)
        
    cm = plt.cm.get_cmap(cmap)
        
    clip = mpy.ImageSequenceClip([256*cm(a) for a in normed], fps)
    return mpy.ipython_display(clip,  maxlen=len(arr)/fps + 1)
    