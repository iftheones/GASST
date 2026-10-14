"""Render real PaviaU input cubes on CPU; no training or synthetic textures."""
from pathlib import Path
import sys, os, json, argparse, hashlib
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / '.figure_deps'), str(ROOT / '.torch_deps')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

OUT = ROOT / 'outputs' / 'PU_cubes'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'pdf.fonttype':42})

def save(fig, name):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, transparent=True)
    plt.close(fig)

def canvas(width, height):
    fig = plt.figure(figsize=(width/120, height/120))
    ax = fig.add_axes([0,0,1,1], xlim=(0,width), ylim=(height,0), aspect='equal')
    ax.axis('off')
    return fig, ax

def cube(ax, data, rgb, x, y, w, h, dx, dy, norm, cmap=None):
    """Top = first spatial row; right = last spatial column; depth = band index."""
    bands = data.shape[2]
    if cmap is None:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'reference_spectral', [(0,'#35104f'),(.16,'#611487'),
            (.25,'#491091'),(.31,'#1315ad'),(.38,'#0632ce'),
            (.43,'#00699f'),(.49,'#00a967'),(.60,'#21d52e'),
            (.74,'#a5e21c'),(.86,'#f4dd16'),
            (.95,'#f05222'),(1,'#a91f3c')], N=256)
    top = Affine2D.from_values(1,0,dx/bands,-dy/bands,x,y) + ax.transData
    side = Affine2D.from_values(dx/bands,-dy/bands,0,1,x+w,y) + ax.transData
    ax.imshow(data[0,:,:].T, extent=(0,w,bands,0), transform=top,
              cmap=cmap, norm=norm, interpolation='nearest', zorder=1)
    ax.imshow(data[:,-1,:], extent=(0,bands,h,0), transform=side,
              cmap=cmap, norm=norm, interpolation='nearest', zorder=2)
    ax.imshow(rgb, extent=(x,x+w,y+h,y), interpolation='nearest', zorder=3)

def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--row', type=int, default=184, help='Top-left context row, zero based')
    parser.add_argument('--col', type=int, default=112, help='Top-left context column, zero based')
    parser.add_argument('--context-size', type=int, default=84)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        OUT=args.output_dir.resolve()
        OUT.mkdir(parents=True,exist_ok=True)
    source = ROOT/'datasets/pu/PaviaU.mat'
    data = loadmat(source)['paviaU']
    print('PaviaU:', data.shape, data.dtype, flush=True)
    # Display composite only: no wavelength calibration is supplied in the MAT file.
    channels = [55,33,12]
    selected = data[:,:,channels].astype(np.float32)
    lo, hi = np.percentile(selected, [2,98], axis=(0,1))
    rgb = np.clip((selected-lo)/(hi-lo),0,1)
    Image.fromarray(np.uint8(rgb*255)).save(OUT/'PU_RGB_native.png')
    if args.preview_only:
        return
    r,c,s = args.row,args.col,args.context_size
    assert 0 <= r < data.shape[0]-s and 0 <= c < data.shape[1]-s
    cr,cc = r+s//2-5,c+s//2-5
    context = data[r:r+s,c:c+s]
    patch = data[cr:cr+10,cc:cc+10]
    # Shared intensity scaling for full scene and both crops.
    low,high = np.percentile(data,[2,98])
    norm = matplotlib.colors.PowerNorm(gamma=0.8,vmin=low,vmax=high,clip=True)
    h,w,bands = data.shape
    fig,ax = canvas(w+110,h+110)
    cube(ax,data,rgb,10,100,w,h,90,90,norm)
    save(fig,'PU_HSI_cube')
    for name,d,im in [('PU_patch_10x10',patch,rgb[cr:cr+10,cc:cc+10]),
                      (f'PU_context_{s}x{s}',context,rgb[r:r+s,c:c+s])]:
        fig,ax = canvas(310,310)
        cube(ax,d,im,10,80,220,220,70,70,norm)
        save(fig,name)
    for name,pr,pc,ps in [('PU_input_panel',cr,cc,10),('PU_context_panel',r,c,s)]:
        fig,ax = canvas(760,735)
        cube(ax,data,rgb,18,110,w,h,90,90,norm)
        ax.add_patch(Rectangle((18+pc,110+pr),ps,ps,fill=False,
                              edgecolor='#ed2525',linewidth=1.6,zorder=5))
        cube(ax,data[pr:pr+ps,pc:pc+ps],rgb[pr:pr+ps,pc:pc+ps],
             535,300,165,165,45,45,norm)
        ax.add_patch(FancyArrowPatch((18+pc+ps,110+pr), (552,270),
                     connectionstyle='arc3,rad=-0.35',arrowstyle='-|>',
                     mutation_scale=15,color='#ed2525',linewidth=1.5,zorder=6))
        save(fig,name)
    # Compact opaque preview for quick inspection, not the production asset.
    with Image.open(OUT/'PU_context_panel.png') as im:
        im.thumbnail((900,900))
        bg=Image.new('RGB',im.size,'white'); bg.paste(im,mask=im.getchannel('A'))
        bg.save(OUT/'preview.png')
    np.save(OUT/'PU_patch_10x10_raw.npy',patch)
    metadata = {
        'source':str(source), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'shape_H_W_B':list(data.shape), 'display_RGB_bands_zero_based':channels,
        'display_RGB_bands_one_based':[i+1 for i in channels],
        'RGB_transform':'Independent whole-scene 2nd/98th percentile stretch; clip to [0,1]; no gamma',
        'RGB_low':lo.tolist(),'RGB_high':hi.tolist(),
        'spectral_faces':'Actual first-row and last-column band intensities; reference_spectral purple-deep-blue-green-yellow-red map with a narrower blue interval and no pale-blue stops, defined in script; shared 2/98 percentile limits with PowerNorm gamma=0.8; display only',
        'spectral_limits':[float(low),float(high)],
        'depth':'All 103 bands in original index order; drawn depth is schematic, not spatial distance',
        'model_patch_zero_based_row_col_size':[cr,cc,10],
        'context_crop_zero_based_row_col_size':[r,c,s],
        'render':'Nearest-neighbor display; 600 dpi exports do not add spatial resolution',
        'formats':'Text-free transparent PNG, SVG and PDF with embedded raster textures; arrows remain vector',
        'model':'Input cube visualization is independent of model weights',
        'versions':{'numpy':np.__version__,'matplotlib':matplotlib.__version__},
    }
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('Saved figures to',OUT,flush=True)

if __name__=='__main__': main()
