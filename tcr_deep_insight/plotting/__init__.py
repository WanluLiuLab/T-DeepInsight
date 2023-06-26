from ._plotting import *
from scanpy.plotting import *
from . import _palette as palette

from pathlib import Path
MODULE_PATH = Path(__file__).parent

from matplotlib import font_manager
font_files = font_manager.findSystemFonts(fontpaths=['./tcr_deep_insight/plotting/fonts/'])
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
