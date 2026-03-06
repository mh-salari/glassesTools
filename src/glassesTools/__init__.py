# try if we have the dependencies for the GUI submodule and import it if so
try:
    import imgui_bundle
except ImportError:
    _has_GUI = False
else:
    _has_GUI = True
    from . import gui as gui

# ensure ffmpeg binaries needed by various submodules are on path
import ffmpeg as _ffmpeg

from . import importing as importing
from . import validation as validation
from .version import __author__ as __author__
from .version import __description__ as __description__
from .version import __email__ as __email__
from .version import __url__ as __url__
from .version import __version__ as __version__

if not _ffmpeg.is_on_path():
    _ffmpeg.add_to_path()
