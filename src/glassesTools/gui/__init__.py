# all the files in this module need imgui_bundle, so catch here if its not installed
try:
    import imgui_bundle
except ImportError:
    raise ImportError(
        "imgui_bundle (or one of its dependencies) is not installed, GUI functionality is not available. You must install glassesTools with the [GUI] extra if you wish to use the GUI."
    ) from None

from . import file_picker as file_picker
from . import msg_box as msg_box
from . import recording_table as recording_table
from . import signal_sync as signal_sync
from . import timeline as timeline
from . import utils as utils
from . import video_player as video_player
from . import worldgaze as worldgaze
