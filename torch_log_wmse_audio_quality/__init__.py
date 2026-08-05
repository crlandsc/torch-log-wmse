# Alias for torch_log_wmse_audio_quality
from torch_log_wmse import *

# `from ... import *` skips names beginning with an underscore, so __version__ has to be
# re-exported explicitly or `torch_log_wmse_audio_quality.__version__` raises AttributeError.
from torch_log_wmse import __version__
