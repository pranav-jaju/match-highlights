"""match-highlights: automatic sports highlight detection from match audio.

Pipeline: audio -> per-frame signals (loudness, sound events, commentary)
-> excitement score -> peak events -> highlight reel.
"""

__version__ = "0.1.0"

# Every signal in the package is resampled onto this frame grid (seconds).
FRAME_HOP_S = 0.5
