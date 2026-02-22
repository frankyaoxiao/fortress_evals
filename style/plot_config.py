#!/usr/bin/env python3
"""
Custom plotting configuration module.
Usage: 
    import matplotlib.pyplot as plt
    from plot_config import setup_style
    setup_style()
"""

import os
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# Custom color palette
COLORS = ["#DB8A48", "#BBAB8B", "#696554", "#F7D67E", "#B6998B"]

# Font paths (adjust these to your system)
user_path = "/home/fxiao/.local/share/fonts"
SUISSE_REGULAR = user_path + "/SuisseIntl-Regular.ttf"
SUISSE_BOLD = user_path + "/SuisseIntl-Bold.ttf"

_title_font_properties = None


def setup_style(style_file=None, verbose=False):
    """
    Apply custom plotting style with optional custom fonts.
    
    Parameters
    ----------
    style_file : str, optional
        Path to .mplstyle file. If None, applies programmatic defaults.
    verbose : bool, default False
        Print warnings about missing fonts.
    
    Returns
    -------
    title_font : FontProperties or None
        FontProperties object for bold titles, if available.
    """
    global _title_font_properties
    
    # Load style file if provided
    if style_file and os.path.exists(style_file):
        mpl.style.use(style_file)
    
    # Attempt to load custom regular font
    if os.path.exists(SUISSE_REGULAR):
        try:
            fm.fontManager.addfont(SUISSE_REGULAR)
            family = fm.FontProperties(fname=SUISSE_REGULAR).get_name()
            mpl.rcParams['font.family'] = family
            mpl.rcParams['font.sans-serif'] = [family] + [
                f for f in mpl.rcParams.get('font.sans-serif', []) if f != family
            ]
            if verbose:
                print(f"✓ Loaded custom font: {family}")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load {SUISSE_REGULAR}: {e}")
    elif verbose:
        print(f"✗ Font file not found: {SUISSE_REGULAR}")
    
    # Attempt to load custom bold font for titles
    if os.path.exists(SUISSE_BOLD):
        try:
            fm.fontManager.addfont(SUISSE_BOLD)
            _title_font_properties = FontProperties(fname=SUISSE_BOLD)
            if verbose:
                print(f"✓ Loaded bold font for titles")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load {SUISSE_BOLD}: {e}")
    elif verbose:
        print(f"✗ Bold font file not found: {SUISSE_BOLD}")
    
    # Set color cycle
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=COLORS)
    
    return _title_font_properties


def get_title_font():
    """Return FontProperties for bold titles, or None if not available."""
    return _title_font_properties


def apply_suptitle(fig, text, fontsize=14, y=0.93, ax=None):
    """
    Apply a centered suptitle with custom bold font if available.
    
    Parameters
    ----------
fig : matplotlib.figure.Figure
        The figure object.
    text : str
        Title text.
    fontsize : int, default 14
        Font size for title.
    y : float, default 0.93
        Vertical position (0-1).
    ax : matplotlib.axes.Axes, optional
        If provided, center title over this axes instead of full figure.
    """
    center_x = 0.5
    if ax is not None:
        bbox = ax.get_position()
        center_x = (bbox.x0 + bbox.x1) / 2
    
    if _title_font_properties is not None:
        fig.suptitle(text, fontproperties=_title_font_properties, 
                     fontsize=fontsize, y=y, x=center_x)
    else:
        fig.suptitle(text, fontweight='bold', 
                     fontsize=fontsize, y=y, x=center_x)