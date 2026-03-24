import matplotlib.pyplot as plt
import seaborn as sns

try:
    import mplcyberpunk
    HAS_CYBERPUNK = True
except ImportError:
    HAS_CYBERPUNK = False

def apply_premium_theme(is_cyberpunk=False):
    """
    Applies a premium, modern dark theme to matplotlib and seaborn.
    Call this right before creating figures.
    """
    if is_cyberpunk and HAS_CYBERPUNK:
        plt.style.use("cyberpunk")
        # Ensure we still have a nice font for cyberpunk
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "savefig.facecolor": plt.rcParams["axes.facecolor"],
            "savefig.edgecolor": plt.rcParams["axes.facecolor"]
        })
    else:
        # Sleek bespoke dark mode
        sns.set_theme(
            style="darkgrid",
            rc={
                "figure.facecolor": "#121212",
                "axes.facecolor": "#1e1e1e",
                "axes.edgecolor": "#333333",
                "axes.labelcolor": "#e0e0e0",
                "text.color": "#e0e0e0",
                "xtick.color": "#e0e0e0",
                "ytick.color": "#e0e0e0",
                "grid.color": "#333333",
                "grid.linestyle": "--",
                "grid.alpha": 0.5,
                "font.family": "sans-serif",
                "font.sans-serif": ["Inter", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
                "axes.titlesize": 14,
                "axes.labelsize": 11,
                "lines.linewidth": 2.0,
                "savefig.facecolor": "#121212",
                "savefig.edgecolor": "#121212",
                "legend.facecolor": "#1e1e1e",
                "legend.edgecolor": "#333333",
                "legend.labelcolor": "#e0e0e0"
            }
        )

def add_cyberpunk_glow(ax=None):
    """
    Adds glowing lines if mplcyberpunk is available.
    """
    if HAS_CYBERPUNK:
        import mplcyberpunk
        if ax:
            mplcyberpunk.add_glow_effects(ax=ax)
        else:
            mplcyberpunk.add_glow_effects()
