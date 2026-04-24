import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap

# Add src to path for utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import apply_poster_style

def get_gray_to_color_cmap(hue, lightness=0.6, max_saturation=0.9):
    # Constant lightness, saturation from 0 (gray) to max_saturation (vibrant)
    sats = np.linspace(0.0, max_saturation, 256)
    colors = [colorsys.hls_to_rgb(hue, lightness, s) for s in sats]
    return ListedColormap(colors)

def main():
    apply_poster_style()
    
    # 12 distinct hues (Red, Orange, Yellow, Green, Cyan, Blue, Purple, Magenta, etc.)
    hues = np.linspace(0.0, 1.0, 13)[:-1] # 12 steps
    names = ["Red", "Orange", "Yellow", "Lime", "Green", "Teal", "Cyan", "Azure", "Blue", "Purple", "Magenta", "Pink"]

    # Create dummy data for visualization
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    data = np.sin(X/2) + np.cos(Y/2)

    fig, axes = plt.subplots(4, 3, figsize=(24, 28))
    axes = axes.flatten()

    for i, h in enumerate(hues):
        cmap = get_gray_to_color_cmap(h)
        im = axes[i].imshow(data, cmap=cmap, aspect='auto')
        axes[i].set_title(f"Palette #{i+1}: Gray to {names[i]}\n(Hue={h:.2f})", fontsize=25, pad=15)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=5.0)
    
    gallery_path = os.path.join(os.path.dirname(__file__), "palette_gallery_gray.png")
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 Gray-to-Color gallery saved to {gallery_path}")

if __name__ == "__main__":
    main()
