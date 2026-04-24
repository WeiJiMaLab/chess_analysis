import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap

# Add src to path for utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import apply_poster_style

def get_isoluminant_cmap(h1, h2, lightness=0.6, saturation=0.8):
    hues = np.linspace(h1, h2, 256)
    colors = [colorsys.hls_to_rgb(h, lightness, saturation) for h in hues]
    return ListedColormap(colors)

def main():
    apply_poster_style()
    
    # 12 random hue pairs
    np.random.seed(42) # For reproducibility of this specific gallery
    n_palettes = 12
    hue_pairs = []
    while len(hue_pairs) < n_palettes:
        h1 = np.random.random()
        h2 = h1 + (np.random.random() * 0.4 - 0.2) # Keep hues somewhat related but distinct
        hue_pairs.append((h1, h2 % 1.0))

    # Create dummy data for visualization
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    data = np.sin(X/2) + np.cos(Y/2)

    fig, axes = plt.subplots(4, 3, figsize=(24, 28))
    axes = axes.flatten()

    for i, (h1, h2) in enumerate(hue_pairs):
        cmap = get_isoluminant_cmap(h1, h2)
        im = axes[i].imshow(data, cmap=cmap, aspect='auto')
        axes[i].set_title(f"Palette #{i+1}\n(h1={h1:.2f}, h2={h2:.2f})", fontsize=25, pad=15)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=5.0)
    
    gallery_path = os.path.join(os.path.dirname(__file__), "palette_gallery.png")
    plt.savefig(gallery_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 Palette gallery saved to {gallery_path}")
    print("\nAvailable Hue Pairs:")
    for i, (h1, h2) in enumerate(hue_pairs):
        print(f"#{i+1}: h1={h1:.3f}, h2={h2:.3f}")

if __name__ == "__main__":
    main()
