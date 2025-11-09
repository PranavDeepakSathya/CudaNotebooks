import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
class Draw:
    """
    Visualizes a 1D realized layout array as a 2D tile,
    given a specified number of columns.
    """
    
    def __init__(self, layout_data: np.ndarray, cols: int):
        """
        Initializes the drawer with layout data and display columns.
        
        :param layout_data: 1D numpy array of physical addresses.
        :param cols: The number of columns for the 2D display grid.
        """
        if not isinstance(layout_data, np.ndarray) or layout_data.ndim != 1:
            raise ValueError("layout_data must be a 1D numpy array.")
        if not isinstance(cols, int) or cols <= 0:
            raise ValueError("cols must be a positive integer.")
            
        self.data = layout_data
        self.N = self.data.shape[0]
        self.cols = cols
        self.rows = int(np.ceil(self.N / self.cols))
        
        # Pad the data with NaN to fit the grid perfectly
        padded_size = self.rows * self.cols
        self.padded_data = np.full(padded_size, np.nan)
        self.padded_data[:self.N] = self.data
        
        # Reshape to the 2D grid for plotting
        self.grid_data = self.padded_data.reshape(self.rows, self.cols)

    def show(self, title: str = "Memory Layout"):
        """
        Displays the 2D plot of the memory layout.
        
        :param title: The title for the plot.
        """
        fig, ax = plt.subplots(figsize=(max(8, self.cols * 0.6), max(6, self.rows * 0.6)))
        
        # Get min/max for color normalization, ignoring NaNs
        min_val = np.nanmin(self.data)
        max_val = np.nanmax(self.data)
        
        # Use a colormap (e.g., 'viridis', 'jet', 'plasma')
        # 'jet' often shows good contrast for address patterns
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=min_val, vmax=max_val)
        
        # Display the grid as an image
        im = ax.imshow(self.grid_data, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Create a color bar
        cbar = fig.colorbar(im, ax=ax, extend='neither')
        cbar.set_label('Physical Address')
        
        # Annotate each cell with its physical address
        # and its logical index
        for r in range(self.rows):
            for c in range(self.cols):
                logical_index = r * self.cols + c
                if logical_index < self.N:
                    physical_address = self.data[logical_index]
                    
                    # Determine text color based on cell brightness
                    # We get the normalized color, then check its perceived luminance
                    rgba = cmap(norm(physical_address))
                    luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                    text_color = 'w' if luminance < 0.5 else 'k'
                    
                    # Add Physical Address (Addr) only
                    text = f"{int(physical_address)}"
                    ax.text(c, r, text, ha='center', va='center', color=text_color, fontsize=10)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Display Column")
        ax.set_ylabel("Display Row")
        
        # Set ticks to match grid
        ax.set_xticks(np.arange(self.cols))
        ax.set_yticks(np.arange(self.rows))
        ax.set_xticklabels(np.arange(self.cols))
        ax.set_yticklabels(np.arange(self.rows))
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Show grid lines
        ax.set_xticks(np.arange(self.cols+1)-.5, minor=True)
        ax.set_yticks(np.arange(self.rows+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        fig.tight_layout()
        plt.show()