import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class Draw:
    """
    Visualizes a 1D realized layout array as a 2D tile,
    with a specified fill order (row-major or column-major).
    """
    
    def __init__(self, layout_data: np.ndarray, display_dim: int, order: str = 'row'):
        """
        Initializes the drawer with layout data and display shape.
        
        :param layout_data: 1D numpy array of physical addresses.
        :param display_dim: The number of columns (if order='row') 
                            or rows (if order='col').
        :param order: 'row' (default) or 'col'.
                      'row' fills the grid row-by-row (C order).
                      'col' fills the grid column-by-column (F order).
        """
        if not isinstance(layout_data, np.ndarray) or layout_data.ndim != 1:
            raise ValueError("layout_data must be a 1D numpy array.")
        if not isinstance(display_dim, int) or display_dim <= 0:
            raise ValueError("display_dim must be a positive integer.")
        if order not in ['row', 'col']:
            raise ValueError("order must be 'row' or 'col'.")
            
        self.data = layout_data
        self.N = self.data.shape[0]
        self.order = order
        
        # 1. Determine grid shape (rows, cols) and reshape order
        if self.order == 'row':
            self.cols = display_dim
            self.rows = int(np.ceil(self.N / self.cols))
            self.reshape_order = 'C'  # 'C' for row-major fill
        else:  # order == 'col'
            self.rows = display_dim
            self.cols = int(np.ceil(self.N / self.rows))
            self.reshape_order = 'F'  # 'F' for column-major fill
            
        # 2. Pad the data with NaN to fit the grid perfectly
        padded_size = self.rows * self.cols
        self.padded_data = np.full(padded_size, np.nan)
        self.padded_data[:self.N] = self.data
        
        # 3. Reshape to the 2D grid using the specified fill order
        self.grid_data = self.padded_data.reshape(self.rows, self.cols, 
                                                  order=self.reshape_order)

    def show(self, title: str = "Memory Layout"):
        """
        Displays the 2D plot of the memory layout.
        
        :param title: The title for the plot.
        """
        fig, ax = plt.subplots(figsize=(max(8, self.cols * 0.6), max(6, self.rows * 0.6)))
        
        min_val = np.nanmin(self.data)
        max_val = np.nanmax(self.data)
        
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=min_val, vmax=max_val)
        
        im = ax.imshow(self.grid_data, cmap=cmap, norm=norm, interpolation='nearest')
        
        cbar = fig.colorbar(im, ax=ax, extend='neither')
        cbar.set_label('Physical Address')
        
        # Annotate each cell
        # We can now just iterate and plot the value in grid_data[r, c]
        # The 'nan' check handles the padding automatically.
        for r in range(self.rows):
            for c in range(self.cols):
                physical_address = self.grid_data[r, c]
                
                if not np.isnan(physical_address):
                    rgba = cmap(norm(physical_address))
                    luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                    text_color = 'w' if luminance < 0.5 else 'k'
                    
                    text = f"{int(physical_address)}"
                    ax.text(c, r, text, ha='center', va='center', color=text_color, fontsize=10)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Display Column")
        ax.set_ylabel("Display Row")
        
        ax.set_xticks(np.arange(self.cols))
        ax.set_yticks(np.arange(self.rows))
        ax.set_xticklabels(np.arange(self.cols))
        ax.set_yticklabels(np.arange(self.rows))
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_xticks(np.arange(self.cols+1)-.5, minor=True)
        ax.set_yticks(np.arange(self.rows+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        fig.tight_layout()
        plt.show()