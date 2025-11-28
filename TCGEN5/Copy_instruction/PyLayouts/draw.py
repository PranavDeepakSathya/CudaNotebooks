import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm

class Draw:
    """
    Visualizes a 1D realized layout array as a 2D tile,
    with a specified fill order (row-major or column-major).
    Supports Address gradient coloring and Bank ID (mod 32) discrete coloring.
    """
    
    # A default palette of 32 distinct colors for the banks
    DEFAULT_BANK_COLORS = [
        "#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C", "#98DF8A", "#D62728", "#FF9896",
        "#9467BD", "#C5B0D5", "#8C564B", "#C49C94", "#E377C2", "#F7B6D2", "#7F7F7F", "#C7C7C7",
        "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5", "#393B79", "#637939", "#8C6D31", "#843C39",
        "#7B4173", "#5254A3", "#8CA252", "#BD9E39", "#AD494A", "#A55194", "#6B6ECF", "#B5CF6B"
    ]

    def __init__(self, layout_data: np.ndarray, display_dim: int, order: str = 'row'):
        """
        Initializes the drawer with layout data and display shape.
        
        :param layout_data: 1D numpy array of physical addresses.
        :param display_dim: The number of columns (if order='row') 
                            or rows (if order='col').
        :param order: 'row' (default) or 'col'.
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
        
        # 1. Determine grid shape
        if self.order == 'row':
            self.cols = display_dim
            self.rows = int(np.ceil(self.N / self.cols))
            self.reshape_order = 'C'
        else:
            self.rows = display_dim
            self.cols = int(np.ceil(self.N / self.rows))
            self.reshape_order = 'F'
            
        # 2. Pad data with NaN
        padded_size = self.rows * self.cols
        self.padded_data = np.full(padded_size, np.nan)
        self.padded_data[:self.N] = self.data
        
        # 3. Reshape to grid
        self.grid_data = self.padded_data.reshape(self.rows, self.cols, 
                                                  order=self.reshape_order)

    def show(self, title: str = None, mode: str = 'address', bank_colors: list = None):
        """
        Displays the 2D plot of the memory layout.
        
        :param title: The title for the plot.
        :param mode: 'address' (gradient based on value) or 'bank' (discrete based on % 32).
        :param bank_colors: List of 32 hex strings. Used only if mode='bank'.
        """
        fig, ax = plt.subplots(figsize=(max(8, self.cols * 0.6), max(6, self.rows * 0.6)))
        
        # --- Determine Coloring Strategy ---
        if mode == 'bank':
            # 1. Prepare Colors
            colors = bank_colors if bank_colors else self.DEFAULT_BANK_COLORS
            if len(colors) < 32:
                raise ValueError("Must provide at least 32 colors for bank mode.")
            
            # Create a discrete colormap
            cmap = ListedColormap(colors[:32])
            # Bounds 0 to 32 ensuring integers map clearly to colors
            norm = BoundaryNorm(np.arange(33), cmap.N) 
            
            # 2. Prepare Data (Bank ID)
            # We copy grid_data, replace NaNs with 0 temporarily to perform modulo safely,
            # then put NaNs back.
            plot_data = self.grid_data.copy()
            nan_mask = np.isnan(plot_data)
            
            # Fill NaNs with 0 temporarily so modulo works
            temp_filled = np.nan_to_num(plot_data, nan=0)
            plot_data = temp_filled.astype(int) % 32
            
            # Convert back to float and restore NaNs so they show as white/transparent
            plot_data = plot_data.astype(float)
            plot_data[nan_mask] = np.nan
            
            if title is None:
                title = "Memory Layout (Bank Coloring)"
            label_suffix = " (Bank ID)"

        else: # mode == 'address'
            plot_data = self.grid_data
            min_val = np.nanmin(self.data)
            max_val = np.nanmax(self.data)
            cmap = plt.get_cmap('jet')
            norm = Normalize(vmin=min_val, vmax=max_val)
            
            if title is None:
                title = "Memory Layout (Address Gradient)"
            label_suffix = ""

        # --- Plotting ---
        # 'nearest' interpolation ensures sharp squares
        im = ax.imshow(plot_data, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Colorbar setup
        cbar = fig.colorbar(im, ax=ax, extend='neither')
        if mode == 'bank':
            cbar.set_label('Bank ID (Address % 32)')
            cbar.set_ticks(np.arange(0.5, 32, 1)) # Center ticks
            # Optional: Show fewer ticks if 32 is too crowded
            cbar.set_ticklabels(range(32) if self.rows < 20 else [str(i) if i%2==0 else '' for i in range(32)])
        else:
            cbar.set_label('Physical Address')
        
        # --- Annotation ---
        for r in range(self.rows):
            for c in range(self.cols):
                physical_address = self.grid_data[r, c] # Always display actual address
                
                if not np.isnan(physical_address):
                    # Determine background color of this cell to decide text color
                    if mode == 'bank':
                        val_for_color = int(physical_address) % 32
                    else:
                        val_for_color = physical_address
                        
                    rgba = cmap(norm(val_for_color))
                    
                    # Calculate luminance: 0.299R + 0.587G + 0.114B
                    luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                    text_color = 'w' if luminance < 0.5 else 'k'
                    
                    text = f"{int(physical_address)}"
                    ax.text(c, r, text, ha='center', va='center', color=text_color, fontsize=9)

        # --- Formatting ---
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Display Column")
        ax.set_ylabel("Display Row")
        
        ax.set_xticks(np.arange(self.cols))
        ax.set_yticks(np.arange(self.rows))
        ax.set_xticklabels(np.arange(self.cols))
        ax.set_yticklabels(np.arange(self.rows))
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Minor grid lines for cell separation
        ax.set_xticks(np.arange(self.cols+1)-.5, minor=True)
        ax.set_yticks(np.arange(self.rows+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        fig.tight_layout()
        plt.show()

