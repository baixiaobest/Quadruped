import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from RL.Logger import Logger, KeyType

class LoggerUI:
    def __init__(self, logger):
        self.logger = logger
        self.window = tk.Tk()
        self.window.title("Logger Data Visualizer")
        self.window.geometry("1000x700")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Helvetica', 12))
        self.style.configure('TButton', font=('Helvetica', 12))
        self.style.configure('TRadiobutton', font=('Helvetica', 12))
        self.style.configure('TCombobox', font=('Helvetica', 12))
        
        # Main container
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Key Selection (left panel)
        key_frame = ttk.Frame(main_frame)
        key_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Episode Keys list
        episode_frame = ttk.LabelFrame(key_frame, text="Episode Keys")
        episode_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.episode_listbox = tk.Listbox(episode_frame, selectmode=tk.MULTIPLE, 
                                        font=('Helvetica', 12), height=15, exportselection=False)
        self.episode_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Update Keys list
        update_frame = ttk.LabelFrame(key_frame, text="Update Keys")
        update_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.update_listbox = tk.Listbox(update_frame, selectmode=tk.MULTIPLE, 
                                       font=('Helvetica', 12), height=15, exportselection=False)
        self.update_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configuration (right panel)
        config_frame = ttk.LabelFrame(main_frame, text="Plot Configuration")
        config_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        # X-axis Type
        self.x_axis_var = tk.StringVar(value="episode")
        ttk.Label(config_frame, text="X-axis Type:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Radiobutton(config_frame, text="Episode/Update Round", variable=self.x_axis_var, 
                       value="episode", command=self.update_ui).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(config_frame, text="Step", variable=self.x_axis_var, 
                       value="step", command=self.update_ui).grid(row=1, column=1, sticky="w")

        # Aggregation Type
        self.agg_var = tk.StringVar(value="mean")
        ttk.Label(config_frame, text="Aggregation:").grid(row=2, column=0, sticky="w", pady=5)
        self.agg_menu = ttk.Combobox(config_frame, values=["mean", "sum", "mean_std", "min", "max"], 
                                    textvariable=self.agg_var, state="readonly", width=15)
        self.agg_menu.grid(row=3, column=0, columnspan=2, sticky="w")

        # Identifier Selection
        self.identifier_var = tk.StringVar()
        ttk.Label(config_frame, text="Episode/Update Round:").grid(row=4, column=0, sticky="w", pady=5)
        self.identifier_menu = ttk.Combobox(config_frame, textvariable=self.identifier_var, 
                                          state="readonly", width=15)
        self.identifier_menu.grid(row=5, column=0, columnspan=2, sticky="w")

        # Plot button
        ttk.Button(config_frame, text="Generate Plots", command=self.generate_plots).grid(
            row=6, column=0, columnspan=2, pady=15)

        # Event bindings for mutual exclusive selection
        self.episode_listbox.bind('<<ListboxSelect>>', self.on_episode_select)
        self.update_listbox.bind('<<ListboxSelect>>', self.on_update_select)

        # Initialize data
        self.update_key_lists()
        self.update_ui()

    def update_key_lists(self):
        """Populate the episode and update key listboxes"""
        self.episode_listbox.delete(0, tk.END)
        self.update_listbox.delete(0, tk.END)
        
        for key in self.logger.get_keys():
            key_type = self.logger.get_key_type(key)
            if key_type == KeyType.EPISODE:
                self.episode_listbox.insert(tk.END, key)
            elif key_type == KeyType.UPDATE:
                self.update_listbox.insert(tk.END, key)

    def on_episode_select(self, event):
        """Handle episode list selection"""
        self.update_listbox.selection_clear(0, tk.END)
        self.update_ui()

    def on_update_select(self, event):
        """Handle update list selection"""
        self.episode_listbox.selection_clear(0, tk.END)
        self.update_ui()

    def update_ui(self):
        """Update UI elements based on selections"""
        if self.x_axis_var.get() == "episode":
            self.agg_menu.config(state="readonly")
            self.identifier_menu.config(state="disabled")
        else:
            self.agg_menu.config(state="disabled")
            self.identifier_menu.config(state="readonly")
            self.update_identifier_list()

    def update_identifier_list(self):
        """Update identifier dropdown based on selected keys"""
        identifiers = set()
        current_keys = self.get_selected_keys()
        
        if current_keys:
            key_type = self.logger.get_key_type(current_keys[0])
            for key in current_keys:
                identifiers.update(self.logger.get_data(key).keys())
                
            if key_type == KeyType.EPISODE:
                all_identifiers = self.logger.episodes
            else:
                all_identifiers = self.logger.update_rounds
                
            identifiers = sorted(identifiers.intersection(all_identifiers))
        
        self.identifier_menu["values"] = sorted(identifiers)

    def get_selected_keys(self):
        """Get currently selected keys from either list"""
        episode_keys = [self.episode_listbox.get(i) for i in self.episode_listbox.curselection()]
        update_keys = [self.update_listbox.get(i) for i in self.update_listbox.curselection()]
        return episode_keys + update_keys

    def generate_plots(self):
        """Generate plots for selected keys"""
        selected_keys = self.get_selected_keys()
        if not selected_keys:
            return

        x_axis = self.x_axis_var.get()
        aggregation = self.agg_var.get() if x_axis == "episode" else None
        identifier = int(self.identifier_var.get()) if x_axis == "step" else None
        
        for key in selected_keys:
            key_type = self.logger.get_key_type(key)
            try:
                if key_type == KeyType.EPISODE:
                    if x_axis == "episode":
                        self.logger.plot_episode(key, aggregation=aggregation)
                    else:
                        self.logger.plot_episode(key, episode=identifier)
                else:
                    if x_axis == "episode":
                        self.logger.plot_update(key, aggregation=aggregation)
                    else:
                        self.logger.plot_update(key, update_round=identifier)
            except ValueError as e:
                print(f"Error plotting {key}: {e}")

        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })
        plt.show()

    def run(self):
        self.window.mainloop()

# Example Usage
if __name__ == "__main__":
    # Create a logger and populate with dummy data
    logger = Logger()
    # Log episode-type data
    logger.log_episode("reward", [1, 2, 3], episode=0)
    logger.log_episode("reward", [4, 5], episode=1)
    logger.log_episode("reward", [5, 4], episode=2)
    logger.log_episode("reward", [3, 2, 1, 3], episode=3)
    logger.log_episode("reward", [4, 5, 8, 10, 11, 10], episode=4)
    # Log update-type data
    logger.log_update("loss", [0.5, 0.3], update_round=0)
    logger.log_update("loss", [0.2, 0.4, 0.6], update_round=1)
    logger.log_update("loss", [0.3, 0.5, 0.7], update_round=2)
    logger.log_update("loss", [0.5, 0.6, 0.8], update_round=3)

    # Launch UI
    ui = LoggerUI(logger)
    ui.run()