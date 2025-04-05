import sys
import os
# Add the parent directory (root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from RL.Logger import Logger

import tkinter as tk
from tkinter import ttk, font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LoggerUI:
    def __init__(self, logger):
        self.logger = logger
        self.window = tk.Tk()
        self.window.title("Logger Data Visualizer")
        self.window.geometry("800x600")
        
        # Configure styles with larger fonts
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Helvetica', 12))
        self.style.configure('TButton', font=('Helvetica', 12))
        self.style.configure('TRadiobutton', font=('Helvetica', 12))
        self.style.configure('TCombobox', font=('Helvetica', 12))
        
        # Main container
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Key Selection (left panel)
        self.key_frame = ttk.LabelFrame(main_frame, text="Select Keys")
        self.key_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.key_listbox = tk.Listbox(self.key_frame, selectmode=tk.MULTIPLE, 
                                    font=('Helvetica', 12), width=25, height=15)
        self.key_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
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
        ttk.Radiobutton(config_frame, text="Episode", variable=self.x_axis_var, 
                       value="episode", command=self.update_ui).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(config_frame, text="Step", variable=self.x_axis_var, 
                       value="step", command=self.update_ui).grid(row=1, column=1, sticky="w")

        # Aggregation Type
        self.agg_var = tk.StringVar(value="mean")
        ttk.Label(config_frame, text="Aggregation:").grid(row=2, column=0, sticky="w", pady=5)
        self.agg_menu = ttk.Combobox(config_frame, values=["mean", "sum", "mean_std"], 
                                    textvariable=self.agg_var, state="readonly", width=12)
        self.agg_menu.grid(row=3, column=0, columnspan=2, sticky="w")

        # Episode Selection
        self.episode_var = tk.StringVar()
        ttk.Label(config_frame, text="Episode:").grid(row=4, column=0, sticky="w", pady=5)
        self.episode_menu = ttk.Combobox(config_frame, textvariable=self.episode_var, 
                                        state="readonly", width=12)
        self.episode_menu.grid(row=5, column=0, columnspan=2, sticky="w")

        # Generate Button
        ttk.Button(config_frame, text="Generate Plots", command=self.generate_plots).grid(
            row=6, column=0, columnspan=2, pady=15)

        # Initialize data
        self.update_key_list()
        self.update_ui()

    def update_key_list(self):
        self.key_listbox.delete(0, tk.END)
        for key in self.logger.get_keys():
            self.key_listbox.insert(tk.END, key)

    def update_ui(self):
        """Update UI elements based on selections"""
        if self.x_axis_var.get() == "episode":
            self.agg_menu.config(state="readonly")
            self.episode_menu.config(state="disabled")
        else:
            self.agg_menu.config(state="disabled")
            self.episode_menu.config(state="readonly")
            self.update_episode_list()

    def update_episode_list(self):
        """Update episode dropdown with available episodes from selected keys"""
        episodes = set()
        for key in self.logger.get_keys():
            episodes.update(self.logger.get_data(key).keys())
        self.episode_menu["values"] = sorted(episodes)

    def generate_plots(self):
        """Generate plots based on current selections"""
        selected_keys = [self.key_listbox.get(i) for i in self.key_listbox.curselection()]
        x_axis = self.x_axis_var.get()
        aggregation = self.agg_var.get() if x_axis == "episode" else None
        episode = int(self.episode_var.get()) if x_axis == "step" else None

        if not selected_keys:
            return

        for key in selected_keys:
            self.logger.plot(
                key=key,
                x_axis=x_axis,
                aggregation=aggregation,
                episode=episode,
                skip_none=True
            )
            
            # Configure plot fonts
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
    logger.log("reward", [1, 2, 3], episode=0)
    logger.log("reward", [4, 5], episode=1)
    logger.log("loss", [0.5, 0.3], episode=0)
    logger.log("loss", [0.2, 0.4, 0.6], episode=1)

    # logger.plot('reward', episode=0, x_axis='step')

    # Launch UI
    ui = LoggerUI(logger)
    ui.run()