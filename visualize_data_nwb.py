# Import necessary libraries
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import glob
from matplotlib.patches import Rectangle, Patch

# Import pynwb for NWB file handling
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd


class DataVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Visualizer")
        self.geometry("1100x700")
        self.data_dir = ""  # Directory where data files are stored
        self.available_dates = []
        self.target_styles = ["CO", "RD"]
        self.create_widgets()
        # Store legend handles and labels
        self.legend_handles = []
        self.legend_labels = []

    def create_widgets(self):
        # Create a main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a LabelFrame for Data Selection
        data_selection_frame = ttk.LabelFrame(main_frame, text="Data Selection")
        data_selection_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Data directory selection
        data_dir_label = ttk.Label(data_selection_frame, text="Data Directory:")
        data_dir_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.data_dir_entry = ttk.Entry(data_selection_frame, width=50)
        self.data_dir_entry.grid(row=0, column=1, columnspan=4, padx=5, pady=5)
        browse_button = ttk.Button(
            data_selection_frame, text="Browse", command=self.browse_data_dir
        )
        browse_button.grid(row=0, column=5, padx=5, pady=5)

        # Date selection
        date_label = ttk.Label(data_selection_frame, text="Select Date:")
        date_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.date_var = tk.StringVar()
        self.date_dropdown = ttk.Combobox(
            data_selection_frame, textvariable=self.date_var
        )
        self.date_dropdown.grid(row=1, column=1, padx=5, pady=5)
        self.date_dropdown.bind("<<ComboboxSelected>>", self.update_trial_range)

        # Target style selection
        target_style_label = ttk.Label(data_selection_frame, text="Target Style:")
        target_style_label.grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.target_style_var = tk.StringVar(value="CO")
        self.target_style_dropdown = ttk.Combobox(
            data_selection_frame,
            textvariable=self.target_style_var,
            values=self.target_styles,
        )
        self.target_style_dropdown.grid(row=1, column=3, padx=5, pady=5)
        self.target_style_dropdown.bind("<<ComboboxSelected>>", self.update_trial_range)

        # Trial range selection
        trial_range_label = ttk.Label(data_selection_frame, text="Trial Range:")
        trial_range_label.grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)

        start_trial_label = ttk.Label(data_selection_frame, text="Start Trial:")
        start_trial_label.grid(row=1, column=5, sticky=tk.W, padx=5, pady=5)
        self.start_trial_var = tk.StringVar()
        self.start_trial_dropdown = ttk.Combobox(
            data_selection_frame, textvariable=self.start_trial_var
        )
        self.start_trial_dropdown.grid(row=1, column=6, padx=5, pady=5)

        end_trial_label = ttk.Label(data_selection_frame, text="End Trial:")
        end_trial_label.grid(row=1, column=7, sticky=tk.W, padx=5, pady=5)
        self.end_trial_var = tk.StringVar()
        self.end_trial_dropdown = ttk.Combobox(
            data_selection_frame, textvariable=self.end_trial_var
        )
        self.end_trial_dropdown.grid(row=1, column=8, padx=5, pady=5)

        # Configure column weights for responsive layout
        for i in range(9):
            data_selection_frame.columnconfigure(i, weight=1)

        # Load and plot button
        load_button = ttk.Button(
            main_frame, text="Load and Plot", command=self.load_and_plot
        )
        load_button.pack(side=tk.TOP, pady=10)

        # Canvas for plots
        self.figure = plt.Figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas.draw()
        # Add the toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def browse_data_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir = directory
            self.data_dir_entry.delete(0, tk.END)
            self.data_dir_entry.insert(0, directory)
            self.scan_dates()

    def scan_dates(self):
        # Scan the data directory for available dates
        pattern = os.path.join(self.data_dir, "*.nwb")
        files = glob.glob(pattern)
        dates = set()
        for file in files:
            basename = os.path.basename(file)
            parts = basename.split("_")
            if len(parts) >= 2:
                date_part = parts[0]
                dates.add(date_part)
        self.available_dates = sorted(list(dates))
        self.date_dropdown["values"] = self.available_dates
        if self.available_dates:
            self.date_var.set(self.available_dates[0])
            self.update_trial_range()

    def update_trial_range(self, event=None):
        date = self.date_var.get()
        target_style = self.target_style_var.get()
        if date and target_style:
            try:
                nwb_file = os.path.join(self.data_dir, f"{date}_{target_style}.nwb")
                io = NWBHDF5IO(nwb_file, "r")
                nwb_data = io.read()
                trials = nwb_data.trials
                trial_nums = trials["trial_number"][:]
                trial_nums = sorted(set(trial_nums))
                trial_nums_str = [str(int(num)) for num in trial_nums]
                self.start_trial_dropdown["values"] = trial_nums_str
                self.end_trial_dropdown["values"] = trial_nums_str
                if trial_nums:
                    self.start_trial_var.set(trial_nums_str[0])
                    # By default, set a range of 10 trials if available
                    if len(trial_nums) > 15:
                        self.end_trial_var.set(trial_nums_str[15])
                    else:
                        self.end_trial_var.set(trial_nums_str[-1])
                io.close()
            except FileNotFoundError:
                messagebox.showerror("Error", "NWB file not found.")
                self.start_trial_dropdown["values"] = []
                self.end_trial_dropdown["values"] = []
        else:
            self.start_trial_dropdown["values"] = []
            self.end_trial_dropdown["values"] = []

    def load_and_plot(self):
        # Get selected options
        date = self.date_var.get()
        target_style = self.target_style_var.get()
        start_trial_str = self.start_trial_var.get()
        end_trial_str = self.end_trial_var.get()
        if not date or not target_style:
            messagebox.showerror("Error", "Please select a date and target style.")
            return
        try:
            if start_trial_str and end_trial_str:
                start_trial = int(start_trial_str)
                end_trial = int(end_trial_str)
                if start_trial > end_trial:
                    raise ValueError(
                        "Start trial number must be less than or equal to end trial number."
                    )
            else:
                start_trial = end_trial = None
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid trial range: {e}")
            return
        # Load data
        try:
            nwb_file = os.path.join(self.data_dir, f"{date}_{target_style}.nwb")
            io = NWBHDF5IO(nwb_file, "r")
            nwb_data = io.read()
        except FileNotFoundError:
            messagebox.showerror("Error", "NWB file not found.")
            return

        # Extract trials data
        trials = nwb_data.trials.to_dataframe()
        if start_trial is not None and end_trial is not None:
            trials = trials[
                (trials["trial_number"] >= start_trial)
                & (trials["trial_number"] <= end_trial)
            ]
        # Get time intervals for the selected trials
        trial_intervals = trials[
            [
                "start_time",
                "stop_time",
                "trial_number",
                "index_target_position",
                "mrp_target_position",
            ]
        ]

        # Extract timeseries data
        behavior_module = nwb_data.processing["behavior"]
        index_position_ts = behavior_module.get_data_interface("index_position")
        index_velocity_ts = behavior_module.get_data_interface("index_velocity")
        mrp_position_ts = behavior_module.get_data_interface("mrp_position")
        mrp_velocity_ts = behavior_module.get_data_interface("mrp_velocity")

        # Extract ecephys data
        ecephys_module = nwb_data.processing["ecephys"]
        spiking_band_power = ecephys_module.get_data_interface("SpikingBandPower")
        threshold_crossings = ecephys_module.get_data_interface("ThresholdCrossings")

        # Convert timeseries data to DataFrames
        index_position_df = self.timeseries_to_df(index_position_ts, "index_position")
        index_velocity_df = self.timeseries_to_df(index_velocity_ts, "index_velocity")
        mrp_position_df = self.timeseries_to_df(mrp_position_ts, "mrp_position")
        mrp_velocity_df = self.timeseries_to_df(mrp_velocity_ts, "mrp_velocity")
        spiking_band_power_df = self.timeseries_to_df(spiking_band_power, "sbp_channel")
        threshold_crossings_df = self.timeseries_to_df(
            threshold_crossings, "tcfr_channel"
        )

        # Merge data into a single DataFrame
        timeseries_df = self.merge_timeseries(
            [
                index_position_df,
                index_velocity_df,
                mrp_position_df,
                mrp_velocity_df,
                spiking_band_power_df,
                threshold_crossings_df,
            ]
        )

        # Add trial numbers to timeseries data based on trial intervals
        timeseries_df = self.assign_trials(timeseries_df, trials)

        # Close NWB file
        io.close()

        # Clear previous legend handles and labels
        self.legend_handles = []
        self.legend_labels = []
        # Generate plots
        self.figure.clf()
        # Create subplots with shared x-axes
        ax1 = self.figure.add_subplot(311)
        ax2 = self.figure.add_subplot(312, sharex=ax1)
        ax3 = self.figure.add_subplot(313, sharex=ax1)
        self.plot_positions(timeseries_df, trial_intervals, ax1)
        self.plot_velocities(timeseries_df, ax2)
        self.plot_channel_averages(timeseries_df, ax3)
        # Adjust layout to prevent overlapping titles
        self.figure.tight_layout(rect=[0, 0.05, 1, 0.95])
        # Add combined legend at the bottom
        if self.legend_handles and self.legend_labels:
            self.figure.legend(
                self.legend_handles,
                self.legend_labels,
                loc="lower center",
                ncol=4,
                bbox_to_anchor=(0.5, 0.0),
            )
        self.canvas.draw()

    def timeseries_to_df(self, ts, column_prefix):
        # Convert a timeseries object to a DataFrame
        data = ts.data[:]
        timestamps = ts.timestamps[:]
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            if data.ndim == 2:
                data = data[:, 0]
            df = pd.DataFrame({column_prefix: data})
        else:
            # For multidimensional data, create column names based on channel index
            num_channels = data.shape[1]
            columns = [f"{column_prefix}_{i}" for i in range(num_channels)]
            df = pd.DataFrame(data, columns=columns)
        df["time"] = timestamps
        return df

    def merge_timeseries(self, df_list):
        # Merge multiple DataFrames on the 'time' column
        from functools import reduce

        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on="time", how="outer"), df_list
        )
        merged_df.sort_values(by="time", inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        return merged_df

    def assign_trials(self, timeseries_df, trials_df):
        # Assign trial numbers to timeseries data based on trial intervals
        timeseries_df["trial_number"] = np.nan
        trial_starts = trials_df["start_time"].values
        trial_stops = trials_df["stop_time"].values
        trial_numbers = trials_df["trial_number"].values
        for start, stop, trial_num in zip(trial_starts, trial_stops, trial_numbers):
            mask = (timeseries_df["time"] >= start) & (timeseries_df["time"] <= stop)
            timeseries_df.loc[mask, "trial_number"] = trial_num
        # Only keep rows where trial_number is not NaN (i.e., within trial intervals)
        timeseries_df = timeseries_df.dropna(subset=["trial_number"]).copy()
        timeseries_df["trial_number"] = timeseries_df["trial_number"].astype(int)
        return timeseries_df

    def plot_positions(self, timeseries_df, trial_intervals, ax):
        lines = timeseries_df.plot(
            x="time", y=["index_position", "mrp_position"], ax=ax, legend=False
        ).lines
        # Get the colors of the lines
        index_line = lines[0]
        mrp_line = lines[1]
        index_color = index_line.get_color()
        mrp_color = mrp_line.get_color()
        # Add target position rectangles
        for idx, trial in trial_intervals.iterrows():
            trial_num = trial["trial_number"]
            start_time = trial["start_time"]
            end_time = trial["stop_time"]
            duration = end_time - start_time
            # Index target rectangle
            rect_index = Rectangle(
                (start_time, trial["index_target_position"] - 0.1),
                width=duration,
                height=0.2,
                color=index_color,
                alpha=0.3,
            )
            ax.add_patch(rect_index)
            # MRP target rectangle
            rect_mrp = Rectangle(
                (start_time, trial["mrp_target_position"] - 0.1),
                width=duration,
                height=0.2,
                color=mrp_color,
                alpha=0.3,
            )
            ax.add_patch(rect_mrp)
        ax.set_title("Position of Index and MRP")
        ax.set_ylabel("Position")
        # Remove x-axis labels to prevent duplication
        ax.set_xlabel("")
        # Collect legend handles and labels
        index_patch = Patch(facecolor=index_color, edgecolor=index_color, alpha=0.3)
        mrp_patch = Patch(facecolor=mrp_color, edgecolor=mrp_color, alpha=0.3)
        self.legend_handles.extend([index_line, mrp_line, index_patch, mrp_patch])
        self.legend_labels.extend(
            ["Index Position", "MRP Position", "Index Target", "MRP Target"]
        )

    def plot_velocities(self, timeseries_df, ax):
        lines = timeseries_df.plot(
            x="time", y=["index_velocity", "mrp_velocity"], ax=ax, legend=False
        ).lines
        ax.set_title("Velocity of Index and MRP")
        ax.set_ylabel("Velocity")
        # Remove x-axis labels to prevent duplication
        ax.set_xlabel("")
        # Collect legend handles and labels
        self.legend_handles.extend(lines)
        self.legend_labels.extend(["Index Velocity", "MRP Velocity"])

    def plot_channel_averages(self, timeseries_df, ax):
        # Get SBP and TCFR channels
        sbp_cols = [
            col for col in timeseries_df.columns if col.startswith("sbp_channel")
        ]
        tcfr_cols = [
            col for col in timeseries_df.columns if col.startswith("tcfr_channel")
        ]
        # Compute averages
        timeseries_df["sbp_avg"] = timeseries_df[sbp_cols].mean(axis=1)
        timeseries_df["tcfr_avg"] = timeseries_df[tcfr_cols].mean(axis=1)
        ax2 = ax.twinx()
        sbp_line = timeseries_df.plot(
            x="time", y="sbp_avg", ax=ax, color="blue", legend=False
        ).lines[0]
        tcfr_line = timeseries_df.plot(
            x="time", y="tcfr_avg", ax=ax2, color="orange", legend=False
        ).lines[0]
        ax.set_title("Average SBP and TCFR Channels")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("SBP Average")
        ax2.set_ylabel("TCFR Average")
        # Collect legend handles and labels without duplicates
        self.legend_handles.extend([sbp_line, tcfr_line])
        self.legend_labels.extend(["SBP Average", "TCFR Average"])


if __name__ == "__main__":
    app = DataVisualizer()
    app.mainloop()
