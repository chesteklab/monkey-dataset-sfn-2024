import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Use ttk for better-looking widgets
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use("TkAgg")  # Ensure TkAgg backend is used.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.stats import pearsonr
from pynwb import NWBHDF5IO  # Import pynwb to read NWB files


class RidgeRegressionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Ridge Regression with History")
        master.geometry(
            "1200x800"
        )  # Set the window size to accommodate plots and results

        # Initialize variables
        self.dataset_folder = ""
        self.days = ["Select Day"]  # Provide a default placeholder
        self.target_styles = ["CO", "RD"]

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Create a main frame
        self.main_frame = ttk.Frame(self.master, padding=(10, 10))
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create input frame
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Input Parameters")
        self.input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Organize input widgets in a grid with two columns
        row = 0
        ttk.Label(self.input_frame, text="Dataset Folder:").grid(
            row=row, column=0, sticky="e", pady=5
        )
        self.folder_entry = ttk.Entry(self.input_frame, width=40)
        self.folder_entry.grid(row=row, column=1, pady=5, sticky="w")
        folder_button = ttk.Button(
            self.input_frame, text="Browse", command=self.browse_folder
        )
        folder_button.grid(row=row, column=2, padx=5, pady=5, sticky="w")

        row += 1
        ttk.Label(self.input_frame, text="Training Day:").grid(
            row=row, column=0, sticky="e", pady=5
        )
        self.train_day_var = tk.StringVar(value=self.days[0])
        self.train_day_menu = ttk.Combobox(
            self.input_frame,
            textvariable=self.train_day_var,
            values=self.days,
            state="readonly",
            width=15,
        )
        self.train_day_menu.grid(row=row, column=1, pady=5, sticky="w")

        ttk.Label(self.input_frame, text="Training Target Style:").grid(
            row=row, column=2, sticky="e", pady=5
        )
        self.train_style_var = tk.StringVar(value="CO")
        self.train_style_menu = ttk.Combobox(
            self.input_frame,
            textvariable=self.train_style_var,
            values=self.target_styles,
            state="readonly",
            width=5,
        )
        self.train_style_menu.grid(row=row, column=3, pady=5, sticky="w")

        row += 1
        ttk.Label(self.input_frame, text="Testing Day:").grid(
            row=row, column=0, sticky="e", pady=5
        )
        self.test_day_var = tk.StringVar(value=self.days[0])
        self.test_day_menu = ttk.Combobox(
            self.input_frame,
            textvariable=self.test_day_var,
            values=self.days,
            state="readonly",
            width=15,
        )
        self.test_day_menu.grid(row=row, column=1, pady=5, sticky="w")

        ttk.Label(self.input_frame, text="Testing Target Style:").grid(
            row=row, column=2, sticky="e", pady=5
        )
        self.test_style_var = tk.StringVar(value="CO")
        self.test_style_menu = ttk.Combobox(
            self.input_frame,
            textvariable=self.test_style_var,
            values=self.target_styles,
            state="readonly",
            width=5,
        )
        self.test_style_menu.grid(row=row, column=3, pady=5, sticky="w")

        row += 1
        ttk.Label(self.input_frame, text="History (time bins):").grid(
            row=row, column=0, sticky="e", pady=5
        )
        self.history_entry = ttk.Entry(self.input_frame, width=5)
        self.history_entry.grid(row=row, column=1, pady=5, sticky="w")
        self.history_entry.insert(0, "10")

        ttk.Label(self.input_frame, text="Feature Type:").grid(
            row=row, column=2, sticky="e", pady=5
        )
        self.feature_var = tk.StringVar(value="SBP")
        self.feature_menu = ttk.Combobox(
            self.input_frame,
            textvariable=self.feature_var,
            values=["SBP", "TCFR"],
            state="readonly",
            width=5,
        )
        self.feature_menu.grid(row=row, column=3, pady=5, sticky="w")

        # Run button
        row += 1
        run_button = ttk.Button(
            self.input_frame, text="Run Regression", command=self.run_regression
        )
        run_button.grid(row=row, column=0, columnspan=4, pady=10)

        # Separator
        separator = ttk.Separator(self.main_frame, orient="horizontal")
        separator.pack(fill=tk.X, padx=5, pady=5)

        # Results frame
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Numerical results table
        self.create_results_table(self.results_frame)

        # Plots frame
        self.plots_frame = ttk.Frame(self.results_frame)
        self.plots_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas and toolbar placeholders
        self.figure = None
        self.canvas = None
        self.toolbar = None

    def create_results_table(self, parent):
        columns = (
            "Metric",
            "Index Position",
            "MRP Position",
            "Index Velocity",
            "MRP Velocity",
        )
        self.results_table = ttk.Treeview(
            parent, columns=columns, show="headings", height=2
        )
        for col in columns:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, anchor="center", width=120)
        self.results_table.pack(fill=tk.X, pady=5)

        # Style the Treeview
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))
        style.configure("Treeview", font=("Helvetica", 10))

    def browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.dataset_folder = folder_selected
            self.folder_entry.delete(0, tk.END)
            self.folder_entry.insert(0, self.dataset_folder)
            self.update_days()

    def update_days(self):
        # Modify to look for NWB files instead of CSV files
        try:
            files = os.listdir(self.dataset_folder)
            # Look for files that match the pattern YYYY-MM-DD_CO.nwb or YYYY-MM-DD_RD.nwb
            self.days = sorted(
                list(set([f.split("_")[0] for f in files if f.endswith(".nwb")]))
            )
            if self.days:
                # Update training day Combobox
                self.train_day_menu["values"] = self.days
                self.train_day_var.set(self.days[0])

                # Update testing day Combobox
                self.test_day_menu["values"] = self.days
                self.test_day_var.set(self.days[0])
            else:
                self.train_day_var.set("Select Day")
                self.test_day_var.set("Select Day")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read dataset folder: {e}")

    def run_regression(self):
        try:
            # Get user inputs
            train_day = self.train_day_var.get()
            test_day = self.test_day_var.get()
            train_style = self.train_style_var.get()
            test_style = self.test_style_var.get()
            history = int(self.history_entry.get())
            feature_type = self.feature_var.get()

            # Validate inputs
            if train_day == "Select Day" or test_day == "Select Day":
                messagebox.showerror(
                    "Input Error", "Please select both training and testing days."
                )
                return

            # Load data
            X_train, y_train, time_train = self.load_data(
                train_day, train_style, history, feature_type
            )
            X_test, y_test, time_test = self.load_data(
                test_day, test_style, history, feature_type
            )

            # If training and testing data are the same, split into 80/20
            if train_day == test_day and train_style == test_style:
                split_idx = int(0.8 * len(X_train))
                X_test = X_train[split_idx:]
                y_test = y_train[split_idx:]
                time_test = time_train[split_idx:]
                X_train = X_train[:split_idx]
                y_train = y_train[:split_idx]
                time_train = time_train[:split_idx]

            # Train Ridge Regression model
            model = Ridge()
            model.fit(X_train, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")

            # Calculate Pearson correlation coefficients
            pearson_corr = []
            for i in range(y_test.shape[1]):
                if np.std(y_test[:, i]) == 0 or np.std(y_pred[:, i]) == 0:
                    corr_coef = float("nan")
                else:
                    corr_coef, _ = pearsonr(y_test[:, i], y_pred[:, i])
                pearson_corr.append(corr_coef)

            # Display results
            self.display_results(mse, pearson_corr)

            # Plot results
            self.plot_results(y_test, y_pred, time_test)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def load_data(self, day, target_style, history, feature_type):
        try:
            # Build file path
            nwb_file_path = os.path.join(
                self.dataset_folder, f"{day}_{target_style}.nwb"
            )

            # Load data from NWB file
            io = NWBHDF5IO(nwb_file_path, "r")
            nwbfile = io.read()

            # Get time vector
            # Assume time vector is the timestamps of one of the timeseries
            # Get behavior timeseries
            behavior_module = nwbfile.processing.get("behavior")
            if behavior_module is None:
                raise ValueError("Behavior processing module not found in NWB file.")

            # Get target variables
            target_names = [
                "index_position",
                "mrp_position",
                "index_velocity",
                "mrp_velocity",
            ]
            y_data = []
            time = None  # Initialize time vector
            for name in target_names:
                ts = behavior_module.get_data_interface(name)
                if ts is None:
                    raise ValueError(
                        f"Behavior timeseries '{name}' not found in NWB file."
                    )
                data = ts.data[:]
                y_data.append(data)

                # Get time vector from the first timeseries
                if time is None:
                    if ts.timestamps is not None:
                        time = ts.timestamps[:]
                    else:
                        num_samples = len(ts.data)
                        rate = ts.rate  # Samples per second
                        starting_time = ts.starting_time
                        time = starting_time + np.arange(num_samples) / rate

            y = np.column_stack(y_data)

            # Get feature data
            ecephys_module = nwbfile.processing.get("ecephys")
            if ecephys_module is None:
                raise ValueError("Ecephys processing module not found in NWB file.")

            if feature_type == "TCFR":
                # Use ThresholdCrossings data
                tcfr_ts = ecephys_module.get_data_interface("ThresholdCrossings")
                if tcfr_ts is None:
                    raise ValueError(
                        "ThresholdCrossings timeseries not found in NWB file."
                    )
                X_data = tcfr_ts.data[:]
            elif feature_type == "SBP":
                # Use SpikingBandPower data
                sbp_ts = ecephys_module.get_data_interface("SpikingBandPower")
                if sbp_ts is None:
                    raise ValueError(
                        "SpikingBandPower timeseries not found in NWB file."
                    )
                X_data = sbp_ts.data[:]
            else:
                raise ValueError(f"Invalid feature type '{feature_type}'.")

            # Include history
            X_list = []
            for i in range(history):
                shifted_data = np.roll(X_data, i, axis=0)
                # Zero out the first i rows
                shifted_data[:i, :] = 0
                X_list.append(shifted_data)
            X = np.hstack(X_list)

            # Close the NWB file
            io.close()

            return X, y, time

        except FileNotFoundError:
            raise FileNotFoundError(
                f"NWB file for {day} with target style {target_style} not found."
            )
        except Exception as e:
            raise Exception(f"Failed to load data: {e}")

    def display_results(self, mse, pearson_corr):
        # Clear previous results
        for item in self.results_table.get_children():
            self.results_table.delete(item)

        # Insert new results
        self.results_table.insert(
            "",
            "end",
            values=(
                "MSE",
                f"{mse[0]:.4f}",
                f"{mse[1]:.4f}",
                f"{mse[2]:.4f}",
                f"{mse[3]:.4f}",
            ),
        )
        self.results_table.insert(
            "",
            "end",
            values=(
                "Pearson Corr.",
                f"{pearson_corr[0]:.4f}",
                f"{pearson_corr[1]:.4f}",
                f"{pearson_corr[2]:.4f}",
                f"{pearson_corr[3]:.4f}",
            ),
        )

    def plot_results(self, y_test, y_pred, time_test):
        # Clear previous plots if any
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar is not None:
            self.toolbar.destroy()

        time_in_seconds = time_test

        # Create figure
        self.figure = plt.Figure(figsize=(10, 8))

        # Create subplots with shared x-axis
        axs = self.figure.subplots(2, 2, sharex=True)
        # Adjust layout to make room for x-axis labels and legend
        self.figure.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=2.5)

        # Plot settings
        line_styles = ["-", "--"]
        labels = ["Ground truth", "Predicted"]

        # Plot Index Position
        axs[0, 0].plot(time_in_seconds, y_test[:, 0], line_styles[0])
        axs[0, 0].plot(time_in_seconds, y_pred[:, 0], line_styles[1])
        axs[0, 0].set_title("Index Position")
        axs[0, 0].set_ylabel("Position")

        # Plot MRP Position
        axs[0, 1].plot(time_in_seconds, y_test[:, 1], line_styles[0])
        axs[0, 1].plot(time_in_seconds, y_pred[:, 1], line_styles[1])
        axs[0, 1].set_title("MRP Position")

        # Plot Index Velocity
        axs[1, 0].plot(time_in_seconds, y_test[:, 2], line_styles[0])
        axs[1, 0].plot(time_in_seconds, y_pred[:, 2], line_styles[1])
        axs[1, 0].set_title("Index Velocity")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Velocity")

        # Plot MRP Velocity
        axs[1, 1].plot(time_in_seconds, y_test[:, 3], line_styles[0])
        axs[1, 1].plot(time_in_seconds, y_pred[:, 3], line_styles[1])
        axs[1, 1].set_title("MRP Velocity")
        axs[1, 1].set_xlabel("Time (s)")

        # Remove individual legends
        for ax in axs.flat:
            ax.label_outer()

        # Create a single legend at the bottom
        lines = [
            axs[0, 0].lines[0],
            axs[0, 0].lines[1],
        ]  # Get one set of lines for the legend
        self.figure.legend(lines, labels, loc="lower center", ncol=2)

        # Adjust subplots to make room for legend
        self.figure.subplots_adjust(bottom=0.12)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plots_frame)
        self.canvas.draw()

        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plots_frame)
        self.toolbar.update()

        # Pack canvas and toolbar
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # The rest of the code remains the same...


if __name__ == "__main__":
    root = tk.Tk()
    app = RidgeRegressionGUI(root)
    root.mainloop()
