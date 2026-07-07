import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def generate_sample_sales_data(start_date="2022-01-01", periods=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    t = np.arange(periods)

    trend = 200 + 0.15 * t
    yearly_seasonality = 40 * np.sin(2 * np.pi * (t - 80) / 365)
    day_of_week = dates.dayofweek
    weekly_seasonality = np.where(day_of_week >= 5, 25, 0)
    noise = rng.normal(0, 15, size=periods)

    y = trend + yearly_seasonality + weekly_seasonality + noise
    y = np.maximum(y, 0)

    return pd.DataFrame({"ds": dates, "y": y})


class ForecastApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sales & Demand Forecasting Dashboard")
        self.geometry("1150x750")
        self.minsize(950, 650)

        self.df = None
        self.forecast = None
        self.model = None

        self.build_top_bar()
        self.build_tabs()

    def build_top_bar(self):
        bar = ttk.Frame(self, padding=10)
        bar.pack(side="top", fill="x")

        ttk.Button(bar, text="Use Sample Data", command=self.load_sample_data).pack(side="left", padx=5)
        ttk.Button(bar, text="Upload CSV", command=self.upload_csv).pack(side="left", padx=5)

        ttk.Label(bar, text="Forecast days:").pack(side="left", padx=(20, 5))
        self.horizon_var = tk.IntVar(value=90)
        ttk.Spinbox(bar, from_=7, to=365, textvariable=self.horizon_var, width=6).pack(side="left")

        ttk.Button(bar, text="Run Forecast", command=self.run_forecast_threaded).pack(side="left", padx=20)
        ttk.Button(bar, text="Export Forecast CSV", command=self.export_csv).pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="No data loaded yet.")
        ttk.Label(bar, textvariable=self.status_var, foreground="gray").pack(side="right", padx=10)

    def build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)

        self.tab_data = ttk.Frame(self.tabs)
        self.tab_forecast = ttk.Frame(self.tabs)
        self.tab_components = ttk.Frame(self.tabs)
        self.tab_table = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_data, text="Input Data")
        self.tabs.add(self.tab_forecast, text="Actual vs Predicted")
        self.tabs.add(self.tab_components, text="Trend & Seasonality")
        self.tabs.add(self.tab_table, text="Forecast Table")

        self.data_tree = self.make_tree(self.tab_data)
        self.table_tree = self.make_tree(self.tab_table)

        self.fig1 = Figure(figsize=(9, 5), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab_forecast)
        self.canvas1.get_tk_widget().pack(fill="both", expand=True)

        self.canvas2 = None

    def make_tree(self, parent):
        tree = ttk.Treeview(parent, show="headings")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return tree

    def fill_tree(self, tree, df):
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor="center")
        for _, row in df.iterrows():
            tree.insert("", "end", values=list(row))

    def load_sample_data(self):
        self.df = generate_sample_sales_data()
        self.status_var.set(f"Sample data loaded: {len(self.df)} rows")
        self.fill_tree(self.data_tree, self.df.tail(50))

    def upload_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        raw_df = pd.read_csv(path)
        col_window = tk.Toplevel(self)
        col_window.title("Select columns")
        col_window.geometry("320x160")

        ttk.Label(col_window, text="Date column:").pack(pady=(15, 0))
        date_var = tk.StringVar(value=raw_df.columns[0])
        ttk.Combobox(col_window, textvariable=date_var, values=list(raw_df.columns), state="readonly").pack()

        ttk.Label(col_window, text="Value column:").pack(pady=(10, 0))
        value_var = tk.StringVar(value=raw_df.columns[-1])
        ttk.Combobox(col_window, textvariable=value_var, values=list(raw_df.columns), state="readonly").pack()

        def confirm():
            date_col = date_var.get()
            value_col = value_var.get()
            df = raw_df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna().sort_values("ds").reset_index(drop=True)

            if df.empty:
                messagebox.showerror("Error", "Couldn't parse that date/value column combination.")
                return

            self.df = df
            self.status_var.set(f"Uploaded data loaded: {len(self.df)} rows")
            self.fill_tree(self.data_tree, self.df.tail(50))
            col_window.destroy()

        ttk.Button(col_window, text="Confirm", command=confirm).pack(pady=15)

    def run_forecast_threaded(self):
        if self.df is None:
            messagebox.showwarning("No data", "Load sample data or upload a CSV first.")
            return
        self.status_var.set("Fitting Prophet model...")
        thread = threading.Thread(target=self.run_forecast)
        thread.start()

    def run_forecast(self):
        try:
            horizon = self.horizon_var.get()
            model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
            model.fit(self.df)

            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            self.model = model
            self.forecast = forecast

            self.after(0, self.update_plots)
            self.after(0, lambda: self.status_var.set(f"Forecast complete: {horizon} days ahead"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.status_var.set("Error during forecasting"))

    def update_plots(self):
        self.ax1.clear()
        self.ax1.plot(self.df["ds"], self.df["y"], "k.", alpha=0.5, markersize=3, label="Actual")
        self.ax1.plot(self.forecast["ds"], self.forecast["yhat"], color="royalblue", label="Predicted")
        self.ax1.fill_between(
            self.forecast["ds"], self.forecast["yhat_lower"], self.forecast["yhat_upper"],
            color="royalblue", alpha=0.2, label="Confidence Interval"
        )
        self.ax1.legend()
        self.ax1.set_title("Actual vs Predicted")
        self.ax1.set_xlabel("Date")
        self.ax1.set_ylabel("Value")
        self.fig1.tight_layout()
        self.canvas1.draw()

        try:
            for widget in self.tab_components.winfo_children():
                widget.destroy()
            comp_fig = self.model.plot_components(self.forecast)
            comp_fig.set_size_inches(9, 7)
            self.canvas2 = FigureCanvasTkAgg(comp_fig, master=self.tab_components)
            self.canvas2.get_tk_widget().pack(fill="both", expand=True)
            self.canvas2.draw()
        except Exception as e:
            messagebox.showerror("Components plot error", str(e))

        try:
            table_df = (
                self.forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                .tail(self.horizon_var.get())
                .rename(columns={
                    "ds": "Date", "yhat": "Predicted",
                    "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"
                })
            )
            table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
            for col in ["Predicted", "Lower Bound", "Upper Bound"]:
                table_df[col] = table_df[col].round(2)
            self.fill_tree(self.table_tree, table_df)
        except Exception as e:
            messagebox.showerror("Forecast table error", str(e))

    def export_csv(self):
        if self.forecast is None:
            messagebox.showwarning("No forecast", "Run a forecast first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        table_df = (
            self.forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            .tail(self.horizon_var.get())
            .rename(columns={
                "ds": "Date", "yhat": "Predicted",
                "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"
            })
        )
        table_df.to_csv(path, index=False)
        messagebox.showinfo("Saved", f"Forecast saved to {path}")


if __name__ == "__main__":
    app = ForecastApp()
    app.mainloop()