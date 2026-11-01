import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import partial_dependence
import joblib
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import warnings
import threading
from datetime import datetime
import numpy as np
import scipy.stats as stats
from scipy.stats import skew, kurtosis, shapiro, normaltest
from itertools import combinations

# 忽略警告
warnings.filterwarnings('ignore')

# 应用安全配置
ALLOWED_FILE_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# 全局字体设置
def setup_english_font():
    try:
        plt.rcParams['font.family'] ='Arial'
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        sns.set_palette("deep")
    except Exception as e:
        print(f"Font setup issue: {e}")

setup_english_font()
pd.set_option('display.float_format', '{:.4f}'.format)
plt.rcParams.update({'font.size': 12})

# 项目根目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir) if os.path.basename(current_script_dir) == 'Analyzer' else current_script_dir

COLORS = {
    'primary': '#2c3e50', 'secondary': '#3498db', 'accent': '#e74c3c',
    'success': '#27ae60', 'warning': '#f39c12', 'light': '#ecf0f1',
    'dark': '#2c3e50', 'background': '#f8f9fa', 'text': '#2c3e50',
    'text_light': '#7f8c8d'
}

def save_plot_english(fig, filepath, dpi=600):
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    except:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')

def save_plot_data(data, filepath_csv, index=False):
    os.makedirs(os.path.dirname(filepath_csv), exist_ok=True)
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath_csv, index=index)
    elif isinstance(data, np.ndarray):
        pd.DataFrame(data).to_csv(filepath_csv, index=index)
    elif isinstance(data, dict):
        pd.DataFrame(data).to_csv(filepath_csv, index=index)
    elif isinstance(data, list):
        pd.DataFrame(data).to_csv(filepath_csv, index=index)
    else:
        pd.DataFrame({'data': data}).to_csv(filepath_csv, index=index)

class EnhancedLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, offset_method='auto'):
        self.threshold = threshold
        self.offset_method = offset_method
        self.columns = None
        self.offsets = {}
        self.apply_log = {}
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns = X.columns
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_trans = X.copy()
        for col in X_trans.columns:
            col_skew = skew(X_trans[col].dropna())
            if abs(col_skew) > self.threshold:
                self.apply_log[col] = True
                min_val = X_trans[col].min()
                if self.offset_method == 'auto':
                    if min_val <= 0:
                        offset = abs(min_val) + 1e-10
                        if offset <= 0:
                            offset = 1e-10
                    else:
                        q1 = X_trans[col].quantile(0.01)
                        offset = max(0, q1 * 0.01)
                else:
                    offset = self.offset_method
                self.offsets[col] = offset
                X_trans[col] = np.log1p(X_trans[col] + offset)
            else:
                self.apply_log[col] = False
                self.offsets[col] = 0
        return X_trans.values
    def inverse_transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_inv = X.copy()
        for col_idx, col_name in enumerate(self.columns):
            if col_name in self.apply_log and self.apply_log[col_name]:
                offset = self.offsets.get(col_name, 0)
                X_inv[col_idx] = np.exp(X_inv[col_idx]) - 1 - offset
        return X_inv.values
    def get_feature_names_out(self, input_features=None):
        return self.columns

class EnhancedPhysicalScaleHandler:
    def __init__(self):
        self.target_stats = {}
        self.feature_stats = {}
        self.log_offsets = {}
        self.scaler_params = {}
        self.imputer_params = {}
        self.apply_log = {}
        self.original_dtypes = {}
    def fit_transform_features(self, X):
        X_processed = X.copy()
        for col in X.columns:
            self.original_dtypes[col] = str(X[col].dtype)
            self.feature_stats[col] = {
                'mean': float(X[col].mean()), 'std': float(X[col].std()),
                'min': float(X[col].min()), 'max': float(X[col].max()),
                'skewness': float(skew(X[col].dropna())), 'kurtosis': float(kurtosis(X[col].dropna())),
                'median': float(X[col].median())
            }
        num_cols = X.select_dtypes(include=np.number).columns
        self.imputer_params['strategy'] = 'median'
        self.imputer_params['features'] = list(num_cols)
        imputer = SimpleImputer(strategy='median')
        X_imputed = X.copy()
        X_imputed[num_cols] = imputer.fit_transform(X_imputed[num_cols])
        self.imputer_params['statistics'] = {col: float(imputer.statistics_[i]) for i, col in enumerate(num_cols)}
        enhanced_log_transformer = EnhancedLogTransformer(threshold=0.5, offset_method='auto')
        X_logged = X_imputed.copy()
        for col in num_cols:
            col_data = X_logged[col].values.reshape(-1, 1)
            transformed = enhanced_log_transformer.fit_transform(pd.DataFrame({col: col_data.flatten()}))
            X_logged[col] = transformed.flatten()
            if enhanced_log_transformer.apply_log.get(col, False):
                self.apply_log[col] = True
                self.log_offsets[col] = enhanced_log_transformer.offsets.get(col, 0)
                self.feature_stats[col]['after_log_skewness'] = float(skew(X_logged[col].dropna()))
                self.feature_stats[col]['after_log_kurtosis'] = float(kurtosis(X_logged[col].dropna()))
            else:
                self.apply_log[col] = False
                self.log_offsets[col] = 0
        scaler = RobustScaler(quantile_range=(25, 75))
        X_scaled = X_logged.copy()
        X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])
        self.scaler_params['center'] = {col: float(scaler.center_[i]) for i, col in enumerate(num_cols)}
        self.scaler_params['scale'] = {col: float(scaler.scale_[i]) for i, col in enumerate(num_cols)}
        for col in num_cols:
            idx = list(num_cols).index(col)
            self.feature_stats[col]['scaled_mean'] = float(scaler.center_[idx])
            self.feature_stats[col]['scaled_std'] = float(scaler.scale_[idx])
            self.feature_stats[col]['scaled_skewness'] = float(skew(X_scaled[col].dropna()))
            self.feature_stats[col]['scaled_kurtosis'] = float(kurtosis(X_scaled[col].dropna()))
        return X_scaled, X_imputed, X_logged
    def save_target_stats(self, y, target_name):
        y_array = y.values if hasattr(y, 'values') else y
        self.target_stats = {
            'name': target_name, 'mean': float(np.mean(y_array)), 'std': float(np.std(y_array)),
            'min': float(np.min(y_array)), 'max': float(np.max(y_array)), 'median': float(np.median(y_array)),
            'skewness': float(skew(y_array)), 'kurtosis': float(kurtosis(y_array)), 'n_samples': len(y_array),
            'q1': float(np.percentile(y_array, 25)), 'q3': float(np.percentile(y_array, 75)),
            'iqr': float(np.percentile(y_array, 75) - np.percentile(y_array, 25))
        }
    def transform_target(self, y):
        y_array = y.values if hasattr(y, 'values') else y
        y_skew = abs(skew(y_array))
        if y_skew > 0.5:
            min_val = np.min(y_array)
            offset = abs(min_val) + 1e-10 if min_val <= 0 else 0
            self.target_stats['log_transformed'] = True
            self.target_stats['log_offset'] = float(offset)
            y_transformed = np.log1p(y_array + offset)
            self.target_stats['transformed_mean'] = float(np.mean(y_transformed))
            self.target_stats['transformed_std'] = float(np.std(y_transformed))
            self.target_stats['transformed_skewness'] = float(skew(y_transformed))
            self.target_stats['transformed_kurtosis'] = float(kurtosis(y_transformed))
            return y_transformed
        else:
            self.target_stats['log_transformed'] = False
            self.target_stats['log_offset'] = 0
            return y_array
    def inverse_transform_target(self, y_transformed):
        if self.target_stats.get('log_transformed', False):
            offset = self.target_stats.get('log_offset', 0)
            return np.expm1(y_transformed) - offset
        else:
            return y_transformed
    def inverse_transform_features(self, X_scaled, features=None):
        if not isinstance(X_scaled, pd.DataFrame):
            X_scaled = pd.DataFrame(X_scaled)
        X_original = X_scaled.copy()
        if features is None:
            features = list(self.feature_stats.keys())
        for col in features:
            if col in self.scaler_params.get('center', {}):
                center = self.scaler_params['center'][col]
                scale = self.scaler_params['scale'][col]
                X_original[col] = X_original[col] * scale + center
        for col in features:
            if col in self.apply_log and self.apply_log[col]:
                offset = self.log_offsets.get(col, 0)
                X_original[col] = np.expm1(X_original[col]) - offset
        return X_original
    def save_transform_parameters(self, filepath):
        transform_data = {
            'target_stats': self.target_stats, 'feature_stats': self.feature_stats,
            'log_offsets': self.log_offsets, 'apply_log': self.apply_log,
            'scaler_params': self.scaler_params, 'imputer_params': self.imputer_params,
            'original_dtypes': self.original_dtypes, 'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(transform_data, f, indent=2, ensure_ascii=False)
    def get_target_transform_params(self):
        return self.target_stats

def validate_file_path(file_path):
    if not file_path:
        return False
    _, ext = os.path.splitext(file_path.lower())
    if ext not in ALLOWED_FILE_EXTENSIONS:
        return False
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        return False
    abs_path = os.path.abspath(file_path)
    project_abs_path = os.path.abspath(project_root)
    return abs_path.startswith(project_abs_path)

class DataQualityChecker:
    @staticmethod
    def check_feature_distribution(df, feature_names):
        results = []
        for feature in feature_names:
            if feature not in df.columns:
                continue
            data = df[feature].dropna()
            if len(data) < 10:
                continue
            stats_dict = {
                'feature': feature, 'n': len(data), 'mean': float(np.mean(data)), 'std': float(np.std(data)),
                'min': float(np.min(data)), 'max': float(np.max(data)), 'median': float(np.median(data)),
                'skewness': float(skew(data)), 'kurtosis': float(kurtosis(data)),
                'missing_pct': float(df[feature].isnull().sum() / len(df) * 100),
                'zero_pct': float((df[feature] == 0).sum() / len(df) * 100),
                'constant': float(df[feature].nunique() == 1)
            }
            try:
                _, shapiro_p = shapiro(data)
                _, normaltest_p = normaltest(data)
                stats_dict['shapiro_p'] = float(shapiro_p)
                stats_dict['normaltest_p'] = float(normaltest_p)
                stats_dict['is_normal'] = shapiro_p > 0.05
            except:
                stats_dict['shapiro_p'] = np.nan
                stats_dict['normaltest_p'] = np.nan
                stats_dict['is_normal'] = False
            q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            outliers = data[(data < lower) | (data > upper)]
            stats_dict['outlier_pct'] = float(len(outliers) / len(data) * 100)
            results.append(stats_dict)
        return pd.DataFrame(results)

class RandomForestPreTrainer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Random Forest Pre-Training for Transfer Learning")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS['background'])
        self.setup_scrollable_main_frame()
        self.setup_styles()
        self.create_widgets()
        self.df = None
        self.timestamp = None
        self.is_running = False
        self.df_processed = None
        self.target_variable = None
        self.scale_handler = EnhancedPhysicalScaleHandler()
        self.data_quality_report = None
        self.n_pdp = 20
        self.n_top_2d = 7

    def setup_scrollable_main_frame(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.main_frame, bg=COLORS['background'])
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar = ttk.Scrollbar(self.root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.inner_frame = ttk.Frame(self.canvas, padding="20")
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.inner_frame, anchor=tk.NW)
        self.inner_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self.on_shift_mousewheel)
    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    def on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), background=COLORS['background'], foreground=COLORS['primary'])
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 14), background=COLORS['background'], foreground=COLORS['text_light'])
        self.style.configure('Step.TLabel', font=('Segoe UI', 12, 'bold'), background=COLORS['background'], foreground=COLORS['secondary'])
        self.style.configure('Status.TLabel', font=('Segoe UI', 10), background=COLORS['background'], foreground=COLORS['text'])
        self.style.configure('Primary.TButton', font=('Segoe UI', 11, 'bold'), padding=10, background=COLORS['secondary'], foreground='white')
        self.style.configure('Success.TButton', font=('Segoe UI', 11), padding=8, background=COLORS['success'], foreground='white')
        self.style.configure('Accent.TButton', font=('Segoe UI', 11), padding=8, background=COLORS['accent'], foreground='white')
        self.style.configure('Default.TButton', font=('Segoe UI', 10), padding=6)
        self.style.configure('Horizontal.TProgressbar', troughcolor=COLORS['light'], background=COLORS['success'], borderwidth=0)
    def create_widgets(self):
        main_frame = self.inner_frame
        main_frame.columnconfigure(0, weight=1)
        # 标题
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, pady=(0,20), sticky=(tk.W,tk.E))
        ttk.Label(title_frame, text="Random Forest Pre-Training", style='Title.TLabel').grid(row=0, column=0, pady=5)
        ttk.Label(title_frame, text="Train Random Forest model and prepare for XGBoost Transfer Learning", style='Subtitle.TLabel').grid(row=1, column=0, pady=5)
        ttk.Separator(main_frame, orient='horizontal').grid(row=1, column=0, pady=10, sticky=(tk.W,tk.E))
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="15")
        control_frame.grid(row=2, column=0, pady=10, sticky=(tk.W,tk.E))
        control_frame.columnconfigure(0, weight=1)
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, pady=10)
        self.load_btn = ttk.Button(button_frame, text="📁 Load Data", style='Primary.TButton', command=self.load_data_wrapper)
        self.load_btn.grid(row=0, column=0, padx=5)
        self.analyze_btn = ttk.Button(button_frame, text="📊 Analyze Data", style='Default.TButton', command=self.analyze_data_wrapper, state='disabled')
        self.analyze_btn.grid(row=0, column=1, padx=5)
        self.preprocess_btn = ttk.Button(button_frame, text="⚙️ Preprocess Data", style='Default.TButton', command=self.preprocess_data_wrapper, state='disabled')
        self.preprocess_btn.grid(row=0, column=2, padx=5)
        self.train_btn = ttk.Button(button_frame, text="🤖 Train Model", style='Default.TButton', command=self.train_model_wrapper, state='disabled')
        self.train_btn.grid(row=0, column=3, padx=5)
        self.run_all_btn = ttk.Button(button_frame, text="🚀 Run All Steps", style='Success.TButton', command=self.run_all_wrapper)
        self.run_all_btn.grid(row=0, column=4, padx=5)
        # PDP参数调节
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=1, column=0, pady=5, sticky=(tk.W,tk.E))
        ttk.Label(param_frame, text="PDP Bootstrap Samples (1D):").pack(side=tk.LEFT, padx=5)
        self.n_pdp_var = tk.IntVar(value=20)
        n_pdp_spin = ttk.Spinbox(param_frame, from_=1, to=100, width=5, textvariable=self.n_pdp_var, command=self.update_n_pdp)
        n_pdp_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(param_frame, text="Top Features for 2D PDP:").pack(side=tk.LEFT, padx=5)
        self.n_top_2d_var = tk.IntVar(value=7)
        n_top_spin = ttk.Spinbox(param_frame, from_=2, to=10, width=5, textvariable=self.n_top_2d_var, command=self.update_n_top_2d)
        n_top_spin.pack(side=tk.LEFT, padx=5)
        # 状态区域
        status_frame = ttk.LabelFrame(main_frame, text="Progress Status", padding="15")
        status_frame.grid(row=3, column=0, pady=10, sticky=(tk.W,tk.E))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, style='Horizontal.TProgressbar', length=400, mode='determinate')
        self.progress_bar.grid(row=0, column=0, pady=(0,10), sticky=(tk.W,tk.E))
        self.status_label = ttk.Label(status_frame, text="Ready to start...", style='Status.TLabel')
        self.status_label.grid(row=1, column=0, pady=5)
        steps_frame = ttk.Frame(status_frame)
        steps_frame.grid(row=2, column=0, pady=10)
        self.step_labels = {}
        steps = [("1","Load Data","⭕"),("2","Data Analysis","⭕"),("3","Preprocessing","⭕"),("4","Model Training","⭕"),("5","Transfer Learning","⭕")]
        for i,(num,text,icon) in enumerate(steps):
            step_frame = ttk.Frame(steps_frame)
            step_frame.grid(row=0, column=i, padx=10)
            icon_label = ttk.Label(step_frame, text=icon, font=('Segoe UI',14))
            icon_label.grid(row=0, column=0, padx=5)
            text_label = ttk.Label(step_frame, text=f"Step {num}\n{text}", font=('Segoe UI',9), justify='center')
            text_label.grid(row=1, column=0)
            self.step_labels[num] = icon_label
        # 输出区域
        output_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="10")
        output_frame.grid(row=4, column=0, pady=10, sticky=(tk.W,tk.E,tk.N,tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        output_container = ttk.Frame(output_frame)
        output_container.grid(row=0, column=0, sticky=(tk.W,tk.E,tk.N,tk.S))
        output_container.columnconfigure(0, weight=1)
        output_container.rowconfigure(0, weight=1)
        text_scrollbar = ttk.Scrollbar(output_container)
        text_scrollbar.grid(row=0, column=1, sticky=(tk.N,tk.S))
        self.output_text = tk.Text(output_container, height=15, width=100, font=('Consolas',10), bg='white', fg=COLORS['text'], yscrollcommand=text_scrollbar.set, wrap=tk.WORD)
        self.output_text.grid(row=0, column=0, sticky=(tk.W,tk.E,tk.N,tk.S))
        text_scrollbar.config(command=self.output_text.yview)
        h_scrollbar = ttk.Scrollbar(output_container, orient=tk.HORIZONTAL)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W,tk.E))
        self.output_text.config(xscrollcommand=h_scrollbar.set)
        h_scrollbar.config(command=self.output_text.xview)
        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=5, column=0, pady=20)
        ttk.Button(bottom_frame, text="📋 Copy Log", command=self.copy_log).grid(row=0, column=0, padx=5)
        ttk.Button(bottom_frame, text="🗑️ Clear Log", command=self.clear_log).grid(row=0, column=1, padx=5)
        ttk.Button(bottom_frame, text="📖 View Transfer Learning", command=self.view_transfer_learning).grid(row=0, column=2, padx=5)
        ttk.Button(bottom_frame, text="❌ Exit", style='Accent.TButton', command=self.root.quit).grid(row=0, column=3, padx=5)
    def update_n_pdp(self):
        self.n_pdp = self.n_pdp_var.get()
    def update_n_top_2d(self):
        self.n_top_2d = self.n_top_2d_var.get()
    def update_status(self, message, step=None, status="working"):
        colors = {"working":COLORS['secondary'],"success":COLORS['success'],"error":COLORS['accent'],"warning":COLORS['warning'],"default":COLORS['text']}
        self.status_label.configure(text=message, foreground=colors.get(status, COLORS['text']))
        if step:
            icons = {"working":"🔄","success":"✅","error":"❌","warning":"⚠️"}
            if step in self.step_labels:
                self.step_labels[step].configure(text=icons.get(status, "⭕"))
        self.output_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.output_text.see(tk.END)
        self.root.update()
    def update_progress(self, value):
        self.progress_var.set(value)
        self.root.update()
    def log_message(self, message):
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
    def copy_log(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.output_text.get("1.0", tk.END))
            self.update_status("Log copied to clipboard", status="success")
        except Exception as e:
            self.log_message(f"Error copying log: {e}")
    def clear_log(self):
        try:
            self.output_text.delete("1.0", tk.END)
            self.update_status("Log cleared", status="success")
        except Exception as e:
            self.log_message(f"Error clearing log: {e}")
    def select_target_variable(self, data_columns):
        try:
            select_dialog = tk.Toplevel(self.root)
            select_dialog.title("Select Target Variable")
            select_dialog.geometry("400x400")
            select_dialog.configure(bg=COLORS['background'])
            select_dialog.transient(self.root)
            select_dialog.grab_set()
            select_dialog.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - select_dialog.winfo_width())//2
            y = self.root.winfo_y() + (self.root.winfo_height() - select_dialog.winfo_height())//2
            select_dialog.geometry(f"+{x}+{y}")
            ttk.Label(select_dialog, text="Select Target Variable", font=('Segoe UI',14,'bold')).pack(pady=10)
            ttk.Label(select_dialog, text="Choose the variable you want to predict:", font=('Segoe UI',11)).pack(pady=5)
            ttk.Separator(select_dialog, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
            list_frame = ttk.Frame(select_dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Consolas',10), selectmode=tk.SINGLE, height=12)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=listbox.yview)
            h_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            listbox.config(xscrollcommand=h_scrollbar.set)
            h_scrollbar.config(command=listbox.xview)
            for i, col in enumerate(data_columns):
                listbox.insert(tk.END, f"{i+1:3d}. {col}")
                if i == len(data_columns)-1:
                    listbox.select_set(i)
            button_frame = ttk.Frame(select_dialog)
            button_frame.pack(pady=20)
            selected_var = [None]
            def confirm():
                if listbox.curselection():
                    selected_var[0] = data_columns[listbox.curselection()[0]]
                    select_dialog.destroy()
                else:
                    messagebox.showwarning("Warning", "Please select a target variable")
            def cancel():
                select_dialog.destroy()
            ttk.Button(button_frame, text="Confirm", command=confirm).grid(row=0, column=0, padx=10)
            ttk.Button(button_frame, text="Cancel", command=cancel).grid(row=0, column=1, padx=10)
            self.root.wait_window(select_dialog)
            return selected_var[0]
        except Exception as e:
            self.log_message(f"Error in target variable selection: {e}")
            return None
    def load_data_wrapper(self):
        if self.is_running:
            self.update_status("Another process is running. Please wait...", status="warning")
            return
        thread = threading.Thread(target=self.load_data_thread)
        thread.daemon = True
        thread.start()
    def load_data_thread(self):
        self.is_running = True
        try:
            self.update_status("Selecting data file...", step="1", status="working")
            self.update_progress(10)
            file_path = filedialog.askopenfilename(
                title="Select Data File",
                filetypes=[
                    ("Excel .xlsx", "*.xlsx"),
                    ("Excel .xls", "*.xls"),
                    ("CSV", "*.csv"),
                    ("All files", "*"),
                ],
                initialdir=project_root,
            )
            if not file_path:
                self.update_status("No file selected", status="warning")
                self.update_progress(0)
                self.is_running = False
                return
            if not validate_file_path(file_path):
                messagebox.showerror("Error", "Invalid file type or size")
                self.update_status("Invalid file selected", status="error")
                self.update_progress(0)
                self.is_running = False
                return
            self.update_status(f"Loading file: {os.path.basename(file_path)}...", status="working")
            self.update_progress(30)
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                selected_sheet = sheet_names[0]
                df = pd.read_excel(file_path, sheet_name=selected_sheet)
                self.log_message(f"Using first worksheet: {selected_sheet}")
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                self.log_message(f"Warning: Data contains {missing_values} missing values")
            self.df = df
            self.update_status(f"Data loaded successfully! Shape: {df.shape}", step="1", status="success")
            self.update_progress(100)
            self.analyze_btn.configure(state='normal')
            self.preprocess_btn.configure(state='normal')
            self.train_btn.configure(state='normal')
            self.log_message(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            self.log_message(f"  File: {os.path.basename(file_path)}")
        except Exception as e:
            self.update_status(f"Error loading data: {str(e)}", status="error")
            self.log_message(f"✗ Error: {str(e)}")
            self.update_progress(0)
        finally:
            self.is_running = False
    def analyze_data_wrapper(self):
        if self.is_running:
            self.update_status("Another process is running. Please wait...", status="warning")
            return
        if self.df is None:
            self.update_status("Please load data first", status="warning")
            return
        thread = threading.Thread(target=self.analyze_data_thread)
        thread.daemon = True
        thread.start()
    def analyze_data_thread(self):
        self.is_running = True
        try:
            self.update_status("Starting data analysis...", step="2", status="working")
            self.update_progress(20)
            if self.timestamp is None:
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data0view_dir = os.path.join(project_root, 'data0view', self.timestamp)
            os.makedirs(data0view_dir, exist_ok=True)
            self.update_status("Generating statistics...", status="working")
            self.update_progress(40)
            self.original_describe_data(self.df, data0view_dir)
            self.update_status("Data analysis completed!", step="2", status="success")
            self.update_progress(100)
            self.log_message(f"✓ Data analysis completed")
            self.log_message(f"  Results saved to: data0view/{self.timestamp}")
        except Exception as e:
            self.update_status(f"Error in data analysis: {str(e)}", status="error")
            self.log_message(f"✗ Error: {str(e)}")
            self.update_progress(0)
        finally:
            self.is_running = False
    def original_describe_data(self, df, data0view_dir):
        try:
            self.log_message("Dataset basic description:")
            self.log_message(str(df.describe()))
            stats_df = pd.concat([df.describe(), df.isnull().sum().to_frame(name='Missing Values')], axis=1)
            stats_df.to_csv(os.path.join(data0view_dir, "data_statistics.csv"))
            plt.figure(figsize=(15,10))
            cols = df.columns
            n_cols = len(cols)
            n_rows = (n_cols+2)//3
            for i, col in enumerate(cols):
                plt.subplot(n_rows, 3, i+1)
                sns.histplot(df[col], kde=True)
                #plt.title(f"{col} Distribution")
                plt.xlabel(col)
                plt.ylabel("Frequency")
            plt.tight_layout()
            plot_path = os.path.join(data0view_dir, "data_distribution_histograms.png")
            save_plot_english(plt.gcf(), plot_path, dpi=600)
            hist_data_dir = os.path.join(data0view_dir, "histogram_data")
            os.makedirs(hist_data_dir, exist_ok=True)
            for col in cols:
                col_data = df[col].dropna().values
                save_plot_data(col_data, os.path.join(hist_data_dir, f"{col}_data.csv"))
            plt.close()
            plt.figure(figsize=(12,10))
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, center=0, square=True, linewidths=.5, cbar_kws={"shrink":.8})
            #plt.title("Variable Correlation Heatmap - Original Data")
            plt.tight_layout()
            corr_path = os.path.join(data0view_dir, "correlation_heatmap_original.png")
            save_plot_english(plt.gcf(), corr_path, dpi=600)
            plt.close()
            corr.to_csv(os.path.join(data0view_dir, "correlation_matrix_original.csv"))
            if len(df) <= 200:
                pair_plot = sns.pairplot(df, diag_kind='kde')
                #pair_plot.fig.suptitle("Variable Scatter Plot Matrix", y=1.02)
                scatter_path = os.path.join(data0view_dir, "variable_scatter_matrix.png")
                save_plot_english(pair_plot.fig, scatter_path, dpi=600)
                df.to_csv(os.path.join(data0view_dir, "variable_scatter_matrix_data.csv"), index=False)
                plt.close()
                self.log_message(f"  Scatter matrix saved: {scatter_path}")
            else:
                self.log_message("  Dataset has more than 200 rows, skipping scatter matrix")
        except Exception as e:
            raise e
    def preprocess_data_wrapper(self):
        if self.is_running:
            self.update_status("Another process is running. Please wait...", status="warning")
            return
        if self.df is None:
            self.update_status("Please load data first", status="warning")
            return
        thread = threading.Thread(target=self.preprocess_data_thread)
        thread.daemon = True
        thread.start()
    def preprocess_data_thread(self):
        self.is_running = True
        try:
            self.update_status("Starting data preprocessing...", step="3", status="working")
            self.update_progress(10)
            if self.timestamp is None:
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data1view_dir = os.path.join(project_root, 'data1view', self.timestamp)
            os.makedirs(data1view_dir, exist_ok=True)
            df_scaled, df_imputed, df_logged = self.fixed_preprocess_data(self.df, data1view_dir)
            self.df_processed = df_scaled
            self.update_status("Data preprocessing completed!", step="3", status="success")
            self.update_progress(100)
            self.log_message(f"✓ Data preprocessing completed")
            self.log_message(f"  Results saved to: data1view/{self.timestamp}")
        except Exception as e:
            self.update_status(f"Error in data preprocessing: {str(e)}", status="error")
            self.log_message(f"✗ Error: {str(e)}")
            self.update_progress(0)
        finally:
            self.is_running = False
    def fixed_preprocess_data(self, df, data1view_dir):
        try:
            df_original = df.copy()
            self.log_message("\n" + "="*60)
            self.log_message("DATA QUALITY CHECK")
            self.log_message("="*60)
            checker = DataQualityChecker()
            quality_report = checker.check_feature_distribution(df, df.columns.tolist())
            quality_report_path = os.path.join(data1view_dir, "data_quality_report.csv")
            quality_report.to_csv(quality_report_path, index=False)
            self.log_message("  Data quality report generated")
            num_cols = df.select_dtypes(include=np.number).columns
            imputer = SimpleImputer(strategy='median')
            df_imputed = df.copy()
            df_imputed[num_cols] = imputer.fit_transform(df_imputed[num_cols])
            df_imputed.to_csv(os.path.join(data1view_dir, "data_after_imputation.csv"), index=False)
            imputed_stats = df_imputed.describe()
            imputed_stats.to_csv(os.path.join(data1view_dir, "imputed_data_statistics.csv"))
            self.log_message("\n" + "="*60)
            self.log_message("SKEWNESS TREATMENT ANALYSIS")
            self.log_message("="*60)
            df_logged = df_imputed.copy()
            log_transformation_report = []
            for col in num_cols:
                original_data = df_logged[col].dropna()
                if len(original_data) < 10:
                    continue
                original_skew = skew(original_data)
                original_kurt = kurtosis(original_data)
                needs_log = abs(original_skew) > 0.5
                if needs_log:
                    min_val = original_data.min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1e-10
                        if min_val > -1e-10:
                            offset = 1e-10
                    else:
                        q1 = original_data.quantile(0.25)
                        offset = max(0, q1 * 0.01)
                    transformed_data = np.log1p(original_data + offset)
                    transformed_skew = skew(transformed_data)
                    transformed_kurt = kurtosis(transformed_data)
                    self.scale_handler.apply_log[col] = True
                    self.scale_handler.log_offsets[col] = offset
                    df_logged[col] = transformed_data
                    log_transformation_report.append({'Feature':col,'Original_Skewness':original_skew,'Original_Kurtosis':original_kurt,'Transformed_Skewness':transformed_skew,'Transformed_Kurtosis':transformed_kurt,'Offset_Used':offset,'Skewness_Improvement':abs(original_skew)-abs(transformed_skew),'Applied_Log':True})
                    self.log_message(f"  Applied log transform to {col}: skew {original_skew:.2f} -> {transformed_skew:.2f}")
                    if abs(transformed_kurt - original_kurt) > 5:
                        self.log_message(f"    WARNING: Large kurtosis change for {col}: {original_kurt:.2f} -> {transformed_kurt:.2f}")
                        if min_val <= 0:
                            df_logged[col] = np.sqrt(original_data + abs(min_val) + 1e-10)
                            sqrt_skew = skew(df_logged[col])
                            self.log_message(f"    Trying sqrt transform: skew {original_skew:.2f} -> {sqrt_skew:.2f}")
                else:
                    self.scale_handler.apply_log[col] = False
                    self.scale_handler.log_offsets[col] = 0
                    log_transformation_report.append({'Feature':col,'Original_Skewness':original_skew,'Original_Kurtosis':original_kurt,'Transformed_Skewness':original_skew,'Transformed_Kurtosis':original_kurt,'Offset_Used':0,'Skewness_Improvement':0,'Applied_Log':False})
                    self.log_message(f"  No transform needed for {col}: skew {original_skew:.2f}")
            log_report_df = pd.DataFrame(log_transformation_report)
            log_report_df.to_csv(os.path.join(data1view_dir, "log_transformation_report.csv"), index=False)
            df_logged.to_csv(os.path.join(data1view_dir, "data_after_log_transformation.csv"), index=False)
            self.log_message("\n" + "="*60)
            self.log_message("STANDARDIZATION ANALYSIS")
            self.log_message("="*60)
            scaler = RobustScaler(quantile_range=(25,75))
            df_scaled = df_logged.copy()
            df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
            self.scale_handler.scaler_params['center'] = {col: float(scaler.center_[i]) for i,col in enumerate(num_cols)}
            self.scale_handler.scaler_params['scale'] = {col: float(scaler.scale_[i]) for i,col in enumerate(num_cols)}
            df_scaled.to_csv(os.path.join(data1view_dir, "preprocessed_data.csv"), index=False)
            scaled_stats = df_scaled.describe()
            scaled_stats.to_csv(os.path.join(data1view_dir, "preprocessed_data_statistics.csv"))
            self.generate_preprocessing_report(df_original, df_imputed, df_logged, df_scaled, data1view_dir)
            return df_scaled, df_imputed, df_logged
        except Exception as e:
            self.log_message(f"✗ Error in fixed_preprocess_data: {e}")
            import traceback
            self.log_message(f"✗ Traceback: {traceback.format_exc()}")
            raise e
    def generate_preprocessing_report(self, df_original, df_imputed, df_logged, df_scaled, data1view_dir):
        try:
            report_dir = os.path.join(data1view_dir, "preprocessing_analysis")
            os.makedirs(report_dir, exist_ok=True)
            self.log_message("\n" + "="*60)
            self.log_message("GENERATING PREPROCESSING REPORT")
            self.log_message("="*60)
            num_cols = df_original.select_dtypes(include=np.number).columns
            stats_comparison = []
            for col in num_cols:
                orig_data = df_original[col].dropna()
                if len(orig_data) < 5:
                    continue
                imputed_data = df_imputed[col]
                logged_data = df_logged[col]
                scaled_data = df_scaled[col]
                stats_dict = {
                    'Feature':col, 'Original_Skew':float(skew(orig_data)), 'Imputed_Skew':float(skew(imputed_data)),
                    'Logged_Skew':float(skew(logged_data)), 'Scaled_Skew':float(skew(scaled_data)),
                    'Original_Kurt':float(kurtosis(orig_data)), 'Imputed_Kurt':float(kurtosis(imputed_data)),
                    'Logged_Kurt':float(kurtosis(logged_data)), 'Scaled_Kurt':float(kurtosis(scaled_data)),
                    'Original_Mean':float(np.mean(orig_data)), 'Imputed_Mean':float(np.mean(imputed_data)),
                    'Logged_Mean':float(np.mean(logged_data)), 'Scaled_Mean':float(np.mean(scaled_data)),
                    'Original_Median':float(np.median(orig_data)), 'Imputed_Median':float(np.median(imputed_data)),
                    'Logged_Median':float(np.median(logged_data)), 'Scaled_Median':float(np.median(scaled_data)),
                    'Skewness_Change_Imputed':float(abs(skew(imputed_data)-skew(orig_data))),
                    'Skewness_Change_Logged':float(abs(skew(logged_data)-skew(orig_data))),
                    'Skewness_Change_Scaled':float(abs(skew(scaled_data)-skew(orig_data))),
                    'Kurtosis_Change_Imputed':float(abs(kurtosis(imputed_data)-kurtosis(orig_data))),
                    'Kurtosis_Change_Logged':float(abs(kurtosis(logged_data)-kurtosis(orig_data))),
                    'Kurtosis_Change_Scaled':float(abs(kurtosis(scaled_data)-kurtosis(orig_data)))
                }
                stats_comparison.append(stats_dict)
            stats_df = pd.DataFrame(stats_comparison)
            stats_df.to_csv(os.path.join(report_dir, "statistical_changes_comparison.csv"), index=False)
            problem_features = []
            kurtosis_problems = stats_df[stats_df['Kurtosis_Change_Logged'] > 3]
            if len(kurtosis_problems) > 0:
                self.log_message(f"\n⚠️  WARNING: {len(kurtosis_problems)} features with large kurtosis changes after log transform:")
                for _,row in kurtosis_problems.iterrows():
                    self.log_message(f"    {row['Feature']}: kurtosis change = {row['Kurtosis_Change_Logged']:.2f}")
                    problem_features.append({'Feature':row['Feature'],'Problem':'Large kurtosis change after log transform','Original_Kurt':row['Original_Kurt'],'Logged_Kurt':row['Logged_Kurt'],'Change':row['Kurtosis_Change_Logged']})
            unchanged_features = stats_df[stats_df['Skewness_Change_Logged'] < 0.1]
            if len(unchanged_features) > 0:
                self.log_message(f"\n⚠️  WARNING: {len(unchanged_features)} features with little skewness change:")
                for _,row in unchanged_features.iterrows():
                    self.log_message(f"    {row['Feature']}: skewness change = {row['Skewness_Change_Logged']:.2f}")
                    problem_features.append({'Feature':row['Feature'],'Problem':'Little skewness change','Original_Skew':row['Original_Skew'],'Logged_Skew':row['Logged_Skew'],'Change':row['Skewness_Change_Logged']})
            if problem_features:
                pd.DataFrame(problem_features).to_csv(os.path.join(report_dir, "problem_features_report.csv"), index=False)
            self.log_message("\n" + "="*60)
            self.log_message("CORRELATION STRUCTURE ANALYSIS")
            self.log_message("="*60)
            corr_original = df_original[num_cols].corr()
            corr_imputed = df_imputed[num_cols].corr()
            corr_logged = df_logged[num_cols].corr()
            corr_scaled = df_scaled[num_cols].corr()
            corr_change_scaled = corr_scaled - corr_original
            corr_original.to_csv(os.path.join(report_dir, "correlation_original.csv"))
            corr_imputed.to_csv(os.path.join(report_dir, "correlation_imputed.csv"))
            corr_logged.to_csv(os.path.join(report_dir, "correlation_logged.csv"))
            corr_scaled.to_csv(os.path.join(report_dir, "correlation_scaled.csv"))
            corr_change_scaled.to_csv(os.path.join(report_dir, "correlation_change_scaled.csv"))
            corr_changes = corr_change_scaled.values[np.triu_indices_from(corr_change_scaled, k=1)]
            corr_stats = {
                'mean_change': float(np.mean(corr_changes)), 'std_change': float(np.std(corr_changes)),
                'max_increase': float(np.max(corr_changes)), 'max_decrease': float(np.min(corr_changes)),
                'abs_mean_change': float(np.mean(np.abs(corr_changes))),
                'pct_significant_changes': float(np.sum(np.abs(corr_changes) > 0.1) / len(corr_changes) * 100),
                'n_total_correlations': int(len(corr_changes))
            }
            self.log_message(f"  Correlation change statistics after standardization:")
            self.log_message(f"    Mean change: {corr_stats['mean_change']:.4f}")
            self.log_message(f"    Mean absolute change: {corr_stats['abs_mean_change']:.4f}")
            self.log_message(f"    Max increase: {corr_stats['max_increase']:.4f}")
            self.log_message(f"    Max decrease: {corr_stats['max_decrease']:.4f}")
            self.log_message(f"    Significant changes (>0.1): {corr_stats['pct_significant_changes']:.1f}%")
            with open(os.path.join(report_dir, "correlation_statistics.json"), 'w') as f:
                json.dump(corr_stats, f, indent=2)
            self.generate_preprocessing_visualizations(df_original, df_scaled, report_dir)
            self.create_comprehensive_preprocessing_report(report_dir, stats_df, corr_stats, problem_features)
            self.log_message("\n" + "="*60)
            self.log_message("✓ PREPROCESSING ANALYSIS COMPLETED")
            self.log_message("="*60)
        except Exception as e:
            self.log_message(f"✗ Error generating preprocessing report: {e}")
    def generate_preprocessing_visualizations(self, df_original, df_scaled, report_dir):
        try:
            vis_dir = os.path.join(report_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            num_cols = df_original.select_dtypes(include=np.number).columns
            top_features = num_cols[:min(9,len(num_cols))]
            fig, axes = plt.subplots(3,3,figsize=(18,15))
            axes = axes.flatten()
            for idx, col in enumerate(top_features):
                if idx < len(axes):
                    axes[idx].hist(df_original[col].dropna(), bins=30, alpha=0.5, color='blue', label='Original', density=True)
                    axes[idx].hist(df_scaled[col], bins=30, alpha=0.5, color='red', label='Processed', density=True)
                    #axes[idx].set_title(f"{col}\nDistribution Comparison")
                    axes[idx].set_xlabel('Value')
                    axes[idx].set_ylabel('Density')
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)
                    dist_compare_data = pd.DataFrame({'original': df_original[col].dropna().values, 'processed': df_scaled[col].values})
                    save_plot_data(dist_compare_data, os.path.join(vis_dir, f"distribution_comparison_{col}.csv"))
            plt.tight_layout()
            save_plot_english(fig, os.path.join(vis_dir, "distribution_comparison.png"), dpi=600)
            plt.close()
            top_features_qq = num_cols[:min(4,len(num_cols))]
            fig, axes = plt.subplots(2,2,figsize=(12,10))
            axes = axes.flatten()
            for idx, col in enumerate(top_features_qq):
                if idx < len(axes):
                    (osm, osr), (slope, intercept, r) = stats.probplot(df_original[col].dropna(), dist="norm")
                    axes[idx].plot(osm, osr, 'bo', markersize=3)
                    axes[idx].plot(osm, slope*osm+intercept, 'r-', linewidth=2)
                    #axes[idx].set_title(f"{col} - Original Q-Q Plot")
                    axes[idx].grid(True, alpha=0.3)
                    qq_data = pd.DataFrame({'theoretical_quantiles':osm, 'sample_quantiles':osr, 'fitted_line':slope*osm+intercept})
                    save_plot_data(qq_data, os.path.join(vis_dir, f"qq_plot_data_{col}.csv"))
            plt.tight_layout()
            save_plot_english(fig, os.path.join(vis_dir, "qq_plots_original.png"), dpi=600)
            plt.close()
            fig, axes = plt.subplots(2,2,figsize=(12,10))
            axes = axes.flatten()
            for idx, col in enumerate(top_features_qq):
                if idx < len(axes):
                    original_vals = df_original[col].dropna().values
                    processed_vals = df_scaled[col].values
                    data_to_plot = [original_vals, processed_vals]
                    bp = axes[idx].boxplot(data_to_plot, labels=['Original','Processed'], patch_artist=True)
                    colors = ['lightblue','lightgreen']
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    #axes[idx].set_title(f"{col}\nBoxplot Comparison")
                    axes[idx].set_ylabel('Value')
                    axes[idx].grid(True, alpha=0.3)
                    box_data = pd.DataFrame({'original':original_vals, 'processed':processed_vals})
                    save_plot_data(box_data, os.path.join(vis_dir, f"boxplot_data_{col}.csv"))
            plt.tight_layout()
            save_plot_english(fig, os.path.join(vis_dir, "boxplot_comparison.png"), dpi=600)
            plt.close()
            self.log_message(f"  Visualizations saved to: {vis_dir}")
        except Exception as e:
            self.log_message(f"  Error generating visualizations: {e}")
    def create_comprehensive_preprocessing_report(self, report_dir, stats_df, corr_stats, problem_features):
        try:
            report_path = os.path.join(report_dir, "comprehensive_preprocessing_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\nCOMPREHENSIVE DATA PREPROCESSING REPORT\n" + "="*80 + "\n\n")
                f.write("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write(f"Total features analyzed: {len(stats_df)}\n\n")
                f.write("PREPROCESSING STEPS SUMMARY\n" + "-"*40 + "\n")
                f.write("1. Missing Value Imputation: Median imputation for numerical columns\n")
                f.write("2. Skewness Treatment: Log transformation for features with |skew| > 0.5\n")
                f.write("3. Standardization: RobustScaler (IQR-based, outlier-resistant)\n")
                f.write("4. Physical Scale Preservation: All transformation parameters saved\n\n")
                f.write("STATISTICAL CHANGES SUMMARY\n" + "-"*40 + "\n")
                skew_improved = len(stats_df[stats_df['Skewness_Change_Logged'] > 0.1])
                f.write(f"Features with significant skewness improvement (>0.1): {skew_improved} / {len(stats_df)}\n")
                avg_skew_improvement = stats_df['Skewness_Change_Logged'].mean()
                f.write(f"Average skewness improvement: {avg_skew_improvement:.3f}\n")
                avg_kurt_change = stats_df['Kurtosis_Change_Logged'].mean()
                f.write(f"Average kurtosis change: {avg_kurt_change:.3f}\n")
                large_kurt_changes = len(stats_df[abs(stats_df['Kurtosis_Change_Logged']) > 3])
                f.write(f"Features with large kurtosis changes (>3): {large_kurt_changes} / {len(stats_df)}\n\n")
                f.write("CORRELATION STRUCTURE CHANGES\n" + "-"*40 + "\n")
                f.write("Impact on feature relationships:\n")
                f.write(f"  Mean correlation change: {corr_stats['mean_change']:.4f}\n")
                f.write(f"  Mean absolute change: {corr_stats['abs_mean_change']:.4f}\n")
                f.write(f"  Maximum increase: {corr_stats['max_increase']:.4f}\n")
                f.write(f"  Maximum decrease: {corr_stats['max_decrease']:.4f}\n")
                f.write(f"  Significant changes (>0.1): {corr_stats['pct_significant_changes']:.1f}%\n")
                f.write(f"  Total correlations analyzed: {corr_stats['n_total_correlations']}\n\n")
                if problem_features:
                    f.write("POTENTIAL ISSUES IDENTIFIED\n" + "-"*40 + "\n")
                    for problem in problem_features:
                        f.write(f"Feature: {problem['Feature']}\n")
                        f.write(f"  Issue: {problem['Problem']}\n")
                        if 'Change' in problem:
                            f.write(f"  Change magnitude: {problem['Change']:.3f}\n")
                        f.write("\n")
                f.write("RECOMMENDATIONS\n" + "-"*40 + "\n")
                if corr_stats['abs_mean_change'] > 0.1:
                    f.write("1. WARNING: Significant correlation changes detected\n")
                    f.write("   - For supervised learning: May affect feature importance interpretation\n")
                    f.write("   - For causal inference: Results may not be directly transferable\n")
                    f.write("   - Consider: Domain-specific feature engineering or alternative preprocessing\n\n")
                if large_kurt_changes > 0:
                    f.write("2. WARNING: Large kurtosis changes in some features\n")
                    f.write("   - Check features with kurtosis change > 3 for data quality issues\n")
                    f.write("   - Consider alternative transformations for these features\n\n")
                f.write("3. For features with little distribution change:\n")
                f.write("   - These features may already be normally distributed\n")
                f.write("   - Or may require different transformation strategies\n\n")
                f.write("4. Physical Scale Preservation:\n")
                f.write("   - All transformation parameters saved in scale_handler\n")
                f.write("   - Can recover original scale for interpretability\n")
                f.write("   - Essential for domain-specific applications\n")
                f.write("\n" + "="*80 + "\nEND OF REPORT\n" + "="*80 + "\n")
            self.log_message(f"  ✓ Comprehensive report saved: {report_path}")
        except Exception as e:
            self.log_message(f"  Error creating comprehensive report: {e}")
    def train_model_wrapper(self):
        if self.is_running:
            self.update_status("Another process is running. Please wait...", status="warning")
            return
        if self.df is None:
            self.update_status("Please load and preprocess data first", status="warning")
            return
        if self.df_processed is not None:
            data_columns = self.df_processed.columns.tolist()
        else:
            data_columns = self.df.columns.tolist()
        selected_target = self.select_target_variable(data_columns)
        if selected_target is None:
            self.update_status("No target variable selected. Using last column by default.", status="warning")
            selected_target = data_columns[-1]
        self.target_variable = selected_target
        self.log_message(f"Selected target variable: {self.target_variable}")
        thread = threading.Thread(target=self.train_model_thread)
        thread.daemon = True
        thread.start()
    def train_model_thread(self):
        self.is_running = True
        try:
            self.update_status("Starting model training...", step="4", status="working")
            self.update_progress(10)
            if self.timestamp is None:
                self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data2evaluation_dir = os.path.join(project_root, 'data2evaluation', self.timestamp)
            data3FIA_dir = os.path.join(project_root, 'data3FIA', self.timestamp)
            data4PDP_dir = os.path.join(project_root, 'data4PDP', self.timestamp)
            best_model_dir = os.path.join(project_root, 'bestmodel', self.timestamp)
            os.makedirs(data2evaluation_dir, exist_ok=True)
            os.makedirs(data3FIA_dir, exist_ok=True)
            os.makedirs(data4PDP_dir, exist_ok=True)
            os.makedirs(best_model_dir, exist_ok=True)
            if self.df is None:
                self.update_status("Please load data first", status="warning")
                self.is_running = False
                return
            df_raw = self.df.copy()
            if self.df_processed is None:
                self.update_status("Please preprocess data first", status="warning")
                self.is_running = False
                return
            df = self.df_processed
            if self.target_variable is None:
                target_var = df.columns[-1]
            else:
                target_var = self.target_variable
                if target_var not in df.columns:
                    target_var = df.columns[-1]
            X = df.drop(columns=[target_var])
            y = df[target_var]
            y_transformed = self.scale_handler.transform_target(y)
            self.scale_handler.save_target_stats(y, target_var)
            self.update_status("Calculating sample weights based on feature completeness...", status="working")
            if target_var in df_raw.columns:
                X_raw = df_raw.drop(columns=[target_var])
            else:
                X_raw = df_raw.copy()
            completeness_scores = 1 - X_raw.isnull().mean(axis=1)
            missing_count = X_raw.isnull().sum(axis=1)
            total_features = X_raw.shape[1]
            penalty_factor = 2.0
            missing_ratio = missing_count / total_features
            sample_weights = np.exp(-penalty_factor * missing_ratio)
            sample_weights = sample_weights / sample_weights.mean()
            weight_stats = {
                'min_weight': float(sample_weights.min()), 'max_weight': float(sample_weights.max()),
                'mean_weight': float(sample_weights.mean()), 'std_weight': float(sample_weights.std()),
                'total_samples': len(sample_weights), 'high_weight_samples': int((sample_weights > 1.5).sum()),
                'low_weight_samples': int((sample_weights < 0.5).sum()),
                'missing_stats': {'mean_missing_ratio': float(missing_ratio.mean()), 'std_missing_ratio': float(missing_ratio.std()),
                                  'complete_samples': int((missing_ratio == 0).sum()), 'incomplete_samples': int((missing_ratio > 0).sum())}
            }
            self.log_message(f"✓ Sample weights calculated based on feature completeness")
            self.log_message(f"  Weight range: {weight_stats['min_weight']:.3f} - {weight_stats['max_weight']:.3f}")
            weights_dir = os.path.join(data2evaluation_dir, "sample_weights")
            os.makedirs(weights_dir, exist_ok=True)
            weights_df = pd.DataFrame({'sample_index': X_raw.index, 'completeness_score': completeness_scores.values,
                                       'missing_count': missing_count.values, 'missing_ratio': missing_ratio.values,
                                       'sample_weight': sample_weights.values})
            weights_df.to_csv(os.path.join(weights_dir, "sample_weights_detail.csv"), index=False)
            with open(os.path.join(weights_dir, "weight_statistics.json"), 'w') as f:
                json.dump(weight_stats, f, indent=2)
            self.visualize_weight_distribution(sample_weights, missing_ratio, completeness_scores, weights_dir)
            X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
                X, y_transformed, sample_weights, test_size=0.2, random_state=42)
            self.log_message(f"  Target variable: {target_var}")
            self.log_message(f"  Training set: {X_train.shape}, Test set: {X_test.shape}")
            self.update_status("Training Random Forest model with sample weights...", status="working")
            self.update_progress(40)
            rf_model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2,
                                             min_samples_leaf=1, max_features='sqrt', random_state=42,
                                             n_jobs=-1, verbose=0, bootstrap=True)
            rf_model.fit(X_train, y_train, sample_weight=weights_train)
            self.log_message("Evaluating model performance...")
            self.update_progress(60)
            y_pred_transformed = rf_model.predict(X_test)
            if self.scale_handler.target_stats.get('log_transformed', False):
                y_pred = self.scale_handler.inverse_transform_target(y_pred_transformed)
                y_test_original = self.scale_handler.inverse_transform_target(y_test)
            else:
                y_pred = y_pred_transformed
                y_test_original = y_test
            r2 = r2_score(y_test_original, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
            mae = mean_absolute_error(y_test_original, y_pred)
            weighted_r2 = 1 - np.sum(weights_test * (y_test_original - y_pred)**2) / np.sum(weights_test * (y_test_original - np.average(y_test_original, weights=weights_test))**2)
            weighted_rmse = np.sqrt(np.average((y_test_original - y_pred)**2, weights=weights_test))
            weighted_mae = np.average(np.abs(y_test_original - y_pred), weights=weights_test)
            self.log_message(f"  Standard R² score: {r2:.4f}")
            self.log_message(f"  Weighted R² score: {weighted_r2:.4f}")
            self.log_message(f"  Standard RMSE: {rmse:.4f}")
            self.log_message(f"  Weighted RMSE: {weighted_rmse:.4f}")
            self.log_message(f"  Standard MAE: {mae:.4f}")
            self.log_message(f"  Weighted MAE: {weighted_mae:.4f}")
            best_params = {'n_estimators':200,'max_depth':None,'min_samples_split':2,'min_samples_leaf':1,'max_features':'sqrt','random_state':42}
            trained_model, results_df = self.fixed_evaluate_model(rf_model, X_train, X_test, y_train, y_test_original, y_pred, best_params, target_var, data2evaluation_dir, sample_weight_train=weights_train)
            weighted_metrics = {'weighted_r2':float(weighted_r2),'weighted_rmse':float(weighted_rmse),'weighted_mae':float(weighted_mae),
                                'standard_r2':float(r2),'standard_rmse':float(rmse),'standard_mae':float(mae),
                                'weight_stats':weight_stats,'target_transformed':self.scale_handler.target_stats.get('log_transformed',False)}
            with open(os.path.join(data2evaluation_dir, "weighted_evaluation_metrics.json"), 'w') as f:
                json.dump(weighted_metrics, f, indent=2)
            self.update_status("Performing feature importance analysis...", status="working")
            self.update_progress(80)
            self.fixed_feature_importance_analysis(rf_model, X_train, X_test, y_train, y_test_original, target_var, data3FIA_dir)
            self.update_status("Performing enhanced partial dependence analysis...", status="working")
            try:
                X_train_original_scale = self.scale_handler.inverse_transform_features(X_train)
                self.fixed_partial_dependence_analysis(rf_model, X_train, X_train_original_scale, target_var, data4PDP_dir)
                self.log_message("✓ Enhanced partial dependence analysis completed")
            except Exception as e:
                self.log_message(f"  Warning: Could not perform partial dependence analysis: {e}")
            self.save_model_and_parameters(rf_model, target_var, X_train, X_test, y_train, y_test_original, weighted_metrics, weight_stats, sample_weights, best_model_dir)
            self.prepare_transfer_learning_files(rf_model, X_train, y_train, target_var, weighted_metrics, sample_weights)
            self.update_status("Model training completed with sample weights!", step="4", status="success")
            self.update_progress(100)
            self.log_message(f"✓ Model training completed with feature-completeness based weighting")
        except Exception as e:
            self.update_status(f"Error in model training: {str(e)}", status="error")
            self.log_message(f"✗ Error: {str(e)}")
            import traceback
            self.log_message(f"✗ Traceback: {traceback.format_exc()}")
            self.update_progress(0)
        finally:
            self.is_running = False
    def visualize_weight_distribution(self, sample_weights, missing_ratio, completeness_scores, weights_dir):
        plt.figure(figsize=(15,10))
        plt.subplot(2,2,1)
        plt.hist(sample_weights, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Sample Weight'); plt.ylabel('Frequency'); plt.grid(True, alpha=0.3)
        plt.subplot(2,2,2)
        plt.scatter(missing_ratio, sample_weights, alpha=0.6, s=20)
        plt.xlabel('Missing Feature Ratio'); plt.ylabel('Sample Weight'); plt.grid(True, alpha=0.3)
        plt.subplot(2,2,3)
        plt.hist(missing_ratio, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Missing Feature Ratio'); plt.ylabel('Frequency'); plt.grid(True, alpha=0.3)
        plt.subplot(2,2,4)
        plt.hist(completeness_scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Completeness Score'); plt.ylabel('Frequency'); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_plot_english(plt.gcf(), os.path.join(weights_dir, "weight_distribution.png"), dpi=600)
        weight_data = pd.DataFrame({'sample_weight':sample_weights, 'missing_ratio':missing_ratio, 'completeness_score':completeness_scores})
        save_plot_data(weight_data, os.path.join(weights_dir, "weight_distribution_data.csv"))
        plt.close()
    def fixed_evaluate_model(self, model, X_train, X_test, y_train_transformed, y_test_original, y_pred_original, best_params, target_name, data2evaluation_dir, sample_weight_train=None):
        try:
            results = []
            if sample_weight_train is not None:
                cv_scores = cross_val_score(model, X_train, y_train_transformed, cv=5, scoring='r2', fit_params={'sample_weight': sample_weight_train})
            else:
                cv_scores = cross_val_score(model, X_train, y_train_transformed, cv=5, scoring='r2')
            for i in range(10):
                model_clone = RandomForestRegressor(**best_params)
                if sample_weight_train is not None:
                    model_clone.fit(X_train, y_train_transformed, sample_weight=sample_weight_train)
                else:
                    model_clone.fit(X_train, y_train_transformed)
                y_pred_iter = model_clone.predict(X_test)
                if self.scale_handler.target_stats.get('log_transformed', False):
                    y_pred_iter_original = self.scale_handler.inverse_transform_target(y_pred_iter)
                else:
                    y_pred_iter_original = y_pred_iter
                r2 = r2_score(y_test_original, y_pred_iter_original)
                mae = mean_absolute_error(y_test_original, y_pred_iter_original)
                mse = mean_squared_error(y_test_original, y_pred_iter_original)
                rmse = np.sqrt(mse)
                evs = explained_variance_score(y_test_original, y_pred_iter_original)
                y_test_nonzero = np.where(y_test_original == 0, 1e-10, y_test_original)
                mre = np.mean(np.abs((y_test_original - y_pred_iter_original) / y_test_nonzero))
                results.append({'Iteration':i+1,'R2_Score':r2,'MAE':mae,'MSE':mse,'RMSE':rmse,'MRE':mre,'Explained_Variance':evs})
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(data2evaluation_dir, "model_evaluation_results.csv"), index=False)
            pd.DataFrame({'CV_Score':cv_scores}).to_csv(os.path.join(data2evaluation_dir, "cross_validation_scores.csv"), index=False)
            avg_metrics = results_df.mean()
            plt.figure(figsize=(10,8))
            plt.scatter(y_test_original, y_pred_original, alpha=0.6, label='Predicted', s=50)
            plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'k--', lw=2, label='Ideal Fit')
            if len(y_test_original) > 1:
                coef = np.polyfit(y_test_original, y_pred_original, 1)
                poly1d_fn = np.poly1d(coef)
                plt.plot(y_test_original, poly1d_fn(y_test_original), 'r-', lw=2, label=f'Actual Fit (y={coef[0]:.3f}x+{coef[1]:.3f})')
            plt.xlabel('True Values (Original Scale)'); plt.ylabel('Predicted Values (Original Scale)')
            #plt.title(f'Random Forest: True vs Predicted\nAverage R² = {avg_metrics["R2_Score"]:.4f}')
            plt.legend(); plt.grid(True, linestyle='--', alpha=0.7)
            true_vs_pred_path = os.path.join(data2evaluation_dir, "true_vs_predicted_original_scale.png")
            save_plot_english(plt.gcf(), true_vs_pred_path, dpi=600)
            true_pred_data = pd.DataFrame({'true_values':y_test_original, 'predicted_values':y_pred_original})
            save_plot_data(true_pred_data, os.path.join(data2evaluation_dir, "true_vs_predicted_data.csv"))
            plt.close()
            residuals = y_test_original - y_pred_original
            plt.figure(figsize=(14,6))
            plt.subplot(1,2,1)
            sns.histplot(residuals, kde=True, color='skyblue')
            plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
            plt.xlabel('Residuals'); plt.ylabel('Frequency')
            plt.subplot(1,2,2)
            plt.scatter(y_pred_original, residuals, alpha=0.6, s=50)
            plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
            plt.xlabel('Predicted Values'); plt.ylabel('Residuals')
            plt.tight_layout()
            residual_path = os.path.join(data2evaluation_dir, "residual_analysis_original_scale.png")
            save_plot_english(plt.gcf(), residual_path, dpi=600)
            residual_data = pd.DataFrame({'predicted_values':y_pred_original, 'residuals':residuals})
            save_plot_data(residual_data, os.path.join(data2evaluation_dir, "residual_data.csv"))
            plt.close()
            weight_info = "with sample weights" if sample_weight_train is not None else "without sample weights"
            target_transform_info = "with log transform" if self.scale_handler.target_stats.get('log_transformed',False) else "without transform"
            summary = f"""Model Evaluation Summary\n========================\nTarget Variable: {target_name}\nTarget Transform: {target_transform_info}\nTraining Samples: {X_train.shape[0]}\nTest Samples: {X_test.shape[0]}\nFeatures: {X_train.shape[1]}\nTraining Method: {weight_info}\n\nCross-Validation Results (5-fold):\n  Mean R²: {cv_scores.mean():.4f}\n  Std R²: {cv_scores.std():.4f}\n\nAverage Evaluation Metrics ({len(results)} iterations):\n  R² Score: {avg_metrics['R2_Score']:.4f}\n  MAE: {avg_metrics['MAE']:.4f}\n  RMSE: {avg_metrics['RMSE']:.4f}\n  MRE: {avg_metrics['MRE']:.4f}\n  Explained Variance: {avg_metrics['Explained_Variance']:.4f}\n\nModel Parameters:\n  {best_params}\n"""
            with open(os.path.join(data2evaluation_dir, "evaluation_summary.txt"), 'w') as f:
                f.write(summary)
            self.log_message(f"  Evaluation results saved to: data2evaluation/{self.timestamp}/")
            return model, results_df
        except Exception as e:
            self.log_message(f"Error in fixed_evaluate_model: {e}")
            raise e
    def fixed_feature_importance_analysis(self, model, X_train, X_test, y_train, y_test, target_var, data3FIA_dir):
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                features_to_show = min(20, len(importances))
                plt.figure(figsize=(10.8, 6))
                plt.barh(range(features_to_show), importances[indices[:features_to_show]][::-1], align="center",
                         color='#35B779')
                plt.yticks(range(features_to_show), X_train.columns[indices[:features_to_show]][::-1])
                plt.tick_params(axis='both', labelsize=15)
                plt.xlabel("Feature Importance", fontsize=17);
                #plt.title(f"Top {features_to_show} Feature Importance - Random Forest")
                plt.tight_layout()
                plt.savefig(os.path.join(data3FIA_dir, "feature_importance.png"), dpi=600, facecolor='white',
                            edgecolor='none')
                plt.close()

                cumulative = np.cumsum(importances[indices])
                plt.figure(figsize=(10.8, 6))
                plt.plot(range(1, len(cumulative) + 1), cumulative, '-', color='#414487', linewidth=2, marker='o')
                plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% threshold')
                plt.tick_params(axis='both', labelsize=15)
                plt.xlabel("Number of Features", fontsize=17);
                plt.ylabel("Cumulative Importance", fontsize=17)
                #plt.title("Cumulative Feature Importance");
                plt.grid(True, alpha=0.3);
                plt.legend(fontsize=15)
                plt.tight_layout()
                plt.savefig(os.path.join(data3FIA_dir, "cumulative_feature_importance.png"), dpi=600, facecolor='white',
                            edgecolor='none')
                importance_data = pd.DataFrame({'Feature_Name':X_train.columns[indices], 'Importance_Score':importances[indices], 'Cumulative_Importance':cumulative, 'Rank':range(1,len(indices)+1)})
                save_plot_data(importance_data, os.path.join(data3FIA_dir, "feature_importance_plot_data.csv"))
                plt.close()
                importance_df = pd.DataFrame({'Feature_Name':X_train.columns[indices], 'Importance_Score':importances[indices],
                                              'Cumulative_Importance':cumulative, 'Importance_Percent':(importances[indices]/importances.sum())*100, 'Rank':range(1,len(indices)+1)})
                importance_df.to_csv(os.path.join(data3FIA_dir, "feature_importance_ranking.csv"), index=False)
                self.generate_feature_importance_report(importance_df, X_train, data3FIA_dir)
            self.perform_shap_analysis(model, X_train, X_train.columns.tolist(), target_var, data3FIA_dir)
            self.log_message(f"  Feature importance analysis saved to: data3FIA/{self.timestamp}/")
        except Exception as e:
            self.log_message(f"Error in fixed_feature_importance_analysis: {e}")
    def generate_feature_importance_report(self, importance_df, X_train, data3FIA_dir):
        try:
            report_dir = os.path.join(data3FIA_dir, "importance_analysis")
            os.makedirs(report_dir, exist_ok=True)
            top_features = importance_df.head(10)['Feature_Name'].tolist()
            report_content = f"""Feature Importance Analysis Report\n==================================\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Features Analyzed: {len(importance_df)}\nTarget Variable: {self.target_variable}\n\nTop 10 Most Important Features:\n{"-"*40}\n"""
            for i,(_,row) in enumerate(importance_df.head(10).iterrows(),1):
                report_content += f"{i:2d}. {row['Feature_Name']:20s} - Importance: {row['Importance_Score']:.4f} ({row['Importance_Percent']:.1f}%)\n"
            report_content += f"""\nCumulative Importance Analysis:\n{"-"*40}\nTop 5 features account for: {importance_df.head(5)['Importance_Percent'].sum():.1f}%\nTop 10 features account for: {importance_df.head(10)['Importance_Percent'].sum():.1f}%\nTop 20 features account for: {importance_df.head(20)['Importance_Percent'].sum():.1f}%\n\nInterpretation:\n1. Features with importance > 5% are considered highly influential\n2. Features with importance < 1% may be candidates for removal\n3. Cumulative importance shows how many features capture most variance\n\nRecommendations:\n1. Focus on top features for model interpretation\n2. Consider feature engineering for high-importance features\n3. Low-importance features may be removed to reduce complexity\n4. Validate importance with domain knowledge\n"""
            with open(os.path.join(report_dir, "feature_importance_report.txt"), 'w') as f:
                f.write(report_content)
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.hist(importance_df['Importance_Score'], bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Importance Score'); plt.ylabel('Frequency'); plt.grid(True, alpha=0.3)
            plt.subplot(1,2,2)
            plt.scatter(range(len(importance_df)), importance_df['Importance_Score'], alpha=0.6)
            plt.xlabel('Feature Rank'); plt.ylabel('Importance Score'); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            dist_plot_path = os.path.join(report_dir, "importance_distribution.png")
            save_plot_english(plt.gcf(), dist_plot_path, dpi=600)
            dist_data = pd.DataFrame({'importance_score':importance_df['Importance_Score'], 'rank':range(1,len(importance_df)+1)})
            save_plot_data(dist_data, os.path.join(report_dir, "importance_distribution_data.csv"))
            plt.close()
            self.log_message(f"  Feature importance report generated")
        except Exception as e:
            self.log_message(f"  Error generating feature importance report: {e}")
    def perform_shap_analysis(self, model, X_train, feature_names, target_var, data3FIA_dir):
        try:
            import shap
        except ImportError:
            self.log_message("  SHAP library not installed. Skipping SHAP analysis. Install with: pip install shap")
            return
        self.log_message("\n" + "="*60)
        self.log_message("SHAP FEATURE IMPORTANCE ANALYSIS")
        self.log_message("="*60)
        shap_dir = os.path.join(data3FIA_dir, "shap_analysis")
        os.makedirs(shap_dir, exist_ok=True)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv(os.path.join(shap_dir, "shap_values.csv"), index=False)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        imp_df = pd.DataFrame({'Feature':feature_names, 'Mean_Abs_SHAP':mean_abs_shap})
        imp_df = imp_df.sort_values('Mean_Abs_SHAP', ascending=False)
        imp_df.to_csv(os.path.join(shap_dir, "shap_feature_importance.csv"), index=False)
        plt.figure(figsize=(10,8))
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        for bar in plt.gca().patches: bar.set_facecolor('#2A788E')
        #plt.title(f"SHAP Feature Importance (Bar) - {target_var}")
        plt.tight_layout()
        save_plot_english(plt.gcf(), os.path.join(shap_dir, "shap_bar_plot.png"), dpi=600)
        plt.close()
        plt.figure(figsize=(12,10))
        shap.summary_plot(shap_values, X_train, show=False, cmap=plt.get_cmap('viridis'))
        #plt.title(f"SHAP Summary Plot - {target_var}")
        plt.tight_layout()
        save_plot_english(plt.gcf(), os.path.join(shap_dir, "shap_summary_plot.png"), dpi=600)
        plt.close()
        self.log_message(f"  SHAP analysis saved to {shap_dir}")
    def fixed_partial_dependence_analysis(self, model, X_train_scaled, X_train_original, target_var, data4PDP_dir):
        self.log_message("="*60)
        self.log_message("Starting Enhanced Partial Dependence Analysis (1D + 2D)")
        self.log_message("="*60)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = X_train_scaled.columns[indices[:min(12, len(X_train_scaled.columns))]]
        else:
            top_features = X_train_scaled.columns[:min(12, len(X_train_scaled.columns))]
        self.log_message(f"\n1. 1D PDP Ensemble Analysis (n_PDP = {self.n_pdp})")
        self.perform_pdp_1d_ensemble(model, X_train_scaled, "Scaled_Data", top_features, data4PDP_dir)
        self.perform_pdp_1d_ensemble(model, X_train_scaled, "Original_Scale", top_features, data4PDP_dir)
        self.log_message(f"\n2. 2D PDP Interaction Analysis (top {self.n_top_2d} features)")
        self.perform_pdp_2d_for_scale(model, X_train_scaled, "Scaled_Data", top_features, data4PDP_dir)
        self.perform_pdp_2d_for_scale(model, X_train_scaled, "Original_Scale", top_features, data4PDP_dir)
        self.create_comprehensive_pdp_report(data4PDP_dir, model, X_train_scaled, target_var)
        self.log_message("\n" + "="*60)
        self.log_message("✓ PDP Analysis Completed Successfully")
        self.log_message("="*60)
    def perform_pdp_1d_ensemble(self, model, X_data, scale_name, top_features, data4PDP_dir):
        try:
            pdp_subdir = os.path.join(data4PDP_dir, f"1D_{scale_name}")
            os.makedirs(pdp_subdir, exist_ok=True)
            n = self.n_pdp
            sample_frac = 0.8
            self.log_message(f"  Analyzing {len(top_features)} features with {n} bootstrap samples")
            for feature in top_features:
                if feature not in X_data.columns:
                    continue
                try:
                    feature_min = X_data[feature].min()
                    feature_max = X_data[feature].max()
                    grid_points = np.linspace(feature_min, feature_max, 50)
                    # 模型始终使用标准化尺度；仅将绘图横坐标转换回原始物理尺度
                    if scale_name == "Original_Scale":
                        grid_points_plot = self.scale_handler.inverse_transform_features(
                            pd.DataFrame({feature: grid_points}),
                            features=[feature]
                        )[feature].to_numpy()
                    else:
                        grid_points_plot = grid_points.copy()
                    all_curves = []
                    for _ in range(n):
                        sample_idx = np.random.choice(len(X_data), size=int(sample_frac*len(X_data)), replace=True)
                        X_sample = X_data.iloc[sample_idx]
                        pdp_vals = []
                        for val in grid_points:
                            X_temp = X_sample.copy()
                            X_temp[feature] = val
                            pdp_vals.append(np.mean(model.predict(X_temp)))
                        all_curves.append(pdp_vals)
                    all_curves = np.array(all_curves)
                    median_curve = np.median(all_curves, axis=0)
                    smoothed = pd.Series(median_curve).rolling(
                        window=3, center=True, min_periods=1
                    ).mean().to_numpy()
                    plt.figure(figsize=(10,6))
                    for i in range(min(n, len(all_curves))):
                        plt.plot(grid_points_plot, all_curves[i], color='gray', alpha=0.2, linewidth=0.8)
                    plt.plot(grid_points_plot, median_curve, color='orange', linewidth=2.5, label='Median PDP')
                    plt.plot(grid_points_plot, smoothed, color='red', linestyle='--', linewidth=2,
                             label='Smoothed Median')
                    #plt.title(f'Partial Dependence for {feature} ({scale_name})\n{n} individual curves + median')
                    plt.xlabel(feature, fontsize=21, labelpad=6);
                    plt.ylabel('Partial Dependence', fontsize=21, labelpad=6);
                    plt.tick_params(axis='both', labelsize=17)
                    plt.legend(); plt.grid(True, alpha=0.3)
                    png_path = os.path.join(pdp_subdir, f"PDP_ensemble_{feature}.png")
                    save_plot_english(plt.gcf(), png_path, dpi=600)
                    data_dict = {
                        'grid_point': grid_points_plot,
                        'grid_point_scaled': grid_points,
                        'median_pdp': median_curve,
                        'smoothed_pdp': smoothed
                    }
                    for i in range(n):
                        data_dict[f'pdp_curve_{i+1}'] = all_curves[i]
                    pd.DataFrame(data_dict).to_csv(os.path.join(pdp_subdir, f"PDP_ensemble_data_{feature}.csv"), index=False)
                    plt.close()
                except Exception as e:
                    self.log_message(f"    Failed for {feature}: {e}")
            self.log_message(f"  ✓ 1D PDP ensemble done for {scale_name}")
        except Exception as e:
            self.log_message(f"  ✗ 1D PDP error: {e}")
    def perform_pdp_2d_for_scale(self, model, X_data, scale_name, top_features, data4PDP_dir):
        try:
            pdp_2d_dir = os.path.join(data4PDP_dir, f"2D_{scale_name}")
            os.makedirs(pdp_2d_dir, exist_ok=True)
            n_top = min(self.n_top_2d, len(top_features))
            selected_features = list(top_features[:n_top])
            if len(selected_features) < 2:
                self.log_message("  Not enough features for 2D PDP (need at least 2).")
                return
            feature_pairs = list(combinations(selected_features, 2))
            self.log_message(f"  Generating 2D PDP for {len(feature_pairs)} feature pairs")
            sample_frac_2d = 0.3
            n_samples_2d = max(100, int(sample_frac_2d * len(X_data)))
            sample_idx = np.random.choice(len(X_data), size=n_samples_2d, replace=False)
            X_sample = X_data.iloc[sample_idx]
            for (f1, f2) in feature_pairs:
                try:
                    n_grid = 30
                    f1_min, f1_max = X_data[f1].min(), X_data[f1].max()
                    f2_min, f2_max = X_data[f2].min(), X_data[f2].max()
                    f1_grid = np.linspace(f1_min, f1_max, n_grid)
                    f2_grid = np.linspace(f2_min, f2_max, n_grid)
                    f1_mesh, f2_mesh = np.meshgrid(f1_grid, f2_grid)
                    # 模型始终使用标准化尺度；Original_Scale 仅把二维/三维图坐标恢复为原始物理值
                    if scale_name == "Original_Scale":
                        f1_grid_plot = self.scale_handler.inverse_transform_features(
                            pd.DataFrame({f1: f1_grid}),
                            features=[f1]
                        )[f1].to_numpy()
                        f2_grid_plot = self.scale_handler.inverse_transform_features(
                            pd.DataFrame({f2: f2_grid}),
                            features=[f2]
                        )[f2].to_numpy()
                    else:
                        f1_grid_plot = f1_grid.copy()
                        f2_grid_plot = f2_grid.copy()
                    f1_mesh_plot, f2_mesh_plot = np.meshgrid(f1_grid_plot, f2_grid_plot)
                    Z = np.zeros_like(f1_mesh)
                    for i in range(n_grid):
                        for j in range(n_grid):
                            X_temp = X_sample.copy()
                            X_temp[f1] = f1_mesh[i,j]
                            X_temp[f2] = f2_mesh[i,j]
                            Z[i,j] = np.mean(model.predict(X_temp))
                    rows = []
                    if scale_name == "Original_Scale":
                        for i in range(n_grid):
                            for j in range(n_grid):
                                rows.append({f1:f1_mesh_plot[i,j], f2:f2_mesh_plot[i,j], 'prediction':Z[i,j]})
                    else:
                        for i, val1 in enumerate(f1_grid):
                            for j, val2 in enumerate(f2_grid):
                                rows.append({f1:val1, f2:val2, 'prediction':Z[i,j]})
                    pd.DataFrame(rows).to_csv(os.path.join(pdp_2d_dir, f"2D_PDP_data_{f1}_{f2}.csv"), index=False)
                    import json
                    data_dict = {'f1_grid':f1_grid_plot.tolist(), 'f2_grid':f2_grid_plot.tolist(), 'Z_values':Z.tolist()}
                    with open(os.path.join(pdp_2d_dir, f"2D_PDP_data_{f1}_{f2}.json"), 'w') as f:
                        json.dump(data_dict, f)
                    plt.figure(figsize=(10,8))
                    contour = plt.contourf(f1_mesh_plot, f2_mesh_plot, Z, levels=20, cmap='viridis')
                    cbar = plt.colorbar(contour, label='Partial Dependence'); cbar.ax.tick_params(labelsize=16); cbar.ax.yaxis.label.set_fontsize(20); cbar.ax.yaxis.labelpad = 7
                    plt.xlabel(f1, fontsize=21); plt.ylabel(f2, fontsize=21); plt.tick_params(axis='both', labelsize=17)
                    #plt.title(f'2D Partial Dependence: {f1} vs {f2} ({scale_name})')
                    plt.tight_layout()
                    save_plot_english(plt.gcf(), os.path.join(pdp_2d_dir, f"2D_PDP_contour_{f1}_{f2}.png"), dpi=600)
                    plt.close()
                    fig = plt.figure(figsize=(12,8))
                    ax = fig.add_subplot(111, projection='3d')
                    surf = ax.plot_surface(f1_mesh_plot, f2_mesh_plot, Z, cmap='coolwarm', alpha=0.8)
                    cbar3d = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5); cbar3d.ax.tick_params(length=plt.rcParams['ytick.major.size'] * 0.8)
                    ax.set_xlabel(f1, labelpad=6);
                    ax.set_ylabel(f2, labelpad=6);
                    ax.set_zlabel('Partial Dependence', labelpad=6)
                    for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis._axinfo['tick']['outward_factor'] *= 0.8
                    #ax.set_title(f'3D PDP: {f1} vs {f2}')
                    save_plot_english(fig, os.path.join(pdp_2d_dir, f"2D_PDP_3D_{f1}_{f2}.png"), dpi=600)
                    plt.close()
                    self.log_message(f"    2D PDP saved for ({f1}, {f2})")
                except Exception as e:
                    self.log_message(f"    Error for pair ({f1}, {f2}): {e}")
            self.log_message(f"  ✓ 2D PDP analysis completed for {scale_name}")
        except Exception as e:
            self.log_message(f"  ✗ 2D PDP error: {e}")
    def create_comprehensive_pdp_report(self, data4PDP_dir, model, X_data, target_var):
        try:
            report_dir = os.path.join(data4PDP_dir, "Comprehensive_Report")
            os.makedirs(report_dir, exist_ok=True)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                top_features = X_data.columns[indices[:min(10, len(X_data.columns))]]
                importance_scores = importances[indices][:len(top_features)]
            else:
                top_features = X_data.columns[:min(10, len(X_data.columns))]
                importance_scores = np.ones(len(top_features))
            report_content = f"""Partial Dependence Analysis Report\n==================================\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\nTarget Variable: {target_var}\nTotal Features Analyzed: {len(X_data.columns)}\nTop Features Analyzed for PDP: {len(top_features)}\n\nAnalysis Scales:\n1. Scaled Data: Preprocessed data used for model training\n2. Original Scale: Data transformed back to original physical scale\n\nTop Features by Importance:\n{"-"*40}\n"""
            for i,(feature,score) in enumerate(zip(top_features, importance_scores),1):
                report_content += f"{i:2d}. {feature:20s} - Importance: {score:.4f}\n"
            report_content += f"""\nInterpretation Guidelines:\n{"-"*40}\n1. 1D PDP Plots:\n   - Show marginal effect of a single feature on predictions\n   - Linear patterns indicate linear relationships\n   - U-shaped patterns suggest optimal values\n   - Steep slopes indicate high sensitivity\n\n2. Scale Comparison:\n   - Scaled PDP: Shows effects in preprocessing space\n   - Original Scale PDP: Shows effects in physical units\n   - Compare both for domain interpretation\n\n3. Key Insights:\n   - Identify non-linear relationships\n   - Detect interaction potential\n   - Understand feature impact magnitude\n   - Validate with domain knowledge\n\nFiles Generated:\n{"-"*40}\n- 1D PDP plots for top features (both scales)\n- 2D contour and 3D surface plots for top feature pairs\n- Feature importance rankings\n- Comprehensive analysis report\n- CSV data files for each PDP plot (traceable)\n\nNext Steps:\n{"-"*40}\n1. Review PDP plots for feature engineering opportunities\n2. Validate findings with domain experts\n3. Use insights for model interpretation\n4. Apply to XGBoost transfer learning feature weighting\n"""
            with open(os.path.join(report_dir, "pdp_analysis_report.txt"), 'w') as f:
                f.write(report_content)
            self.log_message(f"  ✓ Comprehensive PDP report generated")
        except Exception as e:
            self.log_message(f"  ✗ Error creating comprehensive PDP report: {e}")
    def save_model_and_parameters(self, model, target_var, X_train, X_test, y_train, y_test, weighted_metrics, weight_stats, sample_weights, best_model_dir):
        try:
            model_path = os.path.join(best_model_dir, f"RandomForest_{self.timestamp}.pkl")
            joblib.dump(model, model_path)
            weight_info_path = os.path.join(best_model_dir, f"sample_weights_{self.timestamp}.pkl")
            weight_info = {'sample_weights':sample_weights, 'weight_stats':weight_stats, 'training_indices':list(X_train.index) if hasattr(X_train,'index') else list(range(len(X_train)))}
            joblib.dump(weight_info, weight_info_path)
            model_info = {
                'timestamp':self.timestamp, 'model_type':'RandomForestRegressor', 'target_variable':target_var,
                'n_features':X_train.shape[1], 'n_samples':X_train.shape[0], 'training_samples':len(X_train), 'test_samples':len(X_test),
                'standard_r2_score':weighted_metrics['standard_r2'], 'weighted_r2_score':weighted_metrics['weighted_r2'],
                'standard_rmse':weighted_metrics['standard_rmse'], 'weighted_rmse':weighted_metrics['weighted_rmse'],
                'standard_mae':weighted_metrics['standard_mae'], 'weighted_mae':weighted_metrics['weighted_mae'],
                'weight_based_training':True, 'target_transformed':weighted_metrics.get('target_transformed',False),
                'feature_names':list(X_train.columns), 'preprocessing_summary':{'log_transform_applied':self.scale_handler.target_stats.get('log_transformed',False), 'robust_scaling':True, 'physical_scale_preserved':True}
            }
            with open(os.path.join(best_model_dir, f"model_info_{self.timestamp}.json"), 'w') as f:
                json.dump(model_info, f, indent=2)
            transform_params_path = os.path.join(best_model_dir, f"scale_params_{self.timestamp}.json")
            self.scale_handler.save_transform_parameters(transform_params_path)
            self.log_message(f"  Model saved: {model_path}")
            self.log_message(f"  Model info saved: {os.path.join(best_model_dir, f'model_info_{self.timestamp}.json')}")
            self.log_message(f"  Physical scale parameters saved: {transform_params_path}")
        except Exception as e:
            self.log_message(f"  Error saving model and parameters: {e}")
    def prepare_transfer_learning_files(self, model, X_train, y_train, target_var, model_info, sample_weights=None):
        try:
            transfer_dir = os.path.join(project_root, 'transfer_learning', f"RF_for_XGBoost_{self.timestamp}")
            os.makedirs(transfer_dir, exist_ok=True)
            joblib.dump(model, os.path.join(transfer_dir, "RandomForest_transfer_ready.pkl"))
            params_data = {
                "model_type":"RandomForestRegressor","target_variable":target_var,
                "cv_score":float(model_info.get('weighted_r2_score', model_info.get('standard_r2_score',0))),
                "timestamp":self.timestamp,"feature_count":X_train.shape[1],"training_samples":X_train.shape[0],
                "training_date":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"weight_based_training":True,
                "weight_stats":model_info.get('weight_stats',{}),"target_transformed":model_info.get('target_transformed',False),
                "preprocessing_applied":True,"physical_scale_preserved":True
            }
            with open(os.path.join(transfer_dir, "model_parameters.json"), "w") as f:
                json.dump(params_data, f, indent=2)
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({'feature_name':X_train.columns, 'importance_score':feature_importances,
                                              'importance_rank':np.argsort(np.argsort(-feature_importances))+1,
                                              'importance_percent':feature_importances/feature_importances.sum()*100}).sort_values('importance_score', ascending=False)
                importance_df.to_csv(os.path.join(transfer_dir, "feature_importance_detailed.csv"), index=False)
                importance_data = {
                    "feature_names":X_train.columns.tolist(), "importance_scores":feature_importances.tolist(),
                    "importance_ranking":importance_df['feature_name'].tolist(), "top_10_features":importance_df.head(10)['feature_name'].tolist(),
                    "top_10_scores":importance_df.head(10)['importance_score'].tolist(), "feature_count":len(X_train.columns)
                }
                with open(os.path.join(transfer_dir, "feature_importance.json"), "w") as f:
                    json.dump(importance_data, f, indent=2)
            self.scale_handler.save_transform_parameters(os.path.join(transfer_dir, "physical_scale_params.json"))
            preprocessing_report = {
                "preprocessing_steps":["missing_value_imputation","skewness_treatment","robust_scaling"],
                "physical_scale_preservation":True,"correlation_structure_changes":"analyzed_and_reported",
                "distribution_changes":"monitored_and_documented",
                "recommendations_for_transfer":["Use feature importance for XGBoost feature weighting","Consider physical scale parameters for interpretability","Validate predictions in original domain units","Check preprocessing report for distribution changes"]
            }
            with open(os.path.join(transfer_dir, "preprocessing_report.json"), "w") as f:
                json.dump(preprocessing_report, f, indent=2)
            report_content = f"""RANDOM FOREST TRANSFER LEARNING READINESS REPORT\n=================================================\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nModel: RandomForestRegressor\nTimestamp: {self.timestamp}\n\nMODEL INFORMATION\n-----------------\nTarget Variable: {target_var}\nTraining samples: {X_train.shape[0]}\nFeatures: {X_train.shape[1]}\nWeighted R² score: {model_info.get('weighted_r2_score', 'N/A'):.4f}\nWeighted RMSE: {model_info.get('weighted_rmse', 'N/A'):.4f}\n\nPREPROCESSING SUMMARY\n---------------------\n1. Missing values imputed with median\n2. Log transformation applied to skewed features\n3. Robust scaling applied (IQR-based)\n4. Physical scale parameters preserved\n5. Sample weights calculated based on feature completeness\n\nKEY FEATURES FOR XGBoost TRANSFER\n---------------------------------\n1. Feature importance rankings available\n2. Physical scale parameters for interpretability\n3. Sample weights for weighted training\n4. Preprocessing analysis for feature selection\n5. Partial dependence plots for feature understanding\n\nHOW TO USE\n----------\n1. Load feature_importance.json for XGBoost feature weighting\n2. Use physical_scale_params.json to convert predictions to original scale\n3. Review preprocessing_report.json for data transformation insights\n4. Consider sample weights for imbalanced or incomplete data\n\nDIRECTORY STRUCTURE\n-------------------\n{transfer_dir}\n├── RandomForest_transfer_ready.pkl    # Trained model\n├── model_parameters.json              # Model parameters\n├── feature_importance.json            # Feature importance\n├── feature_importance_detailed.csv    # Detailed importance\n├── physical_scale_params.json         # Scale preservation\n└── preprocessing_report.json          # Preprocessing summary\n"""
            with open(os.path.join(transfer_dir, "transfer_learning_readiness_report.txt"), "w", encoding='utf-8') as f:
                f.write(report_content)
            self.log_message(f"  Transfer learning files saved to: {transfer_dir}")
        except Exception as e:
            self.log_message(f"  Warning: Failed to prepare transfer learning files: {e}")
    def run_all_wrapper(self):
        if self.is_running:
            self.update_status("Another process is running. Please wait...", status="warning")
            return
        if self.df is None:
            messagebox.showinfo("Info", "Please load data first using the 'Load Data' button.")
            return
        data_columns = self.df.columns.tolist()
        selected_target = self.select_target_variable(data_columns)
        if selected_target is None:
            self.update_status("No target variable selected. Using last column by default.", status="warning")
            selected_target = data_columns[-1]
        self.target_variable = selected_target
        self.log_message(f"Selected target variable: {self.target_variable}")
        thread = threading.Thread(target=self.run_all_thread)
        thread.daemon = True
        thread.start()
    def run_all_thread(self):
        self.is_running = True
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_message(f"Generated timestamp: {self.timestamp}")
            if self.df is None:
                messagebox.showinfo("Info", "Please load data first using the 'Load Data' button.")
                self.is_running = False
                return
            self.update_status("Step 2/5: Analyzing data...", step="2", status="working")
            self.analyze_data_thread()
            while self.is_running:
                time.sleep(0.1)
            self.is_running = True
            self.update_status("Step 3/5: Preprocessing data...", step="3", status="working")
            self.preprocess_data_thread()
            while self.is_running:
                time.sleep(0.1)
            self.is_running = True
            self.update_status("Step 4/5: Training model...", step="4", status="working")
            self.train_model_thread()
            while self.is_running:
                time.sleep(0.1)
            self.is_running = True
            self.update_status("Step 5/5: Preparing transfer learning...", step="5", status="working")
            summary = self.create_transfer_learning_summary()
            self.log_message("\n" + "="*60)
            self.log_message("TRANSFER LEARNING SUMMARY")
            self.log_message("="*60)
            self.log_message(summary)
            self.update_status("All steps completed successfully!", step="5", status="success")
            self.log_message("\n" + "="*60)
            self.log_message("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
            self.log_message("="*60)
            self.log_message(f"\nAll results saved with timestamp: {self.timestamp}")
            self.log_message(f"Target variable used: {self.target_variable}")
            self.log_message(f"Physical scale preservation: ENABLED")
            self.log_message(f"Sample weight training: ENABLED")
            self.log_message(f"Enhanced preprocessing analysis: ENABLED")
            self.log_message(f"Distribution changes monitored: YES")
            self.log_message(f"Correlation structure analyzed: YES")
            messagebox.showinfo("Complete", f"Random Forest pre-training completed successfully!\n\nTimestamp: {self.timestamp}\nTarget Variable: {self.target_variable}\nPhysical Scale Preservation: ENABLED\nSample Weight Training: ENABLED\nPreprocessing Analysis: COMPREHENSIVE\n\nAll results saved in time-stamped directories.\n\nTransfer learning resources are ready for XGBoost.")
        except Exception as e:
            self.update_status(f"Error in run all: {str(e)}", status="error")
            self.log_message(f"✗ Error: {str(e)}")
        finally:
            self.is_running = False
    def view_transfer_learning(self):
        try:
            transfer_dir = os.path.join(project_root, 'transfer_learning')
            if not os.path.exists(transfer_dir):
                messagebox.showinfo("Info", "No transfer learning models available yet.\n\nPlease run the training first.")
                return
            view_dialog = tk.Toplevel(self.root)
            view_dialog.title("Transfer Learning Resources")
            view_dialog.geometry("800x600")
            view_dialog.configure(bg=COLORS['background'])
            view_dialog.transient(self.root)
            view_dialog.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - view_dialog.winfo_width())//2
            y = self.root.winfo_y() + (self.root.winfo_height() - view_dialog.winfo_height())//2
            view_dialog.geometry(f"+{x}+{y}")
            title_frame = ttk.Frame(view_dialog)
            title_frame.pack(pady=10, fill=tk.X, padx=20)
            ttk.Label(title_frame, text="Transfer Learning Resources", font=('Segoe UI',16,'bold')).pack()
            ttk.Label(title_frame, text="Available Random Forest models for XGBoost transfer", font=('Segoe UI',11)).pack(pady=5)
            ttk.Separator(view_dialog, orient='horizontal').pack(fill=tk.X, padx=20, pady=10)
            content_frame = ttk.Frame(view_dialog)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            text_container = ttk.Frame(content_frame)
            text_container.pack(fill=tk.BOTH, expand=True)
            text_scrollbar = ttk.Scrollbar(text_container)
            text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_area = tk.Text(text_container, font=('Consolas',10), bg='white', fg=COLORS['text'], yscrollcommand=text_scrollbar.set, wrap=tk.WORD)
            text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            text_scrollbar.config(command=text_area.yview)
            models = self.list_available_transfer_learning_models()
            if models:
                text_area.insert(tk.END, f"Found {len(models)} transfer learning model(s):\n\n")
                for i,model in enumerate(models,1):
                    text_area.insert(tk.END, f"Model {i}:\n")
                    text_area.insert(tk.END, f"  Timestamp: {model['timestamp']}\n")
                    text_area.insert(tk.END, f"  Model Type: {model['model_type']}\n")
                    text_area.insert(tk.END, f"  Target Variable: {model.get('target_variable','N/A')}\n")
                    text_area.insert(tk.END, f"  CV Score (R²): {model['cv_score']:.4f}\n")
                    text_area.insert(tk.END, f"  Training Samples: {model['training_samples']}\n")
                    text_area.insert(tk.END, f"  Features: {model['features']}\n")
                    text_area.insert(tk.END, f"  Physical Scale Preservation: {model.get('physical_scale_preserved','N/A')}\n")
                    text_area.insert(tk.END, f"  Weight-Based Training: {model.get('weight_based_training','N/A')}\n")
                    text_area.insert(tk.END, f"  Enhanced Preprocessing: {model.get('enhanced_preprocessing','N/A')}\n")
                    text_area.insert(tk.END, f"  Folder: {model['folder']}\n\n")
            else:
                text_area.insert(tk.END, "No transfer learning models found.\n")
            button_frame = ttk.Frame(view_dialog)
            button_frame.pack(pady=20)
            ttk.Button(button_frame, text="Open Folder", command=lambda: os.startfile(transfer_dir)).grid(row=0, column=0, padx=5)
            ttk.Button(button_frame, text="Close", command=view_dialog.destroy).grid(row=0, column=1, padx=5)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view transfer learning: {str(e)}")
    def list_available_transfer_learning_models(self):
        models = []
        transfer_dir = os.path.join(project_root, 'transfer_learning')
        if os.path.exists(transfer_dir):
            for folder in os.listdir(transfer_dir):
                folder_path = os.path.join(transfer_dir, folder)
                if os.path.isdir(folder_path) and folder.startswith('RF_for_XGBoost_'):
                    params_file = os.path.join(folder_path, 'model_parameters.json')
                    if os.path.exists(params_file):
                        try:
                            with open(params_file,'r') as f:
                                params = json.load(f)
                            has_physical = os.path.exists(os.path.join(folder_path, 'physical_scale_params.json'))
                            has_preproc = os.path.exists(os.path.join(folder_path, 'preprocessing_report.json'))
                            models.append({
                                'folder':folder, 'model_type':params.get('model_type','RandomForestRegressor'),
                                'timestamp':params.get('timestamp', folder.replace('RF_for_XGBoost_','')),
                                'target_variable':params.get('target_variable','N/A'), 'cv_score':params.get('cv_score',0),
                                'training_samples':params.get('training_samples',0), 'features':params.get('feature_count',0),
                                'physical_scale_preserved':has_physical, 'weight_based_training':params.get('weight_based_training',False),
                                'enhanced_preprocessing':has_preproc
                            })
                        except:
                            continue
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models
    def create_transfer_learning_summary(self):
        summary = "Transfer Learning Resources Summary\n" + "="*50 + "\n"
        models = self.list_available_transfer_learning_models()
        if models:
            summary += f"Available models: {len(models)}\n"
            summary += f"Latest model timestamp: {models[0]['timestamp'] if models else 'N/A'}\n"
            summary += f"Latest model target variable: {models[0].get('target_variable','N/A') if models else 'N/A'}\n"
            summary += f"Physical scale preservation: {models[0].get('physical_scale_preserved','N/A')}\n"
            summary += f"Weight-based training: {models[0].get('weight_based_training','N/A')}\n"
            summary += f"Enhanced preprocessing: {models[0].get('enhanced_preprocessing','N/A')}\n"
        else:
            summary += "No transfer learning models available yet.\n"
        summary += "\nHow to use for XGBoost transfer learning:\n"
        summary += "1. Load feature importance from the model\n"
        summary += "2. Use physical_scale_params.json for physical scale predictions\n"
        summary += "3. Apply feature weighting based on importance scores\n"
        summary += "4. Use sample weights for weighted training in XGBoost\n"
        summary += "5. Review preprocessing report for distribution changes\n"
        return summary
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = RandomForestPreTrainer()
    app.run()
