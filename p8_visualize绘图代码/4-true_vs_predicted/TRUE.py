import tkinter as tk
from tkinter import ttk, colorchooser, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import os
import warnings

# ====================== 中文字体自动适配 ======================
def set_chinese_font():
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'PingFang SC', 'STHeiti'
    ]
    for font in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, '测试', fontsize=12)
            fig.canvas.draw()
            plt.close(fig)
            return True
        except:
            continue
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return False

set_chinese_font()

# ====================== 全局数据 ======================
true_vals = np.array([])
pred_vals = np.array([])
slope = None
intercept = None
r2 = None
current_data_file = None

# ====================== 语言资源 ======================
LANG = {
    'zh': {
        'title': '数据对比图',
        'xlabel': 'X 轴',
        'ylabel': 'Y 轴',
        'ideal_label': '理想拟合线 (y = x)',
        'actual_label': '实际拟合线: y = {slope:.3f}x + {intercept:.3f}',
        'r2_text': '$R^2 = {r2:.4f}$',
        'data_label': '数据点',
        'error_load': '无法加载数据文件：{e}',
        'no_data': '无数据，请检查 CSV 文件',
        'choose_csv': '请选择 CSV 数据文件',
        'save_img': '保存图片',
        'load_data': '加载数据',
        'swap_axes': '交换坐标轴',
        'save_success': '图片已保存至：{path}',
        'save_failed': '保存失败：{e}',
        'palette_label': '预设配色方案:',
        'apply_palette': '应用散点颜色',
        'scatter_color': '散点颜色:',
        'bg_color': '绘图背景色:',
        'fit_color': '实际拟合线颜色:',
        'ideal_color': '理想拟合线颜色:',
        'point_size': '点大小:',
        'alpha': '透明度:',
        'line_width': '线宽:',
        'show_grid': '显示网格:',
        'chart_title': '图表标题:',
        'x_label': 'X轴标签:',
        'y_label': 'Y轴标签:',
        'refresh': '刷新绘图',
        'reg_info': '当前回归信息',
        'slope': '斜率',
        'intercept': '截距'
    },
    'en': {
        'title': 'Data Comparison',
        'xlabel': 'X Axis',
        'ylabel': 'Y Axis',
        'ideal_label': 'Ideal fit (y = x)',
        'actual_label': 'Actual fit: y = {slope:.3f}x + {intercept:.3f}',
        'r2_text': '$R^2 = {r2:.4f}$',
        'data_label': 'Data points',
        'error_load': 'Failed to load data file: {e}',
        'no_data': 'No data. Please check CSV file.',
        'choose_csv': 'Select CSV data file',
        'save_img': 'Save Image',
        'load_data': 'Load Data',
        'swap_axes': 'Swap Axes',
        'save_success': 'Image saved to: {path}',
        'save_failed': 'Save failed: {e}',
        'palette_label': 'Preset Palette:',
        'apply_palette': 'Apply to points',
        'scatter_color': 'Scatter color:',
        'bg_color': 'Background color:',
        'fit_color': 'Fit line color:',
        'ideal_color': 'Ideal line color:',
        'point_size': 'Point size:',
        'alpha': 'Alpha:',
        'line_width': 'Line width:',
        'show_grid': 'Show grid:',
        'chart_title': 'Title:',
        'x_label': 'X label:',
        'y_label': 'Y label:',
        'refresh': 'Refresh Plot',
        'reg_info': 'Regression Info',
        'slope': 'Slope',
        'intercept': 'Intercept'
    }
}
current_lang = 'zh'

def get_text(key, **kwargs):
    text = LANG[current_lang].get(key, key)
    return text.format(**kwargs) if kwargs else text

# ====================== 配色方案 ======================
DISCRETE_PALETTES = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind', 'Set1', 'Set2', 'tab10']
SEQUENTIAL_CMAPS = ['viridis', 'plasma', 'coolwarm', 'RdBu_r', 'RdYlBu', 'seismic']
ALL_PALETTES = DISCRETE_PALETTES + SEQUENTIAL_CMAPS

def get_palette_color(palette_name):
    try:
        if palette_name in DISCRETE_PALETTES:
            colors = sns.color_palette(palette_name, n_colors=1)
            return matplotlib.colors.rgb2hex(colors[0])
        else:
            cmap = cm.get_cmap(palette_name)
            rgba = cmap(0.5)
            return matplotlib.colors.rgb2hex(rgba[:3])
    except:
        return '#1f77b4'

# ====================== 数据加载（泛化：第一列为 X，第二列为 Y，列名自动作为轴标签） ======================
def load_data(filename=None):
    global true_vals, pred_vals, slope, intercept, r2, current_data_file
    if filename is None:
        filename = filedialog.askopenfilename(
            title=get_text('choose_csv'),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return False
    try:
        df = pd.read_csv(filename)
        if df.shape[1] < 2:
            raise ValueError("CSV must have at least 2 columns")
        # 强制转换为数值
        true_vals = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        pred_vals = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        # 去除 NaN 行
        mask = ~np.isnan(true_vals) & ~np.isnan(pred_vals)
        true_vals = true_vals[mask]
        pred_vals = pred_vals[mask]
        if len(true_vals) == 0:
            raise ValueError("No valid numeric data in first two columns")

        # 使用列名作为轴标签和标题
        col1_name = str(df.columns[0])
        col2_name = str(df.columns[1])
        xlabel_var.set(col1_name)
        ylabel_var.set(col2_name)
        title_var.set(f"{col1_name} vs {col2_name}")

        # 线性回归
        slope, intercept = np.polyfit(true_vals, pred_vals, 1)
        pred_line = slope * true_vals + intercept
        ss_res = np.sum((pred_vals - pred_line) ** 2)
        ss_tot = np.sum((pred_vals - np.mean(pred_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        current_data_file = filename

        # 更新回归信息显示
        slope_label.config(text=f"{get_text('slope')}: {slope:.4f}")
        intercept_label.config(text=f"{get_text('intercept')}: {intercept:.4f}")
        r2_label.config(text=f"R²: {r2:.4f}")
        return True
    except Exception as e:
        messagebox.showerror("Error", get_text('error_load', e=str(e)))
        return False

# ====================== 交换横纵坐标 ======================
def swap_axes():
    global true_vals, pred_vals, slope, intercept, r2
    if len(true_vals) == 0:
        return
    # 交换数据
    true_vals, pred_vals = pred_vals.copy(), true_vals.copy()
    # 重新计算回归
    slope, intercept = np.polyfit(true_vals, pred_vals, 1)
    pred_line = slope * true_vals + intercept
    ss_res = np.sum((pred_vals - pred_line) ** 2)
    ss_tot = np.sum((pred_vals - np.mean(pred_vals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # 交换轴标签变量
    x_temp = xlabel_var.get()
    xlabel_var.set(ylabel_var.get())
    ylabel_var.set(x_temp)
    # 更新标题以反映新关系
    title_var.set(f"{xlabel_var.get()} vs {ylabel_var.get()}")

    # 更新回归信息显示
    slope_label.config(text=f"{get_text('slope')}: {slope:.4f}")
    intercept_label.config(text=f"{get_text('intercept')}: {intercept:.4f}")
    r2_label.config(text=f"R²: {r2:.4f}")

    update_plot()

# ====================== 绘图更新 ======================
def update_plot():
    if len(true_vals) == 0:
        ax.clear()
        ax.text(0.5, 0.5, get_text('no_data'), ha='center', va='center')
        canvas.draw()
        return

    scatter_color = scatter_color_var.get()
    bg_color = bg_color_var.get()
    point_size = point_size_var.get()
    alpha_val = alpha_var.get()
    fit_line_color = fit_line_color_var.get()
    ideal_line_color = ideal_line_color_var.get()
    line_width = line_width_var.get()
    show_grid = show_grid_var.get()
    title_text = title_var.get()
    xlabel_text = xlabel_var.get()
    ylabel_text = ylabel_var.get()

    ax.clear()
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    ax.scatter(true_vals, pred_vals, c=scatter_color, s=point_size,
               alpha=alpha_val, edgecolors='none', label=get_text('data_label'))

    min_val = min(np.min(true_vals), np.min(pred_vals))
    max_val = max(np.max(true_vals), np.max(pred_vals))
    line_x = np.array([min_val, max_val])
    ax.plot(line_x, line_x, color=ideal_line_color, linestyle='--',
            linewidth=line_width, label=get_text('ideal_label'))

    if slope is not None and intercept is not None:
        reg_y = slope * line_x + intercept
        ax.plot(line_x, reg_y, color=fit_line_color, linestyle='-',
                linewidth=line_width, label=get_text('actual_label', slope=slope, intercept=intercept))

    ax.set_title(title_text, fontsize=12)
    ax.set_xlabel(xlabel_text, fontsize=10)
    ax.set_ylabel(ylabel_text, fontsize=10)
    if show_grid:
        ax.grid(True, linestyle=':', alpha=0.6)

    ax.text(0.95, 0.05, get_text('r2_text', r2=r2),
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    canvas.draw()

# ====================== 颜色选择辅助 ======================
def choose_color(variable, button):
    color_code = colorchooser.askcolor(title="Select Color")[1]
    if color_code:
        variable.set(color_code)
        button.config(bg=color_code)

def add_color_choice(parent, label_key, variable, default_color):
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=3)
    ttk.Label(frame, text=get_text(label_key), width=18).pack(side=tk.LEFT)
    btn = tk.Button(frame, text="Select", bg=default_color,
                    command=lambda: choose_color(variable, btn))
    btn.pack(side=tk.RIGHT, padx=5)
    variable.set(default_color)
    return btn

def apply_preset_color():
    palette = palette_var.get()
    if not palette:
        return
    color = get_palette_color(palette)
    scatter_color_var.set(color)
    for child in control_inner.winfo_children():
        if isinstance(child, ttk.Frame):
            for sub in child.winfo_children():
                if isinstance(sub, tk.Button) and sub.cget('text') in ('Select', '选择颜色'):
                    sub.config(bg=color)
                    break
    update_plot()

# ====================== 保存图片 ======================
def save_figure():
    if len(true_vals) == 0:
        messagebox.showwarning("Warning", get_text('no_data'))
        return
    filetypes = [("PNG image", "*.png"), ("PDF document", "*.pdf"), ("SVG image", "*.svg")]
    filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=filetypes)
    if filename:
        try:
            original_size = fig.get_size_inches()
            fig.set_size_inches(8, 6)
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            fig.set_size_inches(original_size)
            messagebox.showinfo("Success", get_text('save_success', path=filename))
        except Exception as e:
            messagebox.showerror("Error", get_text('save_failed', e=str(e)))

# ====================== 语言切换（不再覆盖标题和轴标签） ======================
def toggle_language():
    global current_lang
    current_lang = 'en' if current_lang == 'zh' else 'zh'
    lang_btn.config(text="中文" if current_lang == 'en' else "English")
    control_frame.config(text="Plot Settings" if current_lang == 'en' else "绘图参数设置")
    load_btn.config(text=get_text('load_data'))
    swap_btn.config(text=get_text('swap_axes'))
    save_btn.config(text=get_text('save_img'))
    palette_label.config(text=get_text('palette_label'))
    apply_palette_btn.config(text=get_text('apply_palette'))
    slope_label.config(text=f"{get_text('slope')}: {slope:.4f}" if slope is not None else f"{get_text('slope')}: --")
    intercept_label.config(text=f"{get_text('intercept')}: {intercept:.4f}" if intercept is not None else f"{get_text('intercept')}: --")
    r2_label.config(text=f"R²: {r2:.4f}" if r2 is not None else "R²: --")
    info_frame.config(text=get_text('reg_info'))
    update_plot()

# ====================== GUI 构建 ======================
root = tk.Tk()
root.title("Data Comparison Plotter")
root.geometry("1300x900")
root.eval('tk::PlaceWindow %s center' % root.winfo_pathname(root.winfo_id()))

# 左侧控制面板（带滚动条）
control_frame = ttk.LabelFrame(root, text="Plot Settings", padding=10)
control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

canvas_control = tk.Canvas(control_frame, borderwidth=0)
scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas_control.yview)
control_inner = ttk.Frame(canvas_control)
canvas_control.create_window((0, 0), window=control_inner, anchor="nw")
canvas_control.configure(yscrollcommand=scrollbar.set)

def configure_scroll_region(event):
    canvas_control.configure(scrollregion=canvas_control.bbox("all"))
control_inner.bind("<Configure>", configure_scroll_region)

canvas_control.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def _on_mousewheel(event):
    canvas_control.yview_scroll(int(-1*(event.delta/120)), "units")
canvas_control.bind_all("<MouseWheel>", _on_mousewheel)

# 右侧绘图区域
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

fig, ax = plt.subplots(figsize=(7, 5.25))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ---- 控件变量 ----
scatter_color_var = tk.StringVar(value='steelblue')
bg_color_var = tk.StringVar(value='white')
point_size_var = tk.IntVar(value=30)
alpha_var = tk.DoubleVar(value=0.7)
fit_line_color_var = tk.StringVar(value='red')
ideal_line_color_var = tk.StringVar(value='green')
line_width_var = tk.DoubleVar(value=1.5)
show_grid_var = tk.BooleanVar(value=True)
title_var = tk.StringVar(value='数据对比图')
xlabel_var = tk.StringVar(value='X 轴')
ylabel_var = tk.StringVar(value='Y 轴')
palette_var = tk.StringVar(value='deep')

# ---- 顶部按钮栏 ----
top_btn_frame = ttk.Frame(control_inner)
top_btn_frame.pack(fill=tk.X, pady=5)
lang_btn = ttk.Button(top_btn_frame, text="English", command=toggle_language)
lang_btn.pack(side=tk.LEFT, padx=2)
load_btn = ttk.Button(top_btn_frame, text=get_text('load_data'), command=lambda: load_data() or update_plot())
load_btn.pack(side=tk.LEFT, padx=2)
swap_btn = ttk.Button(top_btn_frame, text=get_text('swap_axes'), command=swap_axes)
swap_btn.pack(side=tk.LEFT, padx=2)
save_btn = ttk.Button(top_btn_frame, text=get_text('save_img'), command=save_figure)
save_btn.pack(side=tk.LEFT, padx=2)

# ---- 预设配色 ----
palette_frame = ttk.Frame(control_inner)
palette_frame.pack(fill=tk.X, pady=5)
palette_label = ttk.Label(palette_frame, text=get_text('palette_label'), width=18)
palette_label.pack(side=tk.LEFT)
palette_combo = ttk.Combobox(palette_frame, textvariable=palette_var, values=ALL_PALETTES, state="readonly")
palette_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
apply_palette_btn = ttk.Button(palette_frame, text=get_text('apply_palette'), command=apply_preset_color)
apply_palette_btn.pack(side=tk.RIGHT)

# ---- 颜色选择器 ----
add_color_choice(control_inner, 'scatter_color', scatter_color_var, 'steelblue')
add_color_choice(control_inner, 'bg_color', bg_color_var, 'white')
add_color_choice(control_inner, 'fit_color', fit_line_color_var, 'red')
add_color_choice(control_inner, 'ideal_color', ideal_line_color_var, 'green')

# ---- 数值滑块 ----
def add_slider(parent, label_key, var, from_, to_, resolution, fmt):
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=5)
    ttk.Label(frame, text=get_text(label_key), width=18).pack(side=tk.LEFT)
    scale = ttk.Scale(frame, from_=from_, to=to_, orient=tk.HORIZONTAL, variable=var)
    scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    val_label = ttk.Label(frame, text=fmt.format(var.get()))
    val_label.pack(side=tk.RIGHT, padx=3)
    def update_label(*args):
        val_label.config(text=fmt.format(var.get()))
    var.trace('w', update_label)
    return scale

add_slider(control_inner, 'point_size', point_size_var, 5, 100, 1, "{:.0f}")
add_slider(control_inner, 'alpha', alpha_var, 0.1, 1.0, 0.01, "{:.2f}")
add_slider(control_inner, 'line_width', line_width_var, 0.5, 4.0, 0.1, "{:.1f}")

# ---- 网格 ----
frame_grid = ttk.Frame(control_inner)
frame_grid.pack(fill=tk.X, pady=5)
ttk.Label(frame_grid, text=get_text('show_grid'), width=18).pack(side=tk.LEFT)
ttk.Checkbutton(frame_grid, variable=show_grid_var).pack(side=tk.RIGHT)

# ---- 标题和轴标签 ----
ttk.Separator(control_inner, orient='horizontal').pack(fill=tk.X, pady=8)
def add_entry(parent, label_key, var):
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=3)
    ttk.Label(frame, text=get_text(label_key), width=18).pack(side=tk.LEFT)
    ttk.Entry(frame, textvariable=var, width=20).pack(side=tk.RIGHT, expand=True, fill=tk.X)

add_entry(control_inner, 'chart_title', title_var)
add_entry(control_inner, 'x_label', xlabel_var)
add_entry(control_inner, 'y_label', ylabel_var)

ttk.Button(control_inner, text=get_text('refresh'), command=update_plot).pack(pady=15)

# ---- 回归信息 ----
info_frame = ttk.LabelFrame(control_inner, text=get_text('reg_info'), padding=5)
info_frame.pack(fill=tk.X, pady=10)
slope_label = ttk.Label(info_frame, text=f"{get_text('slope')}: --")
slope_label.pack(anchor=tk.W)
intercept_label = ttk.Label(info_frame, text=f"{get_text('intercept')}: --")
intercept_label.pack(anchor=tk.W)
r2_label = ttk.Label(info_frame, text="R²: --")
r2_label.pack(anchor=tk.W)

# ====================== 启动自动加载（如有默认文件） ======================
default_csv = os.path.join(os.path.dirname(__file__), "true_vs_predicted_data.csv")
if os.path.exists(default_csv):
    load_data(default_csv)
    update_plot()
else:
    ax.text(0.5, 0.5, get_text('no_data'), ha='center', va='center')
    canvas.draw()

def on_closing():
    plt.close('all')
    root.quit()
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()