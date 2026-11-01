import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from io import StringIO
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 数据加载 ----------
def load_data():
    root_temp = tk.Tk()
    root_temp.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择相关性矩阵CSV文件",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root_temp.destroy()
    if file_path:
        try:
            df = pd.read_csv(file_path, index_col=0)
            if df.shape[0] != df.shape[1]:
                raise ValueError("必须是方阵")
            print(f"已加载：{file_path}")
            return df.astype(float)
        except Exception as e:
            print(f"文件读取失败：{e}")
    if os.path.exists('correlation_matrix_original.csv'):
        try:
            df = pd.read_csv('correlation_matrix_original.csv', index_col=0)
            if df.shape[0] == df.shape[1]:
                print("已加载默认文件")
                return df.astype(float)
        except:
            pass
    csv_content = """,T,v,t,r,V,Vmicro,SSA,AdsSTP
T,1.0,0.07154468767490951,-0.3851921336158645,-0.48205214111674616,-0.09505753420279513,-0.21039964657174448,-0.07367058672487362,-0.13487713233327456
v,0.07154468767490951,1.0,-0.17211354975188045,-0.2005903533533901,-0.24856168647749877,-0.2687744865095145,-0.2591492414163863,-0.2569615697600365
t,-0.3851921336158645,-0.17211354975188045,1.0,0.0800317118133659,0.11545163452871782,0.021775320708975008,0.06005236031595444,0.08856341882328336
r,-0.48205214111674616,-0.2005903533533901,0.0800317118133659,1.0,0.5458172627313786,0.511551092369855,0.5775854223728024,0.36864492995852133
V,-0.09505753420279513,-0.24856168647749877,0.11545163452871782,0.5458172627313786,1.0,0.7854578275229386,0.8911444471546838,0.6016810085473291
Vmicro,-0.21039964657174448,-0.2687744865095145,0.021775320708975008,0.511551092369855,0.7854578275229386,1.0,0.8468568937203146,0.6299183504316695
SSA,-0.07367058672487362,-0.2591492414163863,0.06005236031595444,0.5775854223728024,0.8911444471546838,0.8468568937203146,1.0,0.6492766496629191
AdsSTP,-0.13487713233327456,-0.2569615697600365,0.08856341882328336,0.36864492995852133,0.6016810085473291,0.6299183504316695,0.6492766496629191,1.0"""
    df = pd.read_csv(StringIO(csv_content), index_col=0)
    return df.astype(float)

# ---------- 保存预览对话框 ----------
class SavePreviewDialog(tk.Toplevel):
    def __init__(self, parent, fig, current_width, current_height, current_dpi, draw_callback):
        super().__init__(parent)
        self.title("保存图片 - 预览与设置")
        self.geometry("700x550")
        self.draw_callback = draw_callback
        self.result = None

        preview_frame = ttk.LabelFrame(self, text="预览 (根据当前设置)")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.preview_fig = Figure(figsize=(5, 4), dpi=80)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, master=preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(ctrl_frame, text="宽度(英寸):").grid(row=0, column=0, padx=5, pady=2)
        self.width_var = tk.DoubleVar(value=current_width)
        width_spin = ttk.Spinbox(ctrl_frame, from_=2, to=30, increment=0.5, textvariable=self.width_var, width=8)
        width_spin.grid(row=0, column=1, padx=5)
        width_spin.bind('<KeyRelease>', lambda e: self.update_preview())

        ttk.Label(ctrl_frame, text="高度(英寸):").grid(row=0, column=2, padx=5)
        self.height_var = tk.DoubleVar(value=current_height)
        height_spin = ttk.Spinbox(ctrl_frame, from_=2, to=30, increment=0.5, textvariable=self.height_var, width=8)
        height_spin.grid(row=0, column=3, padx=5)
        height_spin.bind('<KeyRelease>', lambda e: self.update_preview())

        ttk.Label(ctrl_frame, text="DPI:").grid(row=0, column=4, padx=5)
        self.dpi_var = tk.IntVar(value=current_dpi)
        dpi_spin = ttk.Spinbox(ctrl_frame, from_=72, to=600, increment=1, textvariable=self.dpi_var, width=6)
        dpi_spin.grid(row=0, column=5, padx=5)
        dpi_spin.bind('<KeyRelease>', lambda e: self.update_preview())

        ratio_frame = ttk.Frame(ctrl_frame)
        ratio_frame.grid(row=1, column=0, columnspan=6, pady=5)
        ttk.Label(ratio_frame, text="快速比例:").pack(side=tk.LEFT, padx=5)
        ttk.Button(ratio_frame, text="4:3", command=lambda: self.set_ratio(4/3)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ratio_frame, text="16:9", command=lambda: self.set_ratio(16/9)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ratio_frame, text="1:1", command=lambda: self.set_ratio(1)).pack(side=tk.LEFT, padx=2)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="确定保存", command=self.ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=self.cancel).pack(side=tk.LEFT, padx=10)

        self.update_preview()
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def set_ratio(self, ratio):
        w = self.width_var.get()
        self.height_var.set(w / ratio)
        self.update_preview()

    def update_preview(self):
        w = self.width_var.get()
        h = self.height_var.get()
        dpi = self.dpi_var.get()
        self.preview_fig.set_size_inches(w, h)
        self.preview_fig.set_dpi(dpi)
        self.preview_ax.clear()
        self.draw_callback(self.preview_ax)
        self.preview_fig.tight_layout()
        self.preview_canvas.draw()

    def ok(self):
        self.result = (self.width_var.get(), self.height_var.get(), self.dpi_var.get())
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()

# ---------- 主 GUI ----------
class HeatmapGUI:
    def __init__(self, root, data):
        self.root = root
        self.root.title("稳定热力图 - 固定布局，无意外缩放")
        self.root.geometry("1300x750")
        self.df = data
        self.n = data.shape[0]

        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_canvas = tk.Canvas(main_frame, width=320, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=control_canvas.yview)
        scrollable_frame = ttk.Frame(control_canvas)
        scrollable_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax_pos = [0.15, 0.12, 0.70, 0.78]
        self.ax = self.fig.add_axes(self.ax_pos)
        self.fig.set_constrained_layout(False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(plot_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="保存图片 (可调尺寸+预览)", command=self.save_with_preview).pack(side=tk.LEFT, padx=5)
        ttk.Label(btn_frame, text="  提示：除坐标轴范围外，其他参数不会改变图形位置/大小", font=('Arial', 9)).pack(side=tk.LEFT)

        # 控制变量
        self.cmap_var = tk.StringVar(value='coolwarm')
        self.show_annot_var = tk.BooleanVar(value=True)
        self.annot_fmt_var = tk.StringVar(value='.2f')
        self.annot_fs_var = tk.IntVar(value=8)
        self.linewidths_var = tk.DoubleVar(value=0.5)
        self.xtick_rot_var = tk.IntVar(value=45)
        self.ytick_rot_var = tk.IntVar(value=0)
        self.tick_labelsize_var = tk.IntVar(value=10)
        self.show_cbar_var = tk.BooleanVar(value=True)

        self.vmin_var = tk.DoubleVar(value=-1.0)
        self.vmax_var = tk.DoubleVar(value=1.0)
        self.auto_range_var = tk.BooleanVar(value=False)

        # 坐标轴显示范围：X 轴 0 ~ n+1，Y 轴 0 ~ n
        self.xmin_var = tk.DoubleVar(value=0)
        self.xmax_var = tk.DoubleVar(value=self.n + 1)
        self.ymin_var = tk.DoubleVar(value=0)
        self.ymax_var = tk.DoubleVar(value=self.n)
        self.auto_range_axis_var = tk.BooleanVar(value=True)

        # 构建控件
        row = 0
        ttk.Label(scrollable_frame, text="颜色映射:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
        cmap_combo = ttk.Combobox(scrollable_frame, textvariable=self.cmap_var,
                                  values=['coolwarm', 'RdBu_r', 'viridis', 'plasma', 'RdYlBu', 'seismic'],
                                  state='readonly')
        cmap_combo.grid(row=row, column=1, pady=5, padx=5, sticky=tk.W)
        cmap_combo.bind('<<ComboboxSelected>>', lambda e: self.update_heatmap())
        row += 1

        ttk.Checkbutton(scrollable_frame, text="显示相关系数数值", variable=self.show_annot_var,
                        command=self.update_heatmap).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        row += 1

        fmt_frame = ttk.Frame(scrollable_frame)
        fmt_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        ttk.Label(fmt_frame, text="格式:").pack(side=tk.LEFT)
        fmt_entry = ttk.Entry(fmt_frame, textvariable=self.annot_fmt_var, width=6)
        fmt_entry.pack(side=tk.LEFT, padx=5)
        fmt_entry.bind('<KeyRelease>', lambda e: self.update_heatmap())
        ttk.Label(fmt_frame, text="字体大小:").pack(side=tk.LEFT, padx=(10,0))
        annot_fs_spin = ttk.Spinbox(fmt_frame, from_=4, to=20, textvariable=self.annot_fs_var, width=4,
                                    command=self.update_heatmap)
        annot_fs_spin.pack(side=tk.LEFT, padx=5)
        annot_fs_spin.bind('<KeyRelease>', lambda e: self.update_heatmap())
        row += 1

        range_frame = ttk.LabelFrame(scrollable_frame, text="颜色条范围")
        range_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Checkbutton(range_frame, text="自动范围 (基于数据)", variable=self.auto_range_var,
                        command=self.update_heatmap).pack(anchor=tk.W, padx=5, pady=2)
        inner = ttk.Frame(range_frame)
        inner.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(inner, text="vmin:").pack(side=tk.LEFT)
        vmin_spin = ttk.Spinbox(inner, from_=-1.0, to=1.0, increment=0.05, textvariable=self.vmin_var, width=6,
                                command=self.update_heatmap)
        vmin_spin.pack(side=tk.LEFT, padx=5)
        vmin_spin.bind('<KeyRelease>', lambda e: self.update_heatmap())
        ttk.Label(inner, text="vmax:").pack(side=tk.LEFT, padx=(10,0))
        vmax_spin = ttk.Spinbox(inner, from_=-1.0, to=1.0, increment=0.05, textvariable=self.vmax_var, width=6,
                                command=self.update_heatmap)
        vmax_spin.pack(side=tk.LEFT, padx=5)
        vmax_spin.bind('<KeyRelease>', lambda e: self.update_heatmap())
        row += 1

        line_frame = ttk.Frame(scrollable_frame)
        line_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Label(line_frame, text="格子线宽:").pack(side=tk.LEFT)
        line_spin = ttk.Spinbox(line_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.linewidths_var,
                                width=5, command=self.update_heatmap)
        line_spin.pack(side=tk.LEFT, padx=5)
        line_spin.bind('<KeyRelease>', lambda e: self.update_heatmap())
        row += 1

        rot_frame = ttk.LabelFrame(scrollable_frame, text="坐标轴标签旋转")
        rot_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Label(rot_frame, text="X轴旋转(°):").pack(side=tk.LEFT, padx=5)
        xtick_rot_spin = ttk.Spinbox(rot_frame, from_=0, to=90, increment=5, textvariable=self.xtick_rot_var,
                                     width=4, command=self.update_heatmap)
        xtick_rot_spin.pack(side=tk.LEFT, padx=5)
        xtick_rot_spin.bind('<KeyRelease>', lambda e: self.update_heatmap())
        ttk.Label(rot_frame, text="Y轴旋转(°):").pack(side=tk.LEFT, padx=(10,0))
        ytick_rot_spin = ttk.Spinbox(rot_frame, from_=0, to=90, increment=5, textvariable=self.ytick_rot_var,
                                     width=4, command=self.update_heatmap)
        ytick_rot_spin.pack(side=tk.LEFT, padx=5)
        ytick_rot_spin.bind('<KeyRelease>', lambda e: self.update_heatmap())
        row += 1

        tick_fs_frame = ttk.Frame(scrollable_frame)
        tick_fs_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        ttk.Label(tick_fs_frame, text="刻度标签字体大小:").pack(side=tk.LEFT)
        tick_fs_spin = ttk.Spinbox(tick_fs_frame, from_=6, to=20, textvariable=self.tick_labelsize_var,
                                   width=4, command=self.update_heatmap)
        tick_fs_spin.pack(side=tk.LEFT, padx=5)
        tick_fs_spin.bind('<KeyRelease>', lambda e: self.update_heatmap())
        row += 1

        ttk.Checkbutton(scrollable_frame, text="显示颜色条", variable=self.show_cbar_var,
                        command=self.update_heatmap).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5, padx=5)
        row += 1

        range_axis_frame = ttk.LabelFrame(scrollable_frame, text="坐标轴显示范围 (数据坐标)")
        range_axis_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Checkbutton(range_axis_frame, text="自动全范围 (X:0~n+1, Y:0~n)", variable=self.auto_range_axis_var,
                        command=self.toggle_axis_range).pack(anchor=tk.W, padx=5, pady=2)
        self.axis_range_frame = ttk.Frame(range_axis_frame)
        self.axis_range_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(self.axis_range_frame, text="X范围:").grid(row=0, column=0, padx=2)
        self.xmin_entry = ttk.Entry(self.axis_range_frame, textvariable=self.xmin_var, width=6)
        self.xmin_entry.grid(row=0, column=1, padx=2)
        ttk.Label(self.axis_range_frame, text="~").grid(row=0, column=2)
        self.xmax_entry = ttk.Entry(self.axis_range_frame, textvariable=self.xmax_var, width=6)
        self.xmax_entry.grid(row=0, column=3, padx=2)
        ttk.Label(self.axis_range_frame, text="Y范围:").grid(row=1, column=0, padx=2, pady=2)
        self.ymin_entry = ttk.Entry(self.axis_range_frame, textvariable=self.ymin_var, width=6)
        self.ymin_entry.grid(row=1, column=1, padx=2)
        ttk.Label(self.axis_range_frame, text="~").grid(row=1, column=2)
        self.ymax_entry = ttk.Entry(self.axis_range_frame, textvariable=self.ymax_var, width=6)
        self.ymax_entry.grid(row=1, column=3, padx=2)

        self.xmin_entry.bind('<KeyRelease>', lambda e: self.update_heatmap())
        self.xmax_entry.bind('<KeyRelease>', lambda e: self.update_heatmap())
        self.ymin_entry.bind('<KeyRelease>', lambda e: self.update_heatmap())
        self.ymax_entry.bind('<KeyRelease>', lambda e: self.update_heatmap())
        self.toggle_axis_range()
        row += 1

        ttk.Button(scrollable_frame, text="重置所有参数", command=self.reset_defaults).grid(row=row, column=0, columnspan=2, pady=10, padx=5)
        row += 1
        ttk.Button(scrollable_frame, text="重新加载CSV文件", command=self.reload_data).grid(row=row, column=0, columnspan=2, pady=5, padx=5)
        row += 1

        ttk.Label(scrollable_frame, text="""
核心原则：
- 热力图区域位置/大小完全固定，不会缩放
- 修改颜色、字体、旋转等仅改变内容
- 只有「坐标轴显示范围」改变数据视图
- 保存图片时可调整输出尺寸，不影响主窗口
""", font=('Arial', 8), foreground='gray', justify=tk.LEFT).grid(row=row, column=0, columnspan=2, pady=10, padx=5)

        self.update_heatmap()

    def toggle_axis_range(self):
        if self.auto_range_axis_var.get():
            self.xmin_entry.config(state='disabled')
            self.xmax_entry.config(state='disabled')
            self.ymin_entry.config(state='disabled')
            self.ymax_entry.config(state='disabled')
            self.xmin_var.set(0)
            self.xmax_var.set(self.n + 1)   # X 轴 → n+1
            self.ymin_var.set(0)
            self.ymax_var.set(self.n)       # Y 轴 → n
        else:
            self.xmin_entry.config(state='normal')
            self.xmax_entry.config(state='normal')
            self.ymin_entry.config(state='normal')
            self.ymax_entry.config(state='normal')
        self.update_heatmap()

    def reload_data(self):
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            new_df = pd.read_csv(file_path, index_col=0)
            if new_df.shape != (self.n, self.n):
                messagebox.showerror("错误", f"必须为{self.n}x{self.n}矩阵")
                return
            self.df = new_df.astype(float)
            self.update_heatmap()
            messagebox.showinfo("成功", f"已加载：{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败：{e}")

    def draw_heatmap_on_ax(self, ax, fig_for_cbar=None):
        cmap = self.cmap_var.get()
        annot = self.show_annot_var.get()
        fmt = self.annot_fmt_var.get() if annot else ''
        annot_fs = self.annot_fs_var.get()
        linewidths = self.linewidths_var.get()
        xtick_rot = self.xtick_rot_var.get()
        ytick_rot = self.ytick_rot_var.get()
        tick_labelsize = self.tick_labelsize_var.get()
        show_cbar = self.show_cbar_var.get()
        auto_range = self.auto_range_var.get()

        if auto_range:
            vmin = self.df.min().min()
            vmax = self.df.max().max()
            max_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -max_abs, max_abs
        else:
            vmin = self.vmin_var.get()
            vmax = self.vmax_var.get()

        # 坐标轴显示范围：X轴 0~n+1，Y轴 0~n
        if self.auto_range_axis_var.get():
            xmin, xmax = 0, self.n + 1
            ymin, ymax = 0, self.n
        else:
            xmin = self.xmin_var.get()
            xmax = self.xmax_var.get()
            ymin = self.ymin_var.get()
            ymax = self.ymax_var.get()

        ax.clear()
        sns.heatmap(self.df,
                    annot=annot,
                    fmt=fmt,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=linewidths,
                    linecolor='white',
                    cbar=False,
                    square=True,
                    annot_kws={'size': annot_fs} if annot else None,
                    ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rot, ha='center', fontsize=tick_labelsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rot, fontsize=tick_labelsize)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)   # Y轴反转
        ax.set_autoscale_on(False)

        if show_cbar and fig_for_cbar is not None:
            for cax in fig_for_cbar.axes:
                if cax != ax:
                    fig_for_cbar.delaxes(cax)
            cbar = fig_for_cbar.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation coefficient', fontsize=tick_labelsize)                # 相关系数
            cbar.ax.tick_params(labelsize=tick_labelsize-2)

    def update_heatmap(self):
        for ax in self.fig.axes:
            if ax != self.ax:
                self.fig.delaxes(ax)
        self.draw_heatmap_on_ax(self.ax, self.fig if self.show_cbar_var.get() else None)
        self.ax.set_position(self.ax_pos)
        self.canvas.draw()

    def save_with_preview(self):
        cur_w = self.fig.get_figwidth()
        cur_h = self.fig.get_figheight()
        cur_dpi = self.fig.dpi

        def preview_draw(ax):
            self.draw_heatmap_on_ax(ax, fig_for_cbar=None)

        dialog = SavePreviewDialog(self.root, self.fig, cur_w, cur_h, cur_dpi, preview_draw)
        self.root.wait_window(dialog)
        if dialog.result is None:
            return
        w, h, dpi = dialog.result
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")])
        if not file_path:
            return
        old_size = self.fig.get_size_inches()
        old_dpi = self.fig.dpi
        self.fig.set_size_inches(w, h)
        self.fig.set_dpi(dpi)
        self.update_heatmap()
        self.fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        self.fig.set_size_inches(old_size)
        self.fig.set_dpi(old_dpi)
        self.update_heatmap()
        messagebox.showinfo("保存成功", f"图片已保存至：{file_path}")

    def reset_defaults(self):
        self.cmap_var.set('coolwarm')
        self.show_annot_var.set(True)
        self.annot_fmt_var.set('.2f')
        self.annot_fs_var.set(8)
        self.linewidths_var.set(0.5)
        self.xtick_rot_var.set(45)
        self.ytick_rot_var.set(0)
        self.tick_labelsize_var.set(10)
        self.show_cbar_var.set(True)
        self.auto_range_var.set(False)
        self.vmin_var.set(-1.0)
        self.vmax_var.set(1.0)
        self.auto_range_axis_var.set(True)
        self.toggle_axis_range()
        self.update_heatmap()

if __name__ == '__main__':
    root = tk.Tk()
    data = load_data()
    app = HeatmapGUI(root, data)
    root.mainloop()