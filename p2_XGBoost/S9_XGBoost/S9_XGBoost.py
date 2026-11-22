# -*- coding: utf-8 -*-
"""
XGBoost Transfer Learning System with Small Sample Optimization
从预训练 RandomForest 迁移知识到 XGBoost，并针对小样本优化评估
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无显示环境下也可保存图片
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.inspection import PartialDependenceDisplay
import joblib
import json
import os
import warnings
from datetime import datetime
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, font as tkfont
import threading
from queue import SimpleQueue, Empty
from functools import wraps
import sys
import atexit
import time
import re
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

ALLOWED_FILE_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
MAX_FILE_SIZE = 100 * 1024 * 1024


# ====================== Configuration ======================
class Config:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
    RF_PRETRAIN_DIR = os.path.join(PARENT_ROOT, "p1_RF_preTrain")
    TRANSFER_LEARNING_DIR = os.path.join(RF_PRETRAIN_DIR, "transfer_learning")

    DATA0VIEW_BASE_DIR = os.path.join(PROJECT_ROOT, "data0view")
    DATA1VIEW_BASE_DIR = os.path.join(PROJECT_ROOT, "data1view")
    DATA2EVALUATION_BASE_DIR = os.path.join(PROJECT_ROOT, "data2evaluation")
    DATA3FIA_BASE_DIR = os.path.join(PROJECT_ROOT, "data3FIA")
    DATA4PDP_BASE_DIR = os.path.join(PROJECT_ROOT, "data4PDP")
    BESTMODEL_BASE_DIR = os.path.join(PROJECT_ROOT, "bestmodel")

    RF_EXPORTS = {}

    @staticmethod
    def find_rf_exports(directory):
        """Locate exports in the original project and in both distributed ZIP layouts."""
        root = Path(directory).expanduser()
        exports = {}
        locations = [root, root / "rf_resources", root / "p1_RF_preTrain",
                     root / "p1_RF", root / "p1_RF" / "Main data", root / "Main data"]
        for location in locations:
            for folder in [location, *location.glob("*RF_for_XGBoost_*"),
                           *location.glob("transfer_learning/RF_for_XGBoost_*")]:
                match = re.search(r"RF_for_XGBoost_(\d{8}_\d{6})$", folder.name)
                if (match and (folder / "feature_importance.json").is_file()
                        and (folder / "model_parameters.json").is_file()):
                    exports.setdefault(match.group(1), folder)
        return exports

    @classmethod
    def rf_export(cls, timestamp):
        if timestamp in cls.RF_EXPORTS:
            return Path(cls.RF_EXPORTS[timestamp])
        return Path(cls.TRANSFER_LEARNING_DIR) / f"RF_for_XGBoost_{timestamp}"

    @classmethod
    def data1view_locations(cls, timestamp):
        export = cls.rf_export(timestamp)
        data_root = export.parent.parent if export.parent.name == "transfer_learning" else export.parent
        return [export, data_root / "data1view" / str(timestamp),
                data_root / f"data1view {timestamp}"]

    @staticmethod
    def ensure_directories():
        for d in [
            Config.DATA0VIEW_BASE_DIR, Config.DATA1VIEW_BASE_DIR,
            Config.DATA2EVALUATION_BASE_DIR, Config.DATA3FIA_BASE_DIR,
            Config.DATA4PDP_BASE_DIR, Config.BESTMODEL_BASE_DIR,
        ]:
            os.makedirs(d, exist_ok=True)
        return True

    XGB_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 1,
        'gamma': 0,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'eval_metric': 'rmse',
    }


# ====================== Security ======================
def validate_file_path(file_path):
    if not file_path:
        return False
    _, ext = os.path.splitext(file_path.lower())
    if ext not in ALLOWED_FILE_EXTENSIONS:
        return False
    try:
        if os.path.getsize(file_path) > MAX_FILE_SIZE:
            return False
    except OSError:
        return False
    return os.path.isfile(file_path)


# ====================== Small Sample Evaluator ======================
class SmallSampleEvaluator:
    """小样本数据评估器，保留传统指标的同时增加小样本优化指标"""

    def __init__(self, y_true, y_pred, sample_names=None):
        self.y_true = np.array(y_true, dtype=np.float64)
        self.y_pred = np.array(y_pred, dtype=np.float64)
        self.sample_names = sample_names or [f'Sample_{i+1}' for i in range(len(y_true))]
        if len(self.sample_names) < len(self.y_true):
            self.sample_names = list(self.sample_names) + [
                f'Sample_{i+1}' for i in range(len(self.sample_names), len(self.y_true))
            ]
        self.metrics = {}
        self.is_small_sample = len(y_true) <= 20

    def calculate_all_metrics(self):
        n = len(self.y_true)
        if n == 0:
            self.metrics = {'sample_count': 0, 'is_small_sample': True, 'error': 'No samples'}
            return self.metrics

        r2 = r2_score(self.y_true, self.y_pred)
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(self.y_true, self.y_pred)
        evs = explained_variance_score(self.y_true, self.y_pred)

        y_true_abs = np.abs(self.y_true).copy()
        y_true_abs[y_true_abs == 0] = 1e-8
        mre = float(np.mean(np.abs((self.y_true - self.y_pred) / y_true_abs)))

        errors = self.y_pred - self.y_true
        abs_errors = np.abs(errors)

        with np.errstate(divide='ignore', invalid='ignore'):
            percent_errors = abs_errors / np.abs(self.y_true) * 100
            percent_errors[np.isinf(percent_errors)] = np.nan
            percent_errors[np.isnan(percent_errors)] = 100.0

        smape = float(100.0 / n * np.sum(
            2 * abs_errors / (np.abs(self.y_true) + np.abs(self.y_pred) + 1e-8)
        ))

        thresholds = [0.01, 0.05, 0.1, 0.2]
        accuracy_at_thresholds = {}
        for th in thresholds:
            within = percent_errors <= th * 100
            valid = ~np.isnan(within)
            accuracy_at_thresholds[f'accuracy_{int(th*100)}%'] = (
                float(np.mean(within[valid]) * 100) if np.any(valid) else 0.0
            )

        error_stats = {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'error_range': float(np.ptp(errors)),
        }

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = self.y_pred / self.y_true
            ratios[np.isinf(ratios)] = np.nan
            ratios[np.isnan(ratios)] = 1.0

        ratio_stats = {
            'mean_ratio': float(np.nanmean(ratios)),
            'median_ratio': float(np.nanmedian(ratios)),
            'std_ratio': float(np.nanstd(ratios)),
            'min_ratio': float(np.nanmin(ratios)),
            'max_ratio': float(np.nanmax(ratios)),
        }

        if n > 1:
            try:
                rank_corr, _ = spearmanr(self.y_true, self.y_pred)
                rank_corr = float(rank_corr) if rank_corr is not None and not np.isnan(rank_corr) else float('nan')
            except Exception:
                rank_corr = float('nan')
        else:
            rank_corr = float('nan')

        median_ape = float(np.nanmedian(percent_errors))
        mean_ape = float(np.nanmean(percent_errors))
        max_ape = float(np.nanmax(percent_errors))

        individual = []
        for i in range(n):
            pe = float(percent_errors[i]) if not np.isnan(percent_errors[i]) else None
            rt = float(ratios[i]) if not np.isnan(ratios[i]) else None
            individual.append({
                'sample': self.sample_names[i],
                'true_value': float(self.y_true[i]),
                'predicted_value': float(self.y_pred[i]),
                'absolute_error': float(abs_errors[i]),
                'signed_error': float(errors[i]),
                'percent_error': pe,
                'ratio': rt,
            })

        self.metrics = {
            'is_small_sample': self.is_small_sample,
            'sample_count': n,
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': rmse,
            'mae': float(mae),
            'explained_variance': float(evs),
            'mre': mre,
            'smape': smape,
            'median_ape': median_ape,
            'mean_ape': mean_ape,
            'max_ape': max_ape,
            'rank_correlation': rank_corr,
            'accuracy_at_thresholds': accuracy_at_thresholds,
            'error_statistics': error_stats,
            'ratio_statistics': ratio_stats,
            'individual_errors': individual,
        }
        return self.metrics

    def generate_detailed_report(self):
        self.calculate_all_metrics()
        m = self.metrics
        if m.get('sample_count', 0) == 0:
            return "模型评估报告\n(无有效样本)\n"

        report = f"""
模型评估报告
================================
样本数量: {m['sample_count']}
样本类型: {'小样本' if m['is_small_sample'] else '大样本'}

传统评估指标:
------------------------------------------------
R² Score (决定系数): {m['r2_score']:.4f}
RMSE (均方根误差): {m['rmse']:.4f}
MAE (平均绝对误差): {m['mae']:.4f}
MSE (均方误差): {m['mse']:.4f}
Explained Variance (解释方差): {m['explained_variance']:.4f}
MRE (平均相对误差): {m['mre']:.4f}

小样本优化指标:
------------------------------------------------
对称平均绝对百分比误差 (SMAPE): {m['smape']:.2f}%
平均绝对百分比误差 (MAPE): {m['mean_ape']:.2f}%
中位数绝对百分比误差 (MedAPE): {m['median_ape']:.2f}%
最大百分比误差: {m['max_ape']:.2f}%
排名相关系数: {m['rank_correlation']:.3f}

预测准确率 (在不同误差阈值下):
------------------------------------------------
"""
        for threshold, acc in m['accuracy_at_thresholds'].items():
            thv = int(threshold.split('_')[1].replace('%', ''))
            report += f"  误差在±{thv}%以内: {acc:.1f}%\n"

        es = m['error_statistics']
        rs = m['ratio_statistics']
        report += f"""
误差统计分析:
------------------------------------------------
平均误差: {es['mean_error']:.4f}
中位数误差: {es['median_error']:.4f}
误差标准差: {es['std_error']:.4f}
最小误差: {es['min_error']:.4f}
最大误差: {es['max_error']:.4f}
误差范围: {es['error_range']:.4f}

预测值/真实值比例统计:
------------------------------------------------
平均比例: {rs['mean_ratio']:.3f}
中位数比例: {rs['median_ratio']:.3f}
比例标准差: {rs['std_ratio']:.3f}
最小比例: {rs['min_ratio']:.3f}
最大比例: {rs['max_ratio']:.3f}

详细样本误差分析:
------------------------------------------------
"""
        for e in m['individual_errors']:
            pe = f"{e['percent_error']:.2f}" if e['percent_error'] is not None else "N/A"
            rt = f"{e['ratio']:.3f}" if e['ratio'] is not None else "N/A"
            report += f"""
样本: {e['sample']}
  真实值: {e['true_value']:.4f}
  预测值: {e['predicted_value']:.4f}
  绝对误差: {e['absolute_error']:.4f}
  百分比误差: {pe}%
  预测/真实比例: {rt}
"""
        return report

    def plot_all_visualizations(self, save_path=None):
        if len(self.y_true) < 2:
            return None
        n = len(self.y_true)
        fig = plt.figure(figsize=(16, 12))
        self._plot_scatter_line(plt.subplot(2, 3, 1))
        self._plot_error_bars(plt.subplot(2, 3, 2))
        self._plot_accuracy_matrix(plt.subplot(2, 3, 3))
        self._plot_percent_errors(plt.subplot(2, 3, 4))
        self._plot_error_distribution(plt.subplot(2, 3, 5))
        self._plot_metrics_comparison(plt.subplot(2, 3, 6))
        plt.suptitle(f'模型评估可视化 (样本数: {n})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        return fig

    def _plot_scatter_line(self, ax):
        x = np.arange(len(self.y_true))
        ax.scatter(x, self.y_true, color='blue', s=100, label='真实值', alpha=0.7, edgecolors='black')
        ax.scatter(x, self.y_pred, color='red', s=100, marker='s', label='预测值', alpha=0.7, edgecolors='black')
        for i in range(len(self.y_true)):
            ax.plot([i, i], [self.y_true[i], self.y_pred[i]], color='gray', alpha=0.5, linestyle='--')
        for i, name in enumerate(self.sample_names):
            ax.text(i, max(self.y_true[i], self.y_pred[i]) * 1.02, name, fontsize=8, ha='center', rotation=45)
        ax.set_xlabel('样本索引'); ax.set_ylabel('值'); ax.set_title('真实值 vs 预测值')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_error_bars(self, ax):
        x = np.arange(len(self.y_true))
        errors = self.y_pred - self.y_true
        colors = ['red' if err > 0 else 'blue' for err in errors]
        bars = ax.bar(x, errors, color=colors, alpha=0.7)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + (0.01 if h >= 0 else -0.01),
                    f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('样本索引'); ax.set_ylabel('误差 (预测值 - 真实值)'); ax.set_title('预测误差')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_accuracy_matrix(self, ax):
        thresholds = [0.05, 0.1, 0.2]
        acc_matrix = []
        with np.errstate(divide='ignore', invalid='ignore'):
            pe = np.abs(self.y_pred - self.y_true) / np.abs(self.y_true) * 100
            pe[np.isnan(pe)] = 100
        for t in thresholds:
            acc_matrix.append(pe <= t * 100)
        acc_matrix = np.array(acc_matrix)
        if acc_matrix.size == 0:
            return
        im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto')
        for i in range(len(thresholds)):
            for j in range(len(self.y_true)):
                ax.text(j, i, '✓' if acc_matrix[i, j] else '✗', ha='center', va='center',
                        color='white' if acc_matrix[i, j] else 'black', fontsize=12)
        ax.set_xlabel('样本索引'); ax.set_ylabel('误差阈值')
        ax.set_yticks(range(len(thresholds)))
        ax.set_yticklabels([f'±{int(t*100)}%' for t in thresholds])
        ax.set_title('预测准确性矩阵')
        plt.colorbar(im, ax=ax, ticks=[0, 1], label='准确性')

    def _plot_percent_errors(self, ax):
        x = np.arange(len(self.y_true))
        with np.errstate(divide='ignore', invalid='ignore'):
            pe = np.abs(self.y_pred - self.y_true) / np.abs(self.y_true) * 100
            pe[np.isnan(pe)] = 100
        bars = ax.bar(x, pe, alpha=0.7)
        for i, bar in enumerate(bars):
            if pe[i] <= 10: bar.set_color('green')
            elif pe[i] <= 20: bar.set_color('orange')
            else: bar.set_color('red')
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                    f'{pe[i]:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% 阈值')
        ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20% 阈值')
        ax.set_xlabel('样本索引'); ax.set_ylabel('百分比误差 (%)'); ax.set_title('百分比误差分布')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    def _plot_error_distribution(self, ax):
        errors = self.y_pred - self.y_true
        bins = min(5, len(errors))
        if bins > 0:
            ax.hist(errors, bins=bins, alpha=0.7, edgecolor='black', color='green')
            ax.axvline(x=np.mean(errors), color='red', linestyle='--', label=f'均值: {np.mean(errors):.2f}')
            ax.axvline(x=np.median(errors), color='blue', linestyle='--', label=f'中位数: {np.median(errors):.2f}')
            stats_text = f"误差统计:\n均值: {np.mean(errors):.4f}\n中位数: {np.median(errors):.4f}\n" \
                         f"标准差: {np.std(errors):.4f}\n范围: [{np.min(errors):.4f}, {np.max(errors):.4f}]"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('误差值'); ax.set_ylabel('频率'); ax.set_title('误差分布'); ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_metrics_comparison(self, ax):
        names = ['R²', 'RMSE', 'MAE', 'SMAPE', 'MedAPE']
        vals = [
            self.metrics['r2_score'],
            self.metrics['rmse'],
            self.metrics['mae'],
            self.metrics['smape'] / 100.0,
            self.metrics['median_ape'] / 100.0,
        ]
        x = np.arange(len(names))
        bars = ax.bar(x, vals, alpha=0.7)
        colors = ['green', 'red', 'red', 'orange', 'orange']
        for i, bar in enumerate(bars):
            bar.set_color(colors[i])
            h = bar.get_height()
            label = f'{vals[i]*100:.1f}%' if i >= 3 else f'{vals[i]:.3f}'
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01, label, ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('评估指标'); ax.set_ylabel('指标值'); ax.set_title('评估指标对比')
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45); ax.grid(True, alpha=0.3, axis='y')


# ====================== RF Knowledge Transfer ======================
class RFKnowledgeTransfer:
    def __init__(self, rf_timestamp, load_rf_model=True):
        self.rf_timestamp = rf_timestamp
        self.transfer_dir = Config.rf_export(rf_timestamp)
        self.rf_model = None
        self.feature_importance = None
        self.rf_params = None
        self.target_variable = None
        self.original_features = None
        self.rf_preprocess_stats = None
        self.load_rf_model = load_rf_model

    def load_rf_resources(self):
        try:
            if not self.load_rf_model:
                return True, "RF model loading skipped (continue training mode)"
            if not os.path.exists(self.transfer_dir):
                return False, f"RF transfer directory does not exist: {self.transfer_dir}"

            imp_path = os.path.join(self.transfer_dir, 'feature_importance.json')
            if not os.path.exists(imp_path):
                return False, f"Feature importance file not found: {imp_path}"
            with open(imp_path, 'r', encoding='utf-8') as f:
                self.feature_importance = json.load(f)
            self.original_features = self.feature_importance['feature_names']

            params_path = os.path.join(self.transfer_dir, 'model_parameters.json')
            if not os.path.exists(params_path):
                return False, f"Model parameters file not found: {params_path}"
            with open(params_path, 'r', encoding='utf-8') as f:
                self.rf_params = json.load(f)
            self.target_variable = self.rf_params.get('target_variable')

            return True, "RF resources loaded successfully"
        except Exception as e:
            return False, f"Failed to load RF resources: {e}"

    def get_model_info(self):
        if not self.rf_params:
            return "Model information not loaded"
        cv_score = self.rf_params.get('cv_score', None)
        cv_str = f"{cv_score:.4f}" if isinstance(cv_score, (int, float)) else str(cv_score)
        info = f"""Model Information:
Timestamp: {self.rf_timestamp}
Target Variable: {self.target_variable}
Number of Features: {len(self.original_features) if self.original_features else 'N/A'}
R² Score: {cv_str}
Training Samples: {self.rf_params.get('training_samples', 'N/A')}

Parameter Settings:
"""
        for k, v in self.rf_params.get('best_parameters', {}).items():
            info += f"  {k}: {v}\n"
        if self.feature_importance:
            info += "\nTop 5 Important Features:\n"
            for i, feat in enumerate(self.feature_importance['importance_ranking'][:5], 1):
                idx = self.feature_importance['feature_names'].index(feat)
                score = self.feature_importance['importance_scores'][idx]
                info += f"  {i}. {feat} (Importance: {score:.4f})\n"
        return info

    def get_all_features(self):
        if not self.original_features:
            raise ValueError("Original features not loaded")
        return self.original_features

    def select_top_features(self, n_features=10):
        if not self.feature_importance:
            raise ValueError("Feature importance not loaded")
        ranking = self.feature_importance['importance_ranking']
        return ranking[:min(n_features, len(ranking))]

    def get_feature_weights(self):
        if not self.feature_importance:
            raise ValueError("Feature importance not loaded")
        weights = dict(zip(self.feature_importance['feature_names'], self.feature_importance['importance_scores']))
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    def generate_pseudo_labels(self, X_new):
        if self.rf_model is None:
            model_path = self.transfer_dir / "RandomForest_transfer_ready.pkl"
            if not model_path.is_file():
                raise FileNotFoundError(f"Unlabelled data require the original RF model: {model_path}")
            self.rf_model = joblib.load(model_path)
        missing = [name for name in self.original_features if name not in X_new.columns]
        if missing:
            raise ValueError(f"Missing RF features for pseudo-label generation: {missing}")
        # Preserve the recovered pseudo-label pathway in the RF model's native output units.
        return self.rf_model.predict(X_new.loc[:, self.original_features])


# ====================== Data Preprocessor ======================
def align_input_names(frame, expected):
    aliases = (("v", "speed", "rate", "heating_rate"),
               ("t", "time", "holding_time"), ("r", "ratio", "KOH_ratio"))
    result = frame.copy()
    result.columns = [str(name).strip() for name in result.columns]
    for group in aliases:
        name = next((c for c in group if c in expected), None)
        present = [c for c in group if c in result.columns]
        if name and len(present) > 1:
            raise ValueError(f"Ambiguous columns for {name}: {present}")
        if name and present:
            result.rename(columns={present[0]: name}, inplace=True)
    return result


class DataPreprocessor:
    def __init__(self, rf_timestamp):
        self.rf_timestamp = rf_timestamp
        self.means = None
        self.stds = None
        self.skewed_columns = []
        self._fitted = {}
        self._load_preprocessing_params()

    def _load_preprocessing_params(self):
        for directory in Config.data1view_locations(self.rf_timestamp):
            stats = directory / "preprocessed_data_statistics.csv"
            if stats.is_file():
                summary = pd.read_csv(stats, index_col=0)
                if {"mean", "std"}.issubset(summary.index):
                    self.means, self.stds = summary.loc["mean"], summary.loc["std"]
                    return
            table = directory / "preprocessed_data.csv"
            if table.is_file():
                data = pd.read_csv(table).select_dtypes(include=[np.number])
                self.means, self.stds = data.mean(), data.std()
                return

    def to_state(self):
        return {"version": 1, "columns": self._fitted}

    def restore_state(self, state):
        if state.get("version") != 1 or not state.get("columns"):
            raise ValueError("Invalid saved preprocessing configuration.")
        self._fitted = dict(state["columns"])

    def get_data_info(self, df):
        info = f"Basic Data Information:\nRows: {df.shape[0]}\nColumns: {df.shape[1]}\nData Type Distribution:\n"
        for dtype, count in df.dtypes.value_counts().items():
            info += f"  {dtype}: {count} columns\n"
        info += "\nFirst 5 Rows Preview:\n" + df.head().to_string() + "\n"
        info += "\nStatistical Summary:\n"
        if df.select_dtypes(include=[np.number]).shape[1] > 0:
            info += df.describe().to_string()
        else:
            info += "No numeric columns, cannot generate statistical summary"
        return info

    def preprocess_data(self, df, target_column=None):
        source = df.copy()
        target = None
        if target_column in source.columns:
            target = pd.to_numeric(source.pop(target_column), errors="raise").astype(float)
            if not np.isfinite(target).all():
                raise ValueError(f"The measured target {target_column} must be complete and finite.")
        if source.empty:
            raise ValueError("No input features or samples found.")

        if self.means is not None:
            stats = align_input_names(pd.DataFrame([self.means, self.stds], index=["mean", "std"]),
                                      source.columns)
        else:
            stats = None
        fit = not self._fitted
        state = {} if fit else self._fitted
        scaled = pd.DataFrame(index=source.index)
        for col in (list(source.columns) if fit else [c for c in state if c in source.columns]):
            values = pd.to_numeric(source[col], errors="raise").astype(float)
            if np.isinf(values).any():
                raise ValueError(f"{col} contains an infinite value.")
            median = float(values.median()) if fit else state[col]["median"]
            if not np.isfinite(median):
                raise ValueError(f"{col} has no finite values.")
            values = values.fillna(median)
            if fit:
                offset = None
                if abs(values.skew()) > 0.5:
                    minimum = values.min()
                    offset = 1.0 if minimum > 0 else float(abs(minimum) + 1.0)
            else:
                offset = state[col]["log_offset"]
            if offset is not None:
                if ((values + offset) <= 0).any():
                    raise ValueError(f"{col} is outside the fitted logarithm domain.")
                values = np.log(values + offset)
            if fit:
                if stats is not None and col in stats:
                    mean, std = float(stats.at["mean", col]), float(stats.at["std", col])
                else:
                    mean, std = float(values.mean()), float(values.std())
                if not np.isfinite(std) or std <= 0:
                    mean, std = 0.0, 1.0
                state[col] = {"median": median, "log_offset": offset, "mean": mean, "std": std}
            scaled[col] = (values - state[col]["mean"]) / state[col]["std"]
        self._fitted = state
        self.skewed_columns = [col for col, item in state.items() if item["log_offset"] is not None]
        return scaled, target


# ====================== File Readers ======================
def read_data_file(file_path, log_callback=None):
    if not validate_file_path(file_path):
        raise ValueError(f"Invalid file path: {file_path}")
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == '.csv':
        return read_csv_with_encoding(file_path, log_callback)
    elif ext in ['.xlsx', '.xls']:
        return read_excel_file(file_path, log_callback)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def read_csv_with_encoding(file_path, log_callback=None):
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp1252', 'iso-8859-1', 'utf-8-sig']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            if log_callback:
                log_callback(f"Successfully read with {enc} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if log_callback:
                log_callback(f"Failed with {enc}: {e}")
            continue
    try:
        df = pd.read_csv(file_path)
        if log_callback:
            log_callback("Successfully read with default encoding")
        return df
    except Exception as e:
        raise ValueError(f"Cannot read file, tried multiple encodings: {e}")


def read_excel_file(file_path, log_callback=None):
    try:
        df = pd.read_excel(file_path)
        if log_callback:
            log_callback(f"Successfully read Excel file: {file_path}")
        return df
    except Exception:
        try:
            xls = pd.ExcelFile(file_path)
            df = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])
            if log_callback:
                log_callback(f"Successfully read sheet: {xls.sheet_names[0]}")
            return df
        except Exception as e2:
            raise ValueError(f"Failed to read Excel file: {e2}")


# ====================== XGBoost Transfer Learner ======================
class XGBoostTransferLearner:
    def __init__(self, rf_transfer, log_callback=None, existing_model_path=None, continue_training=False):
        self.rf_transfer = rf_transfer
        self.xgb_model = None
        self.feature_weights = None
        self.selected_features = None
        self.training_history = []
        self.log_callback = log_callback
        self.existing_model_path = existing_model_path
        self.continue_training = continue_training
        if continue_training:
            self.load_existing_model(existing_model_path)

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def load_existing_model(self, model_path):
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Existing XGBoost model not found: {model_path}")
        self.log(f"Loading existing model from: {model_path}")
        self.xgb_model = joblib.load(model_path)
        if not isinstance(self.xgb_model, xgb.XGBRegressor):
            raise ValueError("Select an XGBRegressor model file.")
        self.selected_features = self.xgb_model.get_booster().feature_names
        if not self.selected_features:
            raise ValueError("The saved XGBoost model has no prediction-column names.")
        self.log(f"Loaded model with {len(self.selected_features)} named features.")

    def prepare_training_data(self, X_new, y_new=None, use_pseudo_labels=True, n_features=10):
        self.log("Preparing training data...")
        if self.continue_training and self.selected_features is not None:
            selected_features = self.selected_features
            self.log(f"  Using {len(selected_features)} features from existing model")
        else:
            try:
                selected_features = self.rf_transfer.select_top_features(n_features)
                self.log(f"  Selected Top {len(selected_features)} features")
                if not self.continue_training:
                    try:
                        self.feature_weights = self.rf_transfer.get_feature_weights()
                    except Exception:
                        self.feature_weights = None
            except Exception as e:
                if self.continue_training and self.selected_features is not None:
                    selected_features = self.selected_features
                elif self.continue_training:
                    selected_features = list(X_new.columns)
                else:
                    raise ValueError(f"Failed to get features from RF model: {e}")
        self.selected_features = selected_features

        missing = [name for name in selected_features if name not in X_new.columns]
        if missing:
            raise ValueError(f"Required input features missing: {', '.join(missing)}")
        X_aligned = X_new.loc[:, selected_features]

        if y_new is not None:
            y = y_new.values if isinstance(y_new, pd.Series) else y_new
            label_source = "True labels"
        elif use_pseudo_labels and not self.continue_training:
            self.log("  Generating pseudo-labels using RF model...")
            y = self.rf_transfer.generate_pseudo_labels(X_new)
            label_source = "Pseudo-labels"
        else:
            raise ValueError("Please provide true labels or enable pseudo-labels")
        self.log(f"✓ Data prepared: Features {X_aligned.shape}, Labels {len(y)} ({label_source})")
        return X_aligned, y

    def train(self, X_train, y_train, sample_weight=None, additional_epochs=100):
        previous = None
        if self.continue_training:
            if additional_epochs < 1:
                raise ValueError("Additional training rounds must be a positive integer.")
            previous = self.xgb_model.get_booster()
            params = self.xgb_model.get_params()
            params['n_estimators'] = additional_epochs
            self.log(f"Appending {additional_epochs} boosting rounds to {previous.num_boosted_rounds()} existing rounds.")
        else:
            params = Config.XGB_PARAMS.copy()
            self.log("Training a new XGBoost model.")
        self.xgb_model = xgb.XGBRegressor(**params)
        self.xgb_model.fit(X_train, y_train, xgb_model=previous, verbose=False,
                           sample_weight=sample_weight)

        y_pred = self.xgb_model.predict(X_train)
        r2 = float(r2_score(y_train, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_train, y_pred)))
        mae = float(mean_absolute_error(y_train, y_pred))
        self.log(f"✓ Training fit (in-sample): R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

        res = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'samples': int(len(X_train)),
            'features': int(X_train.shape[1]),
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': rmse ** 2,
            'mre': float(np.mean(np.abs(y_train - y_pred) / np.maximum(np.abs(y_train), 1e-8))),
            'explained_variance': float(explained_variance_score(y_train, y_pred)),
            'n_estimators': self.xgb_model.get_booster().num_boosted_rounds(),
            'continued_training': bool(self.continue_training),
            'additional_epochs': int(additional_epochs) if self.continue_training else 0,
        }
        self.training_history.append(res)
        return res


# ====================== Training Manager ======================
class TrainingManager:
    def __init__(self, rf_timestamp, log_callback=None, load_existing_model=None, continue_training=False):
        self.rf_timestamp = rf_timestamp
        self.log_callback = log_callback
        self.continue_training = continue_training
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.existing_model_info = None
        self.existing_model_path = None

        self.original_df = None
        self.original_target_column = None
        self.original_has_labels = None
        self.sample_names = None
        self.preprocessed_X = None
        self.preprocessed_y = None

        self.create_output_directories()

        self.rf_transfer = RFKnowledgeTransfer(rf_timestamp, load_rf_model=not continue_training)
        self.preprocessor = DataPreprocessor(rf_timestamp)
        self.xgb_learner = None

        self.training_summary = {
            'rf_timestamp': rf_timestamp,
            'xgb_timestamp': self.timestamp,
            'start_time': datetime.now().isoformat(),
            'total_batches': 0,
            'total_samples': 0,
            'batches': [],
            'evaluation_results': {},
            'load_existing_model': load_existing_model if load_existing_model else None,
            'continue_training': continue_training,
            'is_small_sample': False,
            'small_sample_evaluation': {},
            'evaluation_scope': 'in_sample_training_fit',
            'historical_preprocessing_verified': True,
            'input_preprocessing_provenance': 'fitted_for_this_run',
        }

        if load_existing_model and continue_training:
            self.load_existing_model_info(load_existing_model)
            state = self.existing_model_info.get('input_preprocessing')
            if state is None:
                self.training_summary['historical_preprocessing_verified'] = False
                self.training_summary['input_preprocessing_provenance'] = 'legacy_fallback_on_current_data'
                self.log("WARNING: This legacy model has no saved input transform. The recovered script's "
                         "preprocessing will be applied to the current data; consistency with the old "
                         "tree thresholds cannot be verified. The original model will be preserved.")
            else:
                self.preprocessor.restore_state(state)
                self.training_summary['historical_preprocessing_verified'] = self.existing_model_info.get(
                    'historical_preprocessing_verified', True)
                self.training_summary['input_preprocessing_provenance'] = 'loaded_from_parent_model'


    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def load_existing_model_info(self, model_path):
        # Follow the actual model chosen in the GUI; archives may reside outside bestmodel/.
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Existing model not found: {model_path}")
        self.existing_model_path = os.path.abspath(model_path)
        stem = os.path.splitext(os.path.basename(model_path))[0]
        timestamp = stem.removeprefix("XGBoost_model_")
        info_path = os.path.join(os.path.dirname(model_path), f"model_info_{timestamp}.json")
        if not os.path.isfile(info_path):
            raise ValueError("Keep the selected .pkl beside its matching model_info JSON file.")
        with open(info_path, encoding="utf-8") as handle:
            self.existing_model_info = json.load(handle)
        self.log(f"Loaded existing model info: {info_path}")

    def create_output_directories(self):
        Config.ensure_directories()
        self.data0view_dir = os.path.join(Config.DATA0VIEW_BASE_DIR, self.timestamp)
        self.data1view_dir = os.path.join(Config.DATA1VIEW_BASE_DIR, self.timestamp)
        self.data2evaluation_dir = os.path.join(Config.DATA2EVALUATION_BASE_DIR, self.timestamp)
        self.data3FIA_dir = os.path.join(Config.DATA3FIA_BASE_DIR, self.timestamp)
        self.data4PDP_dir = os.path.join(Config.DATA4PDP_BASE_DIR, self.timestamp)
        self.bestmodel_dir = os.path.join(Config.BESTMODEL_BASE_DIR, self.timestamp)
        for d in [self.data0view_dir, self.data1view_dir, self.data2evaluation_dir,
                  self.data3FIA_dir, self.data4PDP_dir, self.bestmodel_dir]:
            os.makedirs(d, exist_ok=True)

    def setup(self):
        self.log("Loading RF knowledge...")
        success, message = self.rf_transfer.load_rf_resources()
        if not success:
            if self.continue_training:
                self.log(f"⚠ RF knowledge loading failed: {message}")
                return True
            raise RuntimeError(message)
        self.log(f"✓ {message}")
        self.log(f"  RF target: {self.rf_transfer.target_variable}")
        self.log(f"  RF features: {len(self.rf_transfer.get_all_features())}")
        return True

    def load_new_data(self, data_path, target_column=None, sample_names=None):
        self.log(f"Loading new data: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file does not exist: {data_path}")
        df = read_data_file(data_path, self.log)
        self.log(f"  Original data: {df.shape[0]} rows × {df.shape[1]} cols")

        is_small = df.shape[0] <= 20
        self.training_summary['is_small_sample'] = is_small
        self.log(f"  样本类型: {'小样本' if is_small else '大样本'}")

        if self.continue_training and self.existing_model_info:
            target_column = self.existing_model_info.get('target_variable')
            self.log(f"  Target from existing model: {target_column}")
        elif target_column is None and self.rf_transfer.target_variable:
            target_column = self.rf_transfer.target_variable
            self.log(f"  Target from RF model: {target_column}")

        has_labels = target_column is not None and target_column in df.columns
        if has_labels:
            self.log(f"  Found target column: {target_column}")
        else:
            self.log(f"  Target column missing, will use pseudo-labels")

        expected = self.rf_transfer.original_features or []
        if self.continue_training and self.existing_model_info:
            expected = self.existing_model_info.get("feature_order") or list(self.preprocessor._fitted)
        df = align_input_names(df, expected)
        id_columns = {"sample", "sample_id", "sample name", "sample_name", "specimen", "id"}
        id_col = next((c for c in df if c.lower() in id_columns and c != target_column
                       and c not in expected), None)
        file_names = df.pop(id_col).astype(str).tolist() if id_col else None
        self.sample_names = sample_names or file_names or [f"Sample_{i+1}" for i in range(len(df))]
        if len(self.sample_names) != len(df):
            raise ValueError("The number of sample names must match the number of data rows.")
        if expected:
            df = df.loc[:, [c for c in df if c in expected or c == target_column]]

        self.original_df = df
        self.original_target_column = target_column
        self.original_has_labels = has_labels
        return df, target_column, has_labels

    def save_original_data_analysis(self):
        if self.original_df is None:
            return
        try:
            self.original_df.to_csv(os.path.join(self.data0view_dir, "original_data.csv"), index=False)
            stats = self.create_data_statistics_summary(self.original_df, "Original Data")
            with open(os.path.join(self.data0view_dir, "data_statistics.txt"), 'w', encoding='utf-8') as f:
                f.write(stats)
            self.create_data_visualization(self.original_df, self.data0view_dir, "Original Data")
            info = {
                'timestamp': self.timestamp, 'rows': int(self.original_df.shape[0]),
                'columns': int(self.original_df.shape[1]),
                'target_column': self.original_target_column, 'has_labels': self.original_has_labels,
                'sample_names': list(self.sample_names),
                'is_small_sample': self.training_summary['is_small_sample'],
                'column_names': list(self.original_df.columns),
                'data_types': {c: str(t) for c, t in self.original_df.dtypes.items()},
            }
            with open(os.path.join(self.data0view_dir, "data_info.json"), 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"  Failed to save original data analysis: {e}")

    def create_data_statistics_summary(self, df, data_type="Data"):
        summary = f"""{data_type} Statistics Summary
================================
Timestamp: {self.timestamp}
Shape: {df.shape[0]} rows × {df.shape[1]} cols
Target: {self.original_target_column}
Contains Labels: {self.original_has_labels}
Sample Type: {'小样本' if df.shape[0] <= 20 else '大样本'}

Column Information ({len(df.columns)} cols):
"""
        for i, c in enumerate(df.columns, 1):
            summary += f"  {i:2d}. {c:25s} ({df[c].dtype}): " \
                       f"{df[c].notnull().sum()} non-null, {df[c].isnull().sum()} null\n"
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary += f"\nNumeric Columns Statistics:\n"
            for c in numeric_cols[:10]:
                s = df[c].describe()
                summary += f"\n  {c}:\n"
                for k in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    if k in s:
                        summary += f"    {k}: {s[k]:.4f}\n"
        return summary

    def create_data_visualization(self, df, output_dir, data_type="Data"):
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return
            n = min(9, len(numeric_cols))
            rows = int(np.ceil(n / 3))
            plt.figure(figsize=(15, 10))
            for i, c in enumerate(numeric_cols[:n], 1):
                plt.subplot(rows, 3, i)
                plt.hist(df[c].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'{c}'); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{data_type}_Histograms.png"), dpi=200, bbox_inches='tight')
            plt.close()

            if len(numeric_cols) > 1:
                plt.figure(figsize=(12, 10))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                plt.title(f'{data_type} Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{data_type}_Correlation_Heatmap.png"),
                            dpi=200, bbox_inches='tight')
                plt.close()
                corr.to_csv(os.path.join(output_dir, f"{data_type}_Correlation_Matrix.csv"))
        except Exception as e:
            self.log(f"  Failed to create {data_type} visualizations: {e}")

    def preprocess_new_data(self, df, target_column=None):
        self.log("Preprocessing new data...")
        X_processed, y_processed = self.preprocessor.preprocess_data(df, target_column)
        self.preprocessed_X = X_processed
        self.preprocessed_y = y_processed
        self.log(f"✓ Preprocessed: Features {X_processed.shape}")
        return X_processed, y_processed

    def save_preprocessed_data_analysis(self):
        if self.preprocessed_X is None:
            return
        try:
            self.preprocessed_X.to_csv(os.path.join(self.data1view_dir, "preprocessed_features.csv"), index=False)
            if self.preprocessed_y is not None:
                y_df = pd.DataFrame({self.original_target_column: self.preprocessed_y})
                y_df.to_csv(os.path.join(self.data1view_dir, "preprocessed_target.csv"), index=False)
                combined = pd.concat([self.preprocessed_X, y_df], axis=1)
                combined.to_csv(os.path.join(self.data1view_dir, "preprocessed_combined.csv"), index=False)
                stats = self.create_data_statistics_summary(combined, "Preprocessed Data")
                with open(os.path.join(self.data1view_dir, "preprocessing_statistics.txt"), 'w', encoding='utf-8') as f:
                    f.write(stats)
                self.create_data_visualization(combined, self.data1view_dir, "Preprocessed Data")
            else:
                stats = self.create_data_statistics_summary(self.preprocessed_X, "Preprocessed Features")
                with open(os.path.join(self.data1view_dir, "preprocessing_statistics.txt"), 'w', encoding='utf-8') as f:
                    f.write(stats)
                self.create_data_visualization(self.preprocessed_X, self.data1view_dir, "Preprocessed Features")
            info = {
                'timestamp': self.timestamp,
                'original_samples': int(self.original_df.shape[0]) if self.original_df is not None else 0,
                'preprocessed_samples': int(self.preprocessed_X.shape[0]),
                'preprocessed_features': int(self.preprocessed_X.shape[1]),
                'target_variable': self.original_target_column,
                'has_labels': self.preprocessed_y is not None,
                'sample_names': list(self.sample_names),
                'is_small_sample': self.training_summary['is_small_sample'],
                'skewed_columns': self.preprocessor.skewed_columns,
                'normalization_used': self.preprocessor.means is not None,
            }
            with open(os.path.join(self.data1view_dir, "preprocessing_info.json"), 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"  Failed to save preprocessed data analysis: {e}")

    def run_training(self, X_all, y_all=None, top_features=10, use_pseudo_labels=True, additional_epochs=100):
        if self.continue_training:
            self.log(f"\nStarting continued training (adding {additional_epochs} epochs)...")
        else:
            self.log("\nStarting new training...")

        if self.continue_training and self.existing_model_path:
            self.xgb_learner = XGBoostTransferLearner(
                self.rf_transfer, self.log,
                existing_model_path=self.existing_model_path,
                continue_training=True)
        else:
            self.xgb_learner = XGBoostTransferLearner(self.rf_transfer, self.log)

        self.training_summary['label_source'] = 'measured' if y_all is not None else 'RF_pseudo_labels'
        X_train, y_train = self.xgb_learner.prepare_training_data(
            X_all, y_all, use_pseudo_labels=use_pseudo_labels, n_features=top_features)

        self.log("Training XGBoost model...")
        res = self.xgb_learner.train(X_train, y_train, additional_epochs=additional_epochs)

        self.training_summary['total_batches'] = 1
        self.training_summary['total_samples'] = int(len(X_train))
        self.training_summary['batches'].append({
            'batch_id': 1, 'samples': int(len(X_train)), 'results': res,
            'timestamp': datetime.now().isoformat(), 'continued_training': bool(self.continue_training)})
        return res

    def evaluate_model_small_sample(self, X_test=None, y_test=None):
        self.log("\nEvaluating model performance (small sample)...")
        if self.xgb_learner is None or self.xgb_learner.xgb_model is None:
            raise ValueError("Model not trained")
        if X_test is None or y_test is None:
            X_test, y_test = self.preprocessed_X, self.preprocessed_y
        if y_test is None:
            raise ValueError("No labels available for evaluation")

        X_aligned = X_test.loc[:, self.xgb_learner.selected_features]

        y_pred = self.xgb_learner.xgb_model.predict(X_aligned)
        evaluator = SmallSampleEvaluator(y_test, y_pred, self.sample_names)
        metrics = evaluator.calculate_all_metrics()
        self.training_summary['small_sample_evaluation'] = metrics
        self.training_summary['evaluation_results'] = {
            'traditional': {k: metrics.get(k, 0) for k in
                            ['r2_score', 'rmse', 'mae', 'mse', 'explained_variance', 'mre']},
            'small_sample': {k: metrics.get(k, 0) for k in
                             ['smape', 'median_ape', 'mean_ape', 'max_ape',
                              'rank_correlation', 'accuracy_at_thresholds']},
        }

        report = evaluator.generate_detailed_report()
        with open(os.path.join(self.data2evaluation_dir,
                                f"small_sample_evaluation_report_{self.timestamp}.txt"),
                  'w', encoding='utf-8') as f:
            f.write('Evaluation on the training samples (in-sample fit).\n' + report)
        evaluator.plot_all_visualizations(
            os.path.join(self.data2evaluation_dir,
                         f"small_sample_visualization_{self.timestamp}.png"))

        self.log(f"  R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        self.log(f"  SMAPE={metrics['smape']:.2f}%, MedAPE={metrics['median_ape']:.2f}%")
        for th, acc in metrics['accuracy_at_thresholds'].items():
            self.log(f"    误差在±{int(th.split('_')[1].replace('%',''))}%以内: {acc:.1f}%")
        return metrics

    def evaluate_model(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test, y_test = self.preprocessed_X, self.preprocessed_y
        if y_test is not None and len(y_test) <= 20:
            return self.evaluate_model_small_sample(X_test, y_test)
        return self._evaluate_model_traditional(X_test, y_test)

    def _evaluate_model_traditional(self, X_test=None, y_test=None):
        self.log("\nEvaluating model performance (traditional)...")
        if self.xgb_learner is None or self.xgb_learner.xgb_model is None:
            raise ValueError("Model not trained")
        if X_test is None or y_test is None:
            X_test, y_test = self.preprocessed_X, self.preprocessed_y
        X_aligned = X_test.loc[:, self.xgb_learner.selected_features]
        y_pred = self.xgb_learner.xgb_model.predict(X_aligned)

        y_true_abs = np.abs(y_test).copy()
        y_true_abs[y_true_abs == 0] = 1e-8
        res = {
            'r2_score': float(r2_score(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'mse': float(mean_squared_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'explained_variance': float(explained_variance_score(y_test, y_pred)),
            'mre': float(np.mean(np.abs((y_test - y_pred) / y_true_abs))),
        }
        self.training_summary['evaluation_results'] = res
        self.log(f"  R²={res['r2_score']:.4f}, RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}")
        return res

    def save_partial_dependence_analysis(self):
        if self.xgb_learner is None or self.xgb_learner.xgb_model is None:
            return
        if self.preprocessed_X is None:
            return
        try:
            features = self.xgb_learner.selected_features[:min(5, len(self.xgb_learner.selected_features))]
            if not features:
                return
            X_for_pdp = self.preprocessed_X.loc[:, self.xgb_learner.selected_features]

            fig, axes = plt.subplots(len(features), 2, figsize=(15, 5 * len(features)))
            if len(features) == 1:
                axes = axes.reshape(1, -1)
            for i, feat in enumerate(features):
                try:
                    ax1 = axes[i, 0]
                    PartialDependenceDisplay.from_estimator(
                        self.xgb_learner.xgb_model, X_for_pdp, [feat], ax=ax1,
                        line_kw={"color": "red", "linewidth": 2})
                    ax1.set_title(f'{feat} - PDP (1D)')
                    ax1.grid(True, alpha=0.3)
                    if i < len(features) - 1:
                        f2 = features[i + 1]
                        ax2 = axes[i, 1]
                        PartialDependenceDisplay.from_estimator(
                            self.xgb_learner.xgb_model, X_for_pdp, [(feat, f2)],
                            ax=ax2, contour_kw={"cmap": "viridis"})
                        ax2.set_title(f'{feat} & {f2} - PDP (2D)')
                    else:
                        ax2 = axes[i, 1]
                        ax2.hist(X_for_pdp[feat].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
                        ax2.set_title(f'{feat} - Value Distribution')
                except Exception as e:
                    self.log(f"  PDP for {feat} failed: {e}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.data4PDP_dir, "partial_dependence_plots.png"),
                        dpi=200, bbox_inches='tight')
            plt.close()

            pdp_info = {
                'timestamp': self.timestamp, 'features_analyzed': features,
                'model_type': 'XGBoostRegressor', 'target_variable': self.original_target_column,
                'samples_used': int(len(X_for_pdp)),
                'is_small_sample': self.training_summary['is_small_sample'],
            }
            with open(os.path.join(self.data4PDP_dir, "pdp_analysis_info.json"), 'w', encoding='utf-8') as f:
                json.dump(pdp_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"  Partial dependence analysis failed: {e}")

    def save_results(self):
        self.log("\nSaving training results...")
        self.training_summary['end_time'] = datetime.now().isoformat()

        if self.xgb_learner and self.xgb_learner.xgb_model:
            self.training_summary['final_model'] = {
                'model_type': 'XGBoostRegressor',
                'n_estimators': self.xgb_learner.xgb_model.get_booster().num_boosted_rounds(),
                'selected_features': list(self.xgb_learner.selected_features) if self.xgb_learner.selected_features else [],
                'feature_count': len(self.xgb_learner.selected_features) if self.xgb_learner.selected_features else 0,
                'target_variable': self.original_target_column,
                'is_small_sample': self.training_summary['is_small_sample'],
            }

        self.save_original_data_analysis()
        self.save_preprocessed_data_analysis()

        if self.xgb_learner and self.xgb_learner.training_history:
            results_list = []
            for i, h in enumerate(self.xgb_learner.training_history, 1):
                if 'mse' not in h:
                    h['mse'] = h.get('rmse', 0) ** 2
                results_list.append({
                    'Iteration': i, 'R2_Score': h.get('r2_score', 0),
                    'MAE': h.get('mae', 0), 'MSE': h.get('mse', 0),
                    'RMSE': h.get('rmse', 0), 'MRE': h.get('mre', 0),
                    'Explained_Variance': h.get('explained_variance', 0),
                    'Continued_Training': h.get('continued_training', False),
                    'Additional_Epochs': h.get('additional_epochs', 0),
                })
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(os.path.join(self.data2evaluation_dir, "model_evaluation_results.csv"), index=False)

            if self.training_summary.get('small_sample_evaluation'):
                ss = self.training_summary['small_sample_evaluation']
                flat = {k: v for k, v in ss.items() if not isinstance(v, (dict, list))}
                for k, v in ss.get('accuracy_at_thresholds', {}).items():
                    flat[k] = v
                pd.DataFrame([flat]).to_csv(
                    os.path.join(self.data2evaluation_dir, "small_sample_evaluation_results.csv"), index=False)

            summary = self.create_evaluation_summary(results_df)
            with open(os.path.join(self.data2evaluation_dir, "evaluation_summary.txt"), 'w', encoding='utf-8') as f:
                f.write(summary)
            self.create_evaluation_plots(results_df)

        if self.xgb_learner and self.xgb_learner.xgb_model:
            self.save_feature_importance_analysis()

        self.save_partial_dependence_analysis()

        if self.xgb_learner and self.xgb_learner.xgb_model:
            model_path = os.path.join(self.bestmodel_dir, f"XGBoost_model_{self.timestamp}.pkl")
            joblib.dump(self.xgb_learner.xgb_model, model_path)
            ss = self.training_summary.get('small_sample_evaluation', {}) or {}
            model_info = {
                'timestamp': self.timestamp, 'model_type': 'XGBoostRegressor',
                'target_variable': self.original_target_column,
                'n_features': len(self.xgb_learner.selected_features) if self.xgb_learner.selected_features else 0,
                'n_samples': self.training_summary['total_samples'],
                'r2_score': self.xgb_learner.training_history[-1]['r2_score'] if self.xgb_learner.training_history else 0,
                'rmse': self.xgb_learner.training_history[-1]['rmse'] if self.xgb_learner.training_history else 0,
                'mae': self.xgb_learner.training_history[-1]['mae'] if self.xgb_learner.training_history else 0,
                'smape': ss.get('smape', None), 'median_ape': ss.get('median_ape', None),
                'is_small_sample': self.training_summary['is_small_sample'],
                'sample_names': list(self.sample_names),
                'parameters': self.xgb_learner.xgb_model.get_params(),
                'boosting_rounds': self.xgb_learner.xgb_model.get_booster().num_boosted_rounds(),
                'feature_order': list(self.xgb_learner.selected_features),
                'input_preprocessing': self.preprocessor.to_state(),
                'input_preprocessing_provenance': self.training_summary['input_preprocessing_provenance'],
                'historical_preprocessing_verified': self.training_summary['historical_preprocessing_verified'],
                'label_source': self.training_summary['label_source'],
                'evaluation_scope': 'in_sample_training_fit',
                'continued_from': self.training_summary.get('load_existing_model'),
                'original_rf_timestamp': self.rf_timestamp,
                'continue_training': self.continue_training,
            }
            with open(os.path.join(self.bestmodel_dir, f"model_info_{self.timestamp}.json"),
                      'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
            self.log(f"✓ Model saved: {model_path}")

        with open(os.path.join(self.data2evaluation_dir, 'training_summary.json'),
                  'w', encoding='utf-8') as f:
            json.dump(self.training_summary, f, indent=2, ensure_ascii=False, default=str)
        for d in [self.data3FIA_dir, self.bestmodel_dir]:
            with open(os.path.join(d, 'training_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(self.training_summary, f, indent=2, ensure_ascii=False, default=str)

        self.log("✓ All results saved")
        self.log(f"  data0view : {self.data0view_dir}")
        self.log(f"  data1view : {self.data1view_dir}")
        self.log(f"  data2eval : {self.data2evaluation_dir}")
        self.log(f"  data3FIA  : {self.data3FIA_dir}")
        self.log(f"  data4PDP  : {self.data4PDP_dir}")
        self.log(f"  bestmodel : {self.bestmodel_dir}")
        return self.training_summary

    def create_evaluation_summary(self, results_df):
        if results_df.empty:
            return "No evaluation data"
        continued = bool(any(results_df['Continued_Training']))
        summary = f"""
XGBoost Transfer Learning Evaluation Summary
================================
RF Source Timestamp   : {self.rf_timestamp}
XGBoost Timestamp     : {self.timestamp}
Sample Type           : {'小样本' if self.training_summary['is_small_sample'] else '大样本'}
"""
        if continued:
            summary += f"Continued From        : {self.training_summary.get('load_existing_model', 'N/A')}\n"
        summary += f"""Target Variable       : {self.original_target_column}
Start Time            : {self.training_summary['start_time']}
Samples               : {self.training_summary['total_samples']}
Features              : {len(self.xgb_learner.selected_features) if self.xgb_learner and self.xgb_learner.selected_features else 0}

In-sample training metrics (batch averages):
  R²            : {results_df['R2_Score'].mean():.4f}
  RMSE          : {results_df['RMSE'].mean():.4f}
  MAE           : {results_df['MAE'].mean():.4f}
  MRE           : {results_df['MRE'].mean():.4f}
  Explained Var : {results_df['Explained_Variance'].mean():.4f}
"""
        ss = self.training_summary.get('small_sample_evaluation', {}) or {}
        if ss:
            smape = ss.get('smape', 'N/A')
            medape = ss.get('median_ape', 'N/A')
            meanape = ss.get('mean_ape', 'N/A')
            rank = ss.get('rank_correlation', 'N/A')
            fmt = lambda v, p=2: f"{v:.{p}f}" if isinstance(v, (int, float)) and not np.isnan(v) else "N/A"
            summary += f"""
Small Sample Metrics:
  SMAPE             : {fmt(smape)}%
  Median APE        : {fmt(medape)}%
  Mean APE          : {fmt(meanape)}%
  Rank Correlation  : {fmt(rank, 3)}

Accuracy at Thresholds:
"""
            for th, acc in ss.get('accuracy_at_thresholds', {}).items():
                thv = int(th.split('_')[1].replace('%', ''))
                summary += f"  ±{thv}%: {acc:.1f}%\n"

        m = self.xgb_learner.xgb_model if self.xgb_learner and self.xgb_learner.xgb_model else None
        summary += f"""
Model Parameters:
  Boosting rounds: {m.get_booster().num_boosted_rounds() if m else 'N/A'}
  Max Depth     : {m.max_depth if m else 'N/A'}
  Learning Rate : {m.learning_rate if m else 'N/A'}

Output Directories:
  data0view : {self.data0view_dir}
  data1view : {self.data1view_dir}
  data2eval : {self.data2evaluation_dir}
  data3FIA  : {self.data3FIA_dir}
  data4PDP  : {self.data4PDP_dir}
  bestmodel : {self.bestmodel_dir}
"""
        return summary

    def create_evaluation_plots(self, results_df):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes[0, 0].plot(results_df['Iteration'], results_df['R2_Score'], 'bo-', linewidth=2)
            axes[0, 0].set_title('R² Score'); axes[0, 0].grid(True, alpha=0.3)
            axes[0, 1].plot(results_df['Iteration'], results_df['RMSE'], 'ro-', linewidth=2)
            axes[0, 1].set_title('RMSE'); axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].plot(results_df['Iteration'], results_df['MAE'], 'go-', linewidth=2)
            axes[1, 0].set_title('MAE'); axes[1, 0].grid(True, alpha=0.3)
            axes[1, 1].bar(results_df['Iteration'], results_df['Explained_Variance'])
            axes[1, 1].set_title('Explained Variance'); axes[1, 1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data2evaluation_dir, 'performance_metrics.png'),
                        dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.log(f"  Failed to create evaluation plots: {e}")

    def save_feature_importance_analysis(self):
        try:
            if not self.xgb_learner or not self.xgb_learner.xgb_model:
                return
            names = self.xgb_learner.selected_features
            scores = self.xgb_learner.xgb_model.feature_importances_
            min_len = min(len(names), len(scores))
            items = sorted([(names[i], float(scores[i])) for i in range(min_len)],
                           key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(items, columns=['Feature_Name', 'Importance_Score'])
            total = df['Importance_Score'].sum()
            df['Importance_Percent'] = (df['Importance_Score'] / total * 100) if total > 0 else 0
            df['Cumulative_Importance'] = df['Importance_Percent'].cumsum()
            df['Rank'] = range(1, len(df) + 1)
            df.to_csv(os.path.join(self.data3FIA_dir, "feature_importance_ranking.csv"), index=False)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            top_n = min(15, len(df))
            plt.barh(range(top_n), df['Importance_Score'].head(top_n)[::-1])
            plt.yticks(range(top_n), df['Feature_Name'].head(top_n)[::-1])
            plt.xlabel('Importance'); plt.title(f'Top {top_n} Feature Importance')
            plt.subplot(2, 1, 2)
            plt.plot(range(1, len(df) + 1), df['Cumulative_Importance'], 'b-', linewidth=2)
            plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% Threshold')
            plt.xlabel('Number of Features'); plt.ylabel('Cumulative Importance (%)')
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data3FIA_dir, "feature_importance.png"),
                        dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.log(f"  Feature importance analysis failed: {e}")


# ====================== GUI ======================
def on_ui_thread(method):
    """Keep native Tk operations on the main thread, including on macOS."""
    @wraps(method)
    def dispatch(self, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return method(self, *args, **kwargs)
        self.ui_events.put((method, args, kwargs))
    return dispatch


class XGBoostTransferGUI:
    def _drain_events(self):
        while not self.window_destroyed:
            try:
                method, args, kwargs = self.ui_events.get_nowait()
            except Empty:
                break
            method(self, *args, **kwargs)
        if not self.window_destroyed:
            self.root.after(75, self._drain_events)

    @on_ui_thread
    def _training_error(self, message):
        if not self.window_destroyed:
            messagebox.showerror("Training Failed", message, parent=self.root)

    @on_ui_thread
    def _finish_training(self):
        self.is_running = False
        if not self.window_destroyed:
            self.run_button.configure(state='normal')

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("XGBoost Transfer Learning System with Small Sample Optimization")
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1220, screen_w - 48)}x{min(800, screen_h - 96)}")
        self.root.minsize(min(420, screen_w - 48), min(320, screen_h - 96))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        atexit.register(self.safe_cleanup)

        self.colors = {
            'primary': '#2c3e50', 'secondary': '#3498db', 'accent': '#e74c3c',
            'success': '#27ae60', 'warning': '#f39c12', 'light': '#ecf0f1',
            'background': '#f8f9fa', 'text': '#2c3e50',
        }
        self.root.configure(bg=self.colors['background'])

        self.rf_timestamp = tk.StringVar()
        self.new_data_path = tk.StringVar()
        self.top_features = tk.IntVar(value=7)
        self.use_pseudo_labels = tk.BooleanVar(value=False)
        self.continue_training = tk.BooleanVar(value=False)
        self.existing_model_path = tk.StringVar()
        self.additional_epochs = tk.IntVar(value=50)
        self.enable_small_sample_eval = tk.BooleanVar(value=True)
        self.sample_names_var = tk.StringVar()

        self.is_running = False
        self.manager = None
        self.rf_transfer = None
        self.data_preprocessor = None
        self.window_destroyed = False
        self.ui_events = SimpleQueue()
        self.root.after(75, self._drain_events)

        self.setup_scrollable_interface()
        self.toggle_training_mode()
        self.scan_available_models()

    def setup_scrollable_interface(self):
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(container, highlightthickness=0, borderwidth=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        ybar.grid(row=0, column=1, sticky="ns")
        self.xbar = ttk.Scrollbar(container, orient="horizontal", command=self.canvas.xview)
        self.xbar.grid(row=1, column=0, sticky="ew")
        self.xbar.grid_remove()
        self.canvas.configure(yscrollcommand=ybar.set, xscrollcommand=self.xbar.set)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.setup_styles()
        self.create_widgets()
        self._layout_pending = False
        self.canvas.bind("<Configure>", self._schedule_layout)
        self.scrollable_frame.bind("<Configure>", self._schedule_layout)
        self.config_panel.bind("<Configure>", self._schedule_layout)
        self.root.bind("<<ThemeChanged>>", self._schedule_layout)
        self._schedule_layout()

        def scroll_canvas(event):
            if event.widget.winfo_toplevel() != self.root or isinstance(event.widget, tk.Text):
                return
            if event.num == 4 or event.num == 5:
                steps = -1 if event.num == 4 else 1
            else:
                steps = -event.delta if sys.platform == "darwin" else -int(event.delta / 120)
            self.canvas.yview_scroll(steps, "units")
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind(sequence, scroll_canvas, add="+")

    def _schedule_layout(self, event=None):
        if not self._layout_pending and not self.window_destroyed:
            self._layout_pending = True
            self.root.after_idle(self._fit_layout)

    def _fit_layout(self):
        self._layout_pending = False
        if self.window_destroyed:
            return
        viewport = self.canvas.winfo_width()
        if viewport <= 1:
            return
        # Tk measures requested sizes in the current platform font and display scale.
        margin, gap = self.page_padding * 2, self.panel_gap
        config_width = self.config_panel.winfo_reqwidth()
        preview_width = self.preview_panel.winfo_reqwidth()
        wide = viewport >= margin + config_width + gap + preview_width
        if self._wide_layout != wide:
            self._wide_layout = wide
            self.page.columnconfigure(0, weight=0 if wide else 1)
            self.page.columnconfigure(1, weight=1 if wide else 0)
            if wide:
                self.preview_panel.grid(row=2, column=1, padx=(gap, 0), pady=5, sticky="nsew")
            else:
                self.preview_panel.grid(row=3, column=0, padx=0, pady=5, sticky="nsew")
        # At very small widths, scrolling keeps complete buttons and fields reachable.
        minimum = max(config_width, self.actions.winfo_reqwidth(), preview_width) + margin
        content_width = max(viewport, minimum)
        if int(float(self.canvas.itemcget(self.canvas_window, "width"))) != content_width:
            self.canvas.itemconfigure(self.canvas_window, width=content_width)
        if content_width > viewport:
            self.xbar.grid()
        else:
            self.xbar.grid_remove()
            if self.canvas.xview()[0]:
                self.canvas.xview_moveto(0)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def setup_styles(self):
        self.style = ttk.Style(self.root)
        base = tkfont.nametofont("TkDefaultFont")
        self.title_font = base.copy()
        base_size = int(base.cget("size"))
        self.title_font.configure(size=base_size + (4 if base_size > 0 else -4), weight="bold")
        self.section_font = base.copy()
        self.section_font.configure(weight="bold")
        self.style.configure("Title.TLabel", font=self.title_font)
        self.style.configure("Section.TLabel", font=self.section_font)

    def create_widgets(self):
        self.page_padding = 12
        self.panel_gap = 12
        self._wide_layout = None
        main = self.page = ttk.Frame(self.scrollable_frame, padding=self.page_padding)
        main.grid(row=0, column=0, sticky="nsew")
        self.scrollable_frame.columnconfigure(0, weight=1)
        self.scrollable_frame.rowconfigure(0, weight=1)
        title = ttk.Label(main, text="XGBoost Transfer Learning System with Small Sample Optimization",
                          style="Title.TLabel", wraplength=520, justify="left")
        title.grid(row=0, column=0, columnspan=2, pady=(0, 8), sticky="w")
        main.bind("<Configure>", lambda e: title.configure(wraplength=max(200, e.width - self.page_padding * 2)))
        ttk.Separator(main, orient="horizontal").grid(row=1, column=0, columnspan=2, pady=(0, 5), sticky="ew")

        cfg = self.config_panel = ttk.LabelFrame(main, text="Configuration", padding=10)
        cfg.grid(row=2, column=0, pady=5, sticky="nsew")
        cfg.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(cfg, text="Training Mode:", style='Section.TLabel').grid(row=r, column=0, sticky=tk.W, pady=5)
        mf = ttk.Frame(cfg); mf.grid(row=r, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Radiobutton(mf, text="New Training (from RF)", variable=self.continue_training, value=False,
                        command=self.toggle_training_mode).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mf, text="Continue Training (from existing XGBoost)", variable=self.continue_training, value=True,
                        command=self.toggle_training_mode).grid(row=1, column=0, sticky="w")

        r += 1
        self.rf_label = ttk.Label(cfg, text="RF Model:", style='Section.TLabel')
        self.rf_label.grid(row=r, column=0, sticky=tk.W, pady=5)
        rf_frame = ttk.Frame(cfg); rf_frame.grid(row=r, column=1, sticky=(tk.W, tk.E), pady=5)
        rf_frame.columnconfigure(0, weight=1)
        self.model_combo = ttk.Combobox(rf_frame, textvariable=self.rf_timestamp, state="readonly", width=18)
        self.model_combo.grid(row=0, column=0, sticky="ew")
        rf_buttons = ttk.Frame(rf_frame)
        rf_buttons.grid(row=1, column=0, pady=(3, 0), sticky="w")
        ttk.Button(rf_buttons, text="RF Folder…", command=self.browse_rf_folder).grid(row=0, column=0)
        ttk.Button(rf_buttons, text="Refresh", command=self.scan_available_models).grid(row=0, column=1, padx=4)
        ttk.Button(rf_buttons, text="Preview", command=self.load_rf_model_preview).grid(row=0, column=2)

        r += 1
        self.existing_model_label = ttk.Label(cfg, text="Existing XGBoost:", style='Section.TLabel')
        self.existing_model_label.grid(row=r, column=0, sticky=tk.W, pady=5)
        ex_frame = ttk.Frame(cfg); ex_frame.grid(row=r, column=1, sticky=(tk.W, tk.E), pady=5)
        ex_frame.columnconfigure(0, weight=1)
        self.existing_entry = ttk.Entry(ex_frame, textvariable=self.existing_model_path, width=12,
                                        state="disabled")
        self.existing_entry.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.existing_browse_btn = ttk.Button(ex_frame, text="Browse...",
                                              command=self.browse_existing_model, state='disabled')
        self.existing_browse_btn.grid(row=0, column=1, padx=5)
        self.existing_preview_btn = ttk.Button(ex_frame, text="Preview",
                                               command=self.load_existing_model_preview, state='disabled')
        self.existing_preview_btn.grid(row=0, column=2, padx=5)

        r += 1
        ttk.Label(cfg, text="New Data File:", style='Section.TLabel').grid(row=r, column=0, sticky=tk.W, pady=5)
        df_frame = ttk.Frame(cfg); df_frame.grid(row=r, column=1, sticky=(tk.W, tk.E), pady=5)
        df_frame.columnconfigure(0, weight=1)
        ttk.Entry(df_frame, textvariable=self.new_data_path, width=12).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ttk.Button(df_frame, text="Browse...", command=self.browse_data_file).grid(row=0, column=1, padx=5)
        ttk.Button(df_frame, text="Preview", command=self.load_data_preview).grid(row=0, column=2, padx=5)

        r += 1
        ttk.Label(cfg, text="Small Sample Options:", style='Section.TLabel').grid(row=r, column=0, sticky=tk.W, pady=5)
        ss_frame = ttk.Frame(cfg); ss_frame.grid(row=r, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Checkbutton(ss_frame, text="Enable Small Sample Evaluation",
                        variable=self.enable_small_sample_eval).grid(row=0, column=0, sticky=tk.W)

        r += 1
        ttk.Label(cfg, text="Sample Names\n(comma-separated):",
                  style='Section.TLabel').grid(row=r, column=0, sticky=tk.W, pady=5)
        sn_frame = ttk.Frame(cfg); sn_frame.grid(row=r, column=1, sticky=(tk.W, tk.E), pady=5)
        sn_frame.columnconfigure(0, weight=1)
        ttk.Entry(sn_frame, textvariable=self.sample_names_var, width=16).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(sn_frame, text="Clear",
                   command=lambda: self.sample_names_var.set("")).grid(row=0, column=1, padx=5)

        r += 1
        ttk.Label(cfg, text="Parameters:", style='Section.TLabel').grid(row=r, column=0, sticky=tk.W, pady=5)
        pf = ttk.Frame(cfg); pf.grid(row=r, column=1, sticky=(tk.W, tk.E), pady=5)
        self.top_features_label = ttk.Label(pf, text="Top Features:")
        self.top_features_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.top_features_spinbox = ttk.Spinbox(pf, from_=1, to=7, textvariable=self.top_features,
                                                width=8)
        self.top_features_spinbox.grid(row=0, column=1, padx=(0, 15))
        ttk.Label(pf, text="Batch: full dataset").grid(row=0, column=2, sticky="w")

        self.additional_epochs_label = ttk.Label(pf, text="Add Epochs:")
        self.additional_epochs_label.grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(3, 0))
        self.additional_epochs_spinbox = ttk.Spinbox(pf, from_=10, to=500, textvariable=self.additional_epochs,
                                                     width=8, state='disabled')
        self.additional_epochs_spinbox.grid(row=1, column=1, padx=(0, 15), pady=(3, 0))
        ttk.Checkbutton(ss_frame, text="Pseudo-Labels",
                        variable=self.use_pseudo_labels).grid(row=1, column=0, sticky="w")

        r += 1
        ctrl = ttk.Frame(cfg); ctrl.grid(row=r, column=0, columnspan=2, pady=15, sticky=(tk.W, tk.E))
        self.run_button = ttk.Button(ctrl, text="Start Training", command=self.start_training)
        self.run_button.grid(row=0, column=0, padx=5)
        ttk.Button(ctrl, text="Open Output Dir", command=self.open_output_dir).grid(row=0, column=1, padx=5)
        ttk.Button(ctrl, text="View History", command=self.view_history).grid(row=0, column=2, padx=5)

        r += 1
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(cfg, variable=self.progress_var, length=120, mode='determinate')
        self.progress_bar.grid(row=r, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        self.status_label = ttk.Label(cfg, text="Ready", wraplength=360)
        self.status_label.grid(row=r + 1, column=0, columnspan=2, pady=5)

        pv = self.preview_panel = ttk.LabelFrame(main, text="Preview", padding=8)
        pv.rowconfigure(0, weight=1); pv.columnconfigure(0, weight=1)
        self.preview_text = scrolledtext.ScrolledText(pv, height=9, width=28, wrap='word', font='TkTextFont',
                                                      bg='white', fg=self.colors['text'])
        self.preview_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        log = ttk.LabelFrame(main, text="Log Output", padding="10")
        log.grid(row=4, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        log.rowconfigure(0, weight=1); log.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log, height=8, width=1, wrap='word', font='TkTextFont',
                                                  bg='white', fg=self.colors['text'])
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        bf = self.actions = ttk.Frame(main); bf.grid(row=5, column=0, columnspan=2, pady=10)
        ttk.Button(bf, text="Clear Log", command=self.clear_log).grid(row=0, column=0, padx=5)
        ttk.Button(bf, text="Copy Log", command=self.copy_log).grid(row=0, column=1, padx=5)
        ttk.Button(bf, text="Clear Preview", command=self.clear_preview).grid(row=0, column=2, padx=5)
        ttk.Button(bf, text="Exit", command=self.quit_program).grid(row=0, column=3, padx=5)

    @on_ui_thread
    def log(self, msg):
        if self.window_destroyed:
            return
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{ts}] {msg}\n")
            self.log_text.see(tk.END)
        except Exception:
            pass

    @on_ui_thread
    def preview(self, msg, clear=False):
        if self.window_destroyed:
            return
        try:
            if clear:
                self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert(tk.END, msg + "\n")
            self.preview_text.see(tk.END)
        except Exception:
            pass

    @on_ui_thread
    def update_status(self, msg):
        if self.window_destroyed:
            return
        try:
            self.status_label.configure(text=msg)
        except Exception:
            pass

    @on_ui_thread
    def update_progress(self, v):
        if self.window_destroyed:
            return
        try:
            self.progress_var.set(v)
        except Exception:
            pass

    def clear_log(self):
        try:
            self.log_text.delete("1.0", tk.END)
        except Exception:
            pass

    def clear_preview(self):
        try:
            self.preview_text.delete("1.0", tk.END)
        except Exception:
            pass

    def copy_log(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.log_text.get("1.0", tk.END))
        except Exception:
            pass

    def toggle_training_mode(self):
        if self.continue_training.get():
            self.model_combo.configure(state='disabled')
            self.rf_label.configure(state='disabled')
            self.top_features_label.configure(state='disabled')
            self.top_features_spinbox.configure(state='disabled')
            self.existing_model_path.set('')
            self.existing_entry.configure(state='readonly')
            self.existing_browse_btn.configure(state='normal')
            self.existing_preview_btn.configure(state='normal')
            self.existing_model_label.configure(state='normal')
            self.additional_epochs_label.configure(state='normal')
            self.additional_epochs_spinbox.configure(state='normal')
        else:
            self.model_combo.configure(state='readonly')
            self.rf_label.configure(state='normal')
            self.top_features_label.configure(state='normal')
            self.top_features_spinbox.configure(state='normal')
            self.existing_model_path.set('')
            self.existing_entry.configure(state='disabled')
            self.existing_browse_btn.configure(state='disabled')
            self.existing_preview_btn.configure(state='disabled')
            self.existing_model_label.configure(state='disabled')
            self.additional_epochs_label.configure(state='disabled')
            self.additional_epochs_spinbox.configure(state='disabled')

    def browse_rf_folder(self):
        folder = filedialog.askdirectory(title="Select an extracted project or RF_for_XGBoost folder")
        if not folder:
            return
        exports = Config.find_rf_exports(folder)
        if not exports:
            messagebox.showerror("No RF export found", "Choose an extracted RF_for_XGBoost folder "
                                 "containing feature_importance.json and model_parameters.json.")
            return
        Config.RF_EXPORTS = exports
        self.rf_timestamp.set(sorted(exports)[-1])
        self.rf_transfer = None
        self.scan_available_models()

    def scan_available_models(self):
        if not Config.RF_EXPORTS:
            for directory in (Config.PROJECT_ROOT, Config.PARENT_ROOT, Config.TRANSFER_LEARNING_DIR):
                Config.RF_EXPORTS.update(Config.find_rf_exports(directory))
                if Config.RF_EXPORTS:
                    break
        timestamps = sorted(Config.RF_EXPORTS, reverse=True)
        self.model_combo["values"] = timestamps
        if self.rf_timestamp.get() not in timestamps:
            self.rf_timestamp.set(timestamps[0] if timestamps else "")
        if timestamps:
            self.log(f"Found {len(timestamps)} RF export(s).")
        else:
            hint = "No RF model found. Click RF Folder… and select the extracted rf_resources folder."
            self.log(hint)
            self.preview(hint, clear=True)

    def browse_existing_model(self):
        p = filedialog.askopenfilename(title="Select XGBoost Model",
                                       filetypes=[("Model Files", "*.pkl"), ("All", "*.*")],
                                       initialdir=Config.BESTMODEL_BASE_DIR if os.path.exists(Config.BESTMODEL_BASE_DIR) else Config.PROJECT_ROOT)
        if p:
            self.existing_model_path.set(p)
            self.log(f"Selected model: {os.path.basename(p)}")

    def load_existing_model_preview(self):
        p = self.existing_model_path.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        try:
            match = re.search(r'XGBoost_model_(\d{8}_\d{6}(?:_\d{6})?)\.pkl', os.path.basename(p))
            if not match:
                self.preview("Invalid model filename", clear=True); return
            ts = match.group(1)
            d = os.path.dirname(p)
            info_file = os.path.join(d, f"model_info_{ts}.json")
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                smape_val = info.get('smape', 'N/A')
                smape_str = f"{smape_val:.2f}" if isinstance(smape_val, (int, float)) else str(smape_val)
                txt = f"""Existing XGBoost Model:
Timestamp: {ts}
Target: {info.get('target_variable', 'N/A')}
Features: {info.get('n_features', 'N/A')}
Samples: {info.get('n_samples', 'N/A')}
R²: {info.get('r2_score', 'N/A')}
RMSE: {info.get('rmse', 'N/A')}
MAE: {info.get('mae', 'N/A')}
SMAPE: {smape_str}%
Is Small Sample: {info.get('is_small_sample', 'N/A')}
Original RF: {info.get('original_rf_timestamp', 'N/A')}

Parameters:
"""
                for k, v in info.get('parameters', {}).items():
                    txt += f"  {k}: {v}\n"
                self.preview(txt, clear=True)
            else:
                self.preview(f"Model info not found: {info_file}", clear=True)
        except Exception as e:
            self.preview(f"Failed to load model info: {e}", clear=True)

    def browse_data_file(self):
        p = filedialog.askopenfilename(title="Select Data File",
                                       filetypes=[("CSV", "*.csv"), ("Excel", ("*.xlsx", "*.xls")), ("All", "*.*")],
                                       initialdir=Config.PROJECT_ROOT)
        if p:
            if validate_file_path(p):
                self.new_data_path.set(p)
                self.log(f"Selected data: {os.path.basename(p)}")
            else:
                messagebox.showerror("Error", "Invalid file type, size, or path")

    def open_output_dir(self):
        if os.path.exists(Config.PROJECT_ROOT):
            try:
                if sys.platform == "win32":
                    os.startfile(Config.PROJECT_ROOT)
                elif sys.platform == "darwin":
                    import subprocess; subprocess.call(["open", Config.PROJECT_ROOT])
                else:
                    import subprocess; subprocess.call(["xdg-open", Config.PROJECT_ROOT])
            except Exception as e:
                messagebox.showinfo("Info", f"Cannot open directory: {e}")

    def view_history(self):
        try:
            v = tk.Toplevel(self.root)
            v.title("Training History")
            v.geometry("700x500")
            v.protocol("WM_DELETE_WINDOW", v.destroy)
            txt = scrolledtext.ScrolledText(v, font='TkTextFont')
            txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            dirs = [
                (Config.DATA0VIEW_BASE_DIR, "Original"),
                (Config.DATA1VIEW_BASE_DIR, "Preprocessed"),
                (Config.DATA2EVALUATION_BASE_DIR, "Evaluation"),
                (Config.DATA3FIA_BASE_DIR, "FeatureImportance"),
                (Config.DATA4PDP_BASE_DIR, "PDP"),
                (Config.BESTMODEL_BASE_DIR, "Model"),
            ]
            records = {}
            for base, typ in dirs:
                if os.path.exists(base):
                    for item in os.listdir(base):
                        ip = os.path.join(base, item)
                        if os.path.isdir(ip):
                            records.setdefault(item, []).append(typ)
            if not records:
                txt.insert(tk.END, "No training history records")
            else:
                for ts in sorted(records.keys(), reverse=True)[:20]:
                    txt.insert(tk.END, f"Timestamp: {ts}\n")
                    txt.insert(tk.END, f"  Contains: {', '.join(records[ts])}\n")
                    # 尝试读取 summary
                    for base, _ in dirs:
                        sp = os.path.join(base, ts, "training_summary.json")
                        if os.path.exists(sp):
                            try:
                                with open(sp, 'r', encoding='utf-8') as f:
                                    s = json.load(f)
                                txt.insert(tk.END, f"  RF: {s.get('rf_timestamp','N/A')}, "
                                                   f"Samples: {s.get('total_samples','N/A')}, "
                                                   f"SmallSample: {s.get('is_small_sample','N/A')}\n")
                            except Exception:
                                pass
                            break
                    txt.insert(tk.END, "\n")
            txt.configure(state='disabled')
        except Exception as e:
            self.log(f"Failed to view history: {e}")

    def load_rf_model_preview(self):
        ts = self.rf_timestamp.get()
        if not ts:
            messagebox.showerror("Error", "Please select an RF model"); return
        try:
            self.rf_transfer = RFKnowledgeTransfer(ts, load_rf_model=True)
            ok, msg = self.rf_transfer.load_rf_resources()
            if ok:
                self.preview(self.rf_transfer.get_model_info(), clear=True)
                self.log(f"✓ RF loaded: {ts}")
                self.update_recommended_params()
            else:
                self.preview(f"RF load failed: {msg}", clear=True)
                self.log(f"✗ RF load failed: {msg}")
        except Exception as e:
            self.preview(f"RF load failed: {e}", clear=True)
            self.log(f"✗ RF load failed: {e}")

    def load_data_preview(self):
        p = self.new_data_path.get()
        if not p or not os.path.exists(p):
            messagebox.showerror("Error", "Please select a valid data file"); return
        try:
            df = read_data_file(p, self.log)
            rf_ts = self.rf_timestamp.get()
            if self.continue_training.get() and self.existing_model_path.get():
                m = re.search(r'XGBoost_model_(\d{8}_\d{6}(?:_\d{6})?)\.pkl', os.path.basename(self.existing_model_path.get()))
                if m:
                    ts = m.group(1)
                    d = os.path.dirname(self.existing_model_path.get())
                    info_f = os.path.join(d, f"model_info_{ts}.json")
                    if os.path.exists(info_f):
                        with open(info_f, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        rf_ts = info.get('original_rf_timestamp', ts)
            if rf_ts:
                self.data_preprocessor = DataPreprocessor(rf_ts)
            if self.data_preprocessor:
                self.preview(self.data_preprocessor.get_data_info(df), clear=True)
            else:
                self.preview(df.head().to_string(), clear=True)
            self.log(f"✓ Data loaded: {df.shape[0]}×{df.shape[1]}")
        except Exception as e:
            self.preview(f"Data load failed: {e}", clear=True)

    def update_recommended_params(self):
        if self.rf_transfer and self.rf_transfer.original_features:
            count = len(self.rf_transfer.original_features)
            self.top_features_spinbox.configure(to=count)
            try:
                requested = self.top_features.get()
            except tk.TclError:
                requested = count
            self.top_features.set(max(1, min(requested, count)))

    def start_training(self):
        if self.is_running:
            messagebox.showwarning("Warning", "Training in progress"); return
        if self.continue_training.get():
            if not self.existing_model_path.get() or not os.path.exists(self.existing_model_path.get()):
                messagebox.showerror("Error", "Select a valid existing model"); return
            if not re.search(r'XGBoost_model_(\d{8}_\d{6}(?:_\d{6})?)\.pkl',
                             os.path.basename(self.existing_model_path.get())):
                messagebox.showerror("Error", "Invalid model filename"); return
        else:
            if not self.rf_timestamp.get():
                messagebox.showerror("Error", "Select an RF model"); return
        if not self.new_data_path.get() or not os.path.exists(self.new_data_path.get()):
            messagebox.showerror("Error", "Select a valid data file"); return

        sample_names = None
        if self.sample_names_var.get():
            sample_names = [s.strip() for s in self.sample_names_var.get().split(',') if s.strip()]

        try:
            options = dict(
                continuing=self.continue_training.get(),
                previous_path=self.existing_model_path.get(),
                rf_timestamp=self.rf_timestamp.get(),
                data_path=self.new_data_path.get(),
                top_features=self.top_features.get(),
                extra_rounds=self.additional_epochs.get() if self.continue_training.get() else 0,
                pseudo=self.use_pseudo_labels.get(),
                small_sample=self.enable_small_sample_eval.get(),
            )
            if not options["continuing"] and options["top_features"] < 1:
                raise ValueError("Top Features must be at least 1.")
            if options["continuing"] and options["extra_rounds"] < 1:
                raise ValueError("Additional rounds must be at least 1.")
        except (tk.TclError, ValueError) as err:
            messagebox.showerror("Training parameters", str(err), parent=self.root)
            return
        self.run_button.configure(state='disabled')
        self.is_running = True
        threading.Thread(target=self.run_training, args=(options, sample_names), daemon=True).start()

    def run_training(self, options, sample_names=None):
        try:
            self.log("\n" + "=" * 70)
            continue_tr = options["continuing"]
            additional_epochs = options["extra_rounds"] if continue_tr else 0
            if continue_tr:
                self.log("XGBoost Continue Training Started")
                match = re.search(r'XGBoost_model_(\d{8}_\d{6}(?:_\d{6})?)\.pkl',
                                  os.path.basename(options["previous_path"]))
                model_ts = match.group(1)
                rf_ts = model_ts
                d = os.path.dirname(options["previous_path"])
                info_f = os.path.join(d, f"model_info_{model_ts}.json")
                if os.path.exists(info_f):
                    try:
                        with open(info_f, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        rf_ts = info.get('original_rf_timestamp', model_ts)
                    except Exception:
                        pass
            else:
                self.log("XGBoost New Training Started")
                rf_ts = options["rf_timestamp"]
                model_ts = None
            self.log("=" * 70)

            top_f = options["top_features"]
            use_pseudo = options["pseudo"]
            enable_ss = options["small_sample"]

            self.update_status("Initializing...")
            self.update_progress(10)

            if continue_tr:
                self.manager = TrainingManager(rf_ts, self.log, load_existing_model=options["previous_path"],
                                               continue_training=True)
            else:
                self.manager = TrainingManager(rf_ts, self.log)

            if not continue_tr:
                self.update_status("Loading RF knowledge...")
                self.update_progress(20)
                self.manager.setup()

            self.update_status("Loading data...")
            self.update_progress(30)
            target_col = None
            if continue_tr and self.manager.existing_model_info:
                target_col = self.manager.existing_model_info.get('target_variable')
            elif not continue_tr and self.manager.rf_transfer.target_variable:
                target_col = self.manager.rf_transfer.target_variable
            new_df, target_col, has_labels = self.manager.load_new_data(
                options["data_path"], target_col, sample_names)

            if not has_labels and not use_pseudo:
                raise ValueError("No target column & pseudo-labels disabled")

            self.update_status("Preprocessing...")
            self.update_progress(40)
            X_proc, y_proc = self.manager.preprocess_new_data(new_df, target_col)

            self.update_status("Training...")
            self.update_progress(60)
            self.manager.run_training(X_proc, y_proc, top_features=top_f,
                                      use_pseudo_labels=use_pseudo and not has_labels,
                                      additional_epochs=additional_epochs)

            self.update_status("Checking training fit...")
            self.update_progress(80)
            if y_proc is None:
                self.log("Measured labels unavailable. Skipping measured-data evaluation.")
            elif enable_ss:
                try:
                    self.manager.evaluate_model_small_sample(X_proc, y_proc)
                except Exception as e:
                    self.log(f"  Small sample eval failed: {e}, using traditional")
                    self.manager._evaluate_model_traditional(X_proc, y_proc)
            else:
                self.manager._evaluate_model_traditional(X_proc, y_proc)

            self.update_status("Saving results...")
            self.update_progress(90)
            self.manager.save_results()

            self.update_status("Complete!")
            self.update_progress(100)
            self.log("=" * 70)
            self.log("Training Complete!")
            if not self.manager.training_summary['historical_preprocessing_verified']:
                self.update_status("Complete with legacy preprocessing warning — see log")
                self.log("WARNING: A legacy parent model's input transform was unavailable; "
                         "the compatibility of these results with the original model is unverified.")
            self.log("=" * 70)
            self.log(f"Total Samples: {self.manager.training_summary.get('total_samples', 0)}")
            self.log(f"Sample Type: {'小样本' if self.manager.training_summary.get('is_small_sample') else '大样本'}")

            if continue_tr and self.manager.existing_model_info:
                old_r2 = self.manager.existing_model_info.get('r2_score')
                trad = self.manager.training_summary.get('evaluation_results', {}).get('traditional', {})
                new_r2 = trad.get('r2_score')
                if isinstance(old_r2, (int, float)) and isinstance(new_r2, (int, float)):
                    self.log(f"R² change: {old_r2:.4f} → {new_r2:.4f}")

            self.log("\nOutput Directories:")
            self.log(f"  data0view : {self.manager.data0view_dir}")
            self.log(f"  data1view : {self.manager.data1view_dir}")
            self.log(f"  data2eval : {self.manager.data2evaluation_dir}")
            self.log(f"  data3FIA  : {self.manager.data3FIA_dir}")
            self.log(f"  data4PDP  : {self.manager.data4PDP_dir}")
            self.log(f"  bestmodel : {self.manager.bestmodel_dir}")

            ss = self.manager.training_summary.get('small_sample_evaluation', {}) or {}
            if ss:
                self.log("\nSmall Sample Metrics:")
                for k in ['smape', 'median_ape', 'mean_ape', 'rank_correlation']:
                    v = ss.get(k)
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        fmt = f"{v:.3f}" if k == 'rank_correlation' else f"{v:.2f}%"
                        self.log(f"  {k}: {fmt}")
                for th, acc in ss.get('accuracy_at_thresholds', {}).items():
                    thv = int(th.split('_')[1].replace('%', ''))
                    self.log(f"  ±{thv}%: {acc:.1f}%")

        except Exception as e:
            self.log(f"\n✗ Training failed: {e}")
            self.log(traceback.format_exc())
            self.update_status(f"Failed: {e}")
            self._training_error(str(e))
        finally:
            self._finish_training()

    def on_closing(self):
        if self.is_running:
            if not messagebox.askyesno("Confirm Exit", "Training is in progress. Exit anyway?"):
                return
        self.window_destroyed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def safe_cleanup(self):
        self.window_destroyed = True

    def quit_program(self):
        self.on_closing()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    try:
        app = XGBoostTransferGUI()
        app.run()
    except Exception as e:
        print(f"Application failed to start: {e}")
        traceback.print_exc()