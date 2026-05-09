# ================================================================
# ДОКУМЕНТИРОВАННАЯ ВЕРСИЯ ФАЙЛА
# Файл: spearman_pearson_app_v02_commented.py
#
# Назначение программы:
#   Графическое приложение (PyQt5) для статистического анализа данных
#   с вычислением корреляций Пирсона и Спирмена, построением
#   тепловых карт, кластеризацией и экспортом результатов.
#
# Основные возможности:
#   • ввод данных по химическим показателям
#   • вычисление корреляции Пирсона и Спирмена
#   • Monte‑Carlo оценка значимости
#   • построение графиков и кластеризации
#   • сохранение CSV и отчетов
#
# Эта версия содержит дополнительные комментарии, объясняющие
# структуру программы, назначение функций и классов.
# ================================================================


# --- Импорт библиотек Python и сторонних модулей ---
import sys

# --- Импорт библиотек Python и сторонних модулей ---
import os

# --- Импорт библиотек Python и сторонних модулей ---
import math

# --- Импорт библиотек Python и сторонних модулей ---
import re

# --- Импорт библиотек Python и сторонних модулей ---
import itertools

# --- Импорт библиотек Python и сторонних модулей ---
import numpy as np

# --- Импорт библиотек Python и сторонних модулей ---
import pandas as pd

# --- Импорт библиотек Python и сторонних модулей ---
from datetime import datetime


# --- Импорт библиотек Python и сторонних модулей ---
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog,
    QMessageBox, QComboBox, QCheckBox, QTextEdit, QProgressBar
)

# --- Импорт библиотек Python и сторонних модулей ---
from PyQt5.QtGui import QFont

# --- Импорт библиотек Python и сторонних модулей ---
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# --- Импорт библиотек Python и сторонних модулей ---
import matplotlib.pyplot as plt

# --- Импорт библиотек Python и сторонних модулей ---
import seaborn as sns


# --- Импорт библиотек Python и сторонних модулей ---
from scipy.cluster.hierarchy import linkage, dendrogram

# --- Импорт библиотек Python и сторонних модулей ---
from scipy.spatial.distance import squareform

# --- Импорт библиотек Python и сторонних модулей ---
from scipy.stats import pearsonr, t


# ============================================================
# НАСТРОЙКИ
# ============================================================

UI_SCALE = 1.0

RANDOM_SEED = 42
MC_ITERATIONS = 10000

# exact permutations считаем только до 7
EXACT_MAX_N = 7

# Размер батча для Monte-Carlo (чем больше — тем быстрее, но больше RAM)
MC_BATCH_SIZE = 512

# Метод кластеризации (ward нельзя для 1-|rho|)
CLUSTER_LINKAGE_METHOD = "average"

BASE_INDICATORS = [
    'Нитраты', 'Хлориды', 'Сульфаты', 'Железо', 'Фториды', 'Марганец',
    'Медь', 'Никель', 'Кадмий', 'Свинец', 'Хром', 'Бор', 'Мышьяк',
    'Калий', 'Кальций', 'Магний', 'Натрий', 'Цинк', 'Кремний', 'Минерализация'
]


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: ensure_reports_dir
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def ensure_reports_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "Отчеты")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ------------------------------------------------
# ФУНКЦИЯ: ensure_csv_dir
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def ensure_csv_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "csv")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ------------------------------------------------
# ФУНКЦИЯ: sanitize_filename
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def sanitize_filename(text):
    text = str(text).strip()
    if text == "":
        return "не_указана"

    text = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("_")

    return text if text else "не_указана"



# ------------------------------------------------
# ФУНКЦИЯ: parse_value
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def parse_value(text):
    text = str(text).strip().replace(",", ".")
    if text == "" or text.lower() == "nan":
        return np.nan
    return float(text)



# ------------------------------------------------
# ФУНКЦИЯ: rankdata_fast
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def rankdata_fast(a):
    return np.argsort(np.argsort(a))



# ------------------------------------------------
# ФУНКЦИЯ: spearman_r_fast
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def spearman_r_fast(x, y):
    rx = rankdata_fast(x)
    ry = rankdata_fast(y)
    return np.corrcoef(rx, ry)[0, 1]



# ------------------------------------------------
# ФУНКЦИЯ: spearman_exact_pvalue
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def spearman_exact_pvalue(rho_obs, n):
    base = np.arange(n)
    perms_total = math.factorial(n)
    extreme = 0

    for perm in itertools.permutations(base):
        perm = np.array(perm, dtype=int)
        r = spearman_r_fast(base, perm)
        if abs(r) >= abs(rho_obs):
            extreme += 1

    return extreme / perms_total


# ============================================================
# ЛИНЕЙНАЯ РЕГРЕССИЯ ДЛЯ ПИРСОНА
# ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: linear_regression_equation
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def linear_regression_equation(x, y):
    """
    Линейная регрессия y = a*x + b
    Возвращает:
        a, b, r2, p_model, n

    p_model — p-value для значимости наклона (a) через t-test
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(x)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return np.nan, np.nan, np.nan, np.nan, len(x)

    # коэффициенты
    a = np.sum((x - x_mean) * (y - y_mean)) / denom
    b = y_mean - a * x_mean

    # прогноз
    y_pred = a * x + b

    # R^2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    if ss_tot < 1e-12:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    # p-value наклона
    n = len(x)
    dof = n - 2

    if dof <= 0:
        return float(a), float(b), float(r2), np.nan, n

    s2 = ss_res / dof
    se_a = np.sqrt(s2 / denom)

    if se_a < 1e-12:
        p_model = np.nan
    else:
        t_stat = a / se_a
        p_model = 2 * (1 - t.cdf(abs(t_stat), df=dof))

    return float(a), float(b), float(r2), float(p_model), n


# ============================================================
# УСКОРЕННЫЙ MONTE-CARLO (ВЕКТОРИЗАЦИЯ)
# ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: _standardize_ranks
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def _standardize_ranks(arr):
    arr = arr.astype(np.float64)
    arr = arr - arr.mean()
    std = arr.std(ddof=0)
    if std < 1e-12:
        return np.zeros_like(arr)
    return arr / std



# ------------------------------------------------
# ФУНКЦИЯ: spearman_mc_pvalue_vectorized
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def spearman_mc_pvalue_vectorized(x, y, iterations=10000, seed=42, batch_size=512, stop_flag=None):
    rng = np.random.default_rng(seed)

    rx = _standardize_ranks(rankdata_fast(x))
    ry = _standardize_ranks(rankdata_fast(y))

    rho_obs = float(np.mean(rx * ry))

    n = len(rx)
    extreme = 0
    done = 0

    base_idx = np.arange(n, dtype=np.int64)

    while done < iterations:
        if stop_flag is not None and stop_flag():
            return rho_obs, None, True

        cur = min(batch_size, iterations - done)

        perms = np.empty((cur, n), dtype=np.int64)
        for k in range(cur):
            perms[k] = rng.permutation(base_idx)

        ry_perm = ry[perms]
        rho_batch = np.mean(ry_perm * rx[None, :], axis=1)

        extreme += int(np.sum(np.abs(rho_batch) >= abs(rho_obs)))
        done += cur

    p = extreme / iterations
    return rho_obs, p, False



# ------------------------------------------------
# ФУНКЦИЯ: lower_triangle_mask
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def lower_triangle_mask(n):
    """
    True = скрываем.
    Здесь мы скрываем ВЕРХНИЙ треугольник + диагональ.
    То есть рисуем только нижний треугольник.
    """
    return np.triu(np.ones((n, n), dtype=bool), k=0)



# ------------------------------------------------
# ФУНКЦИЯ: corr_to_distance_matrix
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def corr_to_distance_matrix(corr_df):
    """
    dist = 1 - |rho|
    + защита от float-мусора (отрицательные, >1)
    """
    corr = corr_df.copy().fillna(0.0)
    dist = 1.0 - corr.abs()

    dist_arr = dist.to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0.0)

    dist_arr = np.clip(dist_arr, 0.0, 1.0)

    return dist_arr


# ============================================================
# КОЛОНКА ПОКАЗАТЕЛЯ
# ============================================================


# ------------------------------------------------
# КЛАСС: IndicatorColumn
# Назначение: компонент пользовательского интерфейса
# или вспомогательный объект приложения.
# ------------------------------------------------
class IndicatorColumn(QWidget):

# ------------------------------------------------
# ФУНКЦИЯ: __init__
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def __init__(self, name, n_rows):
        super().__init__()
        self.name = name

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)

        self.checkbox = QCheckBox("Использовать")
        self.checkbox.setChecked(True)
        self.layout.addWidget(self.checkbox)

        title = QLabel(name)
        title.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(title)

        self.title_label = title

        self.fields = []
        for _ in range(n_rows):
            self.add_field()


# ------------------------------------------------
# ФУНКЦИЯ: set_title
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def set_title(self, new_name):
        self.name = new_name
        self.title_label.setText(new_name)


# ------------------------------------------------
# ФУНКЦИЯ: add_field
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def add_field(self):
        f = QLineEdit()
        f.setFixedWidth(int(90 * UI_SCALE))
        f.setFont(QFont("Arial", int(11 * UI_SCALE)))
        self.layout.addWidget(f)
        self.fields.append(f)


# ------------------------------------------------
# ФУНКЦИЯ: remove_field
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def remove_field(self):
        if self.fields:
            f = self.fields.pop()
            self.layout.removeWidget(f)
            f.deleteLater()


# ------------------------------------------------
# ФУНКЦИЯ: set_rows_count
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def set_rows_count(self, n_rows):
        while len(self.fields) < n_rows:
            self.add_field()
        while len(self.fields) > n_rows:
            self.remove_field()


# ------------------------------------------------
# ФУНКЦИЯ: values
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def values(self):
        out = []
        for f in self.fields:
            try:
                out.append(parse_value(f.text()))
            except Exception:
                out.append(np.nan)
        return np.array(out, dtype=float)


# ------------------------------------------------
# ФУНКЦИЯ: is_active
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def is_active(self):
        return self.checkbox.isChecked()


# ------------------------------------------------
# ФУНКЦИЯ: clear_values
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def clear_values(self):
        for f in self.fields:
            f.setText("")


# ============================================================
# ПОТОК ДЛЯ РАСЧЕТА СПИРМЕНА
# ============================================================


# ------------------------------------------------
# КЛАСС: SpearmanWorker
# Назначение: компонент пользовательского интерфейса
# или вспомогательный объект приложения.
# ------------------------------------------------
class SpearmanWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, object, list)
    error = pyqtSignal(str)
    stopped = pyqtSignal()


# ------------------------------------------------
# ФУНКЦИЯ: __init__
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def __init__(self, data_dict, active_names, alpha, seed, mc_iters):
        super().__init__()
        self.data_dict = data_dict
        self.active_names = active_names
        self.alpha = alpha
        self.seed = seed
        self.mc_iters = mc_iters
        self._stop_requested = False


# ------------------------------------------------
# ФУНКЦИЯ: request_stop
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def request_stop(self):
        self._stop_requested = True


# ------------------------------------------------
# ФУНКЦИЯ: run
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def run(self):
        try:
            active = self.active_names

            rho = pd.DataFrame(index=active, columns=active, dtype=float)
            pval = pd.DataFrame(index=active, columns=active, dtype=float)

            results = []

            total_pairs = len(active) * len(active)
            done_pairs = 0


# ------------------------------------------------
# ФУНКЦИЯ: stop_flag
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
            def stop_flag():
                return self._stop_requested

            for i in active:
                if self._stop_requested:
                    self.stopped.emit()
                    return

                xi = self.data_dict[i]

                for j in active:
                    if self._stop_requested:
                        self.stopped.emit()
                        return

                    yj = self.data_dict[j]

                    mask = ~np.isnan(xi) & ~np.isnan(yj)
                    x = xi[mask]
                    y = yj[mask]

                    if len(x) < 3:
                        rho.loc[i, j] = np.nan
                        pval.loc[i, j] = np.nan
                    else:
                        if len(x) <= EXACT_MAX_N:
                            r = spearman_r_fast(x, y)
                            p = spearman_exact_pvalue(r, len(x))
                        else:
                            pair_seed = abs(hash((i, j, self.seed))) % (2**32)

                            r, p, was_stopped = spearman_mc_pvalue_vectorized(
                                x, y,
                                iterations=self.mc_iters,
                                seed=pair_seed,
                                batch_size=MC_BATCH_SIZE,
                                stop_flag=stop_flag
                            )

                            if was_stopped:
                                self.stopped.emit()
                                return

                        rho.loc[i, j] = r
                        pval.loc[i, j] = p

                        if i != j and (not pd.isna(p)) and (p <= self.alpha):
                            results.append((p, i, j, r))

                    done_pairs += 1
                    percent = int(done_pairs / total_pairs * 100)
                    self.progress.emit(percent)

            results.sort(key=lambda x: x[0])
            self.finished.emit(rho, pval, results)

        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================


# ------------------------------------------------
# КЛАСС: MainWindow
# Назначение: компонент пользовательского интерфейса
# или вспомогательный объект приложения.
# ------------------------------------------------
class MainWindow(QWidget):

# ------------------------------------------------
# ФУНКЦИЯ: __init__
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Корреляция Спирмена (ускоренный Monte-Carlo)")
        self.resize(1600, 920)

        self.n_rows = 3

        self.columns = {}
        self.indicator_order = []
        self.custom_indicators = []

        self.last_corr = None
        self.last_p = None

        # === Пирсон по выборке ===
        self.pearson_corr = None
        self.pearson_p = None
        self.pearson_names = None
        self.pearson_pairs = None

        self.report_text = ""

        self.hm_sig_path = None
        self.hm_full_path = None
        self.den_path = None

        # пути для графиков Пирсона
        self.hm_pearson_full_path = None
        self.hm_pearson_sig_path = None

        self.out_dir = ensure_reports_dir()

        self.csv_dir = ensure_csv_dir()

        self.worker = None

        main = QHBoxLayout(self)

        # ============================================================
        # ЛЕВАЯ ПАНЕЛЬ
        # ============================================================

        left = QVBoxLayout()
        left.setAlignment(Qt.AlignTop)

        self.point_input = QLineEdit()
        self.point_input.setPlaceholderText("Название точки забора")
        self.point_input.setFont(QFont("Arial", int(12 * UI_SCALE)))
        left.addWidget(self.point_input)

        btn_add_row = QPushButton("Добавить строку")
        btn_remove_row = QPushButton("Удалить строку")
        btn_add_row.clicked.connect(self.add_row)
        btn_remove_row.clicked.connect(self.remove_row)

        left.addWidget(btn_add_row)
        left.addWidget(btn_remove_row)

        left.addWidget(QLabel("Название нового показателя:"))

        self.new_indicator_input = QLineEdit()
        self.new_indicator_input.setPlaceholderText("Например: Показатель 21")
        left.addWidget(self.new_indicator_input)

        btn_add_indicator = QPushButton("Добавить показатель")
        btn_remove_indicator = QPushButton("Удалить показатель")

        btn_add_indicator.clicked.connect(self.add_indicator)
        btn_remove_indicator.clicked.connect(self.remove_indicator)

        left.addWidget(btn_add_indicator)
        left.addWidget(btn_remove_indicator)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        cont = QWidget()
        self.scroll_layout = QHBoxLayout(cont)
        self.scroll_layout.setAlignment(Qt.AlignTop)

        self.scroll.setWidget(cont)
        left.addWidget(self.scroll)

        for name in BASE_INDICATORS:
            self._create_indicator_column(name, is_custom=False)
        
        btn_save_csv = QPushButton("Сохранить CSV")
        btn_save_csv.clicked.connect(self.save_csv)
        left.addWidget(btn_save_csv)
        main.addLayout(left, 3)

        # ============================================================
        # ПРАВАЯ ПАНЕЛЬ
        # ============================================================

        right = QVBoxLayout()
        right.setAlignment(Qt.AlignTop)

        btn_csv = QPushButton("Загрузить CSV")
        btn_csv.clicked.connect(self.load_csv)

        self.btn_calc = QPushButton("Рассчитать Спирмена (ρ и p-value)")
        self.btn_calc.clicked.connect(self.calculate_spearman)

        self.btn_stop = QPushButton("Остановить расчет")
        self.btn_stop.clicked.connect(self.stop_calculation)
        self.btn_stop.setEnabled(False)

        btn_heat_sig = QPushButton("График значимых корреляций Спирмена")
        btn_heat_sig.clicked.connect(self.plot_heatmap_significant)

        btn_heat_full = QPushButton("Полный график корреляций Спирмена")
        btn_heat_full.clicked.connect(self.plot_heatmap_full)

        # === Пирсон для выборки ===
        btn_pearson_sample = QPushButton("Рассчитать Пирсона (ρ и p-value) для выборки")
        btn_pearson_sample.clicked.connect(self.calculate_pearson_for_sample)

        # === 2 кнопки построения Пирсона ===
        btn_pearson_plot_full = QPushButton("Полный график корреляций Пирсона")
        btn_pearson_plot_full.clicked.connect(self.plot_pearson_heatmap_full)

        btn_pearson_plot_sig = QPushButton("График значимых корреляций Пирсона")
        btn_pearson_plot_sig.clicked.connect(self.plot_pearson_heatmap_significant)

        btn_cluster = QPushButton("Рассчитать кластеризацию (иерархическая)")
        btn_cluster.clicked.connect(self.calculate_clustering)

        btn_dendro = QPushButton("Построить дендрограмму")
        btn_dendro.clicked.connect(self.plot_dendrogram)

        btn_save = QPushButton("Сохранить отчет")
        btn_save.clicked.connect(self.save_all)

        btn_clear_log = QPushButton("Очистить журнал")
        btn_clear_log.clicked.connect(self.clear_log)

        self.alpha_combo = QComboBox()
        self.alpha_combo.addItems(["0.05", "0.01"])
        self.alpha_combo.setCurrentIndex(0)

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        right.addWidget(btn_csv)
        right.addWidget(self.btn_calc)
        right.addWidget(self.btn_stop)
        right.addWidget(self.progress)

        right.addWidget(btn_heat_full)
        right.addWidget(btn_heat_sig)

        right.addWidget(btn_pearson_sample)
        right.addWidget(btn_pearson_plot_full)
        right.addWidget(btn_pearson_plot_sig)

        right.addWidget(btn_cluster)
        right.addWidget(btn_dendro)

        right.addWidget(QLabel("Уровень значимости:"))
        right.addWidget(self.alpha_combo)

        right.addWidget(btn_save)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setFont(QFont("Consolas", int(10 * UI_SCALE)))
        self.report.setMinimumHeight(int(340 * UI_SCALE))
        right.addWidget(self.report)

        right.addWidget(btn_clear_log)

        main.addLayout(right, 2)

    # ============================================================
    # СОЗДАНИЕ КОЛОНКИ
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: _create_indicator_column
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def _create_indicator_column(self, name, is_custom):
        if name in self.columns:
            return False

        col = IndicatorColumn(name, self.n_rows)
        self.columns[name] = col
        self.indicator_order.append(name)

        if is_custom:
            self.custom_indicators.append(name)

        self.scroll_layout.addWidget(col)
        return True

    # ============================================================
    # ЧИСЛО СТРОК
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: set_rows_count_for_all
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def set_rows_count_for_all(self, n_rows):
        self.n_rows = max(1, int(n_rows))
        for col in self.columns.values():
            col.set_rows_count(self.n_rows)


# ------------------------------------------------
# ФУНКЦИЯ: add_row
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def add_row(self):
        self.set_rows_count_for_all(self.n_rows + 1)


# ------------------------------------------------
# ФУНКЦИЯ: remove_row
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def remove_row(self):
        if self.n_rows > 1:
            self.set_rows_count_for_all(self.n_rows - 1)

    # ============================================================
    # ДОБАВЛЕНИЕ/УДАЛЕНИЕ ПОКАЗАТЕЛЕЙ
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: _next_ault_indicator_name
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def _next_default_indicator_name(self):
        base = 21
        used = set(self.columns.keys())

        k = base
        while True:
            name = "Показатель {}".format(k)
            if name not in used:
                return name
            k += 1


# ------------------------------------------------
# ФУНКЦИЯ: add_indicator
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def add_indicator(self):
        raw = self.new_indicator_input.text().strip()

        if raw == "":
            raw = self._next_default_indicator_name()

        name = raw

        if name in self.columns:
            QMessageBox.warning(self, "Ошибка", "Такой показатель уже существует.")
            return

        ok = self._create_indicator_column(name, is_custom=True)
        if not ok:
            QMessageBox.warning(self, "Ошибка", "Не удалось добавить показатель.")
            return

        self.new_indicator_input.setText(self._next_default_indicator_name())

        self.append_report_block(
            "Добавлен показатель",
            ["Добавлен: {}".format(name)]
        )


# ------------------------------------------------
# ФУНКЦИЯ: remove_indicator
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def remove_indicator(self):
        if not self.custom_indicators:
            QMessageBox.information(self, "Информация", "Нет добавленных показателей.")
            return

        name = self.custom_indicators.pop()

        col = self.columns.pop(name)
        self.indicator_order.remove(name)

        self.scroll_layout.removeWidget(col)
        col.deleteLater()

        self.append_report_block(
            "Удален показатель",
            ["Удален: {}".format(name)]
        )
    
    # ============================================================
# СОХРАНЕНИЕ CSV
# ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: save_csv
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def save_csv(self):

        if not self.indicator_order:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения.")
            return

        data = {}

        for name in self.indicator_order:
            col = self.columns[name]
            data[name] = col.values()

        df = pd.DataFrame(data)

        point = sanitize_filename(self.point_input.text())
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = "{}_{}.csv".format(point, now)

        path = os.path.join(self.csv_dir, filename)

        try:
            df.to_csv(path, index=False, encoding="utf-8")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить CSV:\n{}".format(e))
            return

        QMessageBox.information(
            self,
            "CSV сохранён",
            "Файл сохранён:\n{}".format(path)
        )

        self.append_report_block(
            "CSV сохранён",
            [
                "Файл: {}".format(path),
                "Строк данных: {}".format(df.shape[0]),
                "Показателей: {}".format(df.shape[1])
            ]
        )

    # ============================================================
    # ОТЧЕТ
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: append_report_block
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def append_report_block(self, title, lines):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        block = []
        block.append("=" * 70)
        block.append("{}  |  {}".format(title, now))
        block.append("=" * 70)
        block.extend(lines)
        block.append("")

        self.report_text += "\n".join(block) + "\n"

        # === ВАЖНО: обновляем и автоскроллим вниз ===
        self.report.setPlainText(self.report_text)

        scrollbar = self.report.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ============================================================
    # ПОЛНЫЙ СБРОС ИНТЕРФЕЙСА (ПО УМОЛЧАНИЮ)
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: reset_to_aults
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def reset_to_defaults(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(1500)

        self.set_rows_count_for_all(3)

        for name in list(self.custom_indicators):
            if name in self.columns:
                col = self.columns.pop(name)
                if name in self.indicator_order:
                    self.indicator_order.remove(name)
                self.scroll_layout.removeWidget(col)
                col.deleteLater()

        self.custom_indicators = []

        for base_name in BASE_INDICATORS:
            if base_name not in self.columns:
                self._create_indicator_column(base_name, is_custom=False)

        self.indicator_order = [n for n in BASE_INDICATORS if n in self.columns]

        for name in self.indicator_order:
            col = self.columns[name]
            col.checkbox.setChecked(True)
            col.set_rows_count(3)
            col.clear_values()

        self.point_input.setText("")
        self.new_indicator_input.setText(self._next_default_indicator_name())

        self.alpha_combo.setCurrentIndex(0)

        self.last_corr = None
        self.last_p = None

        self.pearson_corr = None
        self.pearson_p = None
        self.pearson_names = None
        self.pearson_pairs = None

        self.hm_sig_path = None
        self.hm_full_path = None
        self.den_path = None
        self.hm_pearson_full_path = None
        self.hm_pearson_sig_path = None

        self.progress.setValue(0)
        self.btn_calc.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ============================================================
    # ОЧИСТКА ЖУРНАЛА
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: clear_log
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def clear_log(self):

        reply = QMessageBox.question(
            self,
            "Подтверждение очистки",
            "Вы уверены, что хотите очистить журнал?\n\n"
            "⚠️ Все введенные данные и результаты расчетов будут удалены.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.report_text = ""
        self.report.setText("")

        self.reset_to_defaults()

        QMessageBox.information(
            self,
            "Готово",
            "Журнал очищен.\n"
            "Все параметры сброшены по умолчанию:\n"
            "- 20 показателей\n"
            "- 3 строки\n"
            "- пустые значения"
        )

    # ============================================================
    # CSV
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: load_csv
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV", "", "CSV (*.csv)")
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить CSV:\n{}".format(e))
            return

        if df.shape[0] < 1:
            QMessageBox.warning(self, "Ошибка", "CSV пустой")
            return

        self.set_rows_count_for_all(int(df.shape[0]))

        loaded_cols = []
        for name, col in self.columns.items():
            if name in df.columns:
                loaded_cols.append(name)
                vals = df[name].values

                for i in range(self.n_rows):
                    v = vals[i] if i < len(vals) else np.nan
                    if pd.isna(v):
                        col.fields[i].setText("")
                    else:
                        col.fields[i].setText(str(v))

        if not loaded_cols:
            QMessageBox.warning(
                self,
                "Ошибка",
                "В CSV нет ни одной подходящей колонки.\n"
                "Проверьте, что названия совпадают с ожидаемыми."
            )
            return

        self.append_report_block(
            "CSV загружен",
            [
                "Файл: {}".format(path),
                "Строк: {}".format(df.shape[0]),
                "Загружены колонки: {}".format(", ".join(loaded_cols))
            ]
        )

    # ============================================================
    # ОСТАНОВКА ПОТОКА
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: stop_calculation
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def stop_calculation(self):
        if self.worker is None:
            return
        if not self.worker.isRunning():
            return

        self.worker.request_stop()
        self.btn_stop.setEnabled(False)

        self.append_report_block(
            "Остановка расчета",
            ["Пользователь запросил остановку расчета."]
        )

    # ============================================================
    # СПИРМЕН (ПОТОК)
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: calculate_spearman
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def calculate_spearman(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Подождите", "Расчет уже выполняется.")
            return

        alpha = float(self.alpha_combo.currentText())

        active = [n for n in self.indicator_order if self.columns[n].is_active()]
        if len(active) < 2:
            QMessageBox.warning(self, "Ошибка", "Выберите минимум два показателя")
            return

        data_dict = {}
        for name in active:
            data_dict[name] = self.columns[name].values()

        self.progress.setValue(0)
        self.btn_calc.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.append_report_block(
            "Запуск расчета Спирмена",
            [
                "alpha={}".format(alpha),
                "Exact permutations: n <= {}".format(EXACT_MAX_N),
                "Monte-Carlo (vectorized): n > {}, iterations={}, batch_size={}, seed={}".format(
                    EXACT_MAX_N, MC_ITERATIONS, MC_BATCH_SIZE, RANDOM_SEED
                ),
                "Показателей: {}".format(len(active))
            ]
        )

        self.worker = SpearmanWorker(
            data_dict=data_dict,
            active_names=active,
            alpha=alpha,
            seed=RANDOM_SEED,
            mc_iters=MC_ITERATIONS
        )

        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_spearman_finished)
        self.worker.error.connect(self.on_spearman_error)
        self.worker.stopped.connect(self.on_spearman_stopped)

        self.worker.start()


# ------------------------------------------------
# ФУНКЦИЯ: on_spearman_error
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def on_spearman_error(self, msg):
        self.btn_calc.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.warning(self, "Ошибка расчета", msg)


# ------------------------------------------------
# ФУНКЦИЯ: on_spearman_stopped
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def on_spearman_stopped(self):
        self.btn_calc.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(0)

        self.append_report_block(
            "Расчет остановлен",
            ["Расчет был остановлен пользователем."]
        )


# ------------------------------------------------
# ФУНКЦИЯ: on_spearman_finished
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def on_spearman_finished(self, rho, pval, results):
        self.last_corr = rho
        self.last_p = pval

        self.pearson_corr = None
        self.pearson_p = None
        self.pearson_names = None
        self.pearson_pairs = None

        alpha = float(self.alpha_combo.currentText())
        point = sanitize_filename(self.point_input.text())

        lines = []
        lines.append("Точка: {}".format(point))
        lines.append("Уровень значимости: {}".format(alpha))
        lines.append("Строк данных (n_rows): {}".format(self.n_rows))
        lines.append("")

        if not results:
            lines.append("Нет значимых корреляций.")
        else:
            lines.append("Значимые корреляции (отсортированы по p-value):")
            lines.append("")

            idx_map = {name: k for k, name in enumerate(rho.columns)}

            for p, i, j, r in results:
                if idx_map[i] > idx_map[j]:
                    lines.append("{} – {}: ρ={:.3f}, p={:.5f}".format(i, j, r, p))

        self.append_report_block("Расчет Спирмена завершен", lines)

        self.progress.setValue(100)
        self.btn_calc.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ============================================================
    # ПИРСОН ДЛЯ ВЫБОРКИ (по фильтру Спирмена)
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: calculate_pearson_for_sample
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def calculate_pearson_for_sample(self):
        if self.last_corr is None or self.last_p is None:
            QMessageBox.warning(self, "Ошибка", "Сначала рассчитайте Спирмена")
            return

        spearman_thr = 0.7
        spearman_alpha = 0.05

        active = list(self.last_corr.columns)
        data_dict = {name: self.columns[name].values() for name in active}

        selected_pairs = []
        selected_names = set()

        for i_idx in range(len(active)):
            for j_idx in range(i_idx + 1, len(active)):
                i = active[i_idx]
                j = active[j_idx]

                r_s = self.last_corr.loc[i, j]
                p_s = self.last_p.loc[i, j]

                if pd.isna(r_s) or pd.isna(p_s):
                    continue

                if abs(r_s) >= spearman_thr and p_s < spearman_alpha:
                    selected_pairs.append((i, j, r_s, p_s))
                    selected_names.add(i)
                    selected_names.add(j)

        if not selected_pairs:
            QMessageBox.information(
                self,
                "Нет выборки",
                "Не найдено ни одной пары, где |Spearman| >= 0.7 и p-value < 0.05."
            )
            return

        selected_names = sorted(selected_names, key=lambda x: active.index(x))
        self.pearson_names = selected_names
        self.pearson_pairs = selected_pairs

        pearson_corr = pd.DataFrame(index=selected_names, columns=selected_names, dtype=float)
        pearson_p = pd.DataFrame(index=selected_names, columns=selected_names, dtype=float)

        for i in selected_names:
            xi = data_dict[i]
            for j in selected_names:
                yj = data_dict[j]

                mask = ~np.isnan(xi) & ~np.isnan(yj)
                x = xi[mask]
                y = yj[mask]

                if len(x) < 3:
                    pearson_corr.loc[i, j] = np.nan
                    pearson_p.loc[i, j] = np.nan
                else:
                    r, p = pearsonr(x, y)
                    pearson_corr.loc[i, j] = r
                    pearson_p.loc[i, j] = p

        self.pearson_corr = pearson_corr
        self.pearson_p = pearson_p

        lines = []
        lines.append("Условие отбора пар по Спирмену:")
        lines.append("  |rho| >= {:.2f}".format(spearman_thr))
        lines.append("  p-value < {:.2f}".format(spearman_alpha))
        lines.append("")
        lines.append("Выбрано пар: {}".format(len(selected_pairs)))
        lines.append("Показателей в выборке: {}".format(len(selected_names)))
        lines.append("")
        lines.append("Пары (Spearman):")

        for (i, j, rs, ps) in selected_pairs[:50]:
            lines.append("  {} – {}: rho={:.3f}, p={:.5f}".format(i, j, rs, ps))

        if len(selected_pairs) > 50:
            lines.append("  ... (показаны первые 50)")

        sig_pairs = []
        for i_idx in range(len(selected_names)):
            for j_idx in range(i_idx + 1, len(selected_names)):
                i = selected_names[i_idx]
                j = selected_names[j_idx]
                rp = pearson_corr.loc[i, j]
                pp = pearson_p.loc[i, j]
                if (not pd.isna(pp)) and pp < 0.05:
                    sig_pairs.append((pp, i, j, rp))

        sig_pairs.sort(key=lambda x: x[0])

        lines.append("")
        lines.append("Значимые корреляции Пирсона (p < 0.05): {}".format(len(sig_pairs)))
        for (pp, i, j, rp) in sig_pairs[:50]:
            lines.append("  {} – {}: r={:.3f}, p={:.5f}".format(i, j, rp, pp))

        if len(sig_pairs) > 50:
            lines.append("  ... (показаны первые 50)")

        # ============================================================
        # УРАВНЕНИЯ ЛИНЕЙНОЙ РЕГРЕССИИ ДЛЯ ЗНАЧИМЫХ ПАР ПИРСОНА
        # (ФОРМАТ + или - в зависимости от b)
        # + R^2 и p_model
        # ============================================================

        lines.append("")
        lines.append("Уравнения линейной зависимости для значимых пар Пирсона:")
        lines.append("Формат: X – Y: Y = a * X ± b")
        lines.append("")

        eq_count = 0

        for (pp, i, j, rp) in sig_pairs:
            xi = data_dict[i]
            yj = data_dict[j]

            mask = ~np.isnan(xi) & ~np.isnan(yj)
            x = xi[mask]
            y = yj[mask]

            if len(x) < 3:
                continue

            a, b, r2, p_model, n_model = linear_regression_equation(x, y)

            if pd.isna(a) or pd.isna(b):
                continue

            sign = "+" if b >= 0 else "-"
            b_abs = abs(b)

            lines.append(
                "{} (X) – {} (Y): Y = {:.6f} * X {} {:.6f}   | r={:.3f}, p-value={:.5f}, R^2={:.3f}".format(
                    i, j, a, sign, b_abs, rp, pp,
                    r2 if not pd.isna(r2) else float("nan"),
                    p_model if not pd.isna(p_model) else float("nan"),
                    n_model
                )
            )
            eq_count += 1

        if eq_count == 0:
            lines.append("  Нет уравнений (возможна константа в X или мало данных).")

        self.append_report_block("Расчет Пирсона для выборки завершен", lines)

        QMessageBox.information(
            self,
            "Готово",
            "Корреляция Пирсона рассчитана для выборки.\n"
            "Теперь можно строить графики Пирсона."
        )

    # ============================================================
    # ГРАФИК ПИРСОНА (ПОЛНЫЙ)
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: plot_pearson_heatmap_full
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def plot_pearson_heatmap_full(self):
        if self.pearson_corr is None:
            QMessageBox.warning(self, "Ошибка", "Сначала рассчитайте Пирсона для выборки")
            return

        corr = self.pearson_corr.copy()
        n = corr.shape[0]

        mask = lower_triangle_mask(n)

        annot = corr.copy().astype(object)
        for i in annot.index:
            for j in annot.columns:
                r = corr.loc[i, j]
                annot.loc[i, j] = "" if pd.isna(r) else "{:.2f}".format(r)

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr,
            mask=mask,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            linecolor="gray"
        )

        plt.title("Пирсон: корреляции для выборки (нижний треугольник)")
        plt.tight_layout()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        point = sanitize_filename(self.point_input.text())

        filename = "{}_Пирсон_выборка_полный_{}.png".format(point, now)
        self.hm_pearson_full_path = os.path.join(self.out_dir, filename)

        plt.savefig(self.hm_pearson_full_path, dpi=200)
        plt.show()

        self.append_report_block(
            "График Пирсона (полный)",
            ["Сохранено: {}".format(self.hm_pearson_full_path)]
        )

    # ============================================================
    # ГРАФИК ПИРСОНА (ЗНАЧИМЫЙ)
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: plot_pearson_heatmap_significant
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def plot_pearson_heatmap_significant(self):
        if self.pearson_corr is None or self.pearson_p is None:
            QMessageBox.warning(self, "Ошибка", "Сначала рассчитайте Пирсона для выборки")
            return

        alpha = 0.05

        corr = self.pearson_corr.copy()
        pval = self.pearson_p.copy()

        n = corr.shape[0]

        tri_mask = lower_triangle_mask(n)
        nonsig = (pval >= alpha).to_numpy()

        final_mask = tri_mask | nonsig

        annot = corr.copy().astype(object)
        for i in annot.index:
            for j in annot.columns:
                r = corr.loc[i, j]
                p = pval.loc[i, j]
                if pd.isna(r) or pd.isna(p) or (p >= alpha):
                    annot.loc[i, j] = ""
                else:
                    annot.loc[i, j] = "{:.2f}\n({:.3f})".format(r, p)

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr,
            mask=final_mask,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            linecolor="gray"
        )

        plt.title("Пирсон: значимые корреляции (p < 0.05), нижний треугольник")
        plt.tight_layout()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        point = sanitize_filename(self.point_input.text())

        filename = "{}_Пирсон_выборка_значимые_{}.png".format(point, now)
        self.hm_pearson_sig_path = os.path.join(self.out_dir, filename)

        plt.savefig(self.hm_pearson_sig_path, dpi=200)
        plt.show()

        self.append_report_block(
            "График Пирсона (значимые)",
            ["Сохранено: {}".format(self.hm_pearson_sig_path)]
        )

    # ============================================================
    # ГРАФИК ЗНАЧИМЫХ (СПИРМЕН)
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: plot_heatmap_significant
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def plot_heatmap_significant(self):
        if self.last_corr is None or self.last_p is None:
            QMessageBox.warning(self, "Ошибка", "Сначала рассчитайте Спирмена")
            return

        alpha = float(self.alpha_combo.currentText())

        corr = self.last_corr.copy()
        pval = self.last_p.copy()

        n = corr.shape[0]
        tri_mask = lower_triangle_mask(n)
        nonsig_mask = (pval > alpha).to_numpy()

        final_mask = tri_mask | nonsig_mask

        annot = corr.copy().astype(object)
        for i in annot.index:
            for j in annot.columns:
                r = corr.loc[i, j]
                p = pval.loc[i, j]
                if pd.isna(r) or pd.isna(p) or (p > alpha):
                    annot.loc[i, j] = ""
                else:
                    annot.loc[i, j] = "{:.2f}\n({:.3f})".format(r, p)

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr,
            mask=final_mask,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            linecolor="gray"
        )

        plt.title("Спирмен: значимые корреляции (p < {}), нижний треугольник".format(alpha))
        plt.tight_layout()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        point = sanitize_filename(self.point_input.text())

        filename = "{}_значимые_корреляции_{}.png".format(point, now)
        self.hm_sig_path = os.path.join(self.out_dir, filename)

        plt.savefig(self.hm_sig_path, dpi=200)
        plt.show()

        self.append_report_block(
            "График значимых корреляций",
            ["Сохранено: {}".format(self.hm_sig_path)]
        )

    # ============================================================
    # ПОЛНЫЙ ГРАФИК (СПИРМЕН)
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: plot_heatmap_full
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def plot_heatmap_full(self):
        if self.last_corr is None:
            QMessageBox.warning(self, "Ошибка", "Сначала рассчитайте Спирмена")
            return

        corr = self.last_corr.copy()
        n = corr.shape[0]
        tri_mask = lower_triangle_mask(n)

        annot = corr.copy().astype(object)
        for i in annot.index:
            for j in annot.columns:
                r = corr.loc[i, j]
                annot.loc[i, j] = "" if pd.isna(r) else "{:.2f}".format(r)

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr,
            mask=tri_mask,
            annot=annot,
            fmt="",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            linecolor="gray"
        )

        plt.title("Спирмен: полный график корреляций (нижний треугольник)")
        plt.tight_layout()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        point = sanitize_filename(self.point_input.text())

        filename = "{}_полный_график_корреляций_{}.png".format(point, now)
        self.hm_full_path = os.path.join(self.out_dir, filename)

        plt.savefig(self.hm_full_path, dpi=200)
        plt.show()

        self.append_report_block(
            "Полный график корреляций",
            ["Сохранено: {}".format(self.hm_full_path)]
        )

    # ============================================================
    # КЛАСТЕРИЗАЦИЯ
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: calculate_clustering
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def calculate_clustering(self):
        if self.last_corr is None:
            QMessageBox.warning(self, "Ошибка", "Сначала рассчитайте Спирмена")
            return

        dist_arr = corr_to_distance_matrix(self.last_corr)

        condensed = squareform(dist_arr, checks=False)
        link = linkage(condensed, method=CLUSTER_LINKAGE_METHOD)

        lines = []
        lines.append("Использовано расстояние: dist = 1 - |rho|")
        lines.append("Метод linkage: {}".format(CLUSTER_LINKAGE_METHOD))
        lines.append("")
        lines.append("Первые 10 строк linkage (Z):")
        lines.append("формат: [idx1, idx2, dist, sample_count]")
        lines.append("")

        for k in range(min(10, link.shape[0])):
            a, b, d, cnt = link[k]
            lines.append("{:02d}: [{}, {}, {:.4f}, {}]".format(k + 1, int(a), int(b), d, int(cnt)))

        self.append_report_block("Расчет кластеризации (иерархическая)", lines)

    # ============================================================
    # ДЕНДРОГРАММА
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: plot_dendrogram
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def plot_dendrogram(self):
        if self.last_corr is None:
            QMessageBox.warning(self, "Ошибка", "Сначала рассчитайте Спирмена")
            return

        dist_arr = corr_to_distance_matrix(self.last_corr)

        condensed = squareform(dist_arr, checks=False)
        link = linkage(condensed, method=CLUSTER_LINKAGE_METHOD)

        plt.figure(figsize=(14, 8))
        dendrogram(link, labels=list(self.last_corr.columns), leaf_rotation=90)

        plt.title("Дендрограмма показателей (linkage={})".format(CLUSTER_LINKAGE_METHOD))
        plt.tight_layout()

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        point = sanitize_filename(self.point_input.text())

        filename = "{}_дендрограмма_{}.png".format(point, now)
        self.den_path = os.path.join(self.out_dir, filename)

        plt.savefig(self.den_path, dpi=200)
        plt.show()

        self.append_report_block(
            "Дендрограмма",
            ["Сохранено: {}".format(self.den_path)]
        )

    # ============================================================
    # СОХРАНЕНИЕ
    # ============================================================


# ------------------------------------------------
# ФУНКЦИЯ: save_all
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
    def save_all(self):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        point = sanitize_filename(self.point_input.text())

        report_path = os.path.join(self.out_dir, "{}_отчет_{}.txt".format(point, now))

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.report_text.strip() + "\n")

        msg = "Отчет сохранен:\n{}\n\n".format(report_path)
        QMessageBox.information(self, "Готово", msg)



# ------------------------------------------------
# ФУНКЦИЯ: main
# Выполняет отдельную логическую операцию программы.
# Подробности смотрите внутри функции.
# ------------------------------------------------
def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()