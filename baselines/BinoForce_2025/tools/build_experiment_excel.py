from __future__ import annotations

import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "experiment_results.xlsx"

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SECTION_FILL = PatternFill("solid", fgColor="D9EAF7")
HEADER_FONT = Font(color="FFFFFF", bold=True)
BOLD = Font(bold=True)
THIN_BLUE = Side(style="thin", color="9ECAE1")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def value_for_csv(value: str):
    if value is None or value == "":
        return None
    try:
        return int(value) if value.isdigit() else float(value)
    except ValueError:
        return value


def style_sheet(ws, widths: dict[str, int] | None = None, freeze: str = "A2"):
    ws.freeze_panes = freeze
    ws.sheet_view.showGridLines = False
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=THIN_BLUE)
    ws.row_dimensions[1].height = 30
    if widths:
        for col, width in widths.items():
            ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def add_table(ws, ref: str, name: str):
    tab = Table(displayName=name, ref=ref)
    tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
    ws.add_table(tab)


def write_matrix(ws, headers: list[str], rows: list[list[object]], widths: dict[str, int] | None = None):
    ws.append(headers)
    for row in rows:
        ws.append(row)
    style_sheet(ws, widths)
    if rows:
        add_table(ws, f"A1:{get_column_letter(len(headers))}{len(rows)+1}", f"Table{ws.title.replace(' ', '')}")


def main():
    wb = Workbook()
    default = wb.active
    wb.remove(default)

    # README sheet
    ws = wb.create_sheet("说明")
    ws.append(["项目实验数据工作簿", "BinoForce 2025 离线复现"])
    ws.append(["用途", "集中保存本次复现的实验参数、逐行结果、汇总结果与指标定义。"])
    ws.append(["数据来源", "results/binoforce2025_results.csv、method_summary.csv、category_method_summary.csv、dbv_comparison.csv、validation_sweep.json"])
    ws.append(["样本数", 55])
    ws.append(["固定视角数", 5])
    ws.append(["总结果行数", 1925])
    ws.append(["注意", "MonocularForce 是项目内单目消融对照，不是原论文单独提出的方法；quality_score 是项目自定义综合分数。"])
    ws.append(["方法差异详述", "见 docs/method_comparison.md"])
    ws.append(["MonocularForce 说明", "见 docs/monocularforce.md"])
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 110
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows():
        row[0].font = BOLD
        row[0].fill = SECTION_FILL
        row[0].alignment = Alignment(vertical="top", wrap_text=True)
        row[1].alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A2"

    params = [
        ["数据与流程", "samples", 55, "当前读取到的 Annotation 场景数", "src/_pipeline.py::main"],
        ["数据与流程", "views", "main, up, down, left, right", "固定评估/预览视角顺序", "VIEW_ORDER"],
        ["数据与流程", "warmup_frames", 800, "validation sweep 选出的动态预热帧数；不是论文公开迭代次数", "results/validation_sweep.json"],
        ["数据与流程", "eval_frames", 300, "warmup 后用于统计均值的帧数", "DEFAULT_EVAL_FRAMES"],
        ["数据与流程", "fps", 30, "动态相机轨迹帧率", "DEFAULT_FPS"],
        ["数据与流程", "sweep_candidates", "300, 500, 800", "validation subset 候选 warm-up", "DEFAULT_SWEEP_CANDIDATES"],
        ["双目相机", "IPD_METERS", 0.064, "左右虚拟眼间距 6.4 cm；单眼偏移为 ±0.032 m", "论文第 4.1 节；src/_pipeline.py"],
        ["相机", "CAMERA_RADIUS", 10.0, "固定主相机半径", "src/_pipeline.py"],
        ["相机", "FOCAL_LENGTH_MM", 50.0, "透视相机焦距", "src/_pipeline.py"],
        ["相机", "SENSOR_WIDTH_MM", 36.0, "传感器宽度", "src/_pipeline.py"],
        ["相机", "SENSOR_HEIGHT_MM", 24.0, "传感器高度", "src/_pipeline.py"],
        ["相机", "PERTURB_DEGREES", 45.0, "up/down/left/right 相对 main 的视角扰动", "src/_pipeline.py"],
        ["力场", "W_REPULSE", 0.02, "标签间平均排斥力权重", "论文第 4.2 节；src/_pipeline.py"],
        ["力场", "W_ATTRACT", 0.09, "anchor/径向吸引力权重", "论文第 4.2 节；src/_pipeline.py"],
        ["力场", "W_OVERLAP", 0.06, "标签/物体投影重叠力权重", "论文第 4.2 节；src/_pipeline.py"],
        ["力场", "W_LINE", 0.8, "leader line 交叉力权重", "论文第 4.2 节；src/_pipeline.py"],
        ["力场", "W_CIRC", 0.03, "圆形布局力权重", "论文第 4.2 节；src/_pipeline.py"],
        ["力场", "OVERLAP_M", 4.0, "重叠力放大常数", "论文第 4.2 节；src/_pipeline.py"],
        ["力场", "DISPLACEMENT_SCALE", 0.01, "每帧合力到位移的缩放", "论文第 4.2 节；src/_pipeline.py"],
        ["几何 proxy", "object_proxy_eps", "max(0.08*||size||, 0.035)", "布局力使用 anchor 周围局部立方体；评价/预览使用纯净 OBJ", "load_scene"],
        ["评价", "PCK thresholds", "0.05 / 0.10", "相对图像对角线的 manual center 误差阈值", "metric_values"],
        ["评价", "quality_score", "100*(0.7*PCK_005+0.3*PCK_010)-25*OLR-10*LCD", "项目自定义，不是论文原始统计量", "metric_values"],
    ]
    write_matrix(wb.create_sheet("实验参数"), ["类别", "参数", "值", "说明", "来源"], params, {"A": 14, "B": 24, "C": 24, "D": 70, "E": 38})

    raw_headers = ["category", "sample", "view", "method", "num_labels", "PCK_005", "PCK_010", "OLR", "LCD", "DBV", "avg_leader_length", "overlap_pairs", "occluded_points", "intersections", "quality_score"]
    raw_rows = [[value_for_csv(row.get(h, "")) for h in raw_headers] for row in read_csv(RESULTS / "binoforce2025_results.csv")]
    write_matrix(wb.create_sheet("逐行结果"), raw_headers, raw_rows, {"A": 16, "B": 14, "C": 12, "D": 18, "E": 12, "F": 12, "G": 12, "H": 12, "I": 12, "J": 12, "K": 20, "L": 16, "M": 18, "N": 15, "O": 16})
    ws = wb["逐行结果"]
    for row in ws.iter_rows(min_row=2, min_col=6, max_col=15):
        for cell in row:
            cell.number_format = "0.000000"

    method_headers = ["method", "PCK_005", "PCK_010", "OLR", "LCD", "DBV", "avg_leader_length", "overlap_pairs", "occluded_points", "intersections", "quality_score"]
    method_rows = [[value_for_csv(row.get(h, "")) for h in method_headers] for row in read_csv(RESULTS / "method_summary.csv")]
    write_matrix(wb.create_sheet("方法汇总"), method_headers, method_rows, {"A": 20, "B": 12, "C": 12, "D": 12, "E": 12, "F": 12, "G": 20, "H": 16, "I": 18, "J": 15, "K": 16})

    cat_headers = ["category"] + method_headers
    cat_rows = [[value_for_csv(row.get(h, "")) for h in cat_headers] for row in read_csv(RESULTS / "category_method_summary.csv")]
    write_matrix(wb.create_sheet("类别方法汇总"), cat_headers, cat_rows, {"A": 18, "B": 20, "C": 12, "D": 12, "E": 12, "F": 12, "G": 12, "H": 20, "I": 16, "J": 18, "K": 15, "L": 16})

    sweep = read_json(RESULTS / "validation_sweep.json")
    sweep_rows = []
    for rec in sweep["records"]:
        metrics = rec["metrics"]
        sweep_rows.append([rec["warmup_frames"], rec.get("next_warmup_frames"), rec.get("max_relative_change_to_next"), ", ".join(rec["validation_samples"]), *[metrics.get(k) for k in method_headers[1:]]])
    write_matrix(wb.create_sheet("Validation Sweep"), ["warmup_frames", "next_warmup_frames", "max_relative_change_to_next", "validation_samples", *method_headers[1:]], sweep_rows, {"A": 16, "B": 20, "C": 28, "D": 70, "E": 12, "F": 12, "G": 12, "H": 12, "I": 12, "J": 20, "K": 16, "L": 18, "M": 15, "N": 16})

    dbv_headers = ["scene", "with_BinoForce", "without_BinoForce", "reduction"]
    dbv_rows = [[value_for_csv(row.get(h, "")) for h in dbv_headers] for row in read_csv(RESULTS / "dbv_comparison.csv")]
    write_matrix(wb.create_sheet("DBV 对照"), dbv_headers, dbv_rows, {"A": 20, "B": 20, "C": 22, "D": 14})

    metric_rows = [
        ["PCK_005", "越小越好?", "否", "与 manual center 的归一化屏幕距离 ≤ 5% 的标签比例"],
        ["PCK_010", "越小越好?", "否", "与 manual center 的归一化屏幕距离 ≤ 10% 的标签比例"],
        ["OLR", "越小越好", "是", "标签与标签/物体的投影矩形重叠面积比例"],
        ["LCD", "越小越好", "是", "leader line 交叉程度平均值"],
        ["DBV", "越小越好", "是", "左右眼重叠面积差的归一化平均值"],
        ["avg_leader_length", "越小越好", "是", "屏幕空间 leader 长度 / 图像对角线"],
        ["overlap_pairs", "越小越好", "是", "存在投影矩形重叠的标签对数量"],
        ["occluded_points", "越小越好", "是", "标签矩形包含其他 anchor 投影点的数量"],
        ["intersections", "越小越好", "是", "leader line 投影线段相交对数"],
        ["quality_score", "越大越好", "否", "项目自定义综合分数；不可视为论文原始指标"],
    ]
    write_matrix(wb.create_sheet("指标字典"), ["字段", "方向", "是否原始论文指标", "定义"], metric_rows, {"A": 24, "B": 12, "C": 20, "D": 90})

    for ws in wb.worksheets:
        ws.auto_filter.ref = ws.dimensions if ws.title not in {"说明"} else None
        for row in ws.iter_rows():
            for cell in row:
                if cell.row > 1 and isinstance(cell.value, (int, float)):
                    cell.number_format = "0.000000"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
