#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import xlsxwriter


def parse_numeric(value):
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return value
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        try:
            return int(text)
        except ValueError:
            return value
    try:
        return float(text)
    except ValueError:
        return value


def parse_percent(value):
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.endswith("%"):
        text = text[:-1]
    if text == "":
        return value
    try:
        return float(text) / 100.0
    except ValueError:
        return value


def write_table(ws, start_row, start_col, header, rows, percent_cols, fmt_header, fmt_percent, fmt_default):
    ws.write_row(start_row, start_col, header, fmt_header)
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row):
            row_idx = start_row + 1 + r_idx
            col_idx = start_col + c_idx
            if c_idx in percent_cols:
                value = parse_percent(cell)
                if isinstance(value, (int, float)):
                    ws.write_number(row_idx, col_idx, value, fmt_percent)
                else:
                    ws.write(row_idx, col_idx, cell, fmt_default)
                continue

            value = parse_numeric(cell)
            if isinstance(value, (int, float)):
                ws.write_number(row_idx, col_idx, value, fmt_default)
            else:
                ws.write(row_idx, col_idx, cell, fmt_default)


def set_default_columns(ws, header):
    if not header:
        return
    ws.set_column(0, 0, 42)
    if len(header) > 1:
        ws.set_column(1, len(header) - 1, 18)


def write_accuracy(payload, output_path):
    header = payload["header"]
    rows = payload["rows"]
    deviation_columns = payload.get("deviation_columns", [])
    deviation_indexes = [col["index"] for col in deviation_columns]

    workbook = xlsxwriter.Workbook(output_path)
    fmt_title = workbook.add_format({"bold": True, "font_size": 14})
    fmt_header = workbook.add_format({"bold": True, "bg_color": "#E6EEF7"})
    fmt_percent = workbook.add_format({"num_format": "0.00%"})
    fmt_default = workbook.add_format({})

    ws = workbook.add_worksheet("Accuracy")
    ws.write(0, 0, payload.get("title", "accuracy report"), fmt_title)
    ws.write(1, 0, "Generated at:")
    ws.write(1, 1, payload.get("generated_at", ""))
    note = payload.get("note")
    if note:
        ws.write(2, 0, note)

    table_start = 4
    write_table(ws, table_start, 0, header, rows, set(deviation_indexes), fmt_header, fmt_percent, fmt_default)
    set_default_columns(ws, header)
    ws.freeze_panes(table_start + 1, 1)

    charts_ws = workbook.add_worksheet("Charts")
    data_start = table_start + 1
    data_end = table_start + len(rows)
    chart_row = 0
    chart_col = 0

    for idx, col in enumerate(deviation_columns):
        col_index = col["index"]
        title = col.get("title", header[col_index])
        chart = workbook.add_chart({"type": "column"})
        chart.add_series({
            "name": title,
            "categories": ["Accuracy", data_start, 0, data_end, 0],
            "values": ["Accuracy", data_start, col_index, data_end, col_index],
        })
        chart.set_title({"name": title})
        chart.set_y_axis({"num_format": "0.00%"})
        chart.set_legend({"none": True})

        charts_ws.insert_chart(chart_row, chart_col, chart, {"x_scale": 1.2, "y_scale": 1.2})
        if idx % 2 == 0:
            chart_col = 9
        else:
            chart_col = 0
            chart_row += 18

    workbook.close()


def sanitize_sheet_name(name):
    cleaned = "".join(ch if ch not in "[]:*?/\\" else "_" for ch in name)
    return cleaned[:31] if cleaned else "Sheet"


def write_adversary(payload, output_path):
    workbook = xlsxwriter.Workbook(output_path)
    fmt_title = workbook.add_format({"bold": True, "font_size": 14})
    fmt_header = workbook.add_format({"bold": True, "bg_color": "#E6EEF7"})
    fmt_percent = workbook.add_format({"num_format": "0.00%"})
    fmt_default = workbook.add_format({})

    summary_ws = workbook.add_worksheet("Summary")
    summary_ws.write(0, 0, payload.get("title", "adversary report"), fmt_title)
    summary_ws.write(1, 0, "Generated at:")
    summary_ws.write(1, 1, payload.get("generated_at", ""))

    row = 3
    summary_ws.write(row, 0, "Parameters", fmt_header)
    row += 1
    for item in payload.get("params", []):
        summary_ws.write(row, 0, item.get("name", ""))
        summary_ws.write(row, 1, item.get("value", ""))
        row += 1

    row += 1
    summary_ws.write(row, 0, "Summary", fmt_header)
    row += 1
    for item in payload.get("summary", []):
        summary_ws.write(row, 0, item.get("label", ""))
        value = parse_percent(item.get("value", ""))
        if isinstance(value, (int, float)):
            summary_ws.write_number(row, 1, value, fmt_percent)
        else:
            summary_ws.write(row, 1, item.get("value", ""))
        row += 1

    summary_ws.set_column(0, 0, 36)
    summary_ws.set_column(1, 1, 24)

    for table in payload.get("tables", []):
        title = table.get("title", "Report")
        header = table.get("header", [])
        rows = table.get("rows", [])
        ratio_col = table.get("ratio_column", -1)
        name_col = table.get("name_column", 1)
        percent_cols = {ratio_col} if ratio_col >= 0 else set()

        sheet_name = sanitize_sheet_name(title)
        ws = workbook.add_worksheet(sheet_name)
        ws.write(0, 0, title, fmt_title)
        ws.write(1, 0, "Generated at:")
        ws.write(1, 1, payload.get("generated_at", ""))

        table_start = 3
        write_table(ws, table_start, 0, header, rows, percent_cols, fmt_header, fmt_percent, fmt_default)
        ws.freeze_panes(table_start + 1, 1)

        if header:
            ws.set_column(0, 0, 8)
        if len(header) > 1:
            ws.set_column(1, 1, 28)
        if len(header) > 2:
            ws.set_column(2, len(header) - 2, 16)
        if len(header) > 0:
            ws.set_column(len(header) - 1, len(header) - 1, 50)

        data_start = table_start + 1
        data_end = table_start + len(rows)
        if rows and ratio_col >= 0:
            chart = workbook.add_chart({"type": "column"})
            chart.add_series({
                "name": title,
                "categories": [sheet_name, data_start, name_col, data_end, name_col],
                "values": [sheet_name, data_start, ratio_col, data_end, ratio_col],
            })
            chart.set_title({"name": f"{title} Ratio"})
            chart.set_y_axis({"num_format": "0.00%"})
            chart.set_legend({"none": True})
            chart_col = len(header) + 2
            ws.insert_chart(2, chart_col, chart, {"x_scale": 1.2, "y_scale": 1.2})

    workbook.close()


def main():
    parser = argparse.ArgumentParser(description="Generate tokenest Excel reports")
    parser.add_argument("--input", required=True, help="Path to JSON payload")
    parser.add_argument("--output", required=True, help="Path to output xlsx file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Failed to read payload: {exc}", file=sys.stderr)
        return 1

    report_type = payload.get("report_type")
    if report_type == "accuracy":
        write_accuracy(payload, str(output_path))
    elif report_type == "adversary":
        write_adversary(payload, str(output_path))
    else:
        print(f"Unknown report_type: {report_type}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
