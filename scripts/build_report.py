"""Create reproducible PNG evidence and a Vietnamese DOCX from data_audit.json."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import docx
import PIL
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
BLUE, ORANGE, INK, GRAY = "#315A82", "#B66A32", "#182A3A", "#637180"


def dump_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def font(size, bold=False):
    names = (["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"] if bold else
             ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    raise RuntimeError("Cần font Arial hoặc DejaVu Sans để hiển thị tiếng Việt.")


def count(value):
    return f"{value:,}".replace(",", ".")


def bar_chart(filename, title, labels, series, note, percent=False, full_scale=True):
    """Draw labelled audit bars from exact numerical data, without external plotting installs."""
    image = Image.new("RGB", (1500, 660), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 30), title, fill=INK, font=font(33, True))
    top, bottom, left, right = 165, 535, 110, 1440
    largest = max(max(values) for _, values, _ in series)
    if percent:
        ceiling = 100 if full_scale else min(100, max(10, math.ceil(largest * 1.18 / 10) * 10))
    else:
        magnitude = 10 ** math.floor(math.log10(max(1, largest * 1.18)))
        ceiling = next(scale * magnitude for scale in (1, 2, 2.5, 4, 5, 10) if scale * magnitude >= largest * 1.18)
    for tick in range(5):
        value = ceiling * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left, y, right, y), fill="#E5E9ED", width=2)
        label = f"{value:.0f}%" if percent else count(round(value))
        draw.text((left - 18, y - 12), label, anchor="ra", fill=GRAY, font=font(22))
    group_width = (right - left) / len(labels)
    width = min(125, group_width * .7 / len(series))
    for s, (name, values, color) in enumerate(series):
        lx = 115 + s * 450
        draw.rectangle((lx, 107, lx + 25, 132), fill=color)
        draw.text((lx + 38, 103), name, fill=INK, font=font(24))
        for index, value in enumerate(values):
            center = left + group_width * (index + .5)
            x = center + (s - (len(series) - 1) / 2) * (width + 12)
            y = bottom - value / ceiling * (bottom - top)
            if value:
                draw.rectangle((x - width / 2, y, x + width / 2, bottom), fill=color)
            label = f"{value:.1f}%" if percent else count(value)
            draw.text((x, y - 31), label, anchor="ma", fill=INK, font=font(23, True))
    for index, label in enumerate(labels):
        draw.text((left + group_width * (index + .5), bottom + 20), label,
                  anchor="ma", fill=INK, font=font(25))
    draw.text((65, 615), note, fill=GRAY, font=font(21))
    image.save(FIGURES / filename)


def checks_figure(audit):
    labels = ["Đối soát đủ mọi dòng gốc và dòng bị loại", "Văn bản đầu ra không rỗng",
              "Nhãn gốc trong khoảng 1 đến 7", "Mỗi nguồn có đủ hai lớp nhị phân",
              "Không trùng chính xác trong từng nguồn", "Không trùng chính xác giữa hai nguồn"]
    keys = ["rows_accounted_for", "nonempty_text", "labels_in_range", "both_binary_classes_present",
            "no_exact_duplicates_within_sources", "no_exact_duplicates_between_sources"]
    image = Image.new("RGB", (1500, 470), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 25), "Kết quả kiểm tra tự động sau xử lý", fill=INK, font=font(34, True))
    for index, (key, label) in enumerate(zip(keys, labels)):
        y = 105 + index * 52
        draw.text((70, y), "PASS" if audit["checks"][key] else "FAIL",
                  fill="#246B45" if audit["checks"][key] else "#A00000", font=font(27, True))
        draw.text((195, y), label, fill=INK, font=font(27))
    draw.text((65, 430), "Nguồn: reports/data_audit.json → checks; assert trong scripts/prepare_data.py",
              fill=GRAY, font=font(21))
    image.save(FIGURES / "04_checks.png")


def style_document(document):
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(1.8)
    section.left_margin = section.right_margin = Cm(2)
    normal = document.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(10.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.12
    # The bundled default template contains a title rule. Remove paragraph borders.
    for style in document.styles:
        for border in list(style.element.findall(".//" + qn("w:pBdr"))):
            border.getparent().remove(border)
    for name, size in [("Title", 23), ("Subtitle", 11), ("Heading 1", 16), ("Heading 2", 12)]:
        style = document.styles[name]
        style.font.name, style.font.size = "Arial", Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(9)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Minh chứng xử lý dữ liệu  |  ")
    run.font.name, run.font.size = "Arial", Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def table(document, headers, rows, widths):
    result = document.add_table(rows=1, cols=len(headers))
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    result.autofit = False
    for column, width in zip(result.columns, widths):
        column.width = Cm(width)
    for cell, text in zip(result.rows[0].cells, headers):
        cell.text = str(text)
    for row in rows:
        for cell, text in zip(result.add_row().cells, row):
            cell.text = str(text)
    for row_index, row in enumerate(result.rows):
        for i, cell in enumerate(row.cells):
            cell.width = Cm(widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for side in ("top", "left", "bottom", "right"):
                child = OxmlElement(f"w:{side}")
                child.set(qn("w:w"), "90")
                child.set(qn("w:type"), "dxa")
                margins.append(child)
            properties.append(margins)
            borders = OxmlElement("w:tcBorders")
            for side in ("top", "left", "bottom", "right"):
                edge = OxmlElement(f"w:{side}")
                for key, value in (("val", "single"), ("sz", "5"), ("color", "D9D9D9")):
                    edge.set(qn(f"w:{key}"), value)
                borders.append(edge)
            properties.append(borders)
            if row_index == 0:
                fill = OxmlElement("w:shd")
                fill.set(qn("w:fill"), "E8EEF4")
                properties.append(fill)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.05
                if i > 0:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.bold = row_index == 0
        prevent_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(prevent_split)
    document.add_paragraph().paragraph_format.space_after = Pt(1)
    return result


def figure(document, filename, caption):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.add_run().add_picture(str(FIGURES / filename), width=Cm(17))
    paragraph = document.add_paragraph(caption)
    paragraph.paragraph_format.space_after = Pt(9)
    for run in paragraph.runs:
        run.italic, run.font.size = True, Pt(8.5)


def canonical_save(document, path):
    document.core_properties.created = datetime(2026, 11, 14, tzinfo=timezone.utc)
    document.core_properties.modified = datetime(2026, 11, 14, tzinfo=timezone.utc)
    temporary = ROOT / ".cache/tmp/report.tmp.docx"
    document.save(temporary)
    with ZipFile(temporary) as source, ZipFile(path, "w", ZIP_DEFLATED) as target:
        for name in sorted(source.namelist()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            target.writestr(info, source.read(name))
    temporary.unlink()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    audit = json.loads((REPORTS / "data_audit.json").read_text(encoding="utf-8"))
    sources = ["upenn", "stanford"]
    items = [audit["datasets"][source] for source in sources]
    totals = [item["counts"]["final_rows"] for item in items]
    bar_chart("01_counts.png", "Số bài đăng trước và sau xử lý", ["UPenn", "Stanford"],
              [("Dòng đọc từ CSV gốc", [item["counts"]["raw_rows"] for item in items], BLUE),
               ("Bài giữ lại", totals, ORANGE)], "Nguồn: data_audit.json → datasets → counts")
    bar_chart("02_labels.png", "Phân phối hai lớp sau xử lý", ["UPenn", "Stanford"],
              [("Chưa cần ưu tiên", [100 * item["label_counts"].get("0", 0) / n for item, n in zip(items, totals)], BLUE),
               ("Cần ưu tiên", [100 * item["label_counts"].get("1", 0) / n for item, n in zip(items, totals)], ORANGE)],
              "Nhãn cần ưu tiên = 1 khi urgency > 4; đây là nhãn độ khẩn cấp.", percent=True)
    labels = ["1–25", "26–50", "51–100", "101–200", "201–400", ">400"]
    bar_chart("03_lengths.png", "Phân phối độ dài văn bản theo số từ", labels,
              [("UPenn" if source == "upenn" else "Stanford",
                [100 * item["length_histogram"][label] / n for label in labels], color)
               for source, item, n, color in zip(sources, items, totals, [BLUE, ORANGE])],
              "Số từ = số phần tử sau split khoảng trắng; không đồng nhất với số token.", percent=True, full_scale=False)
    checks_figure(audit)

    document = Document()
    style_document(document)
    document.add_paragraph("Minh chứng xử lý dữ liệu diễn đàn học tập", "Title")
    document.add_paragraph("Đề tài ưu tiên phản hồi bài đăng sinh viên bằng học máy", "Subtitle")
    document.add_paragraph(
        "Báo cáo ghi nhận kết quả tải, kiểm tra và chuẩn hóa hai nguồn dữ liệu UPenn và Stanford MOOCPosts. "
        "Mọi số liệu và hình bên dưới được sinh từ hai CSV gốc bằng mã trong Project. "
        "Dữ liệu sau xử lý phục vụ bài toán dự đoán bài cần ưu tiên phản hồi dựa trên độ khẩn cấp.")
    document.add_heading("Dữ liệu đã chuẩn bị", 1)
    table(document, ["Chỉ tiêu", "UPenn", "Stanford"], [
        ["Dòng đọc từ CSV gốc", *[count(item["counts"]["raw_rows"]) for item in items]],
        ["Dòng giữ lại sau xử lý", *[count(n) for n in totals]],
        ["Bài cần ưu tiên", *[count(item["label_counts"].get("1", 0)) for item in items]],
        ["Tỷ lệ cần ưu tiên", *[f"{100 * item['label_counts'].get('1', 0) / n:.2f}%" for item, n in zip(items, totals)]],
    ], [8.2, 4.4, 4.4])
    figure(document, "01_counts.png", "Hình 1. Quy mô đo trực tiếp từ CSV gốc và CSV đã xử lý. Lệnh tái tạo: bash Project/scripts/launchers/run.sh.")
    document.add_paragraph(
        "Nguồn công bố có thể ghi số mẫu khác số dòng đọc thực tế do dòng thiếu trường, dữ liệu trùng "
        "hoặc phiên bản tệp. Báo cáo sử dụng số đếm của phiên bản đã cố định trong mã, kèm SHA-256 để đối soát.")
    document.add_paragraph(f"Phiên bản nguồn GitHub: {audit['source_revision']}")

    document.add_page_break()
    document.add_heading("Quy tắc làm sạch và phân phối nhãn", 1)
    document.add_paragraph(
        "Nhãn gốc phải là số hữu hạn trong khoảng 1 đến 7. Quy đổi priority = 1 khi urgency > 4, "
        "còn lại bằng 0. Các dòng thiếu trường, nhãn không hợp lệ hoặc văn bản rỗng được loại và ghi lý do. "
        "Văn bản được chuẩn hóa Unicode, bỏ một số thẻ HTML và gộp khoảng trắng; giữ từ phủ định và dấu câu.")
    document.add_paragraph(
        "So trùng theo nội dung đã casefold. Trong một nguồn, nhóm có nhãn nhị phân mâu thuẫn bị loại toàn bộ; "
        "nhóm cùng nhãn giữ bản đầu tiên. Giữa hai nguồn, nội dung y hệt được giữ ở UPenn và loại khỏi Stanford "
        "theo quy tắc cố định. Văn bản gốc và ID nguồn vẫn được lưu để truy vết.")
    reasons = [("Dòng thiếu trường", "malformed_or_missing_fields"),
               ("Nhãn thiếu hoặc không hợp lệ", "missing_or_invalid_urgency"),
               ("Văn bản rỗng", "empty_text"),
               ("Bản trùng trong nguồn", "duplicate_within_source"),
               ("Dòng thuộc nhóm nhãn mâu thuẫn", "conflicting_binary_labels"),
               ("Bản trùng giữa hai nguồn", "cross_source_duplicates")]
    table(document, ["Lý do loại dòng", "UPenn", "Stanford"],
          [[label, *[count(item["counts"].get(key, 0)) for item in items]] for label, key in reasons], [9, 4, 4])
    figure(document, "02_labels.png", "Hình 2. Tỷ lệ hai lớp trong tập đã làm sạch. Nguồn: data_audit.json, trường label_counts.")
    document.add_paragraph(
        "Phân phối nhãn là căn cứ chọn chỉ số đánh giá. Các thí nghiệm sau cần báo cáo Precision, Recall, "
        "F1 của lớp cần ưu tiên và Macro-F1; chưa có kết quả mô hình trong giai đoạn này.")

    document.add_page_break()
    document.add_heading("Độ dài văn bản và phạm vi sử dụng", 1)
    document.add_paragraph(
        "Độ dài được tính trên text_clean bằng cách tách theo khoảng trắng. Thống kê này giúp kiểm tra "
        "bài quá ngắn hoặc quá dài trước khi xây dựng TF-IDF và embeddings.")
    table(document, ["Số từ mỗi bài", "UPenn", "Stanford"],
          [[label, *[str(item["word_lengths"][key]) for item in items]]
           for label, key in [("Nhỏ nhất", "min"), ("Trung vị", "median"),
                              ("Trung bình", "mean"), ("Phân vị 95%", "p95"), ("Lớn nhất", "max")]], [9, 4, 4])
    figure(document, "03_lengths.png", "Hình 3. Tỷ lệ bài theo nhóm độ dài. Nguồn: data_audit.json, trường length_histogram.")
    document.add_heading("Thí nghiệm dự kiến", 2)
    document.add_paragraph(
        "So sánh TF-IDF kết hợp Logistic Regression, Linear SVM và Multinomial Naive Bayes với "
        "MiniLM pretrained kết hợp Logistic Regression. MiniLM được dùng để tạo đặc trưng với trọng số giữ nguyên. "
        "Độ dài token và cách xử lý bài dài sẽ được đo riêng bằng tokenizer khi chạy thí nghiệm embeddings.")
    document.add_paragraph(
        "Chưa chia train/test ở bước này. Cần xác minh ID là sinh viên, bài đăng hay khóa học trước khi chọn "
        "quy tắc chia nhóm. Kiểm tra hiện tại chỉ loại nội dung trùng chính xác, chưa loại trùng gần nghĩa "
        "hay quan hệ giữa các bài cùng luồng. Nhãn độ khẩn cấp cũng chưa xác nhận bài đã được trả lời hay chưa.")

    document.add_page_break()
    document.add_heading("Tái tạo minh chứng và kiểm tra nguồn", 1)
    document.add_paragraph(
        "Từ thư mục machine, chạy lệnh sau để tải tệp còn thiếu, xử lý dữ liệu và tạo lại toàn bộ hình cùng file Word. "
        "CSV đã có được dùng lại; không tự cài thư viện và không tải mô hình.")
    paragraph = document.add_paragraph("bash Project/scripts/launchers/run.sh --download")
    paragraph.runs[0].font.name = "Courier New"
    document.add_paragraph("Khi hai CSV gốc đã có, có thể chạy lại hoàn toàn ngoại tuyến bằng bash Project/scripts/launchers/run.sh.")
    figure(document, "04_checks.png", "Hình 4. Các điều kiện đã được kiểm tra bằng assert trên dữ liệu thật trong prepare_data.py.")
    document.add_paragraph(
        "Đối chiếu từng tệp với manifest bằng: python3 Project/scripts/verify_evidence.py. "
        "Manifest ghi SHA-256 của dữ liệu gốc, dữ liệu xử lý, mã nguồn, hình PNG và báo cáo. "
        "rejected_rows.csv ghi nguồn, số thứ tự bản ghi CSV và lý do loại để kiểm tra từng trường hợp.")
    document.add_paragraph(
        "Kiểm tra hai lượt chạy độc lập bằng: python3 Project/scripts/check_reproducibility.py. "
        "Script so SHA-256 của toàn bộ tệp trong manifest, bao gồm cả hình và Word; báo FAIL nếu có khác biệt.")
    table(document, ["Nguồn", "SHA-256 tệp gốc viết tắt"],
          [[source, item["raw_sha256"][:24] + "…"] for source, item in zip(["UPenn", "Stanford"], items)], [3, 14])
    document.add_paragraph("Giá trị SHA-256 đầy đủ và URL tải cố định nằm trong reports/data_audit.json.")
    document.add_heading("Tài liệu nguồn", 2)
    for text in [
        "Švábenský và cộng sự (2023). Towards Generalizable Detection of Urgency of Discussion Forum Posts. https://arxiv.org/abs/2307.07614",
        "Kho dữ liệu và mã của tác giả: https://github.com/pcla-code/forum-posts-urgency",
        "Agrawal và Paepcke (2014). The Stanford MOOC Posts Dataset. Thông tin nguồn được dẫn trong README của kho dữ liệu tác giả.",
    ]:
        paragraph = document.add_paragraph(text)
        for run in paragraph.runs:
            run.font.size = Pt(8.5)
    path = REPORTS / "minh_chung_xu_ly_du_lieu.docx"
    for paragraph in document.paragraphs:
        for border in list(paragraph._p.findall(".//" + qn("w:pBdr"))):
            border.getparent().remove(border)
    canonical_save(document, path)
    dump_json(REPORTS / "environment.json", {"python": sys.version.split()[0], "python_executable": sys.executable,
        "python_docx": docx.__version__, "pillow": PIL.__version__, "system": platform.system(),
        "machine": platform.machine(), "installed_by_this_run": False})
    files = [path, REPORTS / "data_audit.json", REPORTS / "rejected_rows.csv", REPORTS / "environment.json"]
    files += list(FIGURES.glob("*.png")) + list((ROOT / "scripts").glob("*.py"))
    files += [ROOT / name for name in ["README.md", "AGENTS.md", "scripts/launchers/env.sh", "scripts/launchers/run.sh", "requirements-report.txt"]]
    files += list((ROOT / "data/raw").glob("*.csv")) + list((ROOT / "data/processed").glob("*.csv"))
    dump_json(REPORTS / "evidence_manifest.json", {"algorithm": "sha256", "files": {
        str(file.relative_to(ROOT)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(files)}})
    print("Đã tạo 4 hình minh chứng và báo cáo Word từ dữ liệu đã xử lý.")


if __name__ == "__main__":
    main()
