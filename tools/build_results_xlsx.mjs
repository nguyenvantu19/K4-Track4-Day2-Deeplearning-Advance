import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputPath = process.argv[2] ?? "results.xlsx";
const workbook = Workbook.create();
const font = "Arial";
const navy = "#1F4E78";
const paleBlue = "#D9EAF7";
const lightGray = "#F3F6F8";

const columns = {
  Summary: [
    "rank", "exp_id", "stage", "configuration", "macro-F1 val", "top-1 val",
    "ECE val", "p95 latency (ms), batch 1", "relative cost vs I00", "notes",
  ],
  Backbones: [
    "exp_id", "backbone", "pretrained weight tag", "parameters (M)", "GMAC",
    "image size", "epochs", "seed", "macro-F1 val", "top-1 val",
    "train time / epoch (s)", "latency batch 1 (ms)", "notes",
  ],
  Training: [
    "exp_id", "backbone", "changed axis (A-G)", "difference from T00", "seed",
    "macro-F1 val", "top-1 val", "Δ macro-F1 vs T00", "rare-class F1", "notes",
  ],
  Inference: [
    "exp_id", "method", "model/checkpoint", "K (views/models)", "macro-F1 val",
    "top-1 val", "ECE val", "p50 latency (ms), batch 1", "p95 latency (ms), batch 1",
    "p99 latency (ms), batch 1", "throughput (images/s)", "relative cost vs I00", "notes",
  ],
  Final: [
    "exp_id", "configuration", "seed", "macro-F1 val", "macro-F1 test", "top-1 test",
    "ECE test", "mean ± std across seeds", "notes",
  ],
  PerClass: [
    "configuration", "class", "test images", "precision", "recall", "F1", "notes",
  ],
  Latency: [
    "configuration", "GPU", "torch version", "dtype", "image size", "batch", "fused BN",
    "warmup runs", "measured runs", "p50 (ms)", "p95 (ms)", "p99 (ms)", "images/s", "notes",
  ],
};

function styleTable(sheet, headerRow, colCount) {
  const header = sheet.getRangeByIndexes(headerRow - 1, 0, 1, colCount);
  header.format = {
    fill: navy,
    font: { name: font, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  sheet.getRangeByIndexes(headerRow, 0, 40, colCount).format = {
    font: { name: font, size: 10, color: "#1F2937" },
    verticalAlignment: "center",
    borders: { preset: "insideHorizontal", style: "thin", color: "#D9E2F3" },
  };
  sheet.getRangeByIndexes(headerRow, 0, 40, colCount).format.rowHeight = 19;
}

for (const [name, header] of Object.entries(columns)) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.tabColor = name === "Summary" ? navy : "#5B9BD5";
  sheet.getRange("A2").values = [[name === "Summary" ? "DeepWeeds Lab Day 2 — Experiment Summary" : `DeepWeeds Lab Day 2 — ${name}`]];
  sheet.getRange("A2").format = {
    font: { name: font, size: 14, bold: true, color: navy },
    verticalAlignment: "center",
  };
  sheet.getRange("A3").values = [[
    name === "Summary"
      ? "Populate from actual validation/test logs only. Macro-F1 is the primary selection metric."
      : "Enter one row for each actual run. Keep exp_id consistent with curves, predictions, and report.",
  ]];
  sheet.getRange("A3").format = { font: { name: font, size: 10, italic: true, color: "#5B6573" } };
  sheet.getRange("A5").values = [header];
  styleTable(sheet, 5, header.length);
  sheet.getRangeByIndexes(4, 0, 1, header.length).format.rowHeight = 34;
  sheet.getRangeByIndexes(5, 0, 40, header.length).format.fill = lightGray;
  sheet.getRangeByIndexes(5, 0, 40, header.length).format.fill = "#FFFFFF";
  sheet.getRangeByIndexes(5, 0, 40, header.length).format.numberFormat = "0.0000";
  sheet.getRange("A1").format.rowHeight = 8;
  sheet.getRangeByIndexes(0, 0, 45, header.length).format.font = { name: font, size: 10, color: "#1F2937" };
  sheet.getRangeByIndexes(0, 0, 45, header.length).format.autofitColumns();
  for (let column = 0; column < header.length; column += 1) {
    const target = sheet.getRangeByIndexes(0, column, 45, 1);
    target.format.columnWidth = Math.min(28, Math.max(13, header[column].length * 0.95));
  }
  sheet.getRangeByIndexes(5, 0, 40, header.length).format.wrapText = false;
  sheet.freezePanes.freezeRows(5);
}

const summary = workbook.worksheets.getItem("Summary");
summary.getRange("A7:J10").format.fill = paleBlue;
summary.getRange("A7").values = [["Required final checks"]];
summary.getRange("A7").format = { font: { name: font, size: 10, bold: true, color: navy } };
summary.getRange("A8:A10").values = [
  ["• Final and T00/I00 must each have at least three seeds."],
  ["• Test is run exactly once per seed after configuration selection on validation."],
  ["• Copy score/grade output from eval.py into this workbook and the report."],
];
summary.getRange("A8:A10").format = { font: { name: font, size: 10, color: "#1F2937" } };

workbook.recalculate();
const check = await workbook.inspect({
  kind: "table",
  range: "Summary!A2:J10",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 10,
});
if (!check.ndjson.includes("DeepWeeds Lab Day 2")) throw new Error("Workbook verification failed");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 50 },
});
if (!errors.ndjson.includes("matched 0 entries")) throw new Error(`Formula errors detected: ${errors.ndjson}`);
await fs.mkdir(path.dirname(outputPath), { recursive: true });
const previewDir = ".artifact_preview";
await fs.mkdir(previewDir, { recursive: true });
for (const name of Object.keys(columns)) {
  const preview = await workbook.render({ sheetName: name, range: "A1:N15", scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(outputPath);
