#!/usr/bin/env python3
"""
convert_ipynb_to_txt.py

Simple utility to convert Jupyter Notebook (.ipynb) files into human-readable
plain-text (.txt) files. It extracts markdown and code cells (and outputs where
available) and writes them to a .txt file with clear separators.

Usage:
    python convert_ipynb_to_txt.py path/to/notebook.ipynb
    python convert_ipynb_to_txt.py path/to/folder/with/ipynb_files

The script only uses the Python standard library and works on Windows PowerShell.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Dict, Any, Iterable, List


CELL_SEPARATOR = "\n" + ("-" * 80) + "\n"


def extract_text_from_cell(cell: Dict[str, Any]) -> str:
    """Return a human readable text representation of a notebook cell."""
    cell_type = cell.get("cell_type", "unknown")
    source = "".join(cell.get("source", []) or [])
    parts: List[str] = [f"Cell type: {cell_type}"]

    if cell_type == "markdown":
        parts.append("-- Markdown --")
        parts.append(source.rstrip())

    elif cell_type == "code":
        parts.append("-- Code --")
        parts.append(source.rstrip())

        # Try to capture outputs (text and error outputs)
        outputs = cell.get("outputs", []) or []
        if outputs:
            out_lines: List[str] = ["-- Outputs --"]
            for oi, out in enumerate(outputs, start=1):
                out_type = out.get("output_type", "output")
                out_lines.append(f"[{oi}] output_type: {out_type}")

                # text/plain
                text = out.get("text")
                if text:
                    out_lines.append("".join(text).rstrip())

                # data (e.g., {'text/plain': [...]})
                data = out.get("data") or {}
                if isinstance(data, dict):
                    if "text/plain" in data:
                        out_lines.append("".join(data["text/plain"]).rstrip())

                # error traceback
                if out_type == "error":
                    tb = out.get("traceback") or []
                    out_lines.extend([line.rstrip() for line in tb])

            parts.append("\n".join(out_lines))

    else:
        # Generic fallback for other cell types
        parts.append(source.rstrip())

    return "\n".join(parts)


def convert_notebook(ipynb_path: str, txt_path: str) -> None:
    """Read a .ipynb file and write a .txt representation to txt_path."""
    with io.open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", []) or []

    with io.open(txt_path, "w", encoding="utf-8") as out:
        out.write(f"Notebook: {os.path.basename(ipynb_path)}\n")
        out.write(f"Kernelspec: {nb.get('metadata', {}).get('kernelspec', {})}\n")
        out.write(CELL_SEPARATOR)

        for i, cell in enumerate(cells, start=1):
            out.write(f"Cell {i}\n")
            out.write(extract_text_from_cell(cell))
            out.write(CELL_SEPARATOR)


def find_ipynb_files(path: str) -> Iterable[str]:
    """Yield .ipynb file paths from path. If path is a file, yield it.
    If path is a directory, yield .ipynb files in it (non-recursive).
    """
    if os.path.isfile(path):
        if path.lower().endswith(".ipynb"):
            yield path
        else:
            return
    elif os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(".ipynb"):
                yield os.path.join(path, name)
    else:
        return


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Jupyter .ipynb notebooks to plain .txt files."
    )
    parser.add_argument("path", help="Path to a .ipynb file or a directory containing .ipynb files")
    parser.add_argument("-o", "--outdir", help="Output directory for .txt files. Defaults to the same folder as the input.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")

    args = parser.parse_args(argv)

    paths = list(find_ipynb_files(args.path))
    if not paths:
        print(f"No .ipynb files found at: {args.path}")
        return 2

    # Default output directory: the directory where this script lives (Helper Scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or script_dir
    for ipynb in paths:
        base = os.path.splitext(os.path.basename(ipynb))[0]
        target_dir = outdir
        os.makedirs(target_dir, exist_ok=True)
        txt_path = os.path.join(target_dir, base + ".txt")

        if os.path.exists(txt_path) and not args.overwrite:
            print(f"Skipping existing file (use --overwrite to replace): {txt_path}")
            continue

        try:
            convert_notebook(ipynb, txt_path)
            print(f"Converted: {ipynb} -> {txt_path}")
        except Exception as e:
            print(f"Failed to convert {ipynb}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
