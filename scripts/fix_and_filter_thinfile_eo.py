#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:42:31 2026

@author: amurtagh
"""

import json

input_file = "eos_engineering_debug_200.jsonl"
output_file = "eos_engineering_debug_thinfile_fixed.jsonl"

buffer = ""
records_written = 0

with open(input_file) as infile, open(output_file, "w") as outfile:
    for line in infile:
        if line.strip() == "":
            continue
        buffer += line
        if line.strip().endswith("}"):
            try:
                eo = json.loads(buffer)
                # Write only thin-file events
                if eo.get("thin_file_flag") or eo.get("evidence_strength") == "LOW":
                    outfile.write(json.dumps(eo) + "\n")
                    records_written += 1
            except Exception as e:
                print("Error:", e)
            buffer = ""
print(f"Wrote {records_written} thin-file EOs to {output_file}")