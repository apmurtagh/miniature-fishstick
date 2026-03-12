#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:46:01 2026

@author: amurtagh
"""

import json

eo_file = "eos_engineering_debug_thinfile_fixed.jsonl"
narr_file = "artifacts/baselines/lgbm_numeric_v1_subsample/openai_narratives_engineering_debug_with_drivers_top5.jsonl"
out_file = "openai_narratives_engineering_debug_thinfile_with_drivers_top5.jsonl"

thinfile_eids = set()
with open(eo_file) as f:
    for line in f:
        eo = json.loads(line)
        thinfile_eids.add(str(eo["event_id"]))

records_written = 0
with open(narr_file) as infile, open(out_file, "w") as outfile:
    for line in infile:
        narr = json.loads(line)
        if str(narr.get("event_id")) in thinfile_eids:
            outfile.write(json.dumps(narr) + "\n")
            records_written += 1

print(f"Wrote {records_written} thin-file narratives to {out_file}")