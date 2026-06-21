# FIX -> Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

import os
import json
import random
import argparse
import numpy as np
import re # <-- Added import
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# HuggingFace tokenizer for subword mapping. Change to the tokenizer matching the checkpoint if needed.
from transformers import AutoTokenizer

# Import local modules as before
from templates import QuestionTemplates
from generator import DatasetGenerator
from tqdm import tqdm 

# -------------------------
# Configurable constants
# -------------------------
SPLITS = {'val': 1.0}

# JSONL output filenames
GROUNDING_JSONL_TPL = "grounding_{split}.jsonl"
QA_JSONL_TPL = "qa_{split}.jsonl"

# Tokenizer: must match the text encoder used by the checkpoint (Roberta-like for MDETR EB5)
TOKENIZER_NAME = "roberta-base"

# Tiny Gaussian noise sigma (disabled by default). Set >0 to add small noise during generation.
GAUSSIAN_NOISE_SIGMA = 0.0

# Default cell size fallback
DEFAULT_CELL_SIZE = 64

# ============================================================
# Per-bucket generation config
# ============================================================
# Keys:   "depth{N}" -> qtype -> density_key -> form_key -> TOTAL count
# Density keys:  "d03" = grid 5, density 0.3
#                "d07" = grid 5, density 0.7
# Form keys:     "form0" = counting question (template index 0)
#                "form1" = yes/no   question (template index 1)
# CMP only has form0 (single template).
# Split ratios (train/val/test) are applied to each leaf total.
# ============================================================

GENERATION_CONFIG = {
    # --- DEPTH 1 (The Foundation) ---
    "depth1": {
        "A":   {"d03": {"form0": 0, "form1": 0}, "d07": {"form0": 50, "form1": 50}}, 
        
        "CO":  {"d03": {"form0": 0, "form1": 0},  "d07": {"form0": 50, "form1": 50}},
        "SO":  {"d03": {"form0": 0, "form1": 0},  "d07": {"form0": 50, "form1": 50}},
        
        "M":   {"d03": {"form0": 0, "form1": 0},  "d07": {"form0": 50, "form1": 50}},
        
        "CMP": {"d03": {"form0": 0},                "d07": {"form0": 50}}
    },

    # --- DEPTH 2 ---
    "depth2": {
        "CO":  {"d03": {"form0": 0, "form1": 0}, "d07": {"form0": 50, "form1": 50}},
        "SO":  {"d03": {"form0": 0, "form1": 0}, "d07": {"form0": 50, "form1": 50}},
        
        "M":   {"d03": {"form0": 0, "form1": 0}, "d07": {"form0": 50, "form1": 50}},
        
        "CMP": {"d03": {"form0": 0},               "d07": {"form0": 50}}
    },

    # --- DEPTH 3 ---
    "depth3": {
        "M":   {"d03": {"form0": 0, "form1": 0}, "d07": {"form0": 50, "form1": 50}},
        
        "CMP": {"d03": {"form0": 0},               "d07": {"form0": 50}}
    },
}

# Reverse map from density key -> (grid_size, density_float)

_DENSITY_KEY_MAP = {
    "d03": (5, 0.3),
    "d07": (5, 0.7),
}

# -------------------------
# Globals used for JSONL emission
# -------------------------
# Persistent counters and category map across generation
NEXT_IMAGE_ID = 1
NEXT_ANN_ID = 1
# --- Fix 2: Pre-populate CATEGORIES_MAP ---
CATEGORIES = [f"{c} {s}" for c in QuestionTemplates.COLORS for s in QuestionTemplates.SHAPES]
CATEGORIES_MAP: Dict[str, int] = {name: i + 1 for i, name in enumerate(CATEGORIES)}  # maps "color shape" -> category_id (starting from 1)
# --- End Fix 2 ---

count = 0
count_y = 0

# initialize tokenizer (fast)
_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)


# -------------------------
# Helper functions
# -------------------------
def compute_split_quotas(total: int):
    # (Unchanged from your file)
    q = {s: int(total * frac) for s, frac in SPLITS.items()}
    rem = total - sum(q.values())
    if 'train' in q.keys():
        q['train'] += rem
    return q


def ensure_dir(path: str):
    # (Unchanged from your file)
    os.makedirs(path, exist_ok=True)


def save_npz(ex, out_root, split, depth, qtype, idx, grid, dens, local_i):
    # (Unchanged from your file)
    global count, count_y
    sub = os.path.join(out_root, split,
                       f"depth{depth}", qtype,
                       f"form{idx}", f"g{grid}_d{str(dens).replace('.', '')}")
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, f"{local_i:06d}.npz")
    img = np.array(ex['image'])
    if ex['answer'] == 'no':
        count += 1
    if ex['answer'] == 'yes':
        count_y += 1
    np.savez_compressed(path,
        image      = img,
        question   = ex['question'],
        answer     = ex['answer'],
        objects    = ex['objects'],
        anchor_objects = ex['anchor_objects'],
        target_objects = ex['target_objects'],
        cell_mask  = ex['masks']['cell'],
        pixel_mask = ex['masks']['pixel'],
        text_mask  = ex['masks']['text'],
        pos_mask   = ex['masks']['positional']
    )
    return path  # return path for downstream use (JSONL emission)

# -------------------------
# Spurious correlation tracker (Case 1: target = majority)
# -------------------------
from collections import Counter

class SpuriousCorrelationTracker:
    """
    Tracks at generation time whether the target object's color/shape/class
    is the majority among all objects in the scene.
    
    For CO buckets: checks if target color is the majority color.
    For SO buckets: checks if target shape is the majority shape.
    For other buckets (A, M, CMP, BTW): checks if the target (color, shape)
        class is the most frequently occurring class.
    
    Usage:
        tracker = SpuriousCorrelationTracker()
        # Inside generation loop, after generate_example():
        tracker.record(ex, qtype)
        # After generation:
        tracker.report()
        tracker.save_csv("spurious_report.csv")
    """
    def __init__(self):
        # Per-bucket counters: bucket -> {"total": N, "majority": M}
        self.buckets = defaultdict(lambda: {"total": 0, "majority": 0})

    def record(self, ex, qtype, depth, form_idx, grid, dens):
        """
        Record one generated example.
        
        Args:
            ex: dict returned by generate_example(), must have:
                'objects' (list of all scene-relevant objects),
                'target_objects' (list of target objects),
                'anchor_objects' (list of anchor objects)
            qtype: str, e.g. 'CO', 'SO', 'A', 'M', 'CMP'
            depth: int
            form_idx: int
            grid: int
            dens: float
        """
        bucket_key = f"D{depth}_{qtype}_F{form_idx}_g{grid}_d{str(dens).replace('.', '')}"
        self.buckets[bucket_key]["total"] += 1

        all_objects = ex.get('all_scene_objects', ex.get('objects', []))
        target_objects = ex.get('target_objects', [])

        if not all_objects or not target_objects:
            return

        if qtype == 'CO':
            # Check if target color is majority color
            target_color = target_objects[0]['color'].lower()
            color_counts = Counter(obj['color'].lower() for obj in all_objects)
            majority_color = color_counts.most_common(1)[0][0]
            if target_color == majority_color:
                self.buckets[bucket_key]["majority"] += 1

        elif qtype == 'SO':
            # Check if target shape is majority shape
            target_shape = target_objects[0]['shape'].lower()
            shape_counts = Counter(obj['shape'].lower() for obj in all_objects)
            majority_shape = shape_counts.most_common(1)[0][0]
            if target_shape == majority_shape:
                self.buckets[bucket_key]["majority"] += 1

        else:
            # A, M, CMP, BTW: check if target (color, shape) is most common class
            target_class = (target_objects[0]['color'].lower(),
                            target_objects[0]['shape'].lower())
            class_counts = Counter(
                (obj['color'].lower(), obj['shape'].lower()) for obj in all_objects
            )
            majority_class = class_counts.most_common(1)[0][0]
            if target_class == majority_class:
                self.buckets[bucket_key]["majority"] += 1

    def report(self):
        """Print a summary of spurious correlation prevalence per bucket."""
        print("\n===== Spurious Correlation Report (Case 1: target = majority) =====")
        print(f"{'Bucket':<40} {'Total':>8} {'Majority':>10} {'Ratio':>8}")
        print("-" * 70)
        total_all, majority_all = 0, 0
        for bucket in sorted(self.buckets.keys()):
            d = self.buckets[bucket]
            ratio = d["majority"] / d["total"] if d["total"] > 0 else 0.0
            print(f"{bucket:<40} {d['total']:>8} {d['majority']:>10} {ratio:>8.4f}")
            total_all += d["total"]
            majority_all += d["majority"]
        overall = majority_all / total_all if total_all > 0 else 0.0
        print("-" * 70)
        print(f"{'OVERALL':<40} {total_all:>8} {majority_all:>10} {overall:>8.4f}")
        print("=" * 70)

    def save_csv(self, path):
        """Save the report to a CSV file."""
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["bucket", "total", "majority_count", "ratio"])
            for bucket in sorted(self.buckets.keys()):
                d = self.buckets[bucket]
                ratio = d["majority"] / d["total"] if d["total"] > 0 else 0.0
                writer.writerow([bucket, d["total"], d["majority"], f"{ratio:.4f}"])
# -------------------------
# Sentence / span building utilities
# -------------------------
def _pluralize(shape: str) -> str:
    """Simple pluralizer for circle/triangle/square -> adds 's'."""
    return shape + 's'


def _group_objects_by_color_shape(objects: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[int]]:
    grouped = defaultdict(list)
    for idx, obj in enumerate(objects):
        key = (obj['color'].lower(), obj['shape'].lower())
        grouped[key].append(idx)
    return grouped


def _group_objects_by_shape(objects: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    grouped = defaultdict(list)
    for idx, obj in enumerate(objects):
        grouped[obj['shape'].lower()].append(idx)
    return grouped


def _group_objects_by_color(objects: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    # (Unchanged from your file)
    grouped = defaultdict(list)
    for idx, obj in enumerate(objects):
        grouped[obj['color'].lower()].append(idx)
    return grouped

# --- (NEW helper function to find character spans robustly) ---
def _find_char_span(text: str, phrase: str, start_search: int = 0) -> Tuple[int, int]:
    """Helper to find character span, returning (-1, -1) if not found."""
    try:
        # Search for phrase as whole words, case-insensitive
        b_phrase = r"\b" + re.escape(phrase) + r"\b"
        match = None
        # Find the first match *at or after* the start_search index
        for m in re.finditer(b_phrase, text, flags=re.IGNORECASE):
            if m.start() >= start_search:
                match = m
                break
        if match:
            return match.start(), match.end()
    except Exception:
        pass # Fallback

    # Fallback to simple, case-insensitive find
    start = text.lower().find(phrase.lower(), start_search)
    if start != -1:
        end = start + len(phrase)
        return start, end
        
    return -1, -1


# --- (This REPLACES your old _build_sentence_and_word_spans) ---
def _build_sentence_and_word_spans(
    qtype: str, 
    objects: List[Dict[str, Any]], 
    anchor_objects: List[Dict[str, Any]], 
    target_objects: List[Dict[str, Any]], 
    ph_map: Dict[str, str],
    question: str, # Add question as an argument
    form_idx: int # Add form_idx to handle A/form1
):
    """
    Uses the VQA question, identifies anchor/target phrases based on qtype and ph_map,
    maps them to object indices (relative to 'objects' list), calculates word spans.
    """
    
    sentence_text = question # Use the real question
    question_words = sentence_text.split()
    
    # Fix 1: Remove trailing '?' from the last word
    if question_words and question_words[-1].endswith('?'):
        question_words[-1] = question_words[-1][:-1]

    # Map scene objects to their original index (0..N-1)
    obj_to_orig_idx = { (o['x'], o['y'], o['color'], o['shape']): i for i, o in enumerate(objects) }

    # Map target/anchor objects to their original indices
    target_indices = [obj_to_orig_idx.get((t['x'], t['y'], t['color'], t['shape'])) for t in target_objects]
    target_indices = sorted(list(set(i for i in target_indices if i is not None)))
    
    anchor_indices_list = [obj_to_orig_idx.get((a['x'], a['y'], a['color'], a['shape'])) for a in anchor_objects]
    anchor_indices_list = sorted(list(set(i for i in anchor_indices_list if i is not None)))
    
    # This list will hold tuples of (phrase_text, list_of_object_indices)
    phrases_to_ground = []
    
    # --- Casework by qtype (Fixes 3, 4, 5, 6) ---
    
    if qtype == 'A':
        # "How many red circles?" -> {'target': 'red circle'}
        # "Is there any yellow circle?" -> {'target': 'yellow circle'}
        phrase = ph_map['target']
        # --- FIX 1 (A/form1): Only pluralize for form0 (counting) ---
        if form_idx == 0: # Pluralize ONLY for "how many"
             phrase_parts = phrase.split()
             if phrase_parts[-1] in QuestionTemplates.SHAPES:
                 phrase_parts[-1] = _pluralize(phrase_parts[-1])
                 phrase = " ".join(phrase_parts)
        # For form1 ("Is there any..."), phrase is always singular, which matches ph_map['target']
        phrases_to_ground.append((phrase, target_indices))

    elif qtype == 'CMP' and 'anchor1' not in ph_map:
        # Depth 1 CMP: "Are there more red circles than green triangles?"
        # ph_map: {'target1': 'red circle', 'target2': 'green triangle'}
        
        target1_desc = ph_map['target1']
        # --- FIX 3: Get indices from target_objects (which contains ALL relevant objects) ---
        target1_indices = [i for i in target_indices if objects[i]['color'] == target1_desc.split()[0] and objects[i]['shape'] == target1_desc.split()[1]]
        target1_desc = f"{target1_desc.split()[0]} {_pluralize(target1_desc.split()[1])}"
        phrases_to_ground.append((target1_desc, target1_indices))
        
        target2_desc = ph_map['target2']
        # --- FIX 3: Get indices from target_objects ---
        target2_indices = [i for i in target_indices if objects[i]['color'] == target2_desc.split()[0] and objects[i]['shape'] == target2_desc.split()[1]]
        target2_desc = f"{target2_desc.split()[0]} {_pluralize(target2_desc.split()[1])}"
        phrases_to_ground.append((target2_desc, target2_indices))
    
    elif qtype == 'CMP' and 'anchor1' in ph_map:
        # Depth 2/3 CMP: "Are there more [target1 sub-query] than [target2 sub-query]?"
        
        # 1. Ground Anchors
        anchor_keys = sorted([k for k in ph_map if k.startswith('anchor')])
        for i, key in enumerate(anchor_keys):
             phrase = ph_map[key]
             if i < len(anchor_indices_list): # Use the simple ordered list
                 phrases_to_ground.append((phrase, [anchor_indices_list[i]]))

        # 2. Ground Sub-query 1
        target1_desc = ph_map['target1']
        # --- FIX 4: Get indices from target_objects ---
        target1_indices_all = [i for i, o in enumerate(objects) if o['color'] == target1_desc.split()[0] and o['shape'] == target1_desc.split()[1]]
        target1_indices_final = [i for i in target_indices if i in target1_indices_all] # Filter by final targets
        
        plural_target1 = target1_desc
        plural_target1 = f"{target1_desc.split()[0]} {_pluralize(target1_desc.split()[1])}"
        sub_query_1_text = f"{plural_target1} {ph_map['direction1']} {ph_map['anchor1']}"
        if 'direction3' in ph_map and 'anchor3' in ph_map: # Check for D3 keys
             sub_query_1_text += f" and {ph_map['direction2']} {ph_map['anchor2']}"
        phrases_to_ground.append((sub_query_1_text, target1_indices_final))
        
        # 3. Ground Sub-query 2
        target2_desc = ph_map['target2']
        # --- FIX 4: Get indices from target_objects ---
        target2_indices_all = [i for i, o in enumerate(objects) if o['color'] == target2_desc.split()[0] and o['shape'] == target2_desc.split()[1]]
        target2_indices_final = [i for i in target_indices if i in target2_indices_all] # Filter by final targets

        plural_target2 = target2_desc
        plural_target2 = f"{target2_desc.split()[0]} {_pluralize(target2_desc.split()[1])}"
        
        # Fix: Correctly get direction/anchor keys for D2 and D3
        direction_key_2 = 'direction3' if 'direction3' in ph_map else 'direction2'
        anchor_key_2 = 'anchor3' if 'anchor3' in ph_map else 'anchor2'
        
        # Check if these keys exist before trying to access them
        if direction_key_2 in ph_map and anchor_key_2 in ph_map:
            sub_query_2_text = f"{plural_target2} {ph_map[direction_key_2]} {ph_map[anchor_key_2]}"
            if 'direction4' in ph_map and 'anchor4' in ph_map: # D3-specific keys
                 sub_query_2_text += f" and {ph_map['direction4']} {ph_map['anchor4']}"
            phrases_to_ground.append((sub_query_2_text, target2_indices_final))
        else:
            # This case should not be hit if ph_map is correct for D2/D3 CMP
            print(f"Warning: Missing keys for D2/D3 CMP sub-query 2. ph_map: {ph_map}")
        # global DEBUG
        # if not DEBUG:
        #     print(f"DEBUG CMP D2/D3: Question: {question}")
        #     print(f"DEBUG CMP D2/D3: ph_map: {ph_map}")
        #     print(f"DEBUG CMP D2/D3: Objects: {obj_to_orig_idx}")
        #     print(f"DEBUG CMP D2/D3: Anchor objects: {anchor_objects}")
        #     print(f"DEBUG CMP D2/D3: Target objects: {target_objects}")
        #     print(f"DEBUG CMP D2/D3: Anchor Indices List: {anchor_indices_list}")
        #     print(f"DEBUG CMP D2/D3: Target Indices List: {target_indices}")
        #     print(f"DEBUG CMP D2/D3: Target1 Desc: {target1_desc}, Indices: {target1_indices_final}")
        #     print(f"DEBUG CMP D2/D3: Target2 Desc: {target2_desc}, Indices: {target2_indices_final}")
        #     print(f"DEBUG CMP D2/D3: Phrases to ground: {phrases_to_ground}")
        #     DEBUG = 1
    else:
        # Relational types: M, SO, CO, BTW
        
        # 1. Ground Anchors
        anchor_keys = sorted([k for k in ph_map if k.startswith('anchor')])
        for i, key in enumerate(anchor_keys):
            if i < len(anchor_indices_list):
                phrase = ph_map[key]
                # Fix 5: Use "color object" for CO anchors
                if key.startswith('anchor_color'):
                    phrase = f"{phrase} object"
                phrases_to_ground.append((phrase, [anchor_indices_list[i]]))

        # 2. Ground Core Relational Phrase to Targets
        core_phrase = re.sub(r"^(How many|Is there any)\s*", "", question, flags=re.IGNORECASE).strip()
        core_phrase = re.sub(r"\?$", "", core_phrase).strip()
        phrases_to_ground.append((core_phrase, target_indices))

    # --- Convert all collected (phrase, indices) tuples to word spans ---
    word_spans_final = []
    phrase_annid_lists_final = []
    
    # Build a map of word_index -> char_start_index
    char_indices_of_word_starts = [0]
    cumulative_len = 0
    for i, word in enumerate(question_words):
        char_indices_of_word_starts.append(cumulative_len + len(word) + 1) # +1 for space
        cumulative_len += len(word) + 1
    
    processed_char_starts = set()
    
    # Sort by phrase length, longest first, to find "core phrase" before sub-phrases
    sorted_phrases = sorted(phrases_to_ground, key=lambda x: len(x[0]), reverse=True)

    for phrase, obj_indices in sorted_phrases:
        # Allow processing phrases with empty obj_indices (for negative grounding)
        if obj_indices is None:
            print(f"Warning: phrase '{phrase}' has None obj_indices; skipping.")
        char_start, char_end = -1, -1
        current_search_start = 0
        while True: # Find first occurrence that hasn't been used
            start, end = _find_char_span(question, phrase, current_search_start)
            if start == -1: break
            if start not in processed_char_starts:
                char_start, char_end = start, end
                processed_char_starts.add(char_start)
                break
            current_search_start = start + 1
            if current_search_start >= len(question): break

        if char_start != -1:
            # Convert char span to word span
            start_word, end_word = -1, -1
            for i in range(len(question_words)):
                word_start_char = char_indices_of_word_starts[i]
                # End char index of the word (exclusive of space)
                word_end_char = char_indices_of_word_starts[i+1] - 1
                
                if max(char_start, word_start_char) < min(char_end, word_end_char):
                    if start_word == -1:
                        start_word = i
                    end_word = i
            
            if start_word != -1:
                word_spans_final.append((start_word, end_word))
                phrase_annid_lists_final.append(obj_indices) # Keep original indices

    # Sort by word start index for consistency
    sorted_combined = sorted(zip(word_spans_final, phrase_annid_lists_final), key=lambda x: x[0][0])
    if sorted_combined:
        word_spans_final, phrase_annid_lists_final = zip(*sorted_combined)
    else:
        word_spans_final, phrase_annid_lists_final = [], []
    # if qtype == 'CMP' and 'anchor1' in ph_map and 'anchor3' not in ph_map:
    #     print(f"Question: '{question}' | Word Spans: {word_spans_final} | Phrase Ann IDs: {phrase_annid_lists_final}")
    return sentence_text, question_words, list(word_spans_final), list(phrase_annid_lists_final)


def _map_word_spans_to_token_spans(words: List[str], word_spans: List[Tuple[int, int]], tokenizer=None):
    """
    Map word-level spans (indices into 'words' list) to tokenizer subword token spans using the tokenizer.
    Returns list of (token_start, token_end) for each word span (subword token indices, INCLUSIVE).
    """
    if tokenizer is None:
        tokenizer = _tokenizer
    
    # --- MODIFICATION: Use whole sentence tokenization for accuracy ---
    sentence_text = ' '.join(words)
    enc_full = tokenizer(sentence_text, add_special_tokens=True, return_offsets_mapping=True)
    full_offsets = enc_full['offset_mapping']
    
    char_spans = []
    char_idx = 0
    word_start_char = []
    for w in words:
        word_start_char.append(char_idx)
        char_idx += len(w) + 1 # +1 for space
    
    for (w_s, w_e) in word_spans:
        if w_s < len(word_start_char) and w_e < len(words):
            char_start = word_start_char[w_s]
            char_end = word_start_char[w_e] + len(words[w_e])
            char_spans.append((char_start, char_end))
        else:
            char_spans.append((-1, -1))
        
    token_spans = []
    for (char_start, char_end) in char_spans:
        if char_start == -1:
             token_spans.append((0, 0)); continue
             
        tok_s, tok_e = -1, -1
        for i, (start, end) in enumerate(full_offsets):
            if start is None or end is None: continue
            if max(char_start, start) < min(char_end, end):
                if tok_s == -1:
                    tok_s = i
                tok_e = i
        if tok_s != -1:
            token_spans.append((tok_s, tok_e))
        else:
             token_spans.append((0, 0))
            
    return token_spans


# -------------------------
# Grounding / QA JSONL emitter
# -------------------------
def emit_grounding_and_qa(
    out_grounding_jsonl_path: str,
    out_qa_jsonl_path: str,
    img_rel_path: str,
    image_id: int,
    objects: List[Dict[str, Any]],
    anchor_objects: List[Dict[str, Any]],
    target_objects: List[Dict[str, Any]],
    ph_map: Dict[str, str],
    question: str,
    answer,
    cell_size: int,
    categories_map: Dict[str, int], # This argument is now used
    ann_id_start: int,
    qtype: str = None,
    form_idx: int = 0, # --- ADDED form_idx ---
    tokenizer=None
) -> Tuple[int, Dict[str, Any]]:
    """
    Write a grounding JSONL entry and a QA JSONL line for one generated image.
    Returns: (next_ann_id, grounding_entry)
    """
    global _tokenizer
    if tokenizer is None:
        tokenizer = _tokenizer

    annotations = []
    local_ann_ids_map = {} # Maps object index (0..N-1) to annotation id
    ann_id = ann_id_start
    
    # --- Fix 2 (Duplicate Annos): Map ALL object indices to ann_ids first ---
    for i, obj in enumerate(objects):
        local_ann_ids_map[i] = ann_id + i
    
    # Increment ann_id counter *after* mapping all objects in the scene
    ann_id += len(objects)

    # --- Call the new span building function ---
    sentence_text, words, word_spans, phrase_annid_lists = _build_sentence_and_word_spans(
        qtype=qtype,
        objects=objects,
        anchor_objects=anchor_objects,
        target_objects=target_objects,
        ph_map=ph_map,
        question=question, # Pass the real question
        form_idx=form_idx # --- Pass form_idx ---
    )
    
    token_spans = _map_word_spans_to_token_spans(words, word_spans, tokenizer=tokenizer)

    spans_out = []
    referenced_ann_ids = set() # Keep track of which ann_ids are used
    # Convert phrase_annid_lists (object indices) -> actual annotation ids
    for ((w_s, w_e), (t_s, t_e), obj_index_list) in zip(word_spans, token_spans, phrase_annid_lists):
        ann_ids_for_span = [ local_ann_ids_map[obj_idx] for obj_idx in obj_index_list if obj_idx in local_ann_ids_map ]
        # --- Fix 1: Handle empty annotations for 'A' type 'form1' ("Is there...") ---
        if qtype == 'A' and form_idx == 1 and not target_objects:
             spans_out.append({
                'span': [w_s, w_e],
                'token_span': [t_s, t_e],
                'box_ids': [] # Correctly map to empty list
             })
        elif ann_ids_for_span: # Only add span if it maps to valid annotations
            spans_out.append({
                'span': [w_s, w_e],
                'token_span': [t_s, t_e],
                'box_ids': ann_ids_for_span
            })
            referenced_ann_ids.update(ann_ids_for_span) # Track referenced IDs
    # --- Fix 2 (Duplicate Annos): Filter annotations to only those referenced in spans ---
    final_annotations = []
    for i, obj in enumerate(objects):
        ann_id_for_obj = local_ann_ids_map.get(i)
        if ann_id_for_obj in referenced_ann_ids:
            col = int(obj['x'])
            row = int(obj['y'])
            x_min = col * cell_size
            y_min = row * cell_size
            bbox = [int(x_min), int(y_min), int(cell_size), int(cell_size)]
            
            cat_name = f"{obj['color'].lower()} {obj['shape'].lower()}"
            cat_id = CATEGORIES_MAP.get(cat_name, 0) # Fix 2
            
            ann = {
                'id': ann_id_for_obj, # Use the mapped ID
                'image_id': image_id,
                'bbox': bbox,
                'category_id': cat_id, 
                'category_name': cat_name,
                'color': obj['color'].lower(),
                'shape': obj['shape'].lower()
            }
            final_annotations.append(ann)
    # --- End Fix 2 ---

    grounding_entry = {
        'image': {
            'id': image_id,
            'file_name': img_rel_path
        },
        'annotations': final_annotations, # Use the filtered list
        'sentence': {
            'text': sentence_text,
            'words': words,
            'spans': spans_out
        }
    }

    # Append to JSONL files
    Path(out_grounding_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_qa_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_grounding_jsonl_path, 'a') as gf:
        gf.write(json.dumps(grounding_entry) + '\n')
    with open(out_qa_jsonl_path, 'a') as qf:
        qf.write(json.dumps({'image_id': image_id, 'question': question, 'answer': answer}) + '\n')

    return ann_id, grounding_entry
# -------------------------
# Main build_dataset (config-driven)
# -------------------------
def build_dataset(out_root: str):
    """
    Config-driven dataset generation.
    Iterates GENERATION_CONFIG to produce the exact number of samples per bucket.
    """
    global NEXT_IMAGE_ID, NEXT_ANN_ID, CATEGORIES_MAP, _tokenizer, count, count_y

    if not CATEGORIES_MAP:
        print("Warning: CATEGORIES_MAP is empty. Re-populating.")
        CATEGORIES = [f"{c} {s}" for c in QuestionTemplates.COLORS for s in QuestionTemplates.SHAPES]
        CATEGORIES_MAP = {name: i + 1 for i, name in enumerate(CATEGORIES)}

    dg = DatasetGenerator()
    sp_tracker = SpuriousCorrelationTracker()
    os.makedirs(out_root, exist_ok=True)

    # Open split JSONL files once
    split_files = {}
    for split in SPLITS:
        f_ground = open(os.path.join(out_root, GROUNDING_JSONL_TPL.format(split=split)), 'w')
        f_qa     = open(os.path.join(out_root, QA_JSONL_TPL.format(split=split)), 'w')
        split_files[split] = (f_ground, f_qa)

    # ---------- main loop driven by GENERATION_CONFIG ----------
    for depth_key, qtypes in GENERATION_CONFIG.items():
        depth = int(depth_key.replace("depth", ""))          # "depth1" -> 1

        for qtype, densities in qtypes.items():
            for dens_key, forms in densities.items():
                grid, dens = _DENSITY_KEY_MAP[dens_key]      # "d03" -> (5, 0.3)

                for form_key, total in forms.items():
                    idx = int(form_key.replace("form", ""))   # "form0" -> 0

                    # Validate against templates
                    tpl_dict = QuestionTemplates.all_templates().get(depth, {})
                    if qtype not in tpl_dict or idx >= len(tpl_dict[qtype]):
                        print(f"[SKIP] No template for depth{depth}/{qtype}/form{idx}")
                        continue

                    # Compute per-split quotas from this leaf total
                    quotas = {s: int(total * frac) for s, frac in SPLITS.items()}
                    rem = total - sum(quotas.values())
                    quotas['train'] = quotas.get('train', 0) + rem

                    # Configure generator
                    dg.grid_size = grid
                    dg.density   = dens

                    for split, split_quota in quotas.items():
                        if split_quota == 0:
                            continue

                        out_grounding_jsonl_path = os.path.join(out_root, GROUNDING_JSONL_TPL.format(split=split))
                        out_qa_jsonl_path        = os.path.join(out_root, QA_JSONL_TPL.format(split=split))

                        dens_str = str(dens).replace('.', '')
                        sub = os.path.join(split, f"depth{depth}", qtype,
                                           f"form{idx}", f"g{grid}_d{dens_str}")
                        out_image_dir = os.path.join(out_root, 'images', sub)
                        os.makedirs(out_image_dir, exist_ok=True)

                        pbar_desc = f"D{depth} {qtype} f{idx} g{grid} d{dens} {split}"
                        pbar = tqdm(total=split_quota, desc=pbar_desc, leave=False)

                        i = 0
                        while i < split_quota:
                            try:
                                ex = dg.generate_example(depth=depth, qtype=qtype, idx=idx)

                                if GAUSSIAN_NOISE_SIGMA and GAUSSIAN_NOISE_SIGMA > 0.0:
                                    img_arr = np.array(ex['image'])
                                    noise = np.random.normal(0, GAUSSIAN_NOISE_SIGMA, img_arr.shape).astype(np.float32)
                                    img_arr = np.clip(img_arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                                    ex['image'] = img_arr

                                # --- Track spurious correlation ---
                                sp_tracker.record(ex, qtype, depth, idx, grid, dens)
                                npz_path = save_npz(ex, out_root, split, depth, qtype, idx, grid, dens, i)

                                img_arr = np.array(ex['image'])
                                H, W = img_arr.shape[0], img_arr.shape[1]
                                cell_size = W // grid if grid > 0 else DEFAULT_CELL_SIZE

                                img_rel_name = f"{i:06d}.png"
                                img_rel_path = os.path.join('images', sub, img_rel_name)
                                image_save_path = os.path.join(out_image_dir, img_rel_name)

                                from PIL import Image
                                Image.fromarray(np.array(ex['image']).astype(np.uint8)).save(image_save_path)

                                next_ann, grounding_entry = emit_grounding_and_qa(
                                    out_grounding_jsonl_path=out_grounding_jsonl_path,
                                    out_qa_jsonl_path=out_qa_jsonl_path,
                                    img_rel_path=img_rel_path,
                                    image_id=NEXT_IMAGE_ID,
                                    objects=ex['objects'],
                                    anchor_objects=ex['anchor_objects'],
                                    target_objects=ex['target_objects'],
                                    question=ex['question'],
                                    ph_map=ex['ph_map'],
                                    answer=ex['answer'],
                                    cell_size=cell_size,
                                    categories_map=CATEGORIES_MAP,
                                    ann_id_start=NEXT_ANN_ID,
                                    qtype=qtype,
                                    form_idx=idx,
                                    tokenizer=_tokenizer
                                )
                                NEXT_ANN_ID   = next_ann
                                NEXT_IMAGE_ID += 1
                                i += 1
                                pbar.update(1)

                            except Exception:
                                continue   # retry without incrementing i

                        pbar.close()

    # Close file handles
    for f_ground, f_qa in split_files.values():
        f_ground.close()
        f_qa.close()

    sp_tracker.report()
    sp_tracker.save_csv(os.path.join(out_root, "spurious_correlation_report.csv"))
    print("Done.")
    print('negatives:', count)
    print('positives:', count_y)


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output root")
    args = p.parse_args()
    build_dataset(args.out)