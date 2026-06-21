import random
import re

class QuestionTemplates:
    """
    Holds question templates and placeholder sets for CLEVR-Lite.
    """
    COLORS = ['red', 'green', 'blue', 'yellow']
    SHAPES = ['circle', 'square', 'triangle', 'pentagon']
    DIRECTIONS = ['left of', 'right of', 'above', 'below']

    # Depth 1 templates
    DEPTH1_TEMPLATES = {
        'A': [
            "How many {target}s?",
            "Is there any {target}?"
        ],
        'SO': [
            "How many {shape}s are {direction} {anchor_shape}?",
            "Is there any {shape} that is {direction} {anchor_shape}?"
        ],
        'CO': [
            "How many {color} objects are {direction} {anchor_color} object?",
            "Is there any {color} object that is {direction} {anchor_color} object?"
        ],
        'M': [
            "How many {target}s are {direction} {anchor}?",
            "Is there any {target} that is {direction} {anchor}?"
        ],
        'CMP': [
            "Are there more {target1}s than {target2}s?"
        ]#,
        #'AQ': [
        #    "What color is the {shape} that is {direction} {anchor}?",
        #    "What shape is {color} that is {direction} {anchor}?"
        #] 
    }

    # Depth 2 templates
    DEPTH2_TEMPLATES = {
        'SO': [
            "How many {shape}s are {direction1} {anchor_shape1} and {direction2} {anchor_shape2}?",
            "Is there any {shape} that is {direction1} {anchor_shape1} and {direction2} {anchor_shape2}?"
        ],
        'CO': [
            "How many {color} objects are {direction1} {anchor_color1} object and {direction2} {anchor_color2} object?",
            "Is there any {color} object that is {direction1} {anchor_color1} object and {direction2} {anchor_color2} object?"
        ],
        'M': [
            "How many {target}s are {direction1} {anchor1} and {direction2} {anchor2}?",
            "Is there any {target} that is {direction1} {anchor1} and {direction2} {anchor2}?"
        ],
        'CMP': [
            "Are there more {target1}s {direction1} {anchor1} than {target2}s {direction2} {anchor2}?"
        ]
        # 'BTW': [
        #     "How many {target}s are between {anchor1} and {anchor2}?",
        #     "Is there any {target} that is between {anchor1} and {anchor2}?"
        # ]
    }

    # Depth 3 templates
    DEPTH3_TEMPLATES = {
        'M': [
            "How many {target}s that are {direction1} {anchor1}, {direction2} {anchor2}, and {direction3} {anchor3}?",
            "Is there any {target} that is {direction1} {anchor1}, {direction2} {anchor2}, and {direction3} {anchor3}?"
        ],
        'CMP': [
            "Are there more {target1}s {direction1} {anchor1} and {direction2} {anchor2} than {target2}s {direction3} {anchor3} and {direction4} {anchor4}?"
        ]
    }

    @classmethod
    def all_templates(cls):
        return {
            1: cls.DEPTH1_TEMPLATES,
            2: cls.DEPTH2_TEMPLATES,
            3: cls.DEPTH3_TEMPLATES
        }
    
    @classmethod
    def sample_placeholders(cls, template: str):
        """
        Given a template string, detect placeholders and sample distinct concrete values.
        Ensures placeholders of the same class are unique within a question.
        Returns a dict mapping placeholder names to sampled values.
        """
        # Extract placeholder keys
        placeholders = re.findall(r"\{(.*?)\}", template)

        # Group placeholders by type
        color_keys        = [ph for ph in placeholders if ph.startswith('color') or ph.startswith('anchor_color')]
        shape_keys        = [ph for ph in placeholders if ph.startswith('shape') or ph.startswith('anchor_shape')]
        anchor_and_target_keys       = [ph for ph in placeholders if (ph.startswith('anchor') or ph.startswith('target'))
                              and not (ph.startswith('anchor_color') or ph.startswith('anchor_shape'))]
        dir_keys          = [ph for ph in placeholders if ph.startswith('direction')]

        values = {}

        # Sample distinct colors
        if color_keys:
            sampled_colors = random.sample(cls.COLORS, k=len(color_keys))
            for ph, col in zip(color_keys, sampled_colors):
                values[ph] = col

        # Sample distinct shapes
        if shape_keys:
            sampled_shapes = random.sample(cls.SHAPES, k=len(shape_keys))
            for ph, shp in zip(shape_keys, sampled_shapes):
                values[ph] = shp

        # Sample distinct full anchors & targets (color + shape)
        if anchor_and_target_keys:
            combos = [f"{c} {s}" for c in cls.COLORS for s in cls.SHAPES]
            sampled = random.sample(combos, k=len(anchor_and_target_keys))
            for ph, anc_tar in zip(anchor_and_target_keys, sampled):
                values[ph] = anc_tar

        # Sample directions (can repeat)
        for ph in dir_keys:
            values[ph] = random.choice(cls.DIRECTIONS)

        return values

    @classmethod
    def instantiate(cls, depth: int, qtype: str, idx: int):
        """
        Sample and return a fully instantiated question.
        Args:
            depth (int): relational depth (1, 2, or 3)
            qtype (str): one of the keys in the corresponding DEPTHx_TEMPLATES
        Returns:
            question (str), placeholder_map (dict)
        """
        templates = cls.all_templates().get(depth)
        if templates is None or qtype not in templates:
            raise ValueError(f"No templates for depth {depth}, type {qtype}")
        tmpl = templates[qtype][idx]
        ph_map = cls.sample_placeholders(tmpl)
        question = tmpl.format(**ph_map)
        return question, ph_map
    

# testing

def test_question_templates():
    for depth in [1, 2, 3]:
        for qtype in QuestionTemplates.all_templates()[depth].keys():
            for idx in range(len(QuestionTemplates.all_templates()[depth][qtype])):
                question, ph_map = QuestionTemplates.instantiate(depth, qtype, idx)
                print(f"Depth {depth}, Type {qtype}, Question: {question}")
                print(f"Placeholder map: {ph_map}\n")
            print("\n")
        print("\n")



if __name__ == "__main__":
    test_question_templates()
