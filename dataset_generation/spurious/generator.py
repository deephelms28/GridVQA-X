import random
import math
import numpy as np
import re
from templates import QuestionTemplates
from renderer import SceneRenderer
from masks import MaskGenerator
from renderer import SceneRenderer
from PIL import Image

class DatasetGenerator:
    """
    Generates the dataset examples: scenes, questions, answers, and ground‑truth masks.
    """
    def __init__(self, grid_size=5, density=0.3, cell_size=64, draw_grid=False, seed=42):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.density = density
        self.renderer = SceneRenderer(grid_size, cell_size, draw_grid=draw_grid)
        random.seed(seed)
        np.random.seed(seed)

    def sample_scene(self, ph_map, depth, qtype, idx=0):
        """
        Place anchor objects uniquely, then sample target objects within the intersection region
        defined by directional constraints, and fill remaining cells per density.
        Args:
            ph_map (dict): mapping of placeholders to values, includes 'anchor1','anchor2', 'direction1','direction2', and 'target' placeholders.
        Returns:
            objects (list of dict), anchor_coords (dict), target_coords (list of tuples)
        """
        objects = []
        used_positions = set()
        anchor_coords = {}
        num_targets = 1

        total_cells = self.grid_size * self.grid_size
        desired = int(self.density * total_cells)

        # Identify anchors and directions
        anchor_keys = [k for k in ph_map if k.startswith('anchor')]
        dir_keys = [k for k in ph_map if k.startswith('direction')]

        if (qtype == 'CMP'):
            anchor_keys1 = [anchor_keys[i] for i in range(0, int(len(anchor_keys)/2))]
            dir_keys1 = [dir_keys[i] for i in range(0, int(len(dir_keys)/2))]
            anchor_keys2 = [ak for ak in anchor_keys if ak not in anchor_keys1]
            dir_keys2 = [dk for dk in dir_keys if dk not in dir_keys1]
            num_targets = 2
        
        for i in range(num_targets):
            if num_targets == 2:
                if i == 0:
                    anchor_keys = anchor_keys1
                    dir_keys = dir_keys1
                else:
                    anchor_keys = anchor_keys2
                    dir_keys = dir_keys2
            
            # Place anchors, handling one left of/right of and one above/below pair specially
            # Detect one horizontal pair
            horiz_pairs = [(ak, dk) for ak, dk in zip(anchor_keys, dir_keys)
                    if ph_map[dk] in ('left of','right of')]
            horiz_pairs_set = set([ph_map[dk] for (ak, dk) in horiz_pairs])
            handled_horiz = False
            if len(horiz_pairs_set) >= 2:
                left_aks  = [ak for ak, dk in zip(anchor_keys, dir_keys) if ph_map[dk]=='left of']
                right_aks = [ak for ak, dk in zip(anchor_keys, dir_keys) if ph_map[dk]=='right of']
                # Require each left anchor to be >= 2 from the left boundary
                for ak in left_aks:
                    if (qtype == 'SO'):
                        color, shape = random.choice(QuestionTemplates.COLORS), ph_map[ak]
                    elif (qtype == 'CO'):
                        color, shape = ph_map[ak], random.choice(QuestionTemplates.SHAPES)
                    else:
                        color, shape = ph_map[ak].split()
                    while True:
                        x = random.randrange(2, self.grid_size)
                        y = random.randrange(self.grid_size)
                        if (x,y) not in used_positions:
                            anchor_coords[ak] = (x,y)
                            used_positions.add((x,y))
                            objects.append({'x':x,'y':y,'color':color,'shape':shape})
                            break
                if left_aks:
                    max_left_x = min(anchor_coords[ak][0] for ak in left_aks)  # the most restrictive bound
                    # Ensure each right of anchor is at x <= (max_left_x – 2)
                else:
                    max_left_x = self.grid_size - 1  # no constraint from left anchors
                bound = max(0, max_left_x - 2)
                for ak in right_aks:
                    if (qtype == 'SO'):
                        color, shape = random.choice(QuestionTemplates.COLORS), ph_map[ak]
                    elif (qtype == 'CO'):
                        color, shape = ph_map[ak], random.choice(QuestionTemplates.SHAPES)
                    else:
                        color, shape = ph_map[ak].split()
                    while True:
                        x = random.randrange(0, bound+1)
                        y = random.randrange(self.grid_size)
                        if (x,y) not in used_positions:
                            anchor_coords[ak] = (x,y)
                            used_positions.add((x,y))
                            objects.append({'x':x,'y':y,'color':color,'shape':shape})
                            break
                handled_horiz = True

            # Detect one vertical pair
            vert_pairs = [(ak, dk) for ak, dk in zip(anchor_keys, dir_keys)
                    if ph_map[dk] in ('above','below')]
            vert_pairs_set = set([ph_map[dk] for (ak, dk) in vert_pairs])
            handled_vert = False
            if len(vert_pairs_set) >= 2:
                # Collect all vertical pairs
                above_aks = [ak for ak, dk in zip(anchor_keys, dir_keys) if ph_map[dk] == 'above']
                below_aks = [ak for ak, dk in zip(anchor_keys, dir_keys) if ph_map[dk] == 'below']
                # Place all “above” anchors (require y >= 2)
                for ak in above_aks:
                    if (qtype == 'SO'):
                        color, shape = random.choice(QuestionTemplates.COLORS), ph_map[ak]
                    elif (qtype == 'CO'):
                        color, shape = ph_map[ak], random.choice(QuestionTemplates.SHAPES)
                    else:
                        color, shape = ph_map[ak].split()
                    while True:
                        x = random.randrange(self.grid_size)
                        y = random.randrange(2, self.grid_size)
                        if (x, y) not in used_positions:
                            anchor_coords[ak] = (x, y)
                            used_positions.add((x, y))
                            objects.append({'x': x, 'y': y, 'color': color, 'shape': shape})
                            break
                # Determine the most restrictive “above” bound
                if above_aks:
                    min_above_y = min(anchor_coords[ak][1] for ak in above_aks)
                    # every “below” anchor must be y <= (min_above_y - 2)
                    bound_y = max(0, min_above_y - 2)
                else:
                    bound_y = self.grid_size - 1
                # Place all “below” anchors
                for ak in below_aks:
                    if (qtype == 'SO'):
                        color, shape = random.choice(QuestionTemplates.COLORS), ph_map[ak]
                    elif (qtype == 'CO'):
                        color, shape = ph_map[ak], random.choice(QuestionTemplates.SHAPES)
                    else:
                        color, shape = ph_map[ak].split()
                    while True:
                        x = random.randrange(self.grid_size)
                        y = random.randrange(0, bound_y + 1)
                        if (x, y) not in used_positions:
                            anchor_coords[ak] = (x, y)
                            used_positions.add((x, y))
                            objects.append({'x': x, 'y': y, 'color': color, 'shape': shape})
                            break
                handled_vert = True
            # Reset
            if (i == 1):
                anchor_keys = [k for k in ph_map if k.startswith('anchor')]
                dir_keys = [k for k in ph_map if k.startswith('direction')]

        # Place remaining anchors normally
        if qtype == 'BTW':
            for i, ak in enumerate(anchor_keys):
                while True:
                    color, shape = ph_map[ak].split()
                    x = random.randrange(self.grid_size)
                    y = random.randrange(self.grid_size)
                    ok = True
                    if i==1 and (abs(x-objects[0]['x']) <= 1 or abs(y-objects[0]['y']) <= 1):
                        ok = False
                    if ok:
                        anchor_coords[ak] = (x,y)
                        used_positions.add((x,y))
                        objects.append({'x':x,'y':y,'color':color,'shape':shape})
                        break

        for ak, dk in zip(anchor_keys, dir_keys):
            if ak in anchor_coords.keys():
                continue
            # parse the anchor and its required direction
            if (qtype == 'SO'):
                color, shape = random.choice(QuestionTemplates.COLORS), ph_map[ak]
            elif (qtype == 'CO'):
                color, shape = ph_map[ak], random.choice(QuestionTemplates.SHAPES)
            else:
                color, shape = ph_map[ak].split()
            direction = ph_map[dk]
            # sample position ensuring region non-empty
            while True:
                x = random.randrange(self.grid_size)
                y = random.randrange(self.grid_size)
                ok = True
                if direction=='left of' and x<=0: ok=False
                if direction=='right of' and x>=self.grid_size-1: ok=False
                if direction=='above' and y<=0: ok=False
                if direction=='below' and y>=self.grid_size-1: ok=False
                if not ok or (x,y) in used_positions:
                    continue
                anchor_coords[ak] = (x,y)
                used_positions.add((x,y))
                objects.append({'x':x,'y':y,'color':color,'shape':shape})
                break

        valids = []
        target_coords = []
        target_coords1, target_coords2, cmp = [], [], 0 # only useful for CMP type questions
        for t in range(num_targets):
            if (num_targets == 2):
                if (t == 0):
                    anchor_keys = anchor_keys1
                    dir_keys = dir_keys1
                else:
                    anchor_keys = anchor_keys2
                    dir_keys = dir_keys2
            # Compute intersection region for targets
            # Start full grid
            valid = set((i,j) for i in range(self.grid_size) for j in range(self.grid_size))
            if qtype == 'BTW':
                x0,y0 = anchor_coords['anchor1']
                x1,y1 = anchor_coords['anchor2']
                region = set()
                for i in range(min(x0, x1)+1, max(x0, x1)):
                    for j in range(min(y0, y1)+1, max(y0, y1)):
                        region.add((i,j))
                valid = valid & region

            for ak, dk in zip(anchor_keys, dir_keys):
                x0,y0 = anchor_coords[ak]
                direction = ph_map[dk]
                region = set()
                for i,j in valid:
                    if direction=='left of' and i < x0: region.add((i,j))
                    elif direction=='right of' and i > x0: region.add((i,j))
                    elif direction=='above' and j < y0: region.add((i,j))
                    elif direction=='below' and j > y0: region.add((i,j))
                valid = valid & region
            valid -= used_positions 
            valids.append(valid)

            # valid now holds intersection; ensure non-empty
            if not valid:
                print(f"Warning: no valid target region for {qtype} at depth {depth} question={ph_map})")
                for items in anchor_coords.items():
                    print(items)
                print("\n")
                raise ValueError("No valid target region for anchors/directions") # should NEVER happen
            # Determine target placeholder key
            if (qtype == 'SO'):
                target_keys = [k for k in ph_map if k.startswith('shape')]
            elif (qtype == 'CO'):
                target_keys = [k for k in ph_map if k.startswith('color')]
            else:
                target_keys = [k for k in ph_map if k.startswith('target')]
            # Sample targets within valid region
            num_valid = len(valid)
            if num_targets == 2:
                x = 0.6
                if depth == 1 and t==0:
                    if math.isclose(self.density, 0.3): x = self.density/1.65 # found by experimentation
                    else: x = self.density/1.8
                pct = random.uniform(0.1, min(x, (desired - len(objects))/num_valid))
                n_targets = max(1, min(num_valid, int(pct * num_valid)))
            else:
                pct = random.uniform(0.1, min(0.9, (desired - len(objects))/num_valid))
                lst = [i for i in range(1, min(desired - len(objects), num_valid) + 1)]
                n_targets = random.choice(lst)

            if qtype == 'A' and idx==0:
                # half the time sample target to be the majority class
                if random.random() < 0.5:
                    if desired == 7:
                        #------------------------------------4 is getting chosen less--------------------------
                        n_targets = random.choice([4, 5, 6, 7])
                    else:
                        target_probability = random.uniform(0.48, 0.9)
                        n_targets = math.ceil(target_probability * desired)
                else: 
                    # half the time select a distractor for the majority class
                    all_colors_shapes = [(c, s) for c in QuestionTemplates.COLORS for s in QuestionTemplates.SHAPES]
                    tc, ts = ph_map[target_keys[t]].split()
                    all_colors_shapes.remove((tc, ts))
                    major = random.choice(all_colors_shapes)
                    if desired == 7:
                        n_targets = random.choice([1, 2, 3])
                        if n_targets == 3:
                            n_majors = 4
                        elif n_targets == 2:
                            n_majors = random.choice([4, 5])
                        else:
                            n_majors = random.choice([4, 5, 6])
                    else:
                        lst = [i+1 for i in range(8)]
                        n_targets = random.choice(lst)
                        if n_targets == 8:
                            n_majors = 9
                        elif n_targets == 7:
                            n_majors = random.choice([9, 10])
                        elif n_targets == 6:
                            n_majors = random.choice([9, 10, 11])
                        else:
                            n_majors = random.choice([9, 10, 11, 12, 13])
                    # pct_maj = random.uniform(0.5, 0.8)
                    # n_majors = math.ceil(pct_maj * desired)
                    sampled_majors = random.sample(list(valid), k=n_majors)
                    for pos in sampled_majors:
                        objects.append({'x':pos[0],'y':pos[1],'color':major[0],'shape':major[1]})
                        used_positions.add(pos)
                    valid -= used_positions
                    num_valid = len(valid)
                    # n_target_candidates = [i+1 for i in range(desired - len(objects))]
                    # pct = random.uniform(0.1, (desired - len(objects))/num_valid)
                    # n_targets = max(1, min(num_valid, int(pct * desired)))

            
            if idx==1: # existence question
                n_targets = np.random.choice([0, 1, n_targets], p=[0.5, 0.25, 0.25]) # for idx=1, sample 0 targets with half probability
            #print(f"Sampling {n_targets} of type {t} targets for {qtype} at depth {depth} (valid={num_valid})")
            
            # handling majority class sp corr for other non-A buckets
            sampled_majors = []
            n_majors = 0
            suppress = False
            if idx == 0 and qtype != 'A' and qtype != 'CMP':
                n_anchors = len([k for k in ph_map if k.startswith('anchor')])
                remaining = desired - n_anchors  # total non-anchor budget
                half = max(1, remaining // 2)
                if random.random() < 0.5:
                    # SUPPRESS: target minority
                    lst = [i for i in range(1, min(half, num_valid) + 1)]
                    n_targets = random.choice(lst)
                    n_majors = n_targets + 1
                    suppress = True
                else:
                    # BOOST: target majority
                    lower = min(half + 1, num_valid)
                    upper = min(remaining, num_valid)
                    lst = [i for i in range(lower, upper + 1)]
                    n_targets = random.choice(lst)

            sampled_targets = random.sample(list(valid), k=n_targets)
            if not sampled_targets:
                if (qtype == 'SO'):
                    tcolor, tshape = random.choice(QuestionTemplates.COLORS), ph_map[target_keys[0]]
                elif (qtype == 'CO'):
                    tcolor, tshape = ph_map[target_keys[0]], random.choice(QuestionTemplates.SHAPES)
                else:
                    tcolor, tshape = ph_map[target_keys[t]].split()
        
            for pos in sampled_targets:
                if (qtype == 'SO'):
                    tcolor, tshape = random.choice(QuestionTemplates.COLORS), ph_map[target_keys[0]]
                elif (qtype == 'CO'):
                    tcolor, tshape = ph_map[target_keys[0]], random.choice(QuestionTemplates.SHAPES)
                else:
                    tcolor, tshape = ph_map[target_keys[t]].split() 
                objects.append({'x':pos[0],'y':pos[1],'color':tcolor,'shape':tshape})
                if num_targets==2 and t==0: target_coords1.append(pos)
                elif num_targets==2 and t==1: target_coords2.append(pos)
                target_coords.append(pos)
                used_positions.add(pos)

            # Place majority distractor objects (suppress branch of sp corr balancing)
            if suppress and n_majors > 0:
                # Build distractor, excluding anchors
                if qtype == 'SO':
                    tshape_q = ph_map[target_keys[0]]
                    anchor_shapes = set(ph_map[ak] for ak in ph_map
                                        if ak.startswith('anchor'))
                    dist_shapes = [s for s in QuestionTemplates.SHAPES
                                   if s != tshape_q and s not in anchor_shapes]
                    if not dist_shapes:
                        dist_shapes = [s for s in QuestionTemplates.SHAPES
                                       if s != tshape_q]
                    maj_shape = random.choice(dist_shapes)
                elif qtype == 'CO':
                    tcolor_q = ph_map[target_keys[0]]
                    anchor_colors = set(ph_map[ak] for ak in ph_map
                                        if ak.startswith('anchor'))
                    dist_colors = [c for c in QuestionTemplates.COLORS
                                   if c != tcolor_q and c not in anchor_colors]
                    if not dist_colors:
                        dist_colors = [c for c in QuestionTemplates.COLORS
                                       if c != tcolor_q]
                    maj_color = random.choice(dist_colors)
                else:  # M
                    tc_q, ts_q = ph_map[target_keys[t]].split()
                    anchor_pairs = set()
                    for ak in ph_map:
                        if ak.startswith('anchor'):
                            anchor_pairs.add(tuple(ph_map[ak].split()))
                    dist_pool = [(c, s) for c in QuestionTemplates.COLORS
                                 for s in QuestionTemplates.SHAPES
                                 if (c, s) != (tc_q, ts_q)
                                 and (c, s) not in anchor_pairs]
                    if not dist_pool:
                        dist_pool = [(c, s) for c in QuestionTemplates.COLORS
                                     for s in QuestionTemplates.SHAPES
                                     if (c, s) != (tc_q, ts_q)]
                    maj_pair = random.choice(dist_pool)

                # Sample positions from anywhere unused on the grid
                all_unused = [(x, y) for x in range(self.grid_size)
                              for y in range(self.grid_size)
                              if (x, y) not in used_positions]
                n_majors = min(n_majors, len(all_unused))
                sampled_majors = random.sample(all_unused, k=n_majors)
                for pos in sampled_majors:
                    if qtype == 'SO':
                        mc = random.choice(QuestionTemplates.COLORS)
                        objects.append({'x': pos[0], 'y': pos[1],
                                        'color': mc, 'shape': maj_shape})
                    elif qtype == 'CO':
                        ms = random.choice(QuestionTemplates.SHAPES)
                        objects.append({'x': pos[0], 'y': pos[1],
                                        'color': maj_color, 'shape': ms})
                    else:  # M
                        objects.append({'x': pos[0], 'y': pos[1],
                                        'color': maj_pair[0], 'shape': maj_pair[1]})
                    used_positions.add(pos)

        # Reset
        anchor_keys = [k for k in ph_map if k.startswith('anchor')]
        dir_keys = [k for k in ph_map if k.startswith('direction')]

        # Fill remainder up to density
        more = max(0, desired - len(objects))
        all_colors_shapes = [(c, s) for c in QuestionTemplates.COLORS for s in QuestionTemplates.SHAPES]
        # Remove colors/shapes of anchors
        for ak in anchor_keys:
            if (qtype == 'SO'):
                shape = ph_map[ak]
                all_colors_shapes = [(c, s) for (c, s) in all_colors_shapes if s != shape]
            elif (qtype == 'CO'):
                color = ph_map[ak]
                all_colors_shapes = [(c, s) for (c, s) in all_colors_shapes if c != color]
            else:
                color, shape = ph_map[ak].split()
                if (color, shape) in all_colors_shapes:
                    all_colors_shapes.remove((color, shape))

        if qtype == 'CMP':
            tc, ts = ph_map[target_keys[0]].split()
            tc2, ts2 = ph_map[target_keys[1]].split()
        for _ in range(more):
            while True:
                # sample (i,j) half the time from the NOT-valid region, half the time anywhere
                if qtype == 'CMP':
                    combined_valid = valids[0] | valids[1]
                else:
                    combined_valid = valids[0]
                cell_list = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
                not_valid = [cell for cell in cell_list if cell not in combined_valid]
                if random.random() < 0.5:
                    # try to pick from cells outside the combined valid region
                    if not_valid:
                        i, j = random.choice(not_valid)
                        outside = True
                    else:
                        i = random.randrange(self.grid_size)
                        j = random.randrange(self.grid_size)
                        outside = False
                else:
                    i = random.randrange(self.grid_size)
                    j = random.randrange(self.grid_size)
                if (i, j) in not_valid:
                    outside = True
                else:
                    outside = False
                if (i,j) not in used_positions:
                    used_positions.add((i,j))
                    break
            (c, s) = random.choice(all_colors_shapes)
            if qtype == 'CO':
                targets = [(tcolor, shape) for shape in QuestionTemplates.SHAPES]
                non_targets = [item for item in all_colors_shapes if item not in targets]
                (c, s) = random.choice(non_targets)
            elif qtype == 'SO':
                targets = [(color, tshape) for color in QuestionTemplates.COLORS]
                non_targets = [item for item in all_colors_shapes if item not in targets]
                (c, s) = random.choice(non_targets)
            elif qtype == 'M' :
                non_targets = [item for item in all_colors_shapes if item != (tcolor, tshape)]
                (c, s) = random.choice(non_targets)
            elif qtype == 'CMP':
                non_targets = [item for item in all_colors_shapes if item != (tc, ts) and item != (tc2, ts2)]
                (c, s) = random.choice(non_targets)

            flag = False
            if qtype == 'CMP' and ((c, s) == (tc, ts) and (i, j) in valids[0]):
                flag = True
                target_coords1.append((i, j))
                target_coords.append((i, j))
            elif qtype == 'CMP' and ((c, s) == (tc2, ts2) and (i, j) in valids[1]):
                flag = True
                target_coords2.append((i, j))
                target_coords.append((i, j))
            elif (qtype == 'SO' and s == tshape and (i,j) in valid) or (qtype == 'CO' and c == tcolor and (i,j) in valid):
                flag = True
                target_coords.append((i,j))
            elif c == tcolor and s == tshape and (i, j) in valid:
                flag = True
                target_coords.append((i, j))
            if flag and idx == 1:
                if qtype == 'SO' or qtype == 'CO':
                    target_coords.pop()
                else:
                    all_colors_shapes.remove((c,s))
                    (c, s) = random.choice(all_colors_shapes)
                    all_colors_shapes.append((c,s))
                    target_coords.pop()
            objects.append({'x':i,'y':j,'color':c,'shape':s})
        # remove objects from SO and CO templates that bias towards yes for the existence questions
        objects2 = objects.copy()
        if qtype == 'SO' and idx == 1:
            for item in objects2:
                pt = (item['x'], item['y'])
                if pt in valid and not pt in target_coords:
                    if item['shape'] == tshape:
                        objects.remove(item)
        if qtype == 'CO' and idx == 1:
            for item in objects2:
                pt = (item['x'], item['y'])
                if pt in valid and not pt in target_coords:
                    if item['color'] == tcolor:
                        objects.remove(item)
        if qtype == 'CMP': valid = valids[0] | valids[1]
        else: valid = valids[0]
        num_valid = len(valid)
        if len(target_coords1) > len(target_coords2):
            cmp = 1
        #print(f"Sampling {len(target_coords)} targets ({len(target_coords1)} of type 0 and {len(target_coords2)} of type 1) for {qtype} at depth {depth} (valid={num_valid})")
        return objects, anchor_coords, target_coords, valid, valids, cmp


    def generate_example(self, depth, qtype, idx=0):
        """
        Generate a single dataset example.
        Args:
            depth (int): question depth (1,2,3+)
            qtype (str): question type key
        Returns:
            dict with keys: 'image', 'question', 'ph_map', 'objects', 'masks'
        """
        # instantiate question
        question, ph_map = QuestionTemplates.instantiate(depth, qtype, idx)
        # sample scene with placeholders
        objects, anchor_coords, target_coords, intersection_region, regions, cmp = self.sample_scene(ph_map, depth, qtype, idx)

        # render image
        self.renderer.cell_size = self.cell_size
        self.renderer.grid_size = self.grid_size
        image = self.renderer.render(objects)
        
        anchors = list(anchor_coords.values())
        if idx == 0:
            answer = len(target_coords)
        elif idx==1 and len(target_coords) >= 1:
            answer = "yes"
        elif idx==1 and len(target_coords) == 0:
            answer = "no"
        if qtype == 'CMP':
            if cmp == 1:
                answer = "yes"
            else:
                answer = "no"
        # generate masks
        cell_mask = MaskGenerator.cell_mask(self.grid_size, anchors, target_coords)
        pixel_mask = MaskGenerator.pixel_obj_mask(cell_mask, self.grid_size)
        token_str = question.split(" ")
        token_str[-1] = token_str[-1].rstrip("?")
        for i in range(len(token_str)):
            token_str[i] = token_str[i].rstrip("s")            
            token_str[i] = token_str[i].rstrip(",")
        text_mask = MaskGenerator.text_token_mask(token_str)
        pos_mask = MaskGenerator.positional_mask(self.grid_size, intersection_region)

        # TODO: will add an interaction mask as well

        # Bounding box annotations for pre-training MDETR
        objects_in_scene = []
        anchor_objects_in_scene = []
        target_objects_in_scene = []
        anchors = [item[1] for item in anchor_coords.items()]
        if qtype == 'SO':
            relevant_shapes = [ph_map[key] for key in ph_map.keys() if key != 'direction']
            for obj in objects:
                if (obj['shape']) in relevant_shapes and (obj['x'], obj['y']) in anchor_coords.values():
                    anchor_objects_in_scene.append(obj)
                    objects_in_scene.append(obj)
                if (obj['shape']) in relevant_shapes and ((obj['x'], obj['y']) in target_coords):
                    target_objects_in_scene.append(obj)
                    objects_in_scene.append(obj)
        if qtype == 'CO':
            relevant_colors = [ph_map[key] for key in ph_map.keys() if key != 'direction']
            for obj in objects:
                if (obj['color']) in relevant_colors and (obj['x'], obj['y']) in anchor_coords.values():
                    anchor_objects_in_scene.append(obj)
                    objects_in_scene.append(obj)
                if (obj['color']) in relevant_colors and (obj['x'], obj['y']) in target_coords:
                    target_objects_in_scene.append(obj)
                    objects_in_scene.append(obj)
        else:
            relevant_pairs = [tuple(ph_map[key].split()) for key in ph_map.keys() if key != 'direction']
            for obj in objects:
                if (obj['color'], obj['shape']) in relevant_pairs and (obj['x'], obj['y']) in anchor_coords.values():
                    anchor_objects_in_scene.append(obj)
                    objects_in_scene.append(obj)
                if (obj['color'], obj['shape']) in relevant_pairs and (obj['x'], obj['y']) in target_coords:
                    target_objects_in_scene.append(obj)
                    objects_in_scene.append(obj)
    
        masks = {'cell': cell_mask, 'pixel': pixel_mask, 'text': text_mask, 'positional': pos_mask}
        return {
            'image': image,
            'question': question,
            'ph_map': ph_map,
            'objects': objects_in_scene,
            'all_scene_objects': objects,
            'anchor_objects': anchor_objects_in_scene,
            'target_objects': target_objects_in_scene,
            'answer': answer,
            'masks': masks
        }


# testing

if __name__ == '__main__':

    dg = DatasetGenerator(grid_size=5, density=0.7)
    # example = dg.generate_example(3, 'M', 0)
    # print(example['question'])
    # print(example['question'])
    # print(example['objects'])
    # print(example['masks'])
    # example['image'].show()
    ph_map ={'target': 'blue square'}
    qtype = 'A'
    objects, anchor_coords, target_coords, valid, regions, cmp = dg.sample_scene(ph_map, 1, qtype, 1)
    
    print("cmp: ", cmp)

    for map in objects:
        print(map, end=",\n")
    print("\n")

    print("Anchors:")
    for items in anchor_coords.items():
        print(items)
    print("\n")

    print("Targets:")
    for items in target_coords:
        print(items)
    print("\n")

    objects_in_scene = []
    anchor_objects_in_scene = []
    target_objects_in_scene = []
    anchors = [item[1] for item in anchor_coords.items()]
    if qtype == 'SO':
        relevant_shapes = [ph_map[key] for key in ph_map.keys() if key != 'direction']
        for obj in objects:
            if (obj['shape']) in relevant_shapes and (obj['x'], obj['y']) in anchor_coords.values():
                anchor_objects_in_scene.append(obj)
                objects_in_scene.append(obj)
            elif (obj['shape']) in relevant_shapes and ((obj['x'], obj['y']) in target_coords):
                target_objects_in_scene.append(obj)
                objects_in_scene.append(obj)
    if qtype == 'CO':
        relevant_colors = [ph_map[key] for key in ph_map.keys() if key != 'direction']
        for obj in objects:
            if (obj['color']) in relevant_colors and (obj['x'], obj['y']) in anchor_coords.values():
                anchor_objects_in_scene.append(obj)
                objects_in_scene.append(obj)
            elif (obj['color']) in relevant_colors and (obj['x'], obj['y']) in target_coords:
                target_objects_in_scene.append(obj)
                objects_in_scene.append(obj)
    else:
        relevant_pairs = [tuple(ph_map[key].split()) for key in ph_map.keys() if key != 'direction']
        for obj in objects:
            if (obj['color'], obj['shape']) in relevant_pairs and (obj['x'], obj['y']) in anchor_coords.values():
                anchor_objects_in_scene.append(obj)
                objects_in_scene.append(obj)
            elif (obj['color'], obj['shape']) in relevant_pairs and (obj['x'], obj['y']) in target_coords:
                target_objects_in_scene.append(obj)
                objects_in_scene.append(obj)

    for map in objects_in_scene:
        print(map, end=",\n")
    print("\n")
    print("Anchor Objects:")
    for map in anchor_objects_in_scene:
        print(map, end=",\n")
    print("\n")
    print("Target Objects:")
    for map in target_objects_in_scene:
        print(map, end=",\n")
    print("\n")

    renderer = SceneRenderer(grid_size=5, cell_size=64, bg_color='white', draw_grid=True)
    img = renderer.render(objects)
    np_img = np.array(img)
    print(np_img.shape)
    # Save the image
    img.save("./images/scene.png")
