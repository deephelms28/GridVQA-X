import numpy as np
from templates import QuestionTemplates

class MaskGenerator:
    @staticmethod
    def cell_mask(grid_size, anchors, targets):
        mask = np.zeros((grid_size, grid_size), dtype=int)
        for (x,y) in anchors + targets:
            mask[y, x] = 1
        return mask
    
    @staticmethod
    def pixel_obj_mask(cell_mask: np.ndarray,
                       cell_size: int) -> np.ndarray:
        """
        Upsample a (G×G) cell mask to (G·cell_size × G·cell_size) by
        repeating each cell_mask[i,j] into a cell_size×cell_size block.
        """
        G = cell_mask.shape[0]
        H = W = G * cell_size
        pix_mask = np.zeros((H, W), dtype=np.uint8)
        for i in range(G):
            for j in range(G):
                if cell_mask[i, j]:
                    y0, y1 = i*cell_size, (i+1)*cell_size
                    x0, x1 = j*cell_size, (j+1)*cell_size
                    pix_mask[y0:y1, x0:x1] = 1
        return pix_mask

    @staticmethod
    def text_token_mask(tokens) -> np.ndarray:
        """
        Build a length-T binary mask over question tokens.
          - ph_map: color/shape/anchor/direction tokens
          - auxiliary_indices: How/Is/Are tokens (aggregation cues)
        """
        T = len(tokens)
        words = QuestionTemplates.COLORS.copy()
        words.extend(QuestionTemplates.SHAPES)
        words.extend(QuestionTemplates.DIRECTIONS) 
        words.append('between')
        words.append('more')
        words.append('than')
        mask = np.zeros((T,), dtype=np.uint8)
        for idx, token in enumerate(tokens):
            if token in words:
                mask[idx] = 1
        mask[0] = 1
        return mask

    @staticmethod
    def positional_mask(grid_size, intersection_region):
        mask = np.zeros((grid_size, grid_size), dtype=int)
        for (x, y) in intersection_region:
            mask[y, x] = 1
        return mask