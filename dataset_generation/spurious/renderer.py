from PIL import Image, ImageDraw
import numpy as np

class SceneRenderer:
    """
    Renders a grid scene populated with colored shapes.
    Attributes:
        grid_size (int): number of cells per row/column.
        cell_size (int): pixel dimension of each cell.
        bg_color (str): background color.
    """
    def __init__(self, grid_size=5, cell_size=64, bg_color='white', draw_grid=False):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.bg_color = bg_color
        self.draw_grid = draw_grid

    def render(self, objects):
        """
        Draw shapes onto a blank canvas.
        Args:
            objects (list[dict]): each dict has keys 'x', 'y', 'color', 'shape'.
        Returns:
            PIL.Image: rendered scene.
        """
        width = self.grid_size * self.cell_size
        height = self.grid_size * self.cell_size
        img = Image.new('RGB', (width, height), self.bg_color)
        
        # Note: np_img is created here but not used for drawing (PIL Draw is used directly)
        # keeping it in case you perform pixel manipulations later.
        np_img = np.array(img) 
        draw = ImageDraw.Draw(img)

        # Optionally draw grid lines
        if self.draw_grid:
            for i in range(1, self.grid_size):
                # vertical lines
                x = i * self.cell_size
                draw.line([(x, 0), (x, height)], fill='gray')
                # horizontal lines
                y = i * self.cell_size
                draw.line([(0, y), (width, y)], fill='gray')

        for obj in objects:
            x_cell, y_cell = obj['x'], obj['y']
            
            # Calculate bounding box with padding
            x0 = x_cell * self.cell_size + self.cell_size * 0.1
            y0 = y_cell * self.cell_size + self.cell_size * 0.1
            x1 = (x_cell + 1) * self.cell_size - self.cell_size * 0.1
            y1 = (y_cell + 1) * self.cell_size - self.cell_size * 0.1
            
            color = obj['color']
            shape = obj['shape']

            if shape == 'circle':
                draw.ellipse([x0, y0, x1, y1], fill=color, outline='black', width=2)
            
            elif shape == 'square':
                draw.rectangle([x0, y0, x1, y1], fill=color, outline='black', width=2)
            
            elif shape == 'triangle':
                # equilateral triangle logic
                xm = (x0 + x1) / 2
                points = [(xm, y0), (x0, y1), (x1, y1)]
                draw.polygon(points, fill=color, outline='black', width=2)
                
            elif shape == 'pentagon':
                # --- NEW LOGIC FOR PENTAGON ---
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                # Radius is half the width of the bounding box
                radius = (x1 - x0) / 2
                
                pentagon_points = []
                # 5 vertices
                for i in range(5):
                    # Start at -90 degrees (top) to orient the pentagon upright
                    angle = -np.pi / 2 + (i * 2 * np.pi / 5)
                    px = cx + radius * np.cos(angle)
                    py = cy + radius * np.sin(angle)
                    pentagon_points.append((px, py))
                
                draw.polygon(pentagon_points, fill=color, outline='black', width=2)

            else:
                # Fallback: draw a circle
                draw.ellipse([x0, y0, x1, y1], fill=color, outline='black', width=2)

        return img


# testing

def test_scene_renderer():
    # Sample objects
    objects = [
        {'x': 0, 'y': 2, 'color': 'red', 'shape': 'square'},
        {'x': 1, 'y': 1, 'color': 'blue', 'shape': 'circle'},
        {'x': 2, 'y': 0, 'color': 'green', 'shape': 'triangle'},
        
        # Added a Pentagon here
        {'x': 1, 'y': 3, 'color': 'yellow', 'shape': 'pentagon'}, 
        
        {'x': 2, 'y': 4, 'color': 'red', 'shape': 'square'},
        {'x': 3, 'y': 0, 'color': 'red', 'shape': 'triangle'},
        {'x': 3, 'y': 3, 'color': 'red', 'shape': 'triangle'},
        {'x': 4, 'y': 3, 'color': 'cyan', 'shape': 'square'} # changed color to cyan for variety
    ]
    
    # Instantiate renderer
    renderer = SceneRenderer(grid_size=5, cell_size=64, bg_color='white', draw_grid=False)
    
    # Render the scene
    img = renderer.render(objects)
    
    # Display or save the image
    img.show()  # opens in default image viewer
    # img.save('test_scene_with_pentagon.png')

if __name__ == "__main__":
    test_scene_renderer()