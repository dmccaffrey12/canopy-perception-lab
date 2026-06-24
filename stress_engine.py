import cv2
import numpy as np
import random

def apply_gaussian_fog(image: np.ndarray, intensity: float) -> np.ndarray:
    """
    Applies Gaussian Blur and blends the image with a grey overlay to simulate fog.
    intensity: 0 to 10
    """
    if intensity <= 0:
        return image.copy()
        
    H, W, C = image.shape
    
    # 1. Apply Gaussian Blur (kernel size scales with intensity)
    # Ensure kernel size is odd
    ksize = int(intensity * 4)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    # 2. Blend with a light-grey mask to wash out contrast
    # Fog overlay (light grey color: RGB 200, 200, 200)
    fog_overlay = np.full_like(image, 200, dtype=np.uint8)
    
    # Fog density/alpha scales from 0.0 to 0.65
    alpha = (intensity / 10.0) * 0.65
    
    foggy_image = cv2.addWeighted(blurred, 1.0 - alpha, fog_overlay, alpha, 0)
    return foggy_image

def apply_torrential_rain(image: np.ndarray, intensity: float) -> np.ndarray:
    """
    Generates random, angled white lines alpha-blended dynamically across the image.
    intensity: 0 to 10
    """
    if intensity <= 0:
        return image.copy()
        
    H, W, C = image.shape
    rain_layer = np.zeros_like(image)
    
    # Scale number of drops with intensity
    num_drops = int(intensity * 120)
    
    # Draw rain streaks
    # We want angled lines. Let's make them fall from top-left to bottom-right.
    # Angle is close to vertical (e.g. dy is much larger than dx)
    for _ in range(num_drops):
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)
        
        # Length of rain streak scales slightly
        length = random.randint(15, 30)
        
        # Streak vector
        dx = random.randint(2, 6)
        dy = length
        
        # White-ish color for rain drops
        color = (225, 225, 235)
        
        # Draw the line
        cv2.line(rain_layer, (x, y), (x + dx, y + dy), color, thickness=random.randint(1, 2))
        
    # Alpha blend rain layer onto image
    # Rain opacity scales from 0.0 to 0.4
    alpha = (intensity / 10.0) * 0.4
    rainy_image = cv2.addWeighted(image, 1.0, rain_layer, alpha, 0)
    return rainy_image

def apply_canopy_shadow(image: np.ndarray, intensity: float) -> np.ndarray:
    """
    Overlays alternating, high-contrast diagonal exposure bands to simulate tree shadows.
    intensity: 0 to 10
    """
    if intensity <= 0:
        return image.copy()
        
    H, W, C = image.shape
    
    # Coordinates grid
    y_coords = np.arange(H)[:, None]
    x_coords = np.arange(W)[None, :]
    
    # Create diagonal stripes (x + y)
    grid = x_coords + y_coords
    
    # Frequency: we want roughly 8 shadow bands across the smaller dimension
    frequency = (2 * np.pi) / (min(H, W) / 4.0)
    
    # Calculate sine wave ranging from -1.0 to 1.0
    wave = np.sin(grid * frequency)
    
    # Shift wave to 0.0 to 1.0 (0 is dark shadow, 1 is bright sunlit)
    wave_normalized = (wave + 1.0) / 2.0
    
    # Shadow factor: how much light is blocked (max 70% reduction at intensity 10)
    max_shadow_reduction = (intensity / 10.0) * 0.7
    
    # Create 2D shadow mask (multipliers between [1.0 - max_shadow_reduction, 1.0])
    shadow_mask = 1.0 - (wave_normalized * max_shadow_reduction)
    
    # Expand to 3 channels for broadcasting
    shadow_mask_3d = np.expand_dims(shadow_mask, axis=2)
    
    # Apply mask and clip to valid pixel range [0, 255]
    shadowed_image = (image.astype(np.float32) * shadow_mask_3d).clip(0, 255).astype(np.uint8)
    return shadowed_image

def apply_environmental_stress(image: np.ndarray, fog: float, rain: float, shadow: float) -> np.ndarray:
    """
    Applies all active environmental stress factors sequentially.
    """
    stressed = image.copy()
    
    # Order of application: Canopy Shadow -> Gaussian Fog -> Torrential Rain
    # This matches physical atmospheric layering (shadows are on the ground/objects,
    # fog is in the air, rain is in the foreground closer to the lens).
    stressed = apply_canopy_shadow(stressed, shadow)
    stressed = apply_gaussian_fog(stressed, fog)
    stressed = apply_torrential_rain(stressed, rain)
    
    return stressed
