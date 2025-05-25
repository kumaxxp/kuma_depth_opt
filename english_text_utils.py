"""
English text utilities for matplotlib and cv2
Minimal implementation to resolve import dependencies
"""
import cv2
import matplotlib.pyplot as plt

def setup_matplotlib_english():
    """Setup matplotlib for English text rendering"""
    # Basic matplotlib configuration for English
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10

def cv2_put_english_text(img, text, position, font_scale=0.7, color=(255, 255, 255), thickness=1):
    """Put English text on image using cv2"""
    return cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
