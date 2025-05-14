#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix text encoding issues in visualizations
Using English only to ensure compatibility across environments
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import os
from pathlib import Path
import sys

def setup_matplotlib_english():
    """
    Configure matplotlib to use English fonts for all text
    """
    print("Configuring matplotlib for English text display...")
    
    # Simple configuration that works in all environments
    matplotlib.rcParams.update({
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
        'font.family': 'sans-serif',
        'axes.unicode_minus': False  # Ensure minus signs display correctly
    })
    
    print("Font configuration completed")

def cv2_put_english_text(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_AA):
    """
    Draw text on OpenCV images, using English text only for compatibility
    
    Args:
        img: Image to draw on
        text: Text to draw
        org: Bottom-left corner of the text
        fontFace: Font type
        fontScale: Font scale
        color: Text color (B, G, R)
        thickness: Line thickness
        lineType: Line type
        
    Returns:
        Image with text
    """
    # For debugging, replace any non-ASCII characters with their Unicode escape codes
    simplified_text = ""
    for c in text:
        if ord(c) < 128:  # ASCII character
            simplified_text += c
        else:
            # Replace with a description
            simplified_text += "[non-ASCII]"
    
    # Use standard CV2 putText for all text
    cv2.putText(img, simplified_text, org, fontFace, fontScale, color, thickness, lineType)
    return img

# Test code
if __name__ == "__main__":
    # Test matplotlib configuration
    setup_matplotlib_english()
    
    # Matplotlib text display test
    plt.figure(figsize=(8, 4))
    plt.title("Text Display Test - matplotlib")
    plt.text(0.5, 0.5, "Text display test", ha='center', va='center', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("matplotlib_text_test.png")
    plt.close()
    
    # OpenCV text display test
    img = np.ones((300, 600, 3), dtype=np.uint8) * 255
    img = cv2_put_english_text(img, "Text display test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite("opencv_text_test.png", img)
    
    print("Tests completed. Please check the generated image files.")
