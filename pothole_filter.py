"""
Pothole Filter - Advanced filtering to remove false positives
"""
import numpy as np
import cv2


class PotholeFilter:
    """Advanced filter to remove false detections"""
    
    @staticmethod
    def is_likely_pothole(box, conf, frame_shape, min_confidence=0.45):
        """Check if detection is likely a pothole based on 5 rules"""
        
        x1, y1, x2, y2 = box
        frame_height, frame_width = frame_shape[0], frame_shape[1]
        
        # Rule 1: Confidence
        if conf < min_confidence:
            return False, "Low confidence"
        
        # Rule 2: Size Filter
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        frame_area = frame_width * frame_height
        
        min_size = 20
        max_size_ratio = 0.5
        
        if box_width < min_size or box_height < min_size:
            return False, "Too small"
        
        if box_area > (frame_area * max_size_ratio):
            return False, "Too large"
        
        # Rule 3: Aspect Ratio
        aspect_ratio = max(box_width, box_height) / min(box_width, box_height)
        
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False, "Bad aspect ratio"
        
        # Rule 4: Position
        center_y = (y1 + y2) / 2
        y_ratio = center_y / frame_height
        
        if y_ratio < 0.15:
            return False, "In sky region"
        
        if y_ratio > 0.95:
            return False, "In hood region"
        
        return True, "Passed all filters"
    
    @staticmethod
    def filter_detections(boxes, confs, frame_shape, min_confidence=0.45):
        """Filter detections based on multiple rules"""
        
        filtered_boxes = []
        filtered_confs = []
        filter_reasons = []
        
        for box, conf in zip(boxes, confs):
            is_pothole, reason = PotholeFilter.is_likely_pothole(
                box, conf, frame_shape, min_confidence
            )
            
            if is_pothole:
                filtered_boxes.append(box)
                filtered_confs.append(conf)
                filter_reasons.append(reason)
        
        return np.array(filtered_boxes) if filtered_boxes else np.array([]), np.array(filtered_confs) if filtered_confs else np.array([]), filter_reasons
