import os
from natsort import natsorted
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
import matplotlib.pyplot as plt

class YoloToXml:
    def __init__(self, image_dir, label_dir, class_file):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_file = class_file
        
        self.class_index = {}
        
        # Read classes from file
        with open(self.class_file, 'r') as f:
            classes = f.readlines()
            for index, value in enumerate(classes):
                self.class_index[index] = value.strip()
        
    def convert_coordinates(self, size, box):
        dw = 1.0/size[0]
        dh = 1.0/size[1]
        x = box[0]/dw
        w = box[2]/dw
        y = box[1]/dh
        h = box[3]/dh
        xmin = int(x - (w/2.0))
        xmax = int(x + (w/2.0))
        ymin = int(y - (h/2.0))
        ymax = int(y + (h/2.0))
        return (xmin, ymin, xmax, ymax)
        
    def create_xml(self, image_path, txt_path, output_path):
        img = cv2.imread(image_path)
        height, width, depth = img.shape
        
        # Create XML structure with declaration
        root = ET.Element('annotation')
        declaration = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n'
        
        folder = ET.SubElement(root, 'folder')
        folder.text = os.path.basename(os.path.dirname(output_path))
        
        filename = ET.SubElement(root, 'filename')
        filename.text = os.path.basename(image_path)
        
        path = ET.SubElement(root, 'path')
        path.text = image_path
        
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'Unknown'
        
        size = ET.SubElement(root, 'size')
        width_elem = ET.SubElement(size, 'width')
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, 'height')
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, 'depth')
        depth_elem.text = str(depth)
        
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        
        # Read YOLO format file
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            
        # Check if there are any labels
        if not lines:
            # Create an XML with no objects
            obj = ET.SubElement(root, 'object')
            name = ET.SubElement(obj, 'name')
            name.text = '-1'  # Indicates no objects
            
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '1'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin_elem = ET.SubElement(bndbox, 'xmin')
            xmin_elem.text = '0'
            ymin_elem = ET.SubElement(bndbox, 'ymin')
            ymin_elem.text = '0'
            xmax_elem = ET.SubElement(bndbox, 'xmax')
            xmax_elem.text = str(width)
            ymax_elem = ET.SubElement(bndbox, 'ymax')
            ymax_elem.text = str(height)
        else:
            # Process labels
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Convert YOLO coordinates to XML format
                xmin, ymin, xmax, ymax = self.convert_coordinates((width, height), (x, y, w, h))
                
                obj = ET.SubElement(root, 'object')
                
                name = ET.SubElement(obj, 'name')
                name.text = str(class_id)
                
                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'
                
                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '1'
                
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin_elem = ET.SubElement(bndbox, 'xmin')
                xmin_elem.text = str(max(0, xmin))
                ymin_elem = ET.SubElement(bndbox, 'ymin')
                ymin_elem.text = str(max(0, ymin))
                xmax_elem = ET.SubElement(bndbox, 'xmax')
                xmax_elem.text = str(min(width, xmax))
                ymax_elem = ET.SubElement(bndbox, 'ymax')
                ymax_elem.text = str(min(height, ymax))
        
        # Convert to string with pretty formatting
        xml_str = ET.tostring(root, encoding='unicode')
        # Save XML file with declaration
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(declaration + xml_str)

def yolo_to_xml_format(image_dir, label_dir):
    class_file = os.path.join(label_dir, 'classes.txt')
    converter = YoloToXml(image_dir, label_dir, class_file)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(label_dir), 'xml_labels')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all images and labels
    image_files = natsorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    label_files = natsorted([f for f in os.listdir(label_dir) if f.endswith('.txt') and f != 'classes.txt'])
    
    # Ensure we have a label file for each image, using an empty label if missing
    if len(label_files) < len(image_files):
        # Create a list of label files with empty strings for missing files
        label_files = [f'{os.path.splitext(img_file)[0]}.txt' if f'{os.path.splitext(img_file)[0]}.txt' not in label_files else f for f in label_files for img_file in image_files]
    
    for img_file, txt_file in tqdm(zip(image_files, label_files), total=len(image_files), desc="Converting YOLO to XML", unit="image"):
        image_path = os.path.join(image_dir, img_file)
        
        # Check if txt file exists, if not create an empty file
        txt_path = os.path.join(label_dir, txt_file)
        if not os.path.exists(txt_path):
            # Create an empty txt file
            with open(txt_path, 'w') as f:
                pass
        
        xml_file = os.path.splitext(txt_file)[0] + '.xml'
        output_path = os.path.join(output_dir, xml_file)
        
        converter.create_xml(image_path, txt_path, output_path)
        print(f'Converted {txt_file} to {xml_file}')

def compare_xml_labels(xml_dir1, xml_dir2, image_dir):
    """
    Compare XML labels between two directories and visualize differences.
    
    :param xml_dir1: First directory containing XML files
    :param xml_dir2: Second directory containing XML files
    :param image_dir: Directory containing original images
    """
    # Color palette for different scenarios
    colors = {
        'match': (0, 255, 0),     # Green for matching objects
        'id_diff': (0, 0, 255),   # Red for different object IDs
        'missing': (255, 0, 0),   # Blue for missing objects
        'extra': (255, 165, 0)    # Orange for extra objects
    }
    
    # Get sorted list of XML files
    xml_files1 = natsorted([f for f in os.listdir(xml_dir1) if f.endswith('.xml')])
    xml_files2 = natsorted([f for f in os.listdir(xml_dir2) if f.endswith('.xml')])
    
    # Ensure we have matching files
    common_files = set(xml_files1) & set(xml_files2)
    
    # Create output directory for comparison images
    output_dir = os.path.join(os.path.dirname(xml_dir1), 'xml_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Track comparison statistics
    comparison_stats = {
        'total_images': len(common_files),
        'id_mismatches': 0,
        'bbox_mismatches': 0
    }
    
    # Process each common XML file
    for xml_file in tqdm(common_files, desc="Comparing XML Labels", unit="image"):
        # Full paths
        xml_path1 = os.path.join(xml_dir1, xml_file)
        xml_path2 = os.path.join(xml_dir2, xml_file)
        image_path = os.path.join(image_dir, os.path.splitext(xml_file)[0] + '.jpg')
        
        # Parse XML files
        tree1 = ET.parse(xml_path1)
        tree2 = ET.parse(xml_path2)
        
        # Read image
        img = cv2.imread(image_path)
        
        # Extract objects from both XMLs
        objects1 = tree1.findall('.//object')
        objects2 = tree2.findall('.//object')
        
        # Track matched and unmatched objects
        matched_objects = []
        unmatched_objects1 = []
        unmatched_objects2 = []
        
        # Compare objects
        for obj1 in objects1:
            found_match = False
            n1 = obj1.find('name').text
            bbox1 = obj1.find('bndbox')
            xmin1, ymin1 = int(bbox1.find('xmin').text), int(bbox1.find('ymin').text)
            xmax1, ymax1 = int(bbox1.find('xmax').text), int(bbox1.find('ymax').text)
            
            for obj2 in objects2:
                n2 = obj2.find('name').text
                bbox2 = obj2.find('bndbox')
                xmin2, ymin2 = int(bbox2.find('xmin').text), int(bbox2.find('ymin').text)
                xmax2, ymax2 = int(bbox2.find('xmax').text), int(bbox2.find('ymax').text)
                
                # Check for matching objects (similar position and ID)
                iou = calculate_iou((xmin1, ymin1, xmax1, ymax1), (xmin2, ymin2, xmax2, ymax2))
                if iou > 0.7:  # Adjust IOU threshold as needed
                    if n1 == n2:
                        # Matching object
                        cv2.rectangle(img, (xmin1, ymin1), (xmax1, ymax1), colors['match'], 2)
                        matched_objects.append((obj1, obj2))
                    else:
                        # ID mismatch
                        cv2.rectangle(img, (xmin1, ymin1), (xmax1, ymax1), colors['id_diff'], 2)
                        comparison_stats['id_mismatches'] += 1
                    found_match = True
                    break
            
            if not found_match:
                # Object in first XML not found in second
                cv2.rectangle(img, (xmin1, ymin1), (xmax1, ymax1), colors['missing'], 2)
                unmatched_objects1.append(obj1)
        
        # Check for extra objects in second XML
        for obj2 in objects2:
            if not any(obj2 in match for match in matched_objects):
                bbox2 = obj2.find('bndbox')
                xmin2, ymin2 = int(bbox2.find('xmin').text), int(bbox2.find('ymin').text)
                xmax2, ymax2 = int(bbox2.find('xmax').text), int(bbox2.find('ymax').text)
                cv2.rectangle(img, (xmin2, ymin2), (xmax2, ymax2), colors['extra'], 2)
                unmatched_objects2.append(obj2)
        
        # Save comparison image
        output_path = os.path.join(output_dir, xml_file.replace('.xml', '_comparison.jpg'))
        cv2.imwrite(output_path, img)
    
    # Print overall statistics
    print("\nXML Comparison Statistics:")
    print(f"Total Images Compared: {comparison_stats['total_images']}")
    print(f"Images with ID Mismatches: {comparison_stats['id_mismatches']}")
    
    return comparison_stats

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    :param box1: First bounding box (xmin, ymin, xmax, ymax)
    :param box2: Second bounding box (xmin, ymin, xmax, ymax)
    :return: IoU value
    """
    # Compute coordinates of intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    # Compute area of intersection
    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute union area
    union_area = box1_area + box2_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

if __name__ == "__main__":
    image_dir = 'cam0_labeld/images/'
    label_dir = 'cam0_labeld/labels/'
    xml_dir1 = 'cam0_labeld/xml_labels/'
    xml_dir2 = 'cam1_labeld/xml_labels/'
    
    # Convert YOLO to XML for first directory
    yolo_to_xml_format(image_dir, label_dir)
    
    # # Compare XML labels between two directories
    # compare_xml_labels(xml_dir1, xml_dir2, image_dir)