tracking_bbox_mapping = {
    1: (100.0, 100.0, 150.0, 150.0),  # tracking ID 1의 박스
    2: (200.0, 200.0, 250.0, 250.0)   # tracking ID 2의 박스
}


track_id = 1

if track_id in tracking_bbox_mapping:
    bbox = tracking_bbox_mapping[track_id]
    print(bbox)
    
    
    