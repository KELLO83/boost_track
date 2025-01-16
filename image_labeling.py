# YOLO detection
results = model.predict(np_img, device='cuda', classes=[0], augment=True,
                        iou=0.45, conf=0.45)