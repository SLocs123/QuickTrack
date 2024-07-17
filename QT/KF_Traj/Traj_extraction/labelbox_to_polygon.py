import json

test = json.load(open('Export v2 project - SAEdemo - 7_2_2024 (1).ndjson'))


polygons = []
for label_id, label_data in test['projects'].items():
    # Iterate over annotations in the label
    for annotation in label_data['labels'][0]['annotations']['objects']:
        if annotation['annotation_kind'] == 'ImagePolygon':
            polygon_coords = annotation['polygon']
            polygons.append(polygon_coords)

# Printing the extracted polygons
for idx, polygon in enumerate(polygons):
    print(f"Polygon {idx + 1}: {polygon}")
