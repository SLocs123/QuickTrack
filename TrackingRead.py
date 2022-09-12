from pprint import pprint
from openpyxl import Workbook
from operator import getitem
from collections import Counter
import pandas as pd

wb = Workbook()

for index, filename in enumerate(['weights 15, 3, 3, 2domCIE2000.txt']):
    txt_file = open(filename, 'r')
    
    objects = {}
    
    for line in txt_file:
        split_line = line.replace("\n", "").split(" ")
        frame_no = split_line[0]
        obj_id = split_line[1]
        obj_type = split_line[2]
        if obj_type == 'DontCare':
            continue
        if obj_id in objects.keys():
            # object already tracked
            objects[obj_id]['length'] += 1
            objects[obj_id]['frames'].append(frame_no)
        else:
            objects[obj_id] = {
                'initial_frame': frame_no,
                'type': obj_type,
                'length': 1,
                'frames': [frame_no]
            }
    
    clean_objects = {}
    
    for key, obj in objects.items():
        if obj['length'] > 4:
            clean_objects[key] = obj
            clean_objects[key]['length'] -= 4
    # pprint(objects)
    
    # sorted_min = {k: v for k, v in sorted(objects.items(), key=lambda x: x[:]['length'])}
    
    sorted_min = {k: v for k, v in sorted(clean_objects.items(), key=lambda x: getitem(x[1], 'length'))}
    
    lengths = [v['length'] for k, v in sorted_min.items()]
    
    occurances = Counter(lengths)
    
    sorted_occurances = {k: v for k, v in sorted(occurances.items(), key=lambda x: x[0])}
    
    total_occurances = sum(sorted_occurances.values())
    
    average = sum([key * length for key, length in sorted_occurances.items()]) / total_occurances
    
    data = []
    
    for key, value in sorted_occurances.items():
        data.append([key, value])
    
    data.append([total_occurances, average])

    wb.create_sheet('1', index)
    sheet = wb['1']
    
    for row in data:
        sheet.append(row)
    
    wb.save(filename="tracking3.xlsx")
