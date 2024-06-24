import pandas as pd
import ast

def read_csv():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/home/sam/Downloads/Filtered_Detection_Zone_Annotations.csv')

    # Function to convert the string representation of list of dictionaries to actual list of dictionaries
    def convert_to_list_of_coords(polygon_str):
        try:
            list_of_dicts = ast.literal_eval(polygon_str)
            return [[point['x'], point['y']] for point in list_of_dicts]
        except:
            return None

    # Apply the conversion to the 'polygon' column
    df['polygon'] = df['polygon'].apply(convert_to_list_of_coords)

    # Transform the DataFrame into the desired list format
    result = df.apply(lambda row: [row['feature_id'], row['polygon']], axis=1).tolist()

    return result
