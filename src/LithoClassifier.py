# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 06:15:20 2023
Fucntions to be called by the LithoClassifier.ipynb notebook.

These functions are designed to read in driller logs (with a particular 
formatting and file type at this stage), and process the driller descriptions
to automate the identification of primary materials (P1, P2), any material
descriptions (material_descriptors) and colours.

5/8/2023
I am having issues with some misclassifications. In this version, I will
have a function that just does P1 and P2. A second sweep through that 
dataset will do the other interpretations.

Author: 
Dylan Irvine
Charles Darwin University, Research Institute for the Environment and Livelihoods
dylan.irvine@cdu.edu.au

Version:
v1.01

Splitting grain_size out from material_descriptions
 
last updated 1/8/2023

@author: dirvine
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

__version__ = "v1.04"

print('\n=================================')
print('     LithoClassifier '+__version__)
print('=================================\n')
print('Libraries loaded')

# There are issues with either commas, or brackets that are joined to words
# that make them difficult to fine. This function replaces punctuation with
# spaces
def replace_punc(df, column_name, values_to_replace):
    # Check if the column exists
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the file.")
        return

    # Replace the specified values with a space
    for value in values_to_replace:
        df[column_name] = df[column_name].str.replace(value, ' ', regex=False)

    print(f"Replaced values in column '{column_name}' successfully.")
    return df


def df_pre_process(df, to_replace, replace_with):
    # Create a mapping dictionary from the to_replace and replace_with lists
    replace_dict = dict(zip(to_replace, replace_with))
    
    
    def replace_words(text):
        # Convert the input to a string (this handles NaN and other float values)
        text = str(text)
        for word, replacement in replace_dict.items():
            text = text.replace(word, replacement)
        return text
    
    """
    # Define a function to apply the replacements within the text
    def replace_words(text):
        for word, replacement in replace_dict.items():
            text = text.replace(word, replacement)
        return text
    """

    # Apply the replace_words function to the 'DESCR' column
    df['DESCR'] = df['DESCR'].apply(replace_words)
    
    return df

# To extract P1 and P2
def classify_and_save1(df, output_filename, classification_rules):
    # Define the function to classify the description
    def classify_description(descr):
        # Define the result variables
        primary_material = "OTHER NOT CLASSIFIED"
        secondary_material = 'NONE'

        # Iterate through classification_rules to look for exact matches in description
        for category, keywords in classification_rules.items():
            for keyword in keywords:
                if keyword in descr.upper():
                    if primary_material == "OTHER NOT CLASSIFIED":
                        primary_material = category
                    elif secondary_material == 'NONE':
                        secondary_material = category

        # Remaining classifications go here...

        return primary_material, secondary_material, 'NONE', 'NONE', 'NONE', 'NONE'  # temporary return values

    # Apply the new function to the 'DESCR' column
    df[['P1', 'P2', 'Material_Descriptor', 'Colour1', 'Colour2', 'grain_size']] = df['DESCR'].apply(classify_description).apply(pd.Series)

    # Save the DataFrame to the output file
    df.to_csv(output_filename, index=False)

    return df

# NOW DO THE REMAINING CLASSIFICATIONS
def classify_and_save2(df, output_filename, classification_rules, material_descriptors, colours, colour_descriptors, grain_size):
    # Define the function to classify the description
    def classify_description(descr):
        # Define the result variables
        primary_material = "OTHER NOT CLASSIFIED"
        secondary_material = 'NONE'
        descriptors = []
        colours_found = []
        grain_size_found = 'NONE'

        # Split the description into words
        words = re.split(r'\s+', descr.upper())

        # Classify primary and secondary materials (exact string matching)
        found_materials = set()
        for word in words:
            for category, keywords in classification_rules.items():
                for keyword in keywords:
                    if word == keyword and category not in found_materials:
                        found_materials.add(category)
                        if primary_material == "OTHER NOT CLASSIFIED":
                            primary_material = category
                        elif secondary_material == 'NONE':
                            secondary_material = category
                        break

        # Classify descriptors
        for word in words:
            if word in material_descriptors and word not in descriptors:
                descriptors.append(word)

        # Classify grain size. Finds all terms, reports only the first
        grain_size_found = 'NONE'
        encountered_grain_sizes = []  # List to store encountered grain sizes
        for size, size_keywords in grain_size.items():
            if any(keyword in words for keyword in size_keywords):
                encountered_grain_sizes.append(size)

        if encountered_grain_sizes:
            grain_size_found = encountered_grain_sizes[0]

        # Classify colours
        for i, word in enumerate(words):
            if word in colours and len(colours_found) < 2 and word not in colours_found:
                # Check if the previous word is a colour descriptor, if so concatenate the descriptor with the colour
                if i > 0 and words[i - 1] in colour_descriptors:
                    colours_found.append(words[i - 1] + ' ' + word)
                else:
                    colours_found.append(word)

        # If less than two colours were found, add "NONE" for the remaining colours
        while len(colours_found) < 2:
            colours_found.append("NONE")

        # If no descriptors were found, add "NONE"
        if len(descriptors) == 0:
            descriptors.append("NONE")

        return primary_material, secondary_material, ', '.join(descriptors), colours_found[0], colours_found[1], grain_size_found

    # Apply the new function to the 'DESCR' column
    df[['P1', 'P2', 'Material_Descriptor', 'Colour1', 'Colour2', 'grain_size']] = df['DESCR'].apply(classify_description).apply(pd.Series)

    # Move the 'grain_size' column to the desired position
    grain_size_column = df.pop('grain_size')
    target_position = df.columns.get_loc('Material_Descriptor') + 1
    df.insert(target_position, 'grain_size', grain_size_column)

    # Save the DataFrame to the output file
    df.to_csv(output_filename, index=False)

    return df

# a final clean up of the resulting data frame
def tidy_df(df, P1_value, descr_phrases, output_filename):
    """
    This function removes rows from the dataframe where the P1 column contains a certain value 
    and the DESCR column contains any of a list of phrases.
    """
    # Convert descr_phrases to upper case for comparison
    descr_phrases = [phrase.upper() for phrase in descr_phrases]
    
    # Create a mask for the rows to keep
    mask = ~((df['P1'] == P1_value) & (df['DESCR'].str.upper().apply(lambda descr: any(phrase in descr for phrase in descr_phrases))))
    
    # Apply the mask to the DataFrame
    df = df[mask]

    # Save the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)

    print(f"{output_filename} has been cleaned")
    
    return df  


# plot the bar chart
def plot_material_counts(df):
    # Count the number of each type in the P1 column
    material_counts = df['P1'].value_counts()

    # Create a bar chart of the material counts
    plt.figure(figsize=(20/2.54, 8/2.54))
    ax = plt.axes([0.2, 0.2, 0.7, 0.7])
    bars = ax.bar(material_counts.index, material_counts.values, log=True)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')
    
    plt.xticks(rotation=90)  # Rotate x-labels by 90 degrees
    plt.xlabel('Material Type')
    plt.ylabel('Count')
    plt.title('Material Type Counts')
    plt.show()
    #plt.savefig('MaterialCounts.png', dpi=300)

def calculate_score_write_summary(df, fnameout):
    # Grouping by RN
    grouped = df.groupby('RN')

    result = []
    thicknesses = []  # To keep track of thicknesses
    for name, group in grouped:
        # Extracting the required values
        latitude = group['Latitude'].iloc[0]
        longitude = group['Longitude'].iloc[0]
        easting = group['Easting'].iloc[0]
        northing = group['Northing'].iloc[0]
        zone = group['Projecti_1'].iloc[0]

        # Calculating total characters in DESCR
        total_characters = group['DESCR'].apply(len).sum()

        # Calculating the thickness of the materials (but some instances, the bottom depth (last depth), is wrong)
        if len(group) > 1 and group['BOTTOM'].iloc[-1] < group['BOTTOM'].iloc[-2]:
            thickness = group['BOTTOM'].iloc[-2] - group['TOP'].iloc[0]
        else:
            thickness = group['BOTTOM'].iloc[-1] - group['TOP'].iloc[0]

        thicknesses.append(thickness)  # Storing thicknesses for later use

        # Calculating the score
        if thickness != 0:
            score = total_characters / thickness
        else:
            score = 0  # or any appropriate value when thickness is zero

        # Appending the result
        result.append([name, latitude, longitude, easting, northing, zone, thickness, total_characters, score])

    # Creating the result DataFrame
    result_df = pd.DataFrame(result, columns=['RN', 'Latitude', 'Longitude', 'Easting', 'Northing', 'Zone', 'Thickness', 'Total_Characters', 'Score'])

    # Normalizing the score by dividing by the maximum value
    max_score = result_df['Score'].max()
    if max_score != 0:
        result_df['Score'] = result_df['Score'] / max_score

    # Saving to CSV
    result_df.to_csv(fnameout, index=False)

    return result_df


# new functions to increase the outputs of the summary function


def calculate_material_percentage(group, material_types):
    # Initialize the material percentages
    material_percentage = {material: 0 for material in material_types}
    total_thickness = group['BOTTOM'].max() - group['TOP'].min()
    
    # Calculate the percentage for each material type
    for material in material_types:
        material_thickness = group[group['P1'].str.contains(material, regex=True)]['BOTTOM'].subtract(
            group[group['P1'].str.contains(material, regex=True)]['TOP']).sum()
        material_percentage[material] = (material_thickness / total_thickness * 100) if total_thickness else 0
    
    return material_percentage


def calculate_score_write_summary2(df, fnameout, depth_threshold):
    # Adjust layer thicknesses based on the depth threshold
    df['Adjusted_BOTTOM'] = df['BOTTOM'].apply(lambda x: min(x, depth_threshold))
    df['Adjusted_Thickness'] = df['Adjusted_BOTTOM'] - df['TOP']

    # Filter out layers entirely below the threshold
    df = df[df['TOP'] < depth_threshold]

    # Group by RN and Material (P1) and sum the adjusted thicknesses
    grouped_df = df.groupby(['RN', 'P1'])['Adjusted_Thickness'].sum().reset_index()

    # Calculate total adjusted thickness for each RN
    total_adjusted_thickness = df.groupby('RN')['Adjusted_Thickness'].sum()

    # Merge total adjusted thickness back to grouped_df for percentage calculation
    grouped_df = grouped_df.merge(total_adjusted_thickness, on='RN', suffixes=('', '_Total'))

    # Calculate percentage
    grouped_df['Percentage'] = 100 * grouped_df['Adjusted_Thickness'] / grouped_df['Adjusted_Thickness_Total']

    # Pivot to get one line per RN with each material type as a column
    pivot_df = grouped_df.pivot(index='RN', columns='P1', values='Percentage').reset_index()

    # Fill NaN values with 0 (for material types not present in a particular RN)
    pivot_df.fillna(0, inplace=True)
    #pivot_df['Latitude'] = df['Latitude']
    #pivot_df['Longitude'] = df['Longitude']   
    
    # Merge Latitude and Longitude
    lat_lon_df = df[['RN', 'Latitude', 'Longitude']].drop_duplicates()
    final_df = pivot_df.merge(lat_lon_df, on='RN')

    final_df.to_csv(fnameout, index=False)
    return final_df


# this version also pressents the thicknesses (which is needed to de-bug it)
def calculate_score_write_summary3(df, fnameout, depth_threshold):
    # Adjust layer thicknesses based on the depth threshold
    df['Adjusted_BOTTOM'] = df['BOTTOM'].apply(lambda x: min(x, depth_threshold))
    df['Adjusted_Thickness'] = df['Adjusted_BOTTOM'] - df['TOP']

    # Filter out layers entirely below the threshold
    df = df[df['TOP'] < depth_threshold]

    # Group by RN and Material (P1) and sum the adjusted thicknesses
    grouped_df = df.groupby(['RN', 'P1'])['Adjusted_Thickness'].sum().reset_index()

    # Calculate total adjusted thickness for each RN
    total_adjusted_thickness = df.groupby('RN')['Adjusted_Thickness'].sum()

    # Merge total adjusted thickness back to grouped_df for percentage calculation
    grouped_df = grouped_df.merge(total_adjusted_thickness, on='RN', suffixes=('', '_Total'))

    # Calculate percentage
    grouped_df['Percentage'] = 100 * grouped_df['Adjusted_Thickness'] / grouped_df['Adjusted_Thickness_Total']

    # Pivot for percentages
    pivot_percentage = grouped_df.pivot(index='RN', columns='P1', values='Percentage').reset_index()
    pivot_percentage.fillna(0, inplace=True)  # Fill NaN values with 0

    # Pivot for thicknesses
    pivot_thickness = grouped_df.pivot(index='RN', columns='P1', values='Adjusted_Thickness').reset_index()
    pivot_thickness.fillna(0, inplace=True)  # Fill NaN values with 0

    # Merge the two pivots
    final_df = pivot_percentage.merge(pivot_thickness, on='RN', suffixes=('_Percentage', '_Thickness'))

    # Merge Latitude and Longitude
    lat_lon_df = df[['RN', 'Latitude', 'Longitude']].drop_duplicates()
    final_df = final_df.merge(lat_lon_df, on='RN')

    final_df.to_csv(fnameout, index=False)
    return final_df



print('LithoClassifier v1.04 Functions loaded')
print('=================================\n')



"""
OLD CODE, retained, just in case.
# THIS IS THE VERSION OF THE FUNCTION THAT ATTEMPTS TO DO EVERYTHING IN ONE SWEEP
def classify_and_save(df, output_filename, classification_rules, material_descriptors, colours, colour_descriptors, grain_size):
    # Define the function to classify the description
    def classify_description(descr):
        # Define the result variables
        primary_material = "OTHER NOT CLASSIFIED"
        secondary_material = 'NONE'
        descriptors = []
        colours_found = []
        grain_size_found = 'NONE'


        # Check for multi-word matches in classification rules
        for category, keywords in classification_rules.items():
            for keyword in keywords:
                if keyword in descr.upper() and primary_material == "OTHER NOT CLASSIFIED":
                    primary_material = category
                    break
        
        # Split the description into words
        words = re.split(r'\s+', descr.upper())

        # Iterate over the words and check if they match any category
        for i, word in enumerate(words):
            # Classify primary and secondary materials (for single-word terms)
            for category, keywords in classification_rules.items():
                if word in keywords and primary_material == "OTHER NOT CLASSIFIED":
                    primary_material = category
                elif word in keywords and primary_material != category:
                    secondary_material = category

            # Classify descriptors
            if word in material_descriptors and word not in descriptors:
                descriptors.append(word)
                
            # Classify colours
            for colour in colours:
                if word == colour and len(colours_found) < 2 and colour not in colours_found:
                    # Check if the previous word is a colour descriptor, if so concatenate the descriptor with the colour
                    if i > 0 and words[i - 1] in colour_descriptors:
                        colours_found.append(words[i - 1] + ' ' + word)
                    else:
                        colours_found.append(word)

            # Classify grain size
            for size, size_keywords in grain_size.items():
                if word in size_keywords:
                    grain_size_found = size

        # If less than two colours were found, add "NONE" for the remaining colours
        while len(colours_found) < 2:
            colours_found.append("NONE")

        # If no descriptors were found, add "NONE"
        if len(descriptors) == 0:
            descriptors.append("NONE")

        return primary_material, secondary_material, ', '.join(descriptors), colours_found[0], colours_found[1], grain_size_found

    # Apply the new function to the 'DESCR' column
    df[['P1', 'P2', 'Material_Descriptor', 'Colour1', 'Colour2', 'grain_size']] = df['DESCR'].apply(classify_description).apply(pd.Series)

    # Move the 'grain_size' column to the desired position
    grain_size_column = df.pop('grain_size')
    target_position = df.columns.get_loc('Material_Descriptor') + 1
    df.insert(target_position, 'grain_size', grain_size_column)

    # Save the DataFrame to the output file
    df.to_csv(output_filename, index=False)

    return df
"""

"""
def calculate_score_write_summary2(df, fnameout):
    grouped = df.groupby('RN')
    depth_ranges = [(0, 10), (0, 20), (0, 30)]
    material_combinations = {
        "SAND + SANDSTONE": ("SAND", "SANDSTONE"),
        "GRAVEL + COBBLES": ("GRAVEL", "COBBLES"),
        "CLAY + SILT": ("CLAY", "SILT"),
        "BASEMENT + ROCK": ("BASEMENT", "ROCK"),
        "SOIL": ("SOIL",)
    }

    results = []
    for RN, group in grouped:
        max_depth = group['BOTTOM'].max()
        total_thickness = max_depth - group['TOP'].min()
        
        # Calculate overall material percentages
        material_percentages = calculate_material_percentage(group, material_combinations)

        # Calculate material percentages for depth ranges
        depth_material_percentages = {range_depth: calculate_material_percentage(
            group[(group['TOP'] >= range_depth[0]) & (group['BOTTOM'] <= range_depth[1])], material_combinations)
            for range_depth in depth_ranges}

        # Construct the results dictionary
        result_row = {
            "RN": RN,
            "Max Depth": max_depth,
            "Total Thickness": total_thickness,
            **material_percentages
        }

        # Add the depth range material percentages
        for depth_range, percentages in depth_material_percentages.items():
            for material, percentage in percentages.items():
                result_row[f"{depth_range[1]}m {material} Percentage"] = percentage

        # Append to the results list
        results.append(result_row)

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    
    # Write to CSV
    result_df.to_csv(fnameout, index=False)
    return result_df

# Example usage
# df = pd.read_csv('LBW_STRATA_LOG_processed.csv')
# summary_df = calculate_score_write_summary2(df, 'summary2.csv')


def calculate_score_write_summary2(df, fnameout):
    # Calculate thickness for each record
    df['Thickness'] = df['BOTTOM'] - df['TOP']

    # Group by RN and Material (P1) and sum the thicknesses
    grouped_df = df.groupby(['RN', 'P1'])['Thickness'].sum().reset_index()

    # Calculate total thickness for each RN
    total_thickness = df.groupby('RN')['Thickness'].sum()

    # Merge total thickness back to grouped_df for percentage calculation
    grouped_df = grouped_df.merge(total_thickness, on='RN', suffixes=('', '_Total'))

    # Calculate percentage
    grouped_df['Percentage'] = 100 * grouped_df['Thickness'] / grouped_df['Thickness_Total']

    # Pivot to get one line per RN with each material type as a column
    pivot_df = grouped_df.pivot(index='RN', columns='P1', values='Percentage').reset_index()

    # Fill NaN values with 0 (for material types not present in a particular RN)
    pivot_df.fillna(0, inplace=True)

    pivot_df.to_csv(fnameout, index=False)
    return pivot_df
"""