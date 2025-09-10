from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

import plotly.express as px
import datetime
import json
import  os
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.patches as patches
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

"""# Final Code"""

#level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_56_INHIBITION_Level_1/DataJSON/Data.csv'
#level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_57_INHIBITION_Level_2/DataJSON/Data.csv'
#level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_58_INHIBITION_Level_3/DataJSON/Data.csv'

level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_42_INHIBITION_Level_1/DataJSON/Data.csv'
level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_43_INHIBITION_Level_2/DataJSON/Data.csv'
level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_44_INHIBITION_Level_3/DataJSON/Data.csv'

#level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_3_P2/Session_9_10-12-2024_09-28-36/Activity_33_INHIBITION_Level_1/DataJSON/Data.csv'
#level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_3_P2/Session_9_10-12-2024_09-28-36/Activity_34_INHIBITION_Level_2/DataJSON/Data.csv'
#level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_3_P2/Session_9_10-12-2024_09-28-36/Activity_35_INHIBITION_Level_3/DataJSON/Data.csv'

# Define file paths - Updated for Inhibition Game
#level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_56_INHIBITION_Level_1/DataJSON/Data.csv'
#level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_57_INHIBITION_Level_2/DataJSON/Data.csv'
#level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_58_INHIBITION_Level_3/DataJSON/Data.csv'


df_level1 = pd.read_csv(level1_path, sep=';')
df_level2 = pd.read_csv(level2_path, sep=';')
df_level3 = pd.read_csv(level3_path, sep=';')

print("\n Student  - Level 1")
print(f"Shape: {df_level1.shape}")
print(f"Columns: {df_level1.columns.tolist()}")
df_level1.head(5)

print("\n Student  - Level 2")
print(f"Shape: {df_level2.shape}")
print(f"Columns: {df_level2.columns.tolist()}")
df_level2.head(5)

print("\n Student  - Level 3")
print(f"Ddata Shape: {df_level3.shape}")
print(f"Total Columns: {df_level3.columns.tolist()}")
df_level3.head(5)


def clean_eye_data(df):
    df = df.copy()

    df[['EyeTracker-x', 'EyeTracker-y']] = (
        df['EyeTracker']
        .astype(str)
        .str.strip("()")
        .str.replace(" ", "")
        .str.split(",", expand=True)
        .astype(float)
    )

    df_clean = df.dropna(subset=['EyeTracker-x', 'EyeTracker-y']).reset_index(drop=True)

    print(f" Cleaned: {df_clean.shape[0]} rows retained (from {df.shape[0]})")
    return df_clean


df_level1_clean = clean_eye_data(df_level1)
df_level2_clean = clean_eye_data(df_level2)
df_level3_clean = clean_eye_data(df_level3)


print("\n Student - Level 1")
print(f"Shape: {df_level1_clean.shape}")
print(f"Columns: {df_level1_clean.columns.tolist()}")
df_level1_clean.head(5)

print("\n Student - Level 2")
print(f"Shape: {df_level2_clean.shape}")
print(f"Columns: {df_level2_clean.columns.tolist()}")
df_level2_clean.head(5)

print("\n Student - Level 3")
print(f"Shape: {df_level3_clean.shape}")
print(f"Columns: {df_level3_clean.columns.tolist()}")
df_level3_clean.head(5)




# Custom Screen dimensions as per our Toobi
screen_width = 1920
screen_height = 1080

def load_and_clean_data(path):
    try:
        df = pd.read_csv(path, sep=';')
        print(f" Loaded {len(df)} rows from {path}")

        print("DATAFRAME HEAD - BEFORE PREPROCESSING")
        print(df.head())
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")

        # Analyze the data format
        print(f"\nData Analysis:")
        print(f"ObjectName types: {df['ObjectName'].value_counts()}")
        print(f"ObjectState types: {df['ObjectState'].value_counts()}")
        print(f"Label types: {df['Label'].value_counts()}")

        # Removing "Picked Trash" rows here since we dont need them.
        initial_count = len(df)
        picked_trash_mask = df['ObjectState'].astype(str).str.contains('Picked Trash', case=False, na=False)
        df = df[~picked_trash_mask].copy()
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"Removed {removed_count} 'Picked Trash' rows")

        # Parse coordinate columns
        if 'EyeTracker' in df.columns:
            df[['EyeTracker-x', 'EyeTracker-y']] = (
                df['EyeTracker']
                .astype(str)
                .str.strip("()")
                .str.replace(" ", "")
                .str.split(",", expand=True)
                .astype(float)
            )

        if 'GameObjectPos (Screen Coordinates)' in df.columns:
            df[['obj_x', 'obj_y']] = (
                df['GameObjectPos (Screen Coordinates)']
                .astype(str)
                .str.strip("()")
                .str.replace(" ", "")
                .str.split(",", expand=True)
                .astype(float)
            )

        if 'AOI_Origin' in df.columns:
            df[['AOI_Origin_x', 'AOI_Origin_y']] = (
                df['AOI_Origin']
                .astype(str)
                .str.strip("()")
                .str.replace(" ", "")
                .str.split(",", expand=True)
                .astype(float)
            )

        # For this dataset, obj_x and obj_y are the AOI_Origin coordinates
        if 'AOI_Origin_x' not in df.columns and 'obj_x' in df.columns:
            df['AOI_Origin_x'] = df['obj_x']
            df['AOI_Origin_y'] = df['obj_y']

        df_clean = df.copy()

        #  remove rows where both eye tracking AND AOI data are invalid
        eye_invalid = (df_clean['EyeTracker-x'].isna() | df_clean['EyeTracker-y'].isna() |
                      ((df_clean['EyeTracker-x'] == 0) & (df_clean['EyeTracker-y'] == 0)))

        aoi_invalid = (df_clean['AOI_Width'].isna() | df_clean['AOI_Height'].isna() |
                      df_clean['AOI_Origin_x'].isna() | df_clean['AOI_Origin_y'].isna() |
                      (df_clean['AOI_Width'] == 0) | (df_clean['AOI_Height'] == 0))

        # Keep rows that have either valid eye tracking OR valid AOI data
        df_clean = df_clean[~(eye_invalid & aoi_invalid)].copy()

        # Keep only the specified columns after preprocessing
        required_columns = [
            'Timestamp', 'ObjectName', 'ObjectState', 'Label', 'Aux Message',
            'AOI_Width', 'AOI_Height', 'EyeTracker-x', 'EyeTracker-y',
            'obj_x', 'obj_y', 'AOI_Origin_x', 'AOI_Origin_y'
        ]

        # Check which required columns exist in the dataframe
        available_columns = [col for col in required_columns if col in df_clean.columns]
        missing_columns = [col for col in required_columns if col not in df_clean.columns]

        if missing_columns:
            print(f"  Warning: Missing columns: {missing_columns}")

        # Select only the available required columns
        df_clean = df_clean[available_columns].copy()

        print(df_clean.head(10))
        print(f"Columns: {list(df_clean.columns)}")
        print(f"Shape: {df_clean.shape}")

        print(f"  After cleaning: {len(df_clean)} rows ({len(df_clean)/initial_count*100:.1f}% retained)")
        return df_clean.reset_index(drop=True)

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def diagnose_coordinate_systems(df, stage_name):
    # Analyze eye tracking coordinates
    eye_data = df.dropna(subset=['EyeTracker-x', 'EyeTracker-y'])
    eye_data = eye_data[~((eye_data['EyeTracker-x'] == 0) & (eye_data['EyeTracker-y'] == 0))]

    if len(eye_data) > 0:
        print(f"Eye Tracking: X={eye_data['EyeTracker-x'].min():.1f}-{eye_data['EyeTracker-x'].max():.1f}, Y={eye_data['EyeTracker-y'].min():.1f}-{eye_data['EyeTracker-y'].max():.1f}")

    aoi_data = df[
        (df['AOI_Width'].notna()) &
        (df['AOI_Height'].notna()) &
        (df['AOI_Origin_x'].notna()) &
        (df['AOI_Origin_y'].notna()) &
        (df['AOI_Width'] > 0) &
        (df['AOI_Height'] > 0)
    ]

    if len(aoi_data) > 0:
        print(f"AOI Origins: X={aoi_data['AOI_Origin_x'].min():.1f}-{aoi_data['AOI_Origin_x'].max():.1f}, Y={aoi_data['AOI_Origin_y'].min():.1f}-{aoi_data['AOI_Origin_y'].max():.1f}")
        print(f"AOI Sizes: W={aoi_data['AOI_Width'].min():.1f}-{aoi_data['AOI_Width'].max():.1f}, H={aoi_data['AOI_Height'].min():.1f}-{aoi_data['AOI_Height'].max():.1f}")

    return eye_data, aoi_data

def fix_aoi_coordinates(df, stage_name):
    df_fixed = df.copy()

    # Add infromation about the 5 main AOIs as specified
    main_aois = []

    # 1. Blue Bin AOI
    main_aois.append({
        'Timestamp': df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0,
        'ObjectName': 'Blue_Bin',
        'ObjectState': 'Main_AOI',
        'Label': 'Blue_Recycling_Bin',
        'Aux Message': 'Main_Game_AOI',
        'AOI_Width': 304,  # Exact from data
        'AOI_Height': 302,  # Exact from data
        'EyeTracker-x': 0.0,
        'EyeTracker-y': 0.0,
        'obj_x': 494.8,  # Exact from data
        'obj_y': 748.2,  # Exact from data
        'AOI_Origin_x': 507.1,  # Exact from data
        'AOI_Origin_y': 678.8   # Exact from data
    })

    # 2. Yellow Bin AOI
    main_aois.append({
        'Timestamp': df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0,
        'ObjectName': 'Yellow_Bin',
        'ObjectState': 'Main_AOI',
        'Label': 'Yellow_Recycling_Bin',
        'Aux Message': 'Main_Game_AOI',
        'AOI_Width': 252,  # Exact from data
        'AOI_Height': 292,  # Exact from data
        'EyeTracker-x': 0.0,
        'EyeTracker-y': 0.0,
        'obj_x': 801,  # Exact from data
        'obj_y': 752,  # Exact from data
        'AOI_Origin_x': 804.8,  # Exact from data
        'AOI_Origin_y': 682.6   # Exact from data
    })

    # 3. Green Bin AOI
    main_aois.append({
        'Timestamp': df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0,
        'ObjectName': 'Green_Bin',
        'ObjectState': 'Main_AOI',
        'Label': 'Green_Recycling_Bin',
        'Aux Message': 'Main_Game_AOI',
        'AOI_Width': 241,  # Exact from data
        'AOI_Height': 292,  # Exact from data
        'EyeTracker-x': 0.0,
        'EyeTracker-y': 0.0,
        'obj_x': 1079.7,  # Exact from data
        'obj_y': 751.9,   # Exact from data
        'AOI_Origin_x': 1076.5,  # Exact from data
        'AOI_Origin_y': 682.4    # Exact from data
    })

    # 4. Black Bin AOI
    main_aois.append({
        'Timestamp': df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0,
        'ObjectName': 'Black_Bin',
        'ObjectState': 'Main_AOI',
        'Label': 'Black_Waste_Bin',
        'Aux Message': 'Main_Game_AOI',
        'AOI_Width': 250,  # Estimated similar to other bins
        'AOI_Height': 295,  # Estimated similar to other bins
        'EyeTracker-x': 0.0,
        'EyeTracker-y': 0.0,
        'obj_x': 1400,     # Estimated position
        'obj_y': 750,      # Same level as other bins
        'AOI_Origin_x': 1400,  # Following pattern
        'AOI_Origin_y': 680    # Following pattern (obj_y - 70)
    })

    # 5. Item Area
    main_aois.append({
        'Timestamp': df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0,
        'ObjectName': 'Item_Area',
        'ObjectState': 'Main_AOI',
        'Label': 'Item_Appearance_Area',
        'Aux Message': 'Main_Game_AOI',
        'AOI_Width': 400,  # Large enough to capture item variations
        'AOI_Height': 400,
        'EyeTracker-x': 0.0,
        'EyeTracker-y': 0.0,
        'obj_x': 1471.4,   # Exact from data
        'obj_y': 814.8,    # Exact from data
        'AOI_Origin_x': 1271.4,  # obj_x - width/2
        'AOI_Origin_y': 614.8    # obj_y - height/2
    })

    # 6. Check Button AOI
    main_aois.append({
        'Timestamp': df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0,
        'ObjectName': 'Check_Button',
        'ObjectState': 'Main_AOI',
        'Label': 'Check_Button',
        'Aux Message': 'Main_Game_AOI',
        'AOI_Width': 100,  # Reasonable button size
        'AOI_Height': 80,
        'EyeTracker-x': 0.0,
        'EyeTracker-y': 0.0,
        'obj_x': 1225,     # Center of click area
        'obj_y': 90,       # Center of click Y range
        'AOI_Origin_x': 1175,  # obj_x - width/2
        'AOI_Origin_y': 50     # obj_y - height/2
    })

    # 7. Cross Button AOI
    main_aois.append({
        'Timestamp': df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0,
        'ObjectName': 'Cross_Button',
        'ObjectState': 'Main_AOI',
        'Label': 'Cross_Button',
        'Aux Message': 'Main_Game_AOI',
        'AOI_Width': 100,  # Reasonable button size
        'AOI_Height': 80,
        'EyeTracker-x': 0.0,
        'EyeTracker-y': 0.0,
        'obj_x': 590,      # Center of click area
        'obj_y': 125,      # Center of click Y range
        'AOI_Origin_x': 540,   # obj_x - width/2
        'AOI_Origin_y': 85     # obj_y - height/2
    })
    # Add main AOIs to game dataframe
    if main_aois:
        main_aoi_df = pd.DataFrame(main_aois)
        df_fixed = pd.concat([df_fixed, main_aoi_df], ignore_index=True)
        print(f"Added {len(main_aois)} main AOIs")

    aoi_mask = (
        (df_fixed['AOI_Width'].notna()) &
        (df_fixed['AOI_Height'].notna()) &
        (df_fixed['AOI_Origin_x'].notna()) &
        (df_fixed['AOI_Origin_y'].notna()) &
        (df_fixed['AOI_Width'] > 0) &
        (df_fixed['AOI_Height'] > 0)
    )

    if aoi_mask.any():
        df_fixed.loc[aoi_mask, 'AOI_Origin_x'] = np.clip(
            df_fixed.loc[aoi_mask, 'AOI_Origin_x'],
            0,
            screen_width - df_fixed.loc[aoi_mask, 'AOI_Width']
        )
        df_fixed.loc[aoi_mask, 'AOI_Origin_y'] = np.clip(
            df_fixed.loc[aoi_mask, 'AOI_Origin_y'],
            0,
            screen_height - df_fixed.loc[aoi_mask, 'AOI_Height']
        )

        print("Clamped all AOIs to screen bounds")

    # Final Summary
    final_aoi_mask = (
        (df_fixed['AOI_Width'].notna()) &
        (df_fixed['AOI_Height'].notna()) &
        (df_fixed['AOI_Origin_x'].notna()) &
        (df_fixed['AOI_Origin_y'].notna()) &
        (df_fixed['AOI_Width'] > 0) &
        (df_fixed['AOI_Height'] > 0)
    )

    final_aoi_objects = df_fixed[final_aoi_mask]['ObjectName'].value_counts()
    print(f"\nFinal AOI objects: {dict(final_aoi_objects)}")

    return df_fixed

def recalculate_aoi_focus(df, stage_name):
    # Get clean eye tracking data
    eye_data = df.dropna(subset=['EyeTracker-x', 'EyeTracker-y']).copy()
    eye_data = eye_data[~((eye_data['EyeTracker-x'] == 0) & (eye_data['EyeTracker-y'] == 0))]

    if len(eye_data) == 0:
        print("No valid eye tracking data found")
        return 0, {}

    # Get valid AOI data
    aoi_data = df[
        (df['AOI_Width'].notna()) &
        (df['AOI_Height'].notna()) &
        (df['AOI_Origin_x'].notna()) &
        (df['AOI_Origin_y'].notna()) &
        (df['AOI_Width'] > 0) &
        (df['AOI_Height'] > 0)
    ]

    if len(aoi_data) == 0:
        print("No valid AOI data found")
        return 0, {}

    print(f"Analyzing {len(eye_data)} eye tracking points against {len(aoi_data)} AOI regions")

    # Calculate AOI focus with the main 5 AOIs
    in_aoi_count = 0
    aoi_hits = {}

    # Create AOI groups based on the main 5 AOIs
    unique_aois = {}
    for _, aoi_row in aoi_data.iterrows():
        obj_name = str(aoi_row.get('ObjectName', 'Unknown'))

        # Use exact object names for the main AOIs
        if obj_name in ['Blue_Bin', 'Yellow_Bin', 'Green_Bin', 'Black_Bin', 'Item_Area', 'Check_Button', 'Cross_Button']:
            group_key = obj_name
        else:
            continue

        unique_aois[group_key] = {
            'left': aoi_row['AOI_Origin_x'],
            'right': aoi_row['AOI_Origin_x'] + aoi_row['AOI_Width'],
            'top': aoi_row['AOI_Origin_y'],
            'bottom': aoi_row['AOI_Origin_y'] + aoi_row['AOI_Height']
        }

    print(f"Created {len(unique_aois)} main AOI groups: {list(unique_aois.keys())}")

    for _, gaze_row in eye_data.iterrows():
        gaze_x = gaze_row['EyeTracker-x']
        gaze_y = gaze_row['EyeTracker-y']

        hit_any_aoi = False
        for group_name, bounds in unique_aois.items():
            if (bounds['left'] <= gaze_x <= bounds['right'] and
                bounds['top'] <= gaze_y <= bounds['bottom']):

                if not hit_any_aoi:  # Only count first hit to avoid double counting
                    in_aoi_count += 1
                    aoi_hits[group_name] = aoi_hits.get(group_name, 0) + 1
                    hit_any_aoi = True
                    break

    aoi_focus_percentage = (in_aoi_count / len(eye_data)) * 100
    print(f"AOI Focus: {aoi_focus_percentage:.1f}% ({in_aoi_count}/{len(eye_data)} points)")

    if aoi_hits:
        print("  AOI Breakdown:")
        for obj_name, hits in sorted(aoi_hits.items(), key=lambda x: x[1], reverse=True):
            percentage = (hits / len(eye_data)) * 100
            print(f"    {obj_name}: {hits} hits ({percentage:.1f}%)")
    else:
        print("  No AOI hits detected - check coordinate alignment")

    return aoi_focus_percentage, aoi_hits

def create_aoi_visualization(df, stage_name):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'{stage_name} - Eye Tracking and Main AOI Analysis', fontsize=16, fontweight='bold')

    # Eye tracking data
    eye_data = df.dropna(subset=['EyeTracker-x', 'EyeTracker-y'])
    eye_data = eye_data[~((eye_data['EyeTracker-x'] == 0) & (eye_data['EyeTracker-y'] == 0))]

    # Plot 1: Eye tracking heatmap
    ax1 = axes[0]
    if len(eye_data) > 0:
        # Create 2D histogram for heatmap
        H, xedges, yedges = np.histogram2d(eye_data['EyeTracker-x'], eye_data['EyeTracker-y'],
                                          bins=50, range=[[0, screen_width], [0, screen_height]])
        im = ax1.imshow(H.T, origin='lower', extent=[0, screen_width, 0, screen_height],
                       cmap='hot', alpha=0.7, aspect='auto')

        plt.colorbar(im, ax=ax1, label='Gaze Density')
        ax1.scatter(eye_data['EyeTracker-x'], eye_data['EyeTracker-y'],
                   alpha=0.3, s=1, c='cyan', label=f'Eye Tracking ({len(eye_data)} points)')

    ax1.set_xlim(0, screen_width)
    ax1.set_ylim(0, screen_height)
    ax1.set_title('Eye Tracking Heatmap')
    ax1.set_xlabel('X Coordinate (pixels)')
    ax1.set_ylabel('Y Coordinate (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Main AOI regions with eye tracking overlay
    ax2 = axes[1]
    if len(eye_data) > 0:
        ax2.scatter(eye_data['EyeTracker-x'], eye_data['EyeTracker-y'],
                   alpha=0.4, s=2, c='blue', label=f'Eye Tracking ({len(eye_data)} points)')
    aoi_data_fixed = df[
        (df['AOI_Width'].notna()) &
        (df['AOI_Height'].notna()) &
        (df['AOI_Origin_x'].notna()) &
        (df['AOI_Origin_y'].notna()) &
        (df['AOI_Width'] > 0) &
        (df['AOI_Height'] > 0) &
        (df['ObjectName'].isin(['Blue_Bin', 'Yellow_Bin', 'Green_Bin', 'Black_Bin', 'Item_Area', 'Check_Button', 'Cross_Button']))
    ]

    aoi_styles = {
        'Blue_Bin': {'color': 'blue', 'label': 'Blue\nBin', 'alpha': 0.4, 'edgecolor': 'darkblue'},
        'Yellow_Bin': {'color': 'yellow', 'label': 'Yellow\nBin', 'alpha': 0.4, 'edgecolor': 'orange'},
        'Green_Bin': {'color': 'green', 'label': 'Green\nBin', 'alpha': 0.4, 'edgecolor': 'darkgreen'},
        'Black_Bin': {'color': 'black', 'label': 'Black\nBin', 'alpha': 0.4, 'edgecolor': 'gray'},
        'Item_Area': {'color': 'purple', 'label': 'Items\nArea', 'alpha': 0.3, 'edgecolor': 'darkviolet'},
        'Check_Button': {'color': 'lightgreen', 'label': 'Check\nButton', 'alpha': 0.5, 'edgecolor': 'darkgreen'},
        'Cross_Button': {'color': 'lightcoral', 'label': 'Cross\nButton', 'alpha': 0.5, 'edgecolor': 'darkred'}
    }

    for _, aoi_row in aoi_data_fixed.iterrows():
        from matplotlib.patches import Rectangle

        obj_name = aoi_row['ObjectName']
        style = aoi_styles.get(obj_name, {'color': 'gray', 'label': obj_name, 'alpha': 0.3, 'edgecolor': 'black'})

        x = aoi_row['AOI_Origin_x']
        y = aoi_row['AOI_Origin_y']
        width = aoi_row['AOI_Width']
        height = aoi_row['AOI_Height']

        rect = Rectangle((x, y), width, height,
                        linewidth=3, edgecolor=style['edgecolor'],
                        facecolor=style['color'], alpha=style['alpha'])
        ax2.add_patch(rect)
        label_x = x + width/2
        label_y = y + height/2
        text = ax2.text(label_x, label_y, style['label'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=style['color'], alpha=0.8, edgecolor=style['edgecolor']))
        text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    ax2.set_xlim(0, screen_width)
    ax2.set_ylim(0, screen_height)
    ax2.set_title('Main AOI Regions with Eye Tracking Overlay')
    ax2.set_xlabel('X Coordinate (pixels)')
    ax2.set_ylabel('Y Coordinate (pixels)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()

    return fig

def process_level(path, level_num):
    df = load_and_clean_data(path)
    if df is None:
        return None

    stage_name = f"Level {level_num}"

    # Diagnose coordinate issues
    eye_data, aoi_data = diagnose_coordinate_systems(df, stage_name)
    df_fixed = fix_aoi_coordinates(df, stage_name)

    print(df_fixed.head())
    print(f"Final shape: {df_fixed.shape}")
    aoi_focus_percentage, aoi_hits = recalculate_aoi_focus(df_fixed, stage_name)
    viz_fig = create_aoi_visualization(df_fixed, stage_name)

    return {
        'level': level_num,
        'data': df_fixed,
        'aoi_focus': aoi_focus_percentage,
        'aoi_hits': aoi_hits
    }

# Main execution
def main():
    file_paths = [level1_path, level2_path, level3_path]
    results = []

    for i, path in enumerate(file_paths, 1):
        result = process_level(path, i)
        if result:
            results.append(result)

    for result in results:
        level = result['level']
        aoi_focus = result['aoi_focus']
        print(f"Level {level}: AOI Focus = {aoi_focus:.1f}%")

    return results

# Run
if __name__ == "__main__":
    results = main()


