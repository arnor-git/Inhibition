# -*- coding: utf-8 -*-
"""Inhibition Game- Research Paper-V2_Fixation_fixed.ipynb


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import datetime
import json
import  os
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.patches as patches
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""# Final Code"""

import pandas as pd

# Define file paths - Updated for Inhibition Game
#level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_56_INHIBITION_Level_1/DataJSON/Data.csv'
#level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_57_INHIBITION_Level_2/DataJSON/Data.csv'
#level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_58_INHIBITION_Level_3/DataJSON/Data.csv'

level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_42_INHIBITION_Level_1/DataJSON/Data.csv'
level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_43_INHIBITION_Level_2/DataJSON/Data.csv'
level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_44_INHIBITION_Level_3/DataJSON/Data.csv'

#level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_3_P2/Session_9_10-12-2024_09-28-36/Activity_33_INHIBITION_Level_1/DataJSON/Data.csv'
#level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_3_P2/Session_9_10-12-2024_09-28-36/Activity_34_INHIBITION_Level_2/DataJSON/Data.csv'
#level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_3_P2/Session_9_10-12-2024_09-28-36/Activity_35_INHIBITION_Level_3/DataJSON/Data.csv'

import pandas as pd

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
print(f"Shape: {df_level3.shape}")
print(f"Columns: {df_level3.columns.tolist()}")
df_level3.head(5)

import pandas as pd

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

"""Data Pre-Processing"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns


# Screen dimensions
screen_width = 1920
screen_height = 1080

def load_and_clean_data(path):
    """Load and clean CSV data with coordinate parsing"""
    try:
        df = pd.read_csv(path, sep=';')
        print(f" Loaded {len(df)} rows from {path}")

        # Print BEFORE preprocessing

        print("DATAFRAME HEAD - BEFORE PREPROCESSING")
        print(df.head())
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")

        # Analyze the data structure
        print(f"\nData Analysis:")
        print(f"ObjectName types: {df['ObjectName'].value_counts()}")
        print(f"ObjectState types: {df['ObjectState'].value_counts()}")
        print(f"Label types: {df['Label'].value_counts()}")

        # EXCLUDE "Picked Trash" rows as requested
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

        # For this dataset, obj_x and obj_y ARE the AOI_Origin coordinates
        if 'AOI_Origin_x' not in df.columns and 'obj_x' in df.columns:
            df['AOI_Origin_x'] = df['obj_x']
            df['AOI_Origin_y'] = df['obj_y']

        # Clean eye tracking data - keep rows with valid eye tracking OR valid AOI data
        df_clean = df.copy()

        # Only remove rows where both eye tracking AND AOI data are invalid
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

        # Print AFTER preprocessing

        print("DATAFRAME HEAD - AFTER PREPROCESSING")

        print(df_clean.head(10))
        print(f"Columns: {list(df_clean.columns)}")
        print(f"Shape: {df_clean.shape}")

        print(f"  After cleaning: {len(df_clean)} rows ({len(df_clean)/initial_count*100:.1f}% retained)")
        return df_clean.reset_index(drop=True)

    except Exception as e:
        print(f"✗ Error loading {path}: {e}")
        return None

def diagnose_coordinate_systems(df, stage_name):
    """Diagnose coordinate system issues"""

    print(f"COORDINATE DIAGNOSIS - {stage_name}")

    # Analyze eye tracking coordinates
    eye_data = df.dropna(subset=['EyeTracker-x', 'EyeTracker-y'])
    eye_data = eye_data[~((eye_data['EyeTracker-x'] == 0) & (eye_data['EyeTracker-y'] == 0))]

    if len(eye_data) > 0:
        print(f"Eye Tracking: X={eye_data['EyeTracker-x'].min():.1f}-{eye_data['EyeTracker-x'].max():.1f}, Y={eye_data['EyeTracker-y'].min():.1f}-{eye_data['EyeTracker-y'].max():.1f}")

    # Analyze AOI coordinates
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
    """Apply coordinate fixes and create the 5 main AOIs"""
    print(f"\nAPPLYING FIXES - {stage_name}")

    df_fixed = df.copy()

    # Add the 5 main AOIs as specified
    main_aois = []

    # 1. Blue Bin AOI (using exact coordinates from data)
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

    # 2. Yellow Bin AOI (using exact coordinates from data)
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

    # 3. Green Bin AOI (using exact coordinates from data)
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

    # 4. Black Bin AOI (estimated based on pattern from other bins)
    # Pattern analysis: AOI_Origin_x ≈ obj_x, AOI_Origin_y ≈ obj_y - 70
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

    # 5. Item Area (using exact coordinates from data)
    # Items appear at X=1471.4, Y=814.8 consistently
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

    # 6. Check Button AOI (based on actual click coordinates)
    # Check clicks: X=1200-1250, Y=60-120
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

    # 7. Cross Button AOI (based on actual click coordinates)
    # Cross clicks: X=550-630, Y=100-150
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
    # Add main AOIs to dataframe
    if main_aois:
        main_aoi_df = pd.DataFrame(main_aois)
        df_fixed = pd.concat([df_fixed, main_aoi_df], ignore_index=True)
        print(f"Added {len(main_aois)} main AOIs")

    # Clamp all AOIs to screen bounds
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

        print("✓ Clamped all AOIs to screen bounds")

    # Summary of final AOIs
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
    """Recalculate AOI focus with the 5 main AOIs"""
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
            # Skip other AOIs to focus only on the main 5
            continue

        unique_aois[group_key] = {
            'left': aoi_row['AOI_Origin_x'],
            'right': aoi_row['AOI_Origin_x'] + aoi_row['AOI_Width'],
            'top': aoi_row['AOI_Origin_y'],
            'bottom': aoi_row['AOI_Origin_y'] + aoi_row['AOI_Height']
        }

    print(f"Created {len(unique_aois)} main AOI groups: {list(unique_aois.keys())}")

    # Check each gaze point against main AOI groups
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
    """Create visualization showing eye tracking and the main AOI coordinates"""
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

        # Plot heatmap
        im = ax1.imshow(H.T, origin='lower', extent=[0, screen_width, 0, screen_height],
                       cmap='hot', alpha=0.7, aspect='auto')

        # Add colorbar
        plt.colorbar(im, ax=ax1, label='Gaze Density')

        # Overlay scatter plot
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

    # Plot eye tracking data
    if len(eye_data) > 0:
        ax2.scatter(eye_data['EyeTracker-x'], eye_data['EyeTracker-y'],
                   alpha=0.4, s=2, c='blue', label=f'Eye Tracking ({len(eye_data)} points)')

    # AOI regions - focus only on main AOIs
    aoi_data_fixed = df[
        (df['AOI_Width'].notna()) &
        (df['AOI_Height'].notna()) &
        (df['AOI_Origin_x'].notna()) &
        (df['AOI_Origin_y'].notna()) &
        (df['AOI_Width'] > 0) &
        (df['AOI_Height'] > 0) &
        (df['ObjectName'].isin(['Blue_Bin', 'Yellow_Bin', 'Green_Bin', 'Black_Bin', 'Item_Area', 'Check_Button', 'Cross_Button']))
    ]

    # Define colors and labels for main AOIs
    aoi_styles = {
        'Blue_Bin': {'color': 'blue', 'label': 'Blue\nBin', 'alpha': 0.4, 'edgecolor': 'darkblue'},
        'Yellow_Bin': {'color': 'yellow', 'label': 'Yellow\nBin', 'alpha': 0.4, 'edgecolor': 'orange'},
        'Green_Bin': {'color': 'green', 'label': 'Green\nBin', 'alpha': 0.4, 'edgecolor': 'darkgreen'},
        'Black_Bin': {'color': 'black', 'label': 'Black\nBin', 'alpha': 0.4, 'edgecolor': 'gray'},
        'Item_Area': {'color': 'purple', 'label': 'Items\nArea', 'alpha': 0.3, 'edgecolor': 'darkviolet'},
        'Check_Button': {'color': 'lightgreen', 'label': 'Check\nButton', 'alpha': 0.5, 'edgecolor': 'darkgreen'},
        'Cross_Button': {'color': 'lightcoral', 'label': 'Cross\nButton', 'alpha': 0.5, 'edgecolor': 'darkred'}
    }

    # Draw main AOIs
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

        # Position label in center of AOI with better contrast
        label_x = x + width/2
        label_y = y + height/2

        # Use white text with black outline for better visibility
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
    """Process a single level"""
    print(f"PROCESSING LEVEL {level_num}")

    # Load and clean data
    df = load_and_clean_data(path)
    if df is None:
        return None

    stage_name = f"Level {level_num}"

    # Diagnose coordinate issues
    eye_data, aoi_data = diagnose_coordinate_systems(df, stage_name)

    # Apply fixes
    df_fixed = fix_aoi_coordinates(df, stage_name)

    # Print dataframe head after fixes

    print(f"DATAFRAME HEAD - AFTER FIXES ({stage_name})")
    print(df_fixed.head())
    print(f"Final shape: {df_fixed.shape}")

    # Recalculate AOI focus
    aoi_focus_percentage, aoi_hits = recalculate_aoi_focus(df_fixed, stage_name)

    # Create visualization
    viz_fig = create_aoi_visualization(df_fixed, stage_name)

    return {
        'level': level_num,
        'data': df_fixed,
        'aoi_focus': aoi_focus_percentage,
        'aoi_hits': aoi_hits
    }

# Main execution
def main():
    """Main function to process all levels"""
    print("AOI COORDINATE SYSTEM FIX")

    file_paths = [level1_path, level2_path, level3_path]
    results = []

    for i, path in enumerate(file_paths, 1):
        result = process_level(path, i)
        if result:
            results.append(result)

    # Summary
    print("SUMMARY")

    for result in results:
        level = result['level']
        aoi_focus = result['aoi_focus']
        print(f"Level {level}: AOI Focus = {aoi_focus:.1f}%")

    return results

# Run the analysis
if __name__ == "__main__":
    results = main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION and dataset paths

# level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_56_INHIBITION_Level_1/DataJSON/Data.csv'
# level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_57_INHIBITION_Level_2/DataJSON/Data.csv'
# level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_6_p6/Session_14_10-12-2024_14-37-30/Activity_58_INHIBITION_Level_3/DataJSON/Data.csv'

# level1_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_42_INHIBITION_Level_1/DataJSON/Data.csv'
# level2_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_43_INHIBITION_Level_2/DataJSON/Data.csv'
# level3_path = '/content/drive/MyDrive/scoring/Inhibition Game/Student_4_p4/Session_13_10-12-2024_13-40-07/Activity_44_INHIBITION_Level_3/DataJSON/Data.csv'

screen_width  = 1920
screen_height = 1080

# I-VT parameters
MIN_FIXATION_DURATION       = 100   # ms  — physiological minimum
FALLBACK_VELOCITY_THRESHOLD = 300   # px/s — used only if GMM fails


# DATA LOADING & CLEANING

def load_and_clean_data(path):
    """Load CSV, remove invalid rows, parse all coordinate columns."""
    try:
        df = pd.read_csv(path, sep=';')
        print(f"✓ Loaded {len(df)} rows from {path}")

        initial_count = len(df)

        # Remove 'Picked Trash' rows
        mask = df['ObjectState'].astype(str).str.contains('Picked Trash', case=False, na=False)
        df = df[~mask].copy()
        removed = initial_count - len(df)
        if removed > 0:
            print(f"  Removed {removed} 'Picked Trash' rows")

        # Parse EyeTracker coordinates
        if 'EyeTracker' in df.columns:
            df[['EyeTracker-x', 'EyeTracker-y']] = (
                df['EyeTracker'].astype(str).str.strip("()").str.replace(" ", "")
                .str.split(",", expand=True).astype(float)
            )

        # Parse game object screen coordinates
        if 'GameObjectPos (Screen Coordinates)' in df.columns:
            df[['obj_x', 'obj_y']] = (
                df['GameObjectPos (Screen Coordinates)'].astype(str)
                .str.strip("()").str.replace(" ", "")
                .str.split(",", expand=True).astype(float)
            )

        # Parse AOI origin (fall back to obj_x/y if not present)
        if 'AOI_Origin' in df.columns:
            df[['AOI_Origin_x', 'AOI_Origin_y']] = (
                df['AOI_Origin'].astype(str).str.strip("()").str.replace(" ", "")
                .str.split(",", expand=True).astype(float)
            )
        elif 'obj_x' in df.columns:
            df['AOI_Origin_x'] = df['obj_x']
            df['AOI_Origin_y'] = df['obj_y']

        # Drop rows where both eye AND AOI data are invalid
        # Guard: columns may not exist if parsing was skipped
        if 'EyeTracker-x' not in df.columns:
            df['EyeTracker-x'] = np.nan
        if 'EyeTracker-y' not in df.columns:
            df['EyeTracker-y'] = np.nan
        if 'AOI_Width' not in df.columns:
            df['AOI_Width'] = np.nan
        if 'AOI_Height' not in df.columns:
            df['AOI_Height'] = np.nan
        if 'AOI_Origin_x' not in df.columns:
            df['AOI_Origin_x'] = np.nan
        if 'AOI_Origin_y' not in df.columns:
            df['AOI_Origin_y'] = np.nan

        eye_invalid = (
            df['EyeTracker-x'].isna() | df['EyeTracker-y'].isna() |
            ((df['EyeTracker-x'] == 0) & (df['EyeTracker-y'] == 0))
        )
        aoi_invalid = (
            df['AOI_Width'].isna() | df['AOI_Height'].isna() |
            df['AOI_Origin_x'].isna() | df['AOI_Origin_y'].isna() |
            (df['AOI_Width'] == 0) | (df['AOI_Height'] == 0)
        )
        df_clean = df[~(eye_invalid & aoi_invalid)].copy()

        # Keep only columns we need
        wanted = ['Timestamp', 'ObjectName', 'ObjectState', 'Label', 'Aux Message',
                  'AOI_Width', 'AOI_Height', 'EyeTracker-x', 'EyeTracker-y',
                  'obj_x', 'obj_y', 'AOI_Origin_x', 'AOI_Origin_y']
        df_clean = df_clean[[c for c in wanted if c in df_clean.columns]].copy()

        print(f"  After cleaning: {len(df_clean)} rows "
              f"({len(df_clean)/initial_count*100:.1f}% retained)")
        return df_clean.reset_index(drop=True)

    except Exception as e:
        print(f"✗ Error loading {path}: {e}")
        return None


# AOI COORDINATE SETUP

def fix_aoi_coordinates(df):
    """Append the seven fixed game AOIs and clamp all AOI origins to screen bounds."""
    df_fixed = df.copy()

    ts_last = df_fixed['Timestamp'].iloc[-1] if len(df_fixed) > 0 else 0

    main_aois = [
        {'ObjectName': 'Blue_Bin',    'AOI_Width': 304, 'AOI_Height': 302,
         'AOI_Origin_x': 507.1,  'AOI_Origin_y': 678.8,  'obj_x': 494.8,  'obj_y': 748.2},
        {'ObjectName': 'Yellow_Bin',  'AOI_Width': 252, 'AOI_Height': 292,
         'AOI_Origin_x': 804.8,  'AOI_Origin_y': 682.6,  'obj_x': 801.0,  'obj_y': 752.0},
        {'ObjectName': 'Green_Bin',   'AOI_Width': 241, 'AOI_Height': 292,
         'AOI_Origin_x': 1076.5, 'AOI_Origin_y': 682.4,  'obj_x': 1079.7, 'obj_y': 751.9},
        {'ObjectName': 'Black_Bin',   'AOI_Width': 250, 'AOI_Height': 295,
         'AOI_Origin_x': 1400.0, 'AOI_Origin_y': 680.0,  'obj_x': 1400.0, 'obj_y': 750.0},
        {'ObjectName': 'Item_Area',   'AOI_Width': 400, 'AOI_Height': 400,
         'AOI_Origin_x': 1271.4, 'AOI_Origin_y': 614.8,  'obj_x': 1471.4, 'obj_y': 814.8},
        {'ObjectName': 'Check_Button','AOI_Width': 100, 'AOI_Height':  80,
         'AOI_Origin_x': 1175.0, 'AOI_Origin_y':  50.0,  'obj_x': 1225.0, 'obj_y':  90.0},
        {'ObjectName': 'Cross_Button','AOI_Width': 100, 'AOI_Height':  80,
         'AOI_Origin_x':  540.0, 'AOI_Origin_y':  85.0,  'obj_x':  590.0, 'obj_y': 125.0},
    ]

    for aoi in main_aois:
        aoi.update({'Timestamp': ts_last, 'ObjectState': 'Main_AOI',
                    'Label': aoi['ObjectName'], 'Aux Message': 'Main_Game_AOI',
                    'EyeTracker-x': 0.0, 'EyeTracker-y': 0.0})
        # Ensure all columns present in df_fixed exist in the new rows
        for col in df_fixed.columns:
            if col not in aoi:
                aoi[col] = np.nan

    df_fixed = pd.concat([df_fixed, pd.DataFrame(main_aois)], ignore_index=True)

    # Clamp AOI origins so rectangles don't exceed screen
    aoi_mask = (df_fixed['AOI_Width'].notna() & df_fixed['AOI_Height'].notna() &
                df_fixed['AOI_Origin_x'].notna() & df_fixed['AOI_Origin_y'].notna() &
                (df_fixed['AOI_Width'] > 0) & (df_fixed['AOI_Height'] > 0))

    df_fixed.loc[aoi_mask, 'AOI_Origin_x'] = np.clip(
        df_fixed.loc[aoi_mask, 'AOI_Origin_x'], 0,
        screen_width  - df_fixed.loc[aoi_mask, 'AOI_Width'])
    df_fixed.loc[aoi_mask, 'AOI_Origin_y'] = np.clip(
        df_fixed.loc[aoi_mask, 'AOI_Origin_y'], 0,
        screen_height - df_fixed.loc[aoi_mask, 'AOI_Height'])

    return df_fixed


# SECTION 3 — DYNAMIC I-VT FIXATION / SACCADE DETECTION

def calculate_velocities(x, y, timestamps):
    """
    Central-difference velocity (px/s).
    Smoother than simple forward-difference; edge samples are filled.
    """
    n = len(x)
    vel = np.zeros(n)
    for i in range(1, n - 1):
        dt1 = (timestamps[i]   - timestamps[i-1]) / 1000.0
        dt2 = (timestamps[i+1] - timestamps[i])   / 1000.0
        if dt1 > 0 and dt2 > 0:
            vx = ((x[i]-x[i-1])/dt1 + (x[i+1]-x[i])/dt2) / 2
            vy = ((y[i]-y[i-1])/dt1 + (y[i+1]-y[i])/dt2) / 2
            vel[i] = np.sqrt(vx**2 + vy**2)
    if n > 1:
        vel[0]  = vel[1]
        vel[-1] = vel[-2]
    return vel


def compute_gmm_threshold(velocities, fallback=FALLBACK_VELOCITY_THRESHOLD):
    """
    Fit a 2-component GMM to the velocity distribution.
    Returns (threshold, method_name).

    The threshold is the crossover point where p(saccade|v) >= p(fixation|v),
    giving a statistically principled, per-session boundary.
    Falls back to FALLBACK_VELOCITY_THRESHOLD if GMM cannot fit reliably.
    """
    valid = velocities[velocities > 0]
    if len(valid) < 20:
        print(f"  ⚠ Too few samples for GMM — fallback {fallback} px/s")
        return fallback, "fallback"

    try:
        gmm = GaussianMixture(n_components=2, covariance_type='full',
                              max_iter=200, random_state=42)
        gmm.fit(valid.reshape(-1, 1))

        means   = gmm.means_.flatten()                          # shape (2,)
        # covariances_ shape is (2, 1, 1) for full covariance on 1-D data
        stds    = np.sqrt(gmm.covariances_[:, 0, 0])            # shape (2,)
        weights = gmm.weights_.flatten()                         # shape (2,)

        fix_idx  = np.argmin(means)
        sacc_idx = np.argmax(means)
        mu_f, sig_f, w_f = means[fix_idx],  stds[fix_idx],  weights[fix_idx]
        mu_s, sig_s, w_s = means[sacc_idx], stds[sacc_idx], weights[sacc_idx]

        print(f"  GMM → fixation:  μ={mu_f:.1f} σ={sig_f:.1f} w={w_f:.2f}")
        print(f"  GMM → saccade:   μ={mu_s:.1f} σ={sig_s:.1f} w={w_s:.2f}")

        scan  = np.linspace(mu_f, mu_s, 1000)
        p_fix = w_f * np.exp(-0.5*((scan-mu_f)/sig_f)**2) / (sig_f*np.sqrt(2*np.pi))
        p_sac = w_s * np.exp(-0.5*((scan-mu_s)/sig_s)**2) / (sig_s*np.sqrt(2*np.pi))

        cross = np.where(p_sac >= p_fix)[0]
        if len(cross) == 0:
            threshold, method = (mu_f+mu_s)/2, "GMM-midpoint"
        else:
            threshold, method = scan[cross[0]], "GMM-crossover"

        if not (mu_f < threshold < mu_s):
            threshold, method = (mu_f+mu_s)/2, "GMM-midpoint(clamped)"

        print(f"  ✓ Dynamic threshold ({method}): {threshold:.1f} px/s")
        return threshold, method

    except Exception as e:
        print(f"  ⚠ GMM failed ({e}) — fallback {fallback} px/s")
        return fallback, "fallback"


def detect_fixations_saccades(df):
    """
    I-VT classification using a per-session GMM-derived velocity threshold.

    Returns
    -------
    fixations_df : DataFrame  (center_x, center_y, duration, start_time,
                                end_time, dispersion, point_count)
    saccades_df  : DataFrame  (start_x, start_y, end_x, end_y, amplitude,
                                duration, peak_velocity, start_time, end_time)
    threshold    : float   — the GMM threshold used
    """
    eye = df.dropna(subset=['EyeTracker-x', 'EyeTracker-y']).copy()
    eye = eye[~((eye['EyeTracker-x'] == 0) & (eye['EyeTracker-y'] == 0))]
    eye = eye.sort_values('Timestamp').reset_index(drop=True)

    print(f"  Eye data points: {len(eye)}")
    if len(eye) < 5:
        return pd.DataFrame(), pd.DataFrame(), FALLBACK_VELOCITY_THRESHOLD

    velocities = calculate_velocities(
        eye['EyeTracker-x'].values,
        eye['EyeTracker-y'].values,
        eye['Timestamp'].values
    )
    eye['velocity'] = velocities

    threshold, method = compute_gmm_threshold(velocities)
    eye['is_saccade'] = eye['velocity'] >= threshold

    fixations, saccades = [], []
    cur_fix, cur_sac = [], []

    def flush_fix(idx):
        if len(idx) < 2:
            return
        seg = eye.iloc[idx]
        dur = seg['Timestamp'].iloc[-1] - seg['Timestamp'].iloc[0]
        if dur < MIN_FIXATION_DURATION:
            return
        fixations.append({
            'center_x':    seg['EyeTracker-x'].mean(),
            'center_y':    seg['EyeTracker-y'].mean(),
            'duration':    dur,
            'start_time':  seg['Timestamp'].iloc[0],
            'end_time':    seg['Timestamp'].iloc[-1],
            'dispersion':  np.sqrt(seg['EyeTracker-x'].var(ddof=0) + seg['EyeTracker-y'].var(ddof=0)),
            'point_count': len(idx)
        })

    def flush_sac(idx):
        if len(idx) < 2:
            return
        seg = eye.iloc[idx]
        sx, sy = seg['EyeTracker-x'].iloc[0],  seg['EyeTracker-y'].iloc[0]
        ex, ey = seg['EyeTracker-x'].iloc[-1], seg['EyeTracker-y'].iloc[-1]
        amp = euclidean([sx, sy], [ex, ey])
        dur = seg['Timestamp'].iloc[-1] - seg['Timestamp'].iloc[0]
        if dur <= 0:
            return
        saccades.append({
            'start_x': sx, 'start_y': sy, 'end_x': ex, 'end_y': ey,
            'amplitude': amp, 'duration': dur,
            'peak_velocity': seg['velocity'].max(),
            'start_time': seg['Timestamp'].iloc[0],
            'end_time':   seg['Timestamp'].iloc[-1]
        })

    for i, is_sac in enumerate(eye['is_saccade']):
        if is_sac:
            if cur_fix:
                flush_fix(cur_fix); cur_fix = []
            cur_sac.append(i)
        else:
            if cur_sac:
                flush_sac(cur_sac); cur_sac = []
            cur_fix.append(i)

    flush_fix(cur_fix)
    flush_sac(cur_sac)

    fix_df  = pd.DataFrame(fixations)
    sacc_df = pd.DataFrame(saccades)
    print(f"  → Fixations: {len(fix_df)}   Saccades: {len(sacc_df)}")
    return fix_df, sacc_df, threshold


# SECTION 4 — ADVANCED METRICS  (AOI + attention — uses I-VT results)

def _build_aoi_bounds(df):
    """Extract named AOI bounding boxes from the dataframe.
    Uses the last occurrence of each AOI name to prefer the fixed Main_AOI rows
    appended by fix_aoi_coordinates() over any raw data entries.
    """
    aoi_names = ['Blue_Bin', 'Yellow_Bin', 'Green_Bin', 'Black_Bin',
                 'Item_Area', 'Check_Button', 'Cross_Button']
    aoi_data = df[
        df['AOI_Width'].notna() & df['AOI_Height'].notna() &
        df['AOI_Origin_x'].notna() & df['AOI_Origin_y'].notna() &
        (df['AOI_Width'] > 0) & (df['AOI_Height'] > 0)
    ]
    bounds = {}
    for _, row in aoi_data.iterrows():
        name = str(row.get('ObjectName', ''))
        if name in aoi_names:
            # Overwrite — last row wins (Main_AOI rows are appended last)
            bounds[name] = {
                'left':   row['AOI_Origin_x'],
                'right':  row['AOI_Origin_x'] + row['AOI_Width'],
                'top':    row['AOI_Origin_y'],
                'bottom': row['AOI_Origin_y'] + row['AOI_Height'],
            }
    return bounds


def calculate_advanced_metrics(df, level_num, fixations_df, saccades_df):
    """
    Compute AOI-based attention and inhibition metrics.
    Fixation/saccade counts and durations now come from the dynamic I-VT
    results (fixations_df, saccades_df) instead of a hardcoded algorithm.
    """
    eye = df.dropna(subset=['EyeTracker-x', 'EyeTracker-y']).copy()
    eye = eye[~((eye['EyeTracker-x'] == 0) & (eye['EyeTracker-y'] == 0))]
    eye = eye.sort_values('Timestamp')

    if len(eye) == 0:
        return {
            'level': level_num, 'total_gaze_points': 0,
            'fixation_count': 0, 'avg_fixation_duration': 0,
            'avg_saccade_velocity': 0, 'attention_scatter': 0,
            'aoi_transitions': 0, 'task_relevance_ratio': 0,
            'session_duration': 0, 'gaze_efficiency': 0,
        }

    aoi_bounds = _build_aoi_bounds(df)

    # ── AOI transition count
    aoi_transitions, current_aoi = 0, None
    for _, row in eye.iterrows():
        gx, gy = row['EyeTracker-x'], row['EyeTracker-y']
        found = None
        for name, b in aoi_bounds.items():
            if b['left'] <= gx <= b['right'] and b['top'] <= gy <= b['bottom']:
                found = name
                break
        if found != current_aoi:
            if current_aoi is not None:
                aoi_transitions += 1
            current_aoi = found

    # ── Bin vs other attention
    bin_names = {'Blue_Bin', 'Yellow_Bin', 'Green_Bin', 'Black_Bin'}
    bin_attn = other_attn = 0
    for _, row in eye.iterrows():
        gx, gy = row['EyeTracker-x'], row['EyeTracker-y']
        in_bin = any(
            b['left'] <= gx <= b['right'] and b['top'] <= gy <= b['bottom']
            for n, b in aoi_bounds.items() if n in bin_names
        )
        if in_bin:
            bin_attn += 1
        else:
            other_attn += 1

    task_relevance = bin_attn / (bin_attn + other_attn) if (bin_attn + other_attn) > 0 else 0

    # ── Attention scatter
    attention_scatter = float(eye['EyeTracker-x'].std() + eye['EyeTracker-y'].std())

    # ── Use I-VT fixation results (replaces old distance-threshold loop)
    fixation_count     = len(fixations_df)
    avg_fix_duration   = fixations_df['duration'].mean() if fixation_count > 0 else 0

    # ── Use I-VT saccade velocity
    avg_sacc_velocity  = saccades_df['peak_velocity'].mean() if len(saccades_df) > 0 else 0

    session_duration   = (eye['Timestamp'].max() - eye['Timestamp'].min()) / 1000.0
    gaze_efficiency    = fixation_count / len(eye) if len(eye) > 0 else 0

    return {
        'level':                level_num,
        'total_gaze_points':    len(eye),
        'fixation_count':       fixation_count,
        'avg_fixation_duration':avg_fix_duration,
        'avg_saccade_velocity': avg_sacc_velocity,
        'attention_scatter':    attention_scatter,
        'aoi_transitions':      aoi_transitions,
        'task_relevance_ratio': task_relevance,
        'session_duration':     session_duration,
        'gaze_efficiency':      gaze_efficiency,
    }


# SECTION 5 — GAZE PATTERN ANALYSIS

def analyze_gaze_patterns(fixations_df, saccades_df, level_name):
    """Produce qualitative insights from I-VT fixation/saccade results."""
    print(f"\nGAZE PATTERN ANALYSIS — {level_name}")
    insights = []

    if len(fixations_df) > 0:
        avg_dur = fixations_df['duration'].mean()
        print(f"\n  FIXATION ANALYSIS:")
        print(f"    Total: {len(fixations_df)}   Avg duration: {avg_dur:.1f} ms   "
              f"Max: {fixations_df['duration'].max():.1f} ms   "
              f"Total time: {fixations_df['duration'].sum():.1f} ms")

        if avg_dur > 400:
            insights.append("Long fixations suggest deep processing or difficulty")
        elif avg_dur < 200:
            insights.append("Short fixations indicate quick scanning behaviour")
        else:
            insights.append("Normal fixation duration suggests efficient processing")

        regions = {
            'top_left':     (fixations_df['center_x'] <  screen_width/2) & (fixations_df['center_y'] <  screen_height/2),
            'top_right':    (fixations_df['center_x'] >= screen_width/2) & (fixations_df['center_y'] <  screen_height/2),
            'bottom_left':  (fixations_df['center_x'] <  screen_width/2) & (fixations_df['center_y'] >= screen_height/2),
            'bottom_right': (fixations_df['center_x'] >= screen_width/2) & (fixations_df['center_y'] >= screen_height/2),
        }
        region_counts   = {r: fixations_df[mask].shape[0] for r, mask in regions.items()}
        dominant_region = max(region_counts, key=region_counts.get)
        print(f"    Region distribution: {region_counts}  |  Dominant: {dominant_region}")
        if region_counts[dominant_region] > len(fixations_df) * 0.6:
            insights.append(f"Heavy focus on {dominant_region.replace('_',' ')} region")

    if len(saccades_df) > 0:
        avg_amp = saccades_df['amplitude'].mean()
        avg_vel = saccades_df['peak_velocity'].mean()
        print(f"\n  SACCADE ANALYSIS:")
        print(f"    Total: {len(saccades_df)}   Avg amplitude: {avg_amp:.1f} px   "
              f"Avg peak velocity: {avg_vel:.1f} px/s")

        insights.append("Large saccades — broad visual search"   if avg_amp > 300 else
                        "Small saccades — focused examination"   if avg_amp < 100 else
                        "Moderate saccade amplitude")
        insights.append("High saccade velocity — quick shifts"   if avg_vel > 1000 else
                        "Low saccade velocity — deliberate moves" if avg_vel < 500 else
                        "Normal saccade velocity")

    if len(fixations_df) > 0 and len(saccades_df) > 0:
        ratio = len(fixations_df) / len(saccades_df)
        scan_path = saccades_df['amplitude'].sum()
        print(f"\n  COMBINED: F/S ratio={ratio:.2f}   Scan path={scan_path:.1f} px")
        if ratio > 1.5:
            insights.append("High F/S ratio — focused attention")
        elif ratio < 0.5:
            insights.append("Low F/S ratio — active visual exploration")
        if scan_path > 10000:
            insights.append("Extensive scan path — thorough exploration")
        elif scan_path < 3000:
            insights.append("Short scan path — direct, focused attention")

    print(f"\n  KEY INSIGHTS:")
    for ins in insights:
        print(f"    • {ins}")
    return insights


# SECTION 6 — VISUALISATIONS


def create_ivt_visualization(df, fixations_df, saccades_df, level_name):
    """4-panel I-VT visualisation: fixations, saccades, scan path, timeline."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f'{level_name} — I-VT Fixation & Saccade Analysis',
                 fontsize=16, fontweight='bold')

    # Panel 1 — Fixations
    ax1 = axes[0]
    if len(fixations_df) > 0:
        sc = ax1.scatter(fixations_df['center_x'], fixations_df['center_y'],
                         c=fixations_df['duration'], s=fixations_df['duration']/5,
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
        plt.colorbar(sc, ax=ax1, label='Duration (ms)')
        for i, (_, f) in enumerate(fixations_df.iterrows()):
            ax1.annotate(f'{i+1}', (f['center_x'], f['center_y']),
                         ha='center', va='center', fontsize=8,
                         color='white', fontweight='bold')
    ax1.set_xlim(0, screen_width); ax1.set_ylim(0, screen_height)
    ax1.set_title(f'Fixations (n={len(fixations_df)})\nSize & Colour = Duration')
    ax1.set_xlabel('X (px)'); ax1.set_ylabel('Y (px)'); ax1.grid(True, alpha=0.3)

    # Panel 2 — Saccades
    ax2 = axes[1]
    if len(saccades_df) > 0:
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        amps = saccades_df['amplitude'].values
        norm = Normalize(vmin=amps.min(), vmax=amps.max())
        cmap = plt.cm.plasma
        for _, s in saccades_df.iterrows():
            c = cmap(norm(s['amplitude']))
            ax2.arrow(s['start_x'], s['start_y'],
                      s['end_x']-s['start_x'], s['end_y']-s['start_y'],
                      head_width=20, head_length=30, fc=c, ec=c, alpha=0.7, linewidth=2)
        sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        plt.colorbar(sm, ax=ax2, label='Amplitude (px)')
        ax2.set_title(f'Saccades (n={len(saccades_df)})\n'
                      f'Colour = Amplitude ({amps.min():.0f}–{amps.max():.0f} px)')
    else:
        ax2.set_title('Saccades (n=0)')
    ax2.set_xlim(0, screen_width); ax2.set_ylim(0, screen_height)
    ax2.set_xlabel('X (px)'); ax2.set_ylabel('Y (px)'); ax2.grid(True, alpha=0.3)

    # Panel 3 — Scan path
    ax3 = axes[2]
    if len(fixations_df) > 0:
        ax3.scatter(fixations_df['center_x'], fixations_df['center_y'],
                    s=100, c='red', alpha=0.7, edgecolors='black', linewidth=1)
        if len(fixations_df) > 1:
            ax3.plot(fixations_df['center_x'].values,
                     fixations_df['center_y'].values, 'b-', alpha=0.5, linewidth=2)
        for i, (_, f) in enumerate(fixations_df.iterrows()):
            ax3.annotate(f'{i+1}', (f['center_x'], f['center_y']),
                         ha='center', va='center', fontsize=8,
                         color='white', fontweight='bold')
    ax3.set_xlim(0, screen_width); ax3.set_ylim(0, screen_height)
    ax3.set_title('Scan Path'); ax3.set_xlabel('X (px)'); ax3.set_ylabel('Y (px)')
    ax3.grid(True, alpha=0.3)

    # Panel 4 — Timeline
    ax4 = axes[3]
    if len(fixations_df) > 0:
        sacc_t0 = saccades_df['start_time'].min() if len(saccades_df) > 0 else np.inf
        t0 = min(fixations_df['start_time'].min(), sacc_t0)
        if not np.isfinite(t0):
            t0 = fixations_df['start_time'].min()
        for i, (_, f) in enumerate(fixations_df.iterrows()):
            ax4.barh(i, f['duration']/1000, left=(f['start_time']-t0)/1000,
                     height=0.6, color='green', alpha=0.7,
                     label='Fixation' if i == 0 else '')
        if len(saccades_df) > 0:
            for i, (_, s) in enumerate(saccades_df.iterrows()):
                ax4.barh(len(fixations_df)+i, s['duration']/1000,
                         left=(s['start_time']-t0)/1000,
                         height=0.4, color='red', alpha=0.7,
                         label='Saccade' if i == 0 else '')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Event Index')
    ax4.set_title('Temporal Pattern'); ax4.grid(True, alpha=0.3); ax4.legend()

    plt.tight_layout()
    plt.close(fig)   # suppress individual display — shown in combined view
    return fig


def create_statistical_summary(fixations_df, saccades_df, level_name):
    """6-panel statistical summary."""
    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    fig.suptitle(f'{level_name} — Statistical Analysis', fontsize=16, fontweight='bold')

    def hist(ax, data, color, label, xlabel):
        ax.hist(data, bins=20, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--',
                   label=f'Mean: {data.mean():.1f} {label}')
        ax.set_xlabel(xlabel); ax.set_ylabel('Frequency')
        ax.legend(); ax.grid(True, alpha=0.3)

    if len(fixations_df) > 0:
        hist(axes[0], fixations_df['duration'],  'blue',   'ms',   'Duration (ms)')
        axes[0].set_title('Fixation Duration Distribution')
    if len(saccades_df) > 0:
        hist(axes[1], saccades_df['amplitude'],  'orange', 'px',   'Amplitude (px)')
        axes[1].set_title('Saccade Amplitude Distribution')
        hist(axes[2], saccades_df['peak_velocity'], 'green', 'px/s', 'Peak Velocity (px/s)')
        axes[2].set_title('Saccade Velocity Distribution')
    if len(fixations_df) > 0:
        axes[3].scatter(fixations_df['center_x'], fixations_df['center_y'],
                        alpha=0.6, s=50, c='purple')
        axes[3].set_xlim(0, screen_width); axes[3].set_ylim(0, screen_height)
        axes[3].set_xlabel('X (px)'); axes[3].set_ylabel('Y (px)')
        axes[3].set_title('Fixation Spatial Distribution'); axes[3].grid(True, alpha=0.3)
    if len(saccades_df) > 0:
        axes[4].scatter(saccades_df['amplitude'], saccades_df['duration'],
                        alpha=0.6, s=50, c='red')
        axes[4].set_xlabel('Amplitude (px)'); axes[4].set_ylabel('Duration (ms)')
        axes[4].set_title('Saccade Amplitude vs Duration'); axes[4].grid(True, alpha=0.3)
        axes[5].scatter(saccades_df['amplitude'], saccades_df['peak_velocity'],
                        alpha=0.6, s=50, c='blue')
        axes[5].set_xlabel('Amplitude (px)'); axes[5].set_ylabel('Peak Velocity (px/s)')
        axes[5].set_title('Saccade Amplitude vs Velocity'); axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.close(fig)   # suppress individual display — shown in combined view
    return fig


# COMBINED CROSS-LEVEL VISUALISATIONS

def create_combined_ivt_visualization(all_results):
    """
    Combined I-VT visualisation for all levels.
    Layout: rows = levels, columns = [Fixations, Saccades, Scan Path, Timeline]
    All levels shown together in one figure for easy comparison.
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    n_levels = len(all_results)
    fig, axes = plt.subplots(n_levels, 4,
                             figsize=(24, 5 * n_levels),
                             squeeze=False)
    fig.suptitle('I-VT Fixation & Saccade Analysis — All Levels',
                 fontsize=18, fontweight='bold', y=1.01)

    col_titles = ['Fixations\n(Size & Colour = Duration)',
                  'Saccades\n(Colour = Amplitude)',
                  'Scan Path',
                  'Temporal Pattern']
    for j, ct in enumerate(col_titles):
        axes[0][j].set_title(ct, fontsize=12, fontweight='bold', pad=10)

    for row, result in enumerate(all_results):
        fixations_df = result['fixations']
        saccades_df  = result['saccades']
        level_name   = f"Level {result['level']}"

        axes[row][0].set_ylabel(f'{level_name}\nY (px)', fontsize=11,
                                fontweight='bold')

        # Panel 1 — Fixations
        ax = axes[row][0]
        if len(fixations_df) > 0:
            sc = ax.scatter(fixations_df['center_x'], fixations_df['center_y'],
                            c=fixations_df['duration'],
                            s=fixations_df['duration'] / 5,
                            cmap='viridis', alpha=0.7,
                            edgecolors='black', linewidth=0.5)
            plt.colorbar(sc, ax=ax, label='Duration (ms)', shrink=0.8)
            for i, (_, f) in enumerate(fixations_df.iterrows()):
                ax.annotate(f'{i+1}', (f['center_x'], f['center_y']),
                            ha='center', va='center', fontsize=7,
                            color='white', fontweight='bold')
        ax.set_xlim(0, screen_width); ax.set_ylim(0, screen_height)
        ax.set_xlabel('X (px)'); ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.97, f'n={len(fixations_df)}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Panel 2 — Saccades
        ax = axes[row][1]
        if len(saccades_df) > 0:
            amps = saccades_df['amplitude'].values
            norm = Normalize(vmin=amps.min(), vmax=amps.max())
            cmap = plt.cm.plasma
            for _, s in saccades_df.iterrows():
                c = cmap(norm(s['amplitude']))
                ax.arrow(s['start_x'], s['start_y'],
                         s['end_x'] - s['start_x'],
                         s['end_y'] - s['start_y'],
                         head_width=18, head_length=25,
                         fc=c, ec=c, alpha=0.6, linewidth=1.5)
            sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Amplitude (px)', shrink=0.8)
            ax.text(0.02, 0.97,
                    f'n={len(saccades_df)}\n{amps.min():.0f}–{amps.max():.0f} px',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No saccades', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='grey')
        ax.set_xlim(0, screen_width); ax.set_ylim(0, screen_height)
        ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)'); ax.grid(True, alpha=0.3)

        # Panel 3 — Scan path
        ax = axes[row][2]
        if len(fixations_df) > 0:
            ax.scatter(fixations_df['center_x'], fixations_df['center_y'],
                       s=80, c='red', alpha=0.7,
                       edgecolors='black', linewidth=0.5)
            if len(fixations_df) > 1:
                ax.plot(fixations_df['center_x'].values,
                        fixations_df['center_y'].values,
                        'b-', alpha=0.4, linewidth=1.5)
            for i, (_, f) in enumerate(fixations_df.iterrows()):
                ax.annotate(f'{i+1}', (f['center_x'], f['center_y']),
                            ha='center', va='center', fontsize=7,
                            color='white', fontweight='bold')
        ax.set_xlim(0, screen_width); ax.set_ylim(0, screen_height)
        ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)'); ax.grid(True, alpha=0.3)

        # Panel 4 — Timeline
        ax = axes[row][3]
        if len(fixations_df) > 0:
            sacc_t0 = saccades_df['start_time'].min() \
                      if len(saccades_df) > 0 else np.inf
            t0 = min(fixations_df['start_time'].min(), sacc_t0)
            if not np.isfinite(t0):
                t0 = fixations_df['start_time'].min()
            for i, (_, f) in enumerate(fixations_df.iterrows()):
                ax.barh(i, f['duration'] / 1000,
                        left=(f['start_time'] - t0) / 1000,
                        height=0.5, color='green', alpha=0.7,
                        label='Fixation' if i == 0 else '')
            if len(saccades_df) > 0:
                for i, (_, s) in enumerate(saccades_df.iterrows()):
                    ax.barh(len(fixations_df) + i,
                            s['duration'] / 1000,
                            left=(s['start_time'] - t0) / 1000,
                            height=0.3, color='red', alpha=0.7,
                            label='Saccade' if i == 0 else '')
            ax.legend(fontsize=8, loc='upper right')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Event Index')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig


def create_combined_statistical_summary(all_results):
    """
    Combined statistical summary for all levels.
    Layout: rows = levels, columns = [Fix Duration, Sacc Amplitude,
            Sacc Velocity, Fix Spatial, Amp vs Dur, Amp vs Vel]
    """
    n_levels = len(all_results)
    fig, axes = plt.subplots(n_levels, 6,
                             figsize=(28, 5 * n_levels),
                             squeeze=False)
    fig.suptitle('Statistical Summary — All Levels',
                 fontsize=18, fontweight='bold', y=1.01)

    col_titles = ['Fixation Duration\nDistribution',
                  'Saccade Amplitude\nDistribution',
                  'Saccade Velocity\nDistribution',
                  'Fixation Spatial\nDistribution',
                  'Amplitude vs\nDuration',
                  'Amplitude vs\nVelocity']
    for j, ct in enumerate(col_titles):
        axes[0][j].set_title(ct, fontsize=11, fontweight='bold', pad=10)

    def hist(ax, data, color, unit, xlabel):
        ax.hist(data, bins=20, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {data.mean():.1f} {unit}')
        ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('Freq', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for row, result in enumerate(all_results):
        fixations_df = result['fixations']
        saccades_df  = result['saccades']
        level_name   = f"Level {result['level']}"

        axes[row][0].set_ylabel(f'{level_name}\nFrequency',
                                fontsize=11, fontweight='bold')

        # Col 0 — Fixation duration
        if len(fixations_df) > 0:
            hist(axes[row][0], fixations_df['duration'],
                 'steelblue', 'ms', 'Duration (ms)')
        else:
            axes[row][0].text(0.5, 0.5, 'No fixations',
                              transform=axes[row][0].transAxes,
                              ha='center', va='center', color='grey')

        # Col 1 — Saccade amplitude
        if len(saccades_df) > 0:
            hist(axes[row][1], saccades_df['amplitude'],
                 'darkorange', 'px', 'Amplitude (px)')
        else:
            axes[row][1].text(0.5, 0.5, 'No saccades',
                              transform=axes[row][1].transAxes,
                              ha='center', va='center', color='grey')

        # Col 2 — Saccade velocity
        if len(saccades_df) > 0:
            hist(axes[row][2], saccades_df['peak_velocity'],
                 'seagreen', 'px/s', 'Peak Velocity (px/s)')
        else:
            axes[row][2].text(0.5, 0.5, 'No saccades',
                              transform=axes[row][2].transAxes,
                              ha='center', va='center', color='grey')

        # Col 3 — Fixation spatial
        ax = axes[row][3]
        if len(fixations_df) > 0:
            ax.scatter(fixations_df['center_x'], fixations_df['center_y'],
                       alpha=0.6, s=40, c='purple', edgecolors='none')
            ax.set_xlim(0, screen_width); ax.set_ylim(0, screen_height)
            ax.set_xlabel('X (px)', fontsize=9)
            ax.set_ylabel('Y (px)', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No fixations',
                    transform=ax.transAxes, ha='center', va='center', color='grey')

        # Col 4 — Amplitude vs Duration
        ax = axes[row][4]
        if len(saccades_df) > 0:
            ax.scatter(saccades_df['amplitude'], saccades_df['duration'],
                       alpha=0.6, s=40, c='crimson', edgecolors='none')
            ax.set_xlabel('Amplitude (px)', fontsize=9)
            ax.set_ylabel('Duration (ms)', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No saccades',
                    transform=ax.transAxes, ha='center', va='center', color='grey')

        # Col 5 — Amplitude vs Velocity
        ax = axes[row][5]
        if len(saccades_df) > 0:
            ax.scatter(saccades_df['amplitude'], saccades_df['peak_velocity'],
                       alpha=0.6, s=40, c='royalblue', edgecolors='none')
            ax.set_xlabel('Amplitude (px)', fontsize=9)
            ax.set_ylabel('Peak Velocity (px/s)', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No saccades',
                    transform=ax.transAxes, ha='center', va='center', color='grey')

    plt.tight_layout()
    plt.show()
    return fig



def analyze_attention_development(all_results):
    """6-panel attention development chart using unified metrics."""
    print("\n" + "="*60)
    print("ATTENTION DEVELOPMENT ANALYSIS")
    print("="*60)

    metrics_data = [r['metrics'] for r in all_results]
    levels = [m['level'] for m in metrics_data]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle('Attention Development Across Difficulty Levels',
                 fontsize=16, fontweight='bold')

    fix_counts    = [m['fixation_count']       for m in metrics_data]
    avg_fix_dur   = [m['avg_fixation_duration'] for m in metrics_data]
    att_scatter   = [m['attention_scatter']     for m in metrics_data]
    task_rel      = [m['task_relevance_ratio']*100 for m in metrics_data]
    sacc_vel      = [m['avg_saccade_velocity']  for m in metrics_data]
    aoi_trans     = [m['aoi_transitions']       for m in metrics_data]
    gaze_eff      = [m['gaze_efficiency']       for m in metrics_data]
    sess_dur      = [m['session_duration']      for m in metrics_data]

    def dual_axis(ax, y1, y2, c1, c2, l1, l2, title, ylabel1, ylabel2):
        ax2 = ax.twinx()
        color_map = {'y': 'goldenrod', 'c': 'darkcyan', 'b': 'blue',
                     'r': 'red', 'g': 'green', 'm': 'magenta'}
        fc1 = color_map.get(c1, c1)
        fc2 = color_map.get(c2, c2)
        ax.plot(levels, y1,  'o-', color=fc1, linewidth=2, markersize=8, label=l1)
        ax2.plot(levels, y2, 'o-', color=fc2, linewidth=2, markersize=8, label=l2)
        ax.set_xlabel('Level'); ax.set_ylabel(ylabel1, color=fc1)
        ax2.set_ylabel(ylabel2, color=fc2)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor=fc1)
        ax2.tick_params(axis='y', labelcolor=fc2)

    dual_axis(axes[0], fix_counts, avg_fix_dur,   'b', 'r',
              'Fix Count', 'Avg Duration', 'Fixation Patterns',
              'Fixation Count', 'Avg Duration (ms)')
    dual_axis(axes[1], att_scatter, task_rel,      'g', 'm',
              'Scatter', 'Task Relevance %', 'Attention Focus Quality',
              'Attention Scatter', 'Task Relevance (%)')
    dual_axis(axes[2], sacc_vel, aoi_trans,        'c', 'y',
              'Sacc Velocity', 'AOI Transitions', 'Eye Movement Dynamics',
              'Saccade Velocity (px/s)', 'AOI Transitions')

    # Panel 4 — Detailed metrics text
    ax4 = axes[3]
    ax4.axis('off')
    ax4.text(0.5, 0.97, 'METRICS SUMMARY', fontsize=12, fontweight='bold',
             transform=ax4.transAxes, ha='center', va='top')
    level_palette = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue', 'plum']
    for i, m in enumerate(metrics_data):
        yp = 0.82 - i*0.30
        ax4.text(0.05, yp, f"LEVEL {m['level']}", fontsize=11, fontweight='bold',
                 transform=ax4.transAxes, va='top')
        ax4.text(0.05, yp-0.07, f"Gaze pts: {m['total_gaze_points']:.0f}   "
                 f"Fixations: {m['fixation_count']:.0f}", fontsize=9,
                 transform=ax4.transAxes, va='top')
        ax4.text(0.05, yp-0.13, f"Avg fix dur: {m['avg_fixation_duration']:.1f} ms   "
                 f"Scatter: {m['attention_scatter']:.0f}", fontsize=9,
                 transform=ax4.transAxes, va='top')
        ax4.text(0.05, yp-0.19, f"Task Relevance: {m['task_relevance_ratio']*100:.1f}%",
                 fontsize=10, fontweight='bold', transform=ax4.transAxes, va='top',
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=level_palette[i % len(level_palette)], alpha=0.7))

    dual_axis(axes[4], gaze_eff, sess_dur, 'b', 'r',
              'Gaze Efficiency', 'Session Duration',
              'Performance Efficiency', 'Gaze Efficiency', 'Session Duration (s)')

    # Panel 6 — Trend score bar
    ax6 = axes[5]
    trend_scores = []
    for i in range(len(levels)):
        fs = (fix_counts[i]-min(fix_counts))/(max(fix_counts)-min(fix_counts)+1e-9)
        rs = task_rel[i] / 100
        ss = 1 - (att_scatter[i]-min(att_scatter))/(max(att_scatter)-min(att_scatter)+1e-9)
        trend_scores.append((fs+rs+ss)/3*100)
    trend_palette = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue', 'plum']
    bars = ax6.bar(levels, trend_scores,
                   color=[trend_palette[i % len(trend_palette)] for i in range(len(levels))])
    for bar, sc in zip(bars, trend_scores):
        ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{sc:.1f}%', ha='center', va='bottom', fontweight='bold')
    ax6.set_ylim(0, 100); ax6.set_xlabel('Level')
    ax6.set_ylabel('Score (%)'); ax6.set_title('Attention Development Trend')

    plt.tight_layout(); plt.show()

    # Text summary
    print("\nATTENTION DEVELOPMENT SUMMARY:")
    for m in metrics_data:
        print(f"\n  Level {m['level']}:")
        print(f"    Fixations: {m['fixation_count']}  Avg dur: {m['avg_fixation_duration']:.1f} ms")
        print(f"    Scatter: {m['attention_scatter']:.0f}  Task relevance: {m['task_relevance_ratio']*100:.1f}%")
        print(f"    Sacc velocity: {m['avg_saccade_velocity']:.0f} px/s  AOI transitions: {m['aoi_transitions']}")
        print(f"    Efficiency: {m['gaze_efficiency']:.3f}  Duration: {m['session_duration']:.1f}s")

    if len(metrics_data) > 1:
        tr_change  = ((metrics_data[-1]['task_relevance_ratio'] - metrics_data[0]['task_relevance_ratio'])
                      / (metrics_data[0]['task_relevance_ratio'] + 1e-9)) * 100
        sc_change  = ((metrics_data[-1]['attention_scatter']    - metrics_data[0]['attention_scatter'])
                      / (metrics_data[0]['attention_scatter']    + 1e-9)) * 100
        print(f"\n  Task relevance change: {tr_change:+.1f}%")
        print(f"  Attention scatter change: {sc_change:+.1f}%")
        if   tr_change >  10: print("  POSITIVE: Task focus improved significantly")
        elif tr_change < -10: print("  CONCERN:  Task focus declined")
        else:                 print("  STABLE:   Task focus maintained")

    return metrics_data


# PERSONALISED FEEDBACK

def generate_personalized_feedback(all_results, student_id="Student"):
    """4-panel personalised feedback report."""
    print("\n" + "="*60)
    print("PERSONALISED FEEDBACK GENERATION")
    print("="*60)

    metrics_data = [r['metrics'] for r in all_results]
    levels       = [m['level'] for m in metrics_data]
    task_rel     = [m['task_relevance_ratio']*100 for m in metrics_data]
    fix_counts   = [m['fixation_count']  for m in metrics_data]
    att_scatter  = [m['attention_scatter'] for m in metrics_data]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f'Personalised Feedback — {student_id}', fontsize=16, fontweight='bold')

    # Panel 1 — Task focus bars
    ax1 = axes[0]
    perf_cat = ['Excellent' if t>=70 else 'Good' if t>=50
                else 'Needs Improvement' if t>=30 else 'Requires Support' for t in task_rel]
    bar_colors = ['green' if p=='Excellent' else 'lightgreen' if p=='Good'
                  else 'orange' if p=='Needs Improvement' else 'red' for p in perf_cat]
    bars = ax1.bar(levels, task_rel, color=bar_colors, alpha=0.7)
    for bar, cat in zip(bars, perf_cat):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, cat,
                 ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax1.set_ylim(0, 100); ax1.set_xlabel('Level'); ax1.set_ylabel('Task Focus (%)')
    ax1.set_title('Your Focus Performance')

    # Panel 2 — Quality progress
    ax2 = axes[1]
    mn, mx = min(fix_counts), max(fix_counts)
    norm_fix = [(f-mn)/(mx-mn+1e-9)*100 for f in fix_counts]
    mn, mx = min(att_scatter), max(att_scatter)
    norm_sc  = [100-(s-mn)/(mx-mn+1e-9)*100 for s in att_scatter]
    levels_numeric = list(range(len(levels)))
    ax2.plot(levels, norm_fix, 'bo-', linewidth=3, markersize=8, label='Focus Stability')
    ax2.plot(levels, norm_sc,  'ro-', linewidth=3, markersize=8, label='Attention Control')
    ax2.fill_between(levels_numeric, norm_fix, alpha=0.3, color='blue')
    ax2.fill_between(levels_numeric, norm_sc,  alpha=0.3, color='red')
    ax2.set_xticks(levels_numeric)
    ax2.set_xticklabels(levels)
    ax2.set_ylim(0, 100); ax2.set_xlabel('Level'); ax2.set_ylabel('Score (0–100)')
    ax2.set_title('Attention Quality Progress'); ax2.legend(); ax2.grid(True, alpha=0.3)

    # Panel 3 — Teacher dashboard
    ax3 = axes[2]; ax3.axis('off')
    ax3.text(0.5, 0.97, 'TEACHER ANALYSIS DASHBOARD', fontsize=12, fontweight='bold',
             transform=ax3.transAxes, ha='center', va='top')
    for i, m in enumerate(metrics_data):
        yp = 0.85 - i*0.30
        concerns = []
        if m['task_relevance_ratio'] < 0.4:  concerns.append("Low task focus")
        if m['attention_scatter']    > 300:  concerns.append("High distraction")
        if m['gaze_efficiency']      < 0.1:  concerns.append("Poor efficiency")
        if m['aoi_transitions']      > 50:   concerns.append("Hyperactive scanning")
        fc = 'lightgreen' if m['task_relevance_ratio']>0.6 else \
             'lightyellow' if m['task_relevance_ratio']>0.4 else 'lightcoral'
        ax3.text(0.05, yp,      f"LEVEL {m['level']}", fontsize=11, fontweight='bold',
                 transform=ax3.transAxes, va='top')
        ax3.text(0.30, yp,      f"Focus: {m['task_relevance_ratio']*100:.1f}%",
                 fontsize=10, fontweight='bold', transform=ax3.transAxes, va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=fc, alpha=0.8))
        ax3.text(0.05, yp-0.08, f"Scatter: {m['attention_scatter']:.0f}   "
                 f"Efficiency: {m['gaze_efficiency']:.3f}   "
                 f"Duration: {m['session_duration']:.0f}s",
                 fontsize=9, transform=ax3.transAxes, va='top')
        ax3.text(0.05, yp-0.15, f"Concerns: {', '.join(concerns) if concerns else 'None'}",
                 fontsize=9, style='italic', transform=ax3.transAxes, va='top',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.6))

    # Panel 4 — Recommendations
    ax4 = axes[3]; ax4.axis('off')
    avg_tr   = np.mean(task_rel)
    avg_sc   = np.mean(att_scatter)
    avg_trans= np.mean([m['aoi_transitions'] for m in metrics_data])
    recs, strengths, challenges = [], [], []

    if avg_tr < 40:
        recs += ["• Focus training exercises needed", "• Reduce visual distractions"]
    elif avg_tr < 60:
        recs += ["• Practice sustained attention tasks", "• Use attention cuing strategies"]
    else:
        recs += ["• Excellent focus — maintain current strategies"]
    if avg_sc   > 300: recs.append("• Work on impulse control activities")
    if avg_trans > 40: recs += ["• Teach strategic scanning patterns",
                                "• Practice deliberate eye movement control"]
    if len(metrics_data) > 1:
        trend = metrics_data[-1]['task_relevance_ratio'] - metrics_data[0]['task_relevance_ratio']
        if trend >  0.1: recs.append("• Positive improvement trend detected")
        elif trend < -0.1: recs.append("• Declining performance — needs intervention")

    if avg_tr > 60:         strengths.append("Good task focus")
    if avg_sc < 250:        strengths.append("Controlled attention")
    if np.mean([m['gaze_efficiency'] for m in metrics_data]) > 0.15:
        strengths.append("Efficient scanning")
    if avg_tr < 50:         challenges.append("Difficulty maintaining focus")
    if avg_sc > 300:        challenges.append("High distractibility")
    if avg_trans > 45:      challenges.append("Excessive eye movements")

    ax4.text(0.05, 0.97, "RECOMMENDATIONS:", fontsize=11, fontweight='bold',
             transform=ax4.transAxes, va='top')
    ax4.text(0.05, 0.87, "\n".join(recs), fontsize=10, transform=ax4.transAxes, va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax4.text(0.05, 0.55, "STRENGTHS:\n" + "\n".join([f"• {s}" for s in strengths] or ["• Work in progress"]),
             fontsize=10, transform=ax4.transAxes, va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
    ax4.text(0.05, 0.28, "CHALLENGES:\n" + "\n".join([f"• {c}" for c in challenges] or ["• None identified"]),
             fontsize=10, transform=ax4.transAxes, va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout(); plt.show()

    print(f"\nFEEDBACK FOR {student_id}:")
    print(f"  Avg task focus: {avg_tr:.1f}%   Inhibition: "
          f"{'Good' if avg_trans<40 else 'Needs Work'}")
    for m in metrics_data:
        print(f"  Level {m['level']}: {m['task_relevance_ratio']*100:.1f}% focus, "
              f"{m['fixation_count']} fixations")
    for r in recs: print(f"  {r}")

    return {'avg_task_relevance': avg_tr, 'avg_scatter': avg_sc,
            'recommendations': recs, 'strengths': strengths, 'challenges': challenges}


# EARLY WARNING / RISK SYSTEM

def identify_struggling_students(all_results, student_id="Student"):
    """4-panel risk assessment with early warning indicators."""
    print("\n" + "="*60)
    print("STRUGGLING STUDENT IDENTIFICATION SYSTEM")
    print("="*60)

    metrics_data = [r['metrics'] for r in all_results]
    risk_thresholds = {
        'tr_critical': 0.30, 'tr_warning': 0.50,
        'sc_critical': 400,  'sc_warning': 300,
        'tr_critical_transitions': 60, 'tr_warning_transitions': 45,
        'eff_critical': 0.08,'eff_warning': 0.12,
    }

    risk_scores = []
    for m in metrics_data:
        score, factors = 0, []
        if   m['task_relevance_ratio'] < risk_thresholds['tr_critical']:
            score += 3; factors.append("Critical task focus deficit")
        elif m['task_relevance_ratio'] < risk_thresholds['tr_warning']:
            score += 2; factors.append("Low task focus")
        if   m['attention_scatter'] > risk_thresholds['sc_critical']:
            score += 3; factors.append("Poor attention control")
        elif m['attention_scatter'] > risk_thresholds['sc_warning']:
            score += 2; factors.append("Moderate attention scatter")
        if   m['aoi_transitions'] > risk_thresholds['tr_critical_transitions']:
            score += 2; factors.append("Hyperactive scanning")
        elif m['aoi_transitions'] > risk_thresholds['tr_warning_transitions']:
            score += 1; factors.append("Elevated transitions")
        if   m['gaze_efficiency'] < risk_thresholds['eff_critical']:
            score += 2; factors.append("Very low efficiency")
        elif m['gaze_efficiency'] < risk_thresholds['eff_warning']:
            score += 1; factors.append("Low efficiency")
        risk_scores.append({'level': m['level'], 'risk_score': score,
                            'risk_factors': factors, 'metrics': m})

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f'Early Warning System — Risk Assessment for {student_id}',
                 fontsize=16, fontweight='bold')

    levels = [r['level'] for r in risk_scores]
    scores = [r['risk_score'] for r in risk_scores]

    # Panel 1 — Risk bar
    ax1 = axes[0]
    colors = ['green' if s<=2 else 'orange' if s<=5 else 'red' for s in scores]
    bars = ax1.bar(levels, scores, color=colors, alpha=0.7)
    ax1.axhline(2, color='green',  linestyle='--', alpha=0.7, label='Low Risk')
    ax1.axhline(5, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk')
    ax1.axhline(8, color='red',    linestyle='--', alpha=0.7, label='High Risk')
    ax1.set_ylim(0, max(10, max(scores)+1))
    ax1.set_xlabel('Level'); ax1.set_ylabel('Risk Score')
    ax1.set_title('Academic Risk Progression'); ax1.legend()
    for bar, sc in zip(bars, scores):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                 f'{sc}', ha='center', va='bottom', fontweight='bold')

    # Panel 2 — Multi-dimensional profile
    ax2 = axes[1]; ax2.axis('off')
    ax2.text(0.5, 0.97, 'MULTI-DIMENSIONAL RISK PROFILE', fontsize=12,
             fontweight='bold', transform=ax2.transAxes, ha='center', va='top')
    cats  = ['Task Focus', 'Attn Control', 'Efficiency', 'Scanning']
    clvls = ['lightcoral', 'lightyellow', 'lightgreen', 'lightblue', 'plum']
    for i, r in enumerate(risk_scores):
        m  = r['metrics']
        yp = 0.85 - i*0.28
        ns = [min(100, m['task_relevance_ratio']*100),
              min(100, max(0, 100-m['attention_scatter']/5)),
              min(100, m['gaze_efficiency']*1000),
              min(100, max(0, 100-m['aoi_transitions']/2))]
        ax2.text(0.05, yp, f"LEVEL {r['level']}", fontsize=11, fontweight='bold',
                 transform=ax2.transAxes, va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=clvls[i % len(clvls)], alpha=0.8))
        for j, (cat, sc) in enumerate(zip(cats, ns)):
            fc = 'lightgreen' if sc>70 else 'lightyellow' if sc>40 else 'lightcoral'
            ax2.text(0.05, yp-0.05-(j*0.045), f"{cat}: {sc:.0f}/100",
                     fontsize=9, transform=ax2.transAxes, va='top',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=fc, alpha=0.7))

    # Panel 3 — Warning indicators
    ax3 = axes[2]; ax3.axis('off')
    ax3.set_title('Early Warning Indicators', fontsize=12, fontweight='bold')
    for i, r in enumerate(risk_scores):
        yp = 0.90 - i*0.30
        rl = 'LOW' if r['risk_score']<=2 else 'MODERATE' if r['risk_score']<=5 else 'HIGH'
        bg = '#ccffcc' if rl=='LOW' else '#ffffcc' if rl=='MODERATE' else '#ffcccc'
        tc = 'darkgreen' if rl=='LOW' else 'darkorange' if rl=='MODERATE' else 'darkred'
        ax3.text(0.05, yp, f"LEVEL {r['level']}", fontsize=11, fontweight='bold',
                 transform=ax3.transAxes, va='top')
        ax3.text(0.40, yp, f"{rl} RISK  {r['risk_score']}/10", fontsize=10,
                 fontweight='bold', color=tc, transform=ax3.transAxes, va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.8))
        factors = r['risk_factors'] if r['risk_factors'] else ['No significant concerns']
        for j, f in enumerate(factors[:3]):
            ax3.text(0.08, yp-0.08-(j*0.05), f"• {f}", fontsize=9,
                     style='italic', transform=ax3.transAxes, va='top')

    # Panel 4 — Interventions
    ax4 = axes[3]; ax4.axis('off')
    ax4.set_title('Intervention Recommendations', fontsize=12, fontweight='bold')
    all_factors = [f for r in risk_scores for f in r['risk_factors']]
    avg_risk    = np.mean([r['risk_score'] for r in risk_scores])
    interv = []
    if   avg_risk > 6: interv += ["IMMEDIATE INTERVENTION:", "• Reduce cognitive load",
                                  "• Attention training program",
                                  "• Consider executive function assessment"]
    elif avg_risk > 3: interv += ["PREVENTIVE MEASURES:", "• Monitor closely",
                                  "• Focus enhancement activities",
                                  "• Adjust task difficulty gradually"]
    else:              interv += ["MAINTENANCE:", "• Continue current support",
                                  "• Monitor for changes",
                                  "• Gradually increase challenge"]
    if "Critical task focus deficit" in all_factors:
        interv.append("• Sustained attention training")
    if "Hyperactive scanning" in all_factors:
        interv.append("• Systematic visual search strategies")
    if "Very low efficiency" in all_factors:
        interv.append("• Processing speed activities")
    ax4.text(0.05, 0.95, "\n".join(interv), fontsize=10,
             transform=ax4.transAxes, va='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

    plt.tight_layout(); plt.show()

    print(f"\nEARLY WARNING REPORT — {student_id}:")
    avg_risk = np.mean(scores)
    cat = "LOW" if avg_risk<=2 else "MODERATE" if avg_risk<=5 else "HIGH"
    print(f"  Average risk score: {avg_risk:.1f}/10  —  {cat} RISK")
    for r in risk_scores:
        print(f"\n  Level {r['level']} (score {r['risk_score']}/10):")
        for f in (r['risk_factors'] or ['No significant risk factors']):
            print(f"    - {f}")

    return {'overall_risk': avg_risk, 'risk_category': cat,
            'risk_scores': risk_scores, 'interventions': interv}



def create_cross_level_comparison(all_results):
    """6-panel cross-level comparison using unified I-VT metrics."""
    if len(all_results) < 2:
        return None

    fig, axes = plt.subplots(1, 6, figsize=(24, 5))
    fig.suptitle('Cross-Level Eye Tracking Comparison', fontsize=16, fontweight='bold')

    levels      = [f"Level {r['level']}" for r in all_results]
    fix_counts  = [len(r['fixations'])   for r in all_results]
    sacc_counts = [len(r['saccades'])    for r in all_results]
    avg_fd  = [r['fixations']['duration'].mean()       if len(r['fixations'])>0 else 0 for r in all_results]
    avg_sa  = [r['saccades']['amplitude'].mean()       if len(r['saccades'])>0  else 0 for r in all_results]
    avg_sv  = [r['saccades']['peak_velocity'].mean()   if len(r['saccades'])>0  else 0 for r in all_results]
    ratios  = [(len(r['fixations'])/len(r['saccades'])
                if len(r['fixations'])>0 and len(r['saccades'])>0 else 0)
               for r in all_results]

    x, w = np.arange(len(levels)), 0.35
    axes[0].bar(x-w/2, fix_counts,  w, label='Fixations', color='blue',   alpha=0.7)
    axes[0].bar(x+w/2, sacc_counts, w, label='Saccades',  color='red',    alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(levels)
    axes[0].set_title('Counts by Level'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for ax, y, marker, col, title, ylabel in [
        (axes[1], avg_fd, 'o-', 'green',  'Avg Fixation Duration',  'ms'),
        (axes[2], avg_sa, 's-', 'orange', 'Avg Saccade Amplitude',  'px'),
        (axes[3], avg_sv, '^-', 'purple', 'Avg Saccade Velocity',   'px/s'),
    ]:
        ax.plot(levels, y, marker, color=col, linewidth=2, markersize=8)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)

    ratio_colors = ['red' if r<1.0 else 'green' for r in ratios]
    axes[4].bar(levels, ratios, color=ratio_colors, alpha=0.7)
    axes[4].axhline(1.0, color='black', linestyle='--', alpha=0.5)
    axes[4].set_title('F/S Ratio by Level'); axes[4].set_ylabel('Ratio')
    axes[4].grid(True, alpha=0.3)

    eff = []
    for i in range(len(levels)):
        if avg_fd[i]>0 and avg_sa[i]>0:
            eff.append(((avg_fd[i]/max(avg_fd)) + (1-avg_sa[i]/max(avg_sa)))/2)
        else:
            eff.append(0)
    bar_colors = plt.cm.RdYlGn(eff)
    bars = axes[5].bar(levels, eff, color=bar_colors, alpha=0.8)
    for bar, sc in zip(bars, eff):
        axes[5].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f'{sc:.2f}', ha='center', va='bottom', fontweight='bold')
    axes[5].set_title('Visual Processing Efficiency'); axes[5].set_ylabel('Score (0–1)')
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout(); plt.show()
    return fig


# COMPREHENSIVE REPORT

def generate_comprehensive_report(all_results):
    """Print structured final report across all levels."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EYE TRACKING ANALYSIS REPORT")
    print(f"{'='*80}")

    total_fix  = sum(len(r['fixations']) for r in all_results)
    total_sacc = sum(len(r['saccades'])  for r in all_results)
    print(f"\nEXECUTIVE SUMMARY:")
    print(f"  Levels analysed : {len(all_results)}")
    print(f"  Total fixations : {total_fix}")
    print(f"  Total saccades  : {total_sacc}")
    if total_sacc > 0:
        print(f"  Overall F/S ratio: {total_fix/total_sacc:.2f}")

    print(f"\nDETAILED LEVEL ANALYSIS:")
    for r in all_results:
        fix, sacc, m = r['fixations'], r['saccades'], r['metrics']
        print(f"\n  LEVEL {r['level']}:")
        if len(fix) > 0:
            print(f"    Fixations      : {len(fix)}   "
                  f"Avg dur: {fix['duration'].mean():.1f} ms   "
                  f"Max: {fix['duration'].max():.1f} ms")
            print(f"    Attention ctr  : ({fix['center_x'].mean():.0f}, {fix['center_y'].mean():.0f})")
        if len(sacc) > 0:
            print(f"    Saccades       : {len(sacc)}   "
                  f"Avg amp: {sacc['amplitude'].mean():.1f} px   "
                  f"Avg vel: {sacc['peak_velocity'].mean():.1f} px/s")
        print(f"    Task relevance : {m['task_relevance_ratio']*100:.1f}%   "
              f"AOI transitions: {m['aoi_transitions']}   "
              f"Efficiency: {m['gaze_efficiency']:.3f}")
        for ins in r['insights']:
            print(f"    → {ins}")

    if len(all_results) > 1:
        print(f"\nCROSS-LEVEL COGNITIVE INTERPRETATION:")
        fd = [r['fixations']['duration'].mean() if len(r['fixations'])>0 else 0 for r in all_results]
        sa = [r['saccades']['amplitude'].mean() if len(r['saccades'])>0  else 0 for r in all_results]
        if all(d>0 for d in fd):
            if   fd[-1] > fd[0]*1.2: print("  ↑ INCREASING COGNITIVE LOAD across levels")
            elif fd[-1] < fd[0]*0.8: print("  ↓ LEARNING EFFECT — improving efficiency")
            else:                    print("  ↔ STABLE PROCESSING across levels")
        if all(a>0 for a in sa):
            if   sa[-1] > sa[0]*1.3: print("  ↑ BROADER SEARCH at higher levels")
            elif sa[-1] < sa[0]*0.7: print("  ↓ FOCUSED STRATEGY developing across levels")

    print(f"\n{'='*80}")


#   LEVEL PROCESSING  &  MAIN

def process_level(path, level_num):
    """
    Full pipeline for a single level:
      load → clean → AOI setup → I-VT detection → metrics → visualisations
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING LEVEL {level_num}")
    print(f"{'='*60}")

    df = load_and_clean_data(path)
    if df is None:
        return None

    df = fix_aoi_coordinates(df)
    level_name = f"Level {level_num}"

    # Dynamic I-VT detection
    fixations_df, saccades_df, threshold = detect_fixations_saccades(df)

    # Gaze pattern insights
    insights = analyze_gaze_patterns(fixations_df, saccades_df, level_name)

    # AOI + attention metrics — fed by I-VT results
    metrics = calculate_advanced_metrics(df, level_num, fixations_df, saccades_df)

    # Visualisations
    ivt_fig   = create_ivt_visualization(df, fixations_df, saccades_df, level_name)
    stats_fig = create_statistical_summary(fixations_df, saccades_df, level_name)

    return {
        'level':      level_num,
        'data':       df,
        'fixations':  fixations_df,
        'saccades':   saccades_df,
        'threshold':  threshold,
        'insights':   insights,
        'metrics':    metrics,
        'ivt_figure': ivt_fig,
        'stats_figure': stats_fig,
    }


def main(student_id="Student"):
    print("UNIFIED EYE TRACKING ANALYSIS PIPELINE")
    print("Dynamic I-VT  +  AOI Attention Metrics  +  Risk Assessment")
    print("="*60)

    try:
        file_paths = [level1_path, level2_path, level3_path]
    except NameError:
        print("✗ ERROR: level1_path, level2_path, level3_path are not defined.")
        print("  Please uncomment and set the path variables at the top of this file.")
        return None
    results = []

    for i, path in enumerate(file_paths, 1):
        result = process_level(path, i)
        if result:
            results.append(result)

    if not results:
        print("No valid data to analyse.")
        return None

    # ── Combined cross-level visualisations (all levels in one figure)
    print("\nGenerating combined visualisations across all levels...")
    create_combined_ivt_visualization(results)
    create_combined_statistical_summary(results)

    # ── Cross-level analyses
    attention_metrics = analyze_attention_development(results)
    feedback_data     = generate_personalized_feedback(results, student_id)
    warning_data      = identify_struggling_students(results, student_id)

    if len(results) > 1:
        create_cross_level_comparison(results)

    generate_comprehensive_report(results)

    # ── Final summary
    print(f"\n{'='*60}\nANALYSIS COMPLETE\n{'='*60}")
    print(f"  Levels processed   : {len(results)}")
    print(f"  Avg task focus     : {feedback_data['avg_task_relevance']:.1f}%")
    print(f"  Risk category      : {warning_data['risk_category']}")
    print(f"  Recommendations    : {len(feedback_data['recommendations'])}")

    return {
        'results':           results,
        'attention_metrics': attention_metrics,
        'feedback_data':     feedback_data,
        'warning_data':      warning_data,
    }


if __name__ == "__main__":
    comprehensive_results = main(student_id="Student_6")

