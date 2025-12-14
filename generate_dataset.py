"""
Generate Synthetic Fish Species Conservation Status Dataset
Aligned with SDG 14: Life Below Water
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Define categorical options
habitat_types = ['Coral Reef', 'Deep Sea', 'Coastal', 'Open Ocean', 'Estuary']
population_trends = ['Increasing', 'Stable', 'Declining', 'Critical']
fishing_pressure_levels = ['Low', 'Moderate', 'High', 'Very High']
geographic_regions = ['Pacific', 'Atlantic', 'Indian Ocean', 'Arctic', 'Mediterranean']

# Generate features
data = {
    'species_id': range(1, n_samples + 1),
    'habitat_type': np.random.choice(habitat_types, n_samples),
    'population_trend': np.random.choice(population_trends, n_samples),
    'fishing_pressure': np.random.choice(fishing_pressure_levels, n_samples),
    'average_size_cm': np.random.uniform(10, 200, n_samples).round(2),
    'geographic_region': np.random.choice(geographic_regions, n_samples),
    'reproduction_rate': np.random.uniform(0.5, 10.0, n_samples).round(2),
    'depth_range_m': np.random.uniform(5, 2000, n_samples).round(2),
    'water_temperature_c': np.random.uniform(5, 30, n_samples).round(2),
    'population_size_thousands': np.random.uniform(1, 500, n_samples).round(2)
}

df = pd.DataFrame(data)

# Generate conservation status based on logical rules
def assign_conservation_status(row):
    score = 0
    
    # Population trend impact
    if row['population_trend'] == 'Increasing':
        score += 3
    elif row['population_trend'] == 'Stable':
        score += 2
    elif row['population_trend'] == 'Declining':
        score += 1
    else:  # Critical
        score += 0
    
    # Fishing pressure impact (inverse)
    if row['fishing_pressure'] == 'Low':
        score += 3
    elif row['fishing_pressure'] == 'Moderate':
        score += 2
    elif row['fishing_pressure'] == 'High':
        score += 1
    else:  # Very High
        score += 0
    
    # Reproduction rate impact
    if row['reproduction_rate'] > 7:
        score += 2
    elif row['reproduction_rate'] > 4:
        score += 1
    
    # Population size impact
    if row['population_size_thousands'] > 300:
        score += 2
    elif row['population_size_thousands'] > 150:
        score += 1
    
    # Add some randomness
    score += np.random.choice([0, 1, -1])
    
    # Assign status based on total score
    if score >= 7:
        return 'Good'
    elif score >= 4:
        return 'Moderate'
    else:
        return 'Poor'

df['conservation_status'] = df.apply(assign_conservation_status, axis=1)

# Save to CSV
df.to_csv('fish_conservation_data.csv', index=False)

print("✓ Dataset generated successfully!")
print(f"Total samples: {len(df)}")
print(f"\nConservation Status Distribution:")
print(df['conservation_status'].value_counts())
print(f"\nDataset saved as: fish_conservation_data.csv")

