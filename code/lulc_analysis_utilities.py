"""
============================================================
LULC CHANGE ANALYSIS - UTILITY FUNCTIONS
============================================================
Additional helper functions for advanced analysis, validation,
and comparison across multiple datasets.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class LULCAnalysisUtilities:
    """Utility class for additional LULC analysis tasks."""
    
    @staticmethod
    def validate_raster_data(raster_path: Path, 
                            expected_crs: str = None) -> Dict:
        """
        Validate raster file integrity and metadata.
        
        Returns:
            dict with validation results and metadata
        """
        import rasterio
        
        validation = {
            'valid': True,
            'warnings': [],
            'metadata': {}
        }
        
        try:
            with rasterio.open(raster_path) as src:
                validation['metadata'] = {
                    'shape': src.shape,
                    'crs': src.crs,
                    'transform': src.transform,
                    'dtype': src.dtypes[0],
                    'count': src.count,
                    'bounds': src.bounds
                }
                
                # Check for NaN/NoData issues
                data = src.read(1)
                unique_vals = np.unique(data)
                
                validation['metadata']['unique_values'] = len(unique_vals)
                validation['metadata']['value_range'] = (int(unique_vals.min()), int(unique_vals.max()))
                
                # Warnings
                if src.nodata is None:
                    validation['warnings'].append("NoData value not set")
                
                if src.crs is None:
                    validation['warnings'].append("CRS not defined")
                    validation['valid'] = False
                    
                if expected_crs and str(src.crs) != expected_crs:
                    validation['warnings'].append(f"CRS mismatch: expected {expected_crs}")
                
        except Exception as e:
            validation['valid'] = False
            validation['warnings'].append(f"Error reading file: {str(e)}")
        
        return validation
    
    @staticmethod
    def validate_polygon_data(vector_path: Path) -> Dict:
        """
        Validate polygon/shapefile integrity.
        
        Returns:
            dict with validation results
        """
        import geopandas as gpd
        
        validation = {
            'valid': True,
            'warnings': [],
            'metadata': {}
        }
        
        try:
            gdf = gpd.read_file(vector_path)
            
            validation['metadata'] = {
                'count': len(gdf),
                'crs': gdf.crs,
                'bounds': gdf.total_bounds,
                'geometry_types': gdf.geometry.type.unique().tolist()
            }
            
            # Check for issues
            if gdf.crs is None:
                validation['warnings'].append("CRS not defined")
            
            if gdf.geometry.has_z.any():
                validation['warnings'].append("Polygons have Z dimension (3D)")
            
            if not gdf.geometry.is_valid.all():
                invalid_count = (~gdf.geometry.is_valid).sum()
                validation['warnings'].append(f"{invalid_count} invalid geometries")
            
            if gdf.geometry.is_empty.any():
                validation['warnings'].append("Contains empty geometries")
            
        except Exception as e:
            validation['valid'] = False
            validation['warnings'].append(f"Error reading file: {str(e)}")
        
        return validation
    
    @staticmethod
    def compare_datasets(results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare LULC statistics across multiple datasets.
        
        Args:
            results_dict: dict of analysis results keyed by dataset name
            
        Returns:
            DataFrame comparing key metrics across datasets
        """
        comparison_data = []
        
        for dataset_name, result in results_dict.items():
            metrics = result['metrics']
            
            comparison_data.append({
                'Dataset': dataset_name,
                'Total_Area_km2': metrics['Total_Mining_Area_km2'],
                'Stable_Area_km2': metrics['Stable_Area_km2'],
                'Changed_Area_km2': metrics['Changed_Area_km2'],
                'Stability_Pct': metrics['Stability_Percentage'],
                'Change_Pct': metrics['Change_Percentage']
            })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def identify_hotspots(transition_df: pd.DataFrame,
                         threshold_km2: float = 0.1) -> pd.DataFrame:
        """
        Identify significant LULC transition hotspots.
        
        Args:
            transition_df: transition matrix DataFrame
            threshold_km2: minimum area to consider significant
            
        Returns:
            DataFrame of significant transitions
        """
        hotspots = transition_df[
            (transition_df['Change_Type'] == 'Changed') & 
            (transition_df['Area_km2'] >= threshold_km2)
        ].sort_values('Area_km2', ascending=False)
        
        return hotspots
    
    @staticmethod
    def calculate_class_stability_index(transition_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stability index for each LULC class (0-1 scale).
        
        Returns:
            DataFrame with class stability metrics
        """
        class_stats = []
        
        for class_name in transition_df['From_Name'].unique():
            class_data = transition_df[transition_df['From_Name'] == class_name]
            
            # Total area of class in 2000
            total_area = class_data['Area_km2'].sum()
            
            if total_area == 0:
                continue
            
            # Stable area (where class remained same)
            stable_area = class_data[
                (class_data['From_Name'] == class_data['To_Name'])
            ]['Area_km2'].sum()
            
            # Stability index (0 = completely changed, 1 = completely stable)
            stability_index = stable_area / total_area if total_area > 0 else 0
            
            class_stats.append({
                'Class': class_name,
                'Area_2000_km2': total_area,
                'Stable_Area_km2': stable_area,
                'Changed_Area_km2': total_area - stable_area,
                'Stability_Index': stability_index,
                'Stability_Percentage': stability_index * 100
            })
        
        return pd.DataFrame(class_stats).sort_values('Stability_Index', ascending=False)
    
    @staticmethod
    def calculate_class_conversion_rates(transition_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate conversion rates FROM each class TO other classes.
        
        Returns:
            dict mapping source class to conversion details
        """
        conversion_rates = {}
        
        for class_from in transition_df['From_Name'].unique():
            class_data = transition_df[transition_df['From_Name'] == class_from]
            total_area = class_data['Area_km2'].sum()
            
            conversions = []
            
            for class_to in class_data['To_Name'].unique():
                area = class_data[class_data['To_Name'] == class_to]['Area_km2'].sum()
                rate = (area / total_area * 100) if total_area > 0 else 0
                
                conversions.append({
                    'To_Class': class_to,
                    'Area_km2': area,
                    'Conversion_Rate_Pct': rate
                })
            
            conversion_rates[class_from] = {
                'Total_Area_km2': total_area,
                'Conversions': sorted(conversions, key=lambda x: x['Conversion_Rate_Pct'], 
                                    reverse=True)
            }
        
        return conversion_rates


class LULCVisualizations:
    """Advanced visualization functions."""
    
    @staticmethod
    def plot_multi_dataset_comparison(comparison_df: pd.DataFrame,
                                     output_fp: Path):
        """
        Create multi-dataset comparison plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total area comparison
        axes[0, 0].bar(comparison_df['Dataset'], comparison_df['Total_Area_km2'], 
                       color='steelblue')
        axes[0, 0].set_ylabel('Total Area (km²)', fontweight='bold')
        axes[0, 0].set_title('Total Mining Area by Dataset', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Stability comparison
        x = np.arange(len(comparison_df))
        width = 0.35
        axes[0, 1].bar(x - width/2, comparison_df['Stable_Area_km2'], width, 
                       label='Stable', color='#2ecc71')
        axes[0, 1].bar(x + width/2, comparison_df['Changed_Area_km2'], width, 
                       label='Changed', color='#e74c3c')
        axes[0, 1].set_ylabel('Area (km²)', fontweight='bold')
        axes[0, 1].set_title('Stable vs Changed Areas', fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(comparison_df['Dataset'], rotation=45)
        axes[0, 1].legend()
        
        # Stability percentage
        axes[1, 0].barh(comparison_df['Dataset'], comparison_df['Stability_Pct'], 
                        color='#2ecc71')
        axes[1, 0].set_xlabel('Stability (%)', fontweight='bold')
        axes[1, 0].set_title('LULC Stability Percentage', fontweight='bold')
        axes[1, 0].set_xlim(0, 100)
        
        # Change percentage
        axes[1, 1].barh(comparison_df['Dataset'], comparison_df['Change_Pct'], 
                        color='#e74c3c')
        axes[1, 1].set_xlabel('Change (%)', fontweight='bold')
        axes[1, 1].set_title('LULC Change Percentage', fontweight='bold')
        axes[1, 1].set_xlim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_fp, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_fp}")
        plt.close()
    
    @staticmethod
    def plot_class_stability_index(stability_df: pd.DataFrame,
                                   output_fp: Path):
        """
        Create bar chart of class stability indices.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color gradient based on stability
        colors = plt.cm.RdYlGn(stability_df['Stability_Index'].values)
        
        ax.barh(stability_df['Class'], stability_df['Stability_Index'], color=colors)
        ax.set_xlabel('Stability Index (0=Changed, 1=Stable)', fontweight='bold')
        ax.set_title('LULC Class Stability Index (2000-2020)', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for i, (idx, row) in enumerate(stability_df.iterrows()):
            ax.text(row['Stability_Index'] + 0.02, i, f"{row['Stability_Index']:.2f}", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_fp, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_fp}")
        plt.close()
    
    @staticmethod
    def plot_conversion_sankey(transition_df: pd.DataFrame,
                              top_n: int = 10,
                              output_fp: Path = None):
        """
        Create a visual representation of top LULC conversions.
        (Requires plotly for interactive version)
        """
        top_transitions = transition_df[
            transition_df['Change_Type'] == 'Changed'
        ].nlargest(top_n, 'Area_km2')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create labels
        labels = [f"{row['From_Name']} → {row['To_Name']}" 
                 for _, row in top_transitions.iterrows()]
        areas = top_transitions['Area_km2'].values
        
        # Plot
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, areas, color=plt.cm.Spectral(np.linspace(0, 1, len(labels))))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Area (km²)', fontweight='bold')
        ax.set_title(f'Top {top_n} LULC Conversions (2000-2020)', fontweight='bold')
        
        # Add value labels
        for i, area in enumerate(areas):
            ax.text(area + 0.01 * max(areas), i, f'{area:.2f}', 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        if output_fp:
            plt.savefig(output_fp, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_fp}")
        plt.show()
        plt.close()


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_advanced_analysis():
    """
    Example of how to use advanced analysis utilities.
    """
    
    print("\n" + "="*70)
    print("ADVANCED LULC ANALYSIS UTILITIES - EXAMPLE")
    print("="*70)
    
    # Example 1: Validate data files
    print("\n[1] Validating Data Files")
    print("-" * 70)
    
    utils = LULCAnalysisUtilities()
    
    # Example paths (update with real paths)
    # raster_result = utils.validate_raster_data(Path("path/to/2000.tif"))
    # vector_result = utils.validate_polygon_data(Path("path/to/mining_polygons.gpkg"))
    
    print("✓ Data validation complete (see output above)")
    
    # Example 2: Compare multiple datasets
    print("\n[2] Comparing Multiple Datasets")
    print("-" * 70)
    
    # This would use results from analyze_lulc_change()
    # comparison_df = utils.compare_datasets(results)
    # print(comparison_df)
    
    # Example 3: Identify hotspots
    print("\n[3] Identifying Change Hotspots")
    print("-" * 70)
    
    # This would use transition matrix from analysis
    # hotspots = utils.identify_hotspots(transition_df, threshold_km2=0.5)
    # print(hotspots)
    
    # Example 4: Calculate class stability indices
    print("\n[4] Class Stability Analysis")
    print("-" * 70)
    
    # stability = utils.calculate_class_stability_index(transition_df)
    # print(stability)
    
    # Example 5: Conversion rate analysis
    print("\n[5] Class Conversion Rates")
    print("-" * 70)
    
    # conversions = utils.calculate_class_conversion_rates(transition_df)
    # for source_class, details in conversions.items():
    #     print(f"\n{source_class}: {details['Total_Area_km2']:.2f} km²")
    #     for conv in details['Conversions']:
    #         print(f"  → {conv['To_Class']}: {conv['Conversion_Rate_Pct']:.1f}%")
    
    print("\n" + "="*70)
    print("See comments above for actual usage with your data")
    print("="*70)


if __name__ == "__main__":
    example_advanced_analysis()
