"""
Main Runner Script
Executes the data extraction pipeline for any EPANET network file

Usage:
    python run_pipeline.py [inp_file] [output_file] [flow_threshold]
    
Examples:
    python run_pipeline.py data/net3.inp
    python run_pipeline.py data/exnet.inp data/exnet_data.dat
    python run_pipeline.py data/exnet.inp data/exnet_data.dat 0.05
"""

import os
import sys
from data_extraction import WaterNetworkDataExtractor


def run_full_pipeline(inp_file: str = None,
                      output_file: str = None,
                      flow_threshold: float = 0.0):
    """
    Run the data extraction pipeline
    
    Args:
        inp_file: Path to EPANET .inp file (relative to simulation directory)
        output_file: Path to output .dat file (default: data/network_data.dat)
        flow_threshold: Minimum flow to consider (GPM)
    """
    
    print("\n" + "="*80)
    print(" WATER NETWORK SENSOR PLACEMENT - DATA EXTRACTION PIPELINE")
    print("="*80 + "\n")
    
    # -------------------------------------------------------------------------
    # STEP 1: Data Extraction
    # -------------------------------------------------------------------------
    print("STEP 1: DATA EXTRACTION")
    print("-"*80)
    
    try:
        # Initialize extractor
        extractor = WaterNetworkDataExtractor(inp_file)
        
        # Load network
        extractor.load_network()
        
        # Extract network structure
        network_structure = extractor.extract_network_structure()
        
        # Run hydraulic simulation
        sim_results = extractor.run_hydraulic_simulation(save_timeseries=True)
        
        # Build flow patterns
        flow_pattern_data = extractor.build_flow_patterns(sim_results, flow_threshold=flow_threshold)
        
        # Build attack scenarios
        attack_data = extractor.build_attack_scenarios()
        
        # Build optimization data structures
        optimization_data = extractor.build_optimization_data_structures(
            flow_pattern_data,
            attack_data
        )
        
        # Save all data
        extractor.save_data_structures(optimization_data, output_file=output_file)
        
        # Close network
        extractor.close()
        
        print("\n✓ Data extraction completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during data extraction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    # Determine output file path for display
    output_display = output_file if output_file else "data/network_data.dat"
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print("\nGenerated file:")
    print(f"  - {output_display} (AMPL format)")
    
    print("\n" + "="*80)
    print("\nNext steps:")
    print(f"  1. Review the data in {output_display}")
    print("  2. Run the optimization solver with the generated data")
    print("\n" + "="*80 + "\n")
    
    return True


def main(inp_file: str = None,
         output_file: str = None,
         flow_threshold: float = 0.0):
    """
    Main entry point
    
    Args:
        inp_file: Path to EPANET .inp file (relative to simulation directory)
        output_file: Path to output .dat file (default: data/network_data.dat)
        flow_threshold: Minimum flow to consider in GPM (default: 0.0 = any positive flow)
    """
    
    # Check if input file exists
    if not os.path.exists(inp_file):
        print(f"Error: Input file '{inp_file}' not found!")
        return False
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Input file: {inp_file}")
    print(f"  Output file: {output_file if output_file else 'auto (based on input)'}")
    print(f"  Flow threshold: {flow_threshold} GPM")
    
    # Run pipeline
    success = run_full_pipeline(
        inp_file=inp_file,
        output_file=output_file,
        flow_threshold=flow_threshold
    )
    
    if success:
        print("Pipeline executed successfully!")
        return True
    else:
        print("Pipeline failed!")
        return False


if __name__ == "__main__":
    main("./data/net3.inp", output_file="./data/net3_data.dat")
