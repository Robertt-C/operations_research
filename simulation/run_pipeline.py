"""
Main Runner Script
Executes the data extraction pipeline
"""

import os
import sys
from data_extraction import WaterNetworkDataExtractor


def run_full_pipeline(inp_file: str = 'data/net3.inp', 
                      flow_threshold: float = 0.01):
    """
    Run the data extraction pipeline
    
    Args:
        inp_file: Path to EPANET .inp file
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
        extractor.save_data_structures(optimization_data)
        
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
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print("\nGenerated file:")
    print("  - data/network_data.dat (AMPL format)")
    
    print("\n" + "="*80)
    print("\nNext steps:")
    print("  1. Review the data in data/network_data.dat")
    print("  2. Run the AMPL model using solve.run")
    print("\n" + "="*80 + "\n")
    
    return True


def main():
    """Main entry point"""
    
    # Configuration
    INP_FILE = 'data/net3.inp'
    FLOW_THRESHOLD = 0.01  # GPM
    
    # Check if input file exists
    if not os.path.exists(INP_FILE):
        print(f"Error: Input file '{INP_FILE}' not found!")
        print("Please ensure net3.inp is in the data/ directory.")
        sys.exit(1)
    
    # Run pipeline
    success = run_full_pipeline(
        inp_file=INP_FILE,
        flow_threshold=FLOW_THRESHOLD
    )
    
    if success:
        print("Pipeline executed successfully!")
        sys.exit(0)
    else:
        print("Pipeline failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
