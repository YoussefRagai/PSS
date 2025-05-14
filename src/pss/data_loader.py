import os
import msgpack
import pandas as pd

def load_master_seasons():
    """Load the master seasons data."""
    try:
        with open('master_seasons.msgpack', 'rb') as f:
            return msgpack.unpackb(f.read())
    except Exception as e:
        print(f"Error loading master_seasons.msgpack: {e}")
        return None

def load_match_data(match_id):
    """Load data for a specific match."""
    try:
        match_dir = f'analysis/match_{match_id}'
        if not os.path.exists(match_dir):
            return None
            
        data = {}
        # Load details
        with open(f'{match_dir}/match_{match_id}_details.msgpack', 'rb') as f:
            data['details'] = msgpack.unpackb(f.read())
            
        # Load events
        data['events'] = pd.read_parquet(f'{match_dir}/match_{match_id}_events.parquet')
        
        # Load player summary
        data['player_summary'] = pd.read_csv(f'{match_dir}/match_{match_id}_player_summary.csv')
        
        return data
    except Exception as e:
        print(f"Error loading match {match_id}: {e}")
        return None

def get_available_matches():
    """Get list of available match IDs."""
    try:
        return [d.split('_')[1] for d in os.listdir('analysis') if d.startswith('match_')]
    except Exception as e:
        print(f"Error getting available matches: {e}")
        return [] 