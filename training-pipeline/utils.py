import os
from datetime import datetime


def get_filename_with_highest_timestep(directory) -> tuple[str, int]:
    highest_timestep = 0
    highest_timestep_file = ""
    latest_dt = None

    # Iterate over files in the given directory
    files = sorted(os.listdir(directory), key=lambda f: os.path.getctime(os.path.join(directory, f)))

    for filename in files:
        # Only process files ending with '_steps.zip'
        if filename.endswith('_steps.zip'):
            # Split the filename based on underscores
            parts = filename.split('_')
            # Get the timestep value (the part before '_steps.zip')
            timestep = int(parts[-2])
            date_with_v, time = parts[:2]
            
            dt = datetime.strptime("_".join([date_with_v[1:], time]), "%Y%m%d_%H%M")
            # Update the highest timestep and corresponding filename
            if (latest_dt is None or dt > latest_dt) or (dt == latest_dt and timestep > highest_timestep):
                highest_timestep = timestep
                highest_timestep_file = filename

    return highest_timestep_file, highest_timestep