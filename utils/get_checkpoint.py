import os
import glob




def get_checkpoint(output_dir):
    """"Get the second to last checkpoint from the output directory.
    This is useful for resuming training from the last checkpoint, while avoiding the last one which might be corrupted or incomplete.
    Args:
        output_dir (str): The directory where checkpoints are saved.
    Returns:
        str: The path to the second to last checkpoint, or the last one if there are less than 2 checkpoints.
    """
    
    try:
        # Get all checkpoint folders
        checkpoints = glob.glob(f"{output_dir}/checkpoint-*")
        if len(checkpoints) < 2:
            # return the last checkpoint if there are less than 2 checkpoints
            return sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[0]
        # Sort by step number
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
        
        # Return second to last
        return checkpoints[-2]
    except Exception as e:
        print(f"There is no checkpoints in directory {output_dir} or error: {e}")
        return None






