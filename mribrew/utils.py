import os
import re
import sys
from datetime import datetime

class colours:
    # Terminal color codes
    CEND = '\33[0m'
    CBLUE = '\33[94m'
    CGREEN = '\33[92m'
    CRED = '\33[91m'
    CYELLOW = '\33[93m'
    UBOLD = '\33[1m'

class Tee:
    def __init__(self, res_dir):
        self.log_file = open(os.path.join(res_dir, 'logfile.txt'), "a")
        self.original_stdout = sys.stdout
        sys.stdout = self

        # Write the current date and time to the log_file
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_file.write(f"\n---- Logging started at {current_time} ----\n")

    def write(self, obj):
        # Write to console
        self.original_stdout.write(obj)
        
        # Write to log_file with ANSI codes removed
        clean_obj = re.sub(r'\x1b\[\d+m', '', obj)
        self.log_file.write(clean_obj)
        
        # Ensure we are flushing the outputs to the file
        self.log_file.flush()

    def flush(self):
        self.log_file.flush()

def should_use_subset_data():
    ### TO REMOVE!
    """Ask the user whether to use the subset data or not."""
    while True:
        user_input = input("Would you like to use the subset data for all "
                           "processing (for testing purposes)? [Y/n]: ").strip().lower()
        
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'n'.")