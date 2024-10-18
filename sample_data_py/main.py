import os
import importlib.util

def run_script(script_path):
    # Get the script name without extension
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    print(f"Executed: {script_path}")

def main():
    # Directory containing the scripts
    script_dir = './'
    
    # Ensure the output directory exists
    os.makedirs('../sample_data', exist_ok=True)
    
    # Get all Python files in the directory
    python_files = [f for f in os.listdir(script_dir) if f.endswith('.py')]
    
    # Run each script
    for script in python_files:
        script_path = os.path.join(script_dir, script)
        run_script(script_path)

if __name__ == "__main__":
    main()