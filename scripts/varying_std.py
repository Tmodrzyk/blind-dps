import yaml
import numpy as np
import subprocess


for sigma in np.linspace(0.0, 6.0, 20):
    
    # Load the YAML config file
    with open('./configs/gaussian_deblur_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Modify the value of "kernel_std"
    config['kernel_std'] = float(sigma) 

    # Save the modified config back to the file
    with open('./configs/gaussian_deblur_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    save_dir = f"/home/modrzyk/code/blind-dps/results/{sigma}"

    command = "python3 blind_deblur_demo.py " \
            "--img_model_config=configs/custom_model_config_128.yaml " \
            "--kernel_model_config=configs/kernel_model_config.yaml " \
            "--diffusion_config=configs/diffusion_config.yaml " \
            "--task_config=configs/gaussian_deblur_config.yaml " \
            "--reg_ord=1 " \
            "--reg_scale=1.0 " \
            f"--save_dir={save_dir}" 

    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command execution failed with return code {e.returncode}")

    