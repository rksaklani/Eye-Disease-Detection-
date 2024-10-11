import subprocess



def convert_h5_to_json(h5_model_path, output_dir):
    try:
        # Prepare the command for tensorflowjs conversion
        command = [
            'tensorflowjs_converter',
            '--input_format', 'keras',  # Specify input format as Keras (.h5)
            h5_model_path,              # Path to the .h5 file
            output_dir                  # Directory where model.json and weights will be saved
        ]
        
        # Execute the conversion command
        subprocess.run(command, check=True)
        print(f"Model converted successfully to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    # Specify the .h5 model file and output directory
    h5_model_path = './pediatric_eye_disease_model.h5'  # Path to your .h5 model
    output_dir = './static/model_tfjs/'  # Output directory for JSON and weight files

    # Convert the model
    convert_h5_to_json(h5_model_path, output_dir)
