import runpod
import torch
import numpy as np
import os
import json
import base64
from io import BytesIO
import tempfile
import zipfile
from PIL import Image
import cv2

# Import OpenSTL components
from openstl.api import BaseExperiment
from openstl.utils import create_parser
import openstl.methods as methods
import openstl.models as models


def load_model(model_name, config_path=None):
    """Load OpenSTL model based on configuration"""
    try:
        # Default to SimVP with gSTA if no specific config provided
        if config_path is None:
            config_path = "configs/mmnist/simvp/SimVP_gSTA.py"
        
        # Create experiment runner
        args = create_parser().parse_args([
            '--config_file', config_path,
            '--ex_name', 'runpod_inference'
        ])
        
        exp = BaseExperiment(args)
        if exp.method is not None:
            exp.method.eval()
        
        return exp
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def process_video_prediction(input_data, model):
    """Process video prediction using OpenSTL"""
    try:
        # Parse input parameters
        frames = input_data.get('frames', [])
        prediction_length = input_data.get('prediction_length', 10)
        
        # Convert base64 encoded frames to tensors
        input_frames = []
        for frame_b64 in frames:
            # Decode base64 image
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data))
            img_array = np.array(img)
            
            # Convert to tensor and normalize
            if len(img_array.shape) == 3:
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            else:
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
            
            input_frames.append(img_tensor)
        
        # Stack frames into batch
        input_tensor = torch.stack(input_frames).unsqueeze(0)  # [1, T, C, H, W]
        
        # Run prediction
        with torch.no_grad():
            predictions = model.method.test_one_epoch(input_tensor, prediction_length)
        
        # Convert predictions back to images
        output_frames = []
        for i in range(predictions.shape[1]):
            frame = predictions[0, i].cpu().numpy()
            
            # Denormalize and convert to uint8
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            if frame.shape[0] == 1:  # Grayscale
                img = Image.fromarray(frame[0], mode='L')
            else:  # RGB
                img = Image.fromarray(frame.transpose(1, 2, 0), mode='RGB')
            
            # Convert to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            output_frames.append(img_b64)
        
        return {
            'predicted_frames': output_frames,
            'num_predictions': len(output_frames),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error'
        }


def handler(job):
    """Main RunPod handler function"""
    try:
        job_input = job['input']
        task_type = job_input.get('task_type', 'video_prediction')
        model_name = job_input.get('model_name', 'simvp_gsta')
        config_path = job_input.get('config_path', None)
        
        # Load model
        model = load_model(model_name, config_path)
        if model is None:
            return {
                'error': 'Failed to load model',
                'status': 'error'
            }
        
        # Process based on task type
        if task_type == 'video_prediction':
            result = process_video_prediction(job_input, model)
        else:
            result = {
                'error': f'Unsupported task type: {task_type}',
                'status': 'error'
            }
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error'
        }


# Health check function
def health_check():
    """Simple health check to verify the handler is working"""
    try:
        import openstl
        return {
            'status': 'healthy',
            'openstl_version': openstl.__version__ if hasattr(openstl, '__version__') else 'unknown',
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test the handler locally
    print("OpenSTL RunPod Handler Starting...")
    print(json.dumps(health_check(), indent=2))
    
    # Start the RunPod serverless handler
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    }) 