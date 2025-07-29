# OpenSTL on RunPod

This repository is configured to run on RunPod Serverless for spatiotemporal predictive learning tasks.

## Quick Start

1. Deploy this repository on RunPod
2. Send requests to the serverless endpoint

## API Usage

### Video Prediction

Send a POST request with the following JSON structure:

```json
{
  "input": {
    "task_type": "video_prediction",
    "model_name": "simvp_gsta",
    "prediction_length": 10,
    "frames": [
      "base64_encoded_image_1",
      "base64_encoded_image_2",
      "base64_encoded_image_3"
    ]
  }
}
```

### Parameters

- `task_type`: Type of task ("video_prediction")
- `model_name`: Model to use (default: "simvp_gsta")
- `config_path`: Optional custom config path
- `prediction_length`: Number of frames to predict (default: 10)
- `frames`: Array of base64-encoded input frames

### Response

```json
{
  "predicted_frames": [
    "base64_encoded_prediction_1",
    "base64_encoded_prediction_2"
  ],
  "num_predictions": 10,
  "status": "success"
}
```

## Supported Models

- SimVP with gSTA (default)
- ConvLSTM
- PredRNN variants
- And other OpenSTL models via config_path

## Docker Build

The included Dockerfile will:
1. Install PyTorch with CUDA support
2. Install all OpenSTL dependencies
3. Set up the workspace
4. Configure the handler script

## Local Testing

```bash
python handler.py
```

This will start a local test of the handler and show health status. 