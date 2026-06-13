import os
import cv2
import glob
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
from yt_vid_downloader import download_video

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        '''
        Initializer for PyTorch's premade LSTM, defining the 
        encoder and correlating decoder, as well as an output_layer.
        :param: input_dim, the amount of dimensions in the LSTM:
                - X, Y coordinates
        :param: hidden_dim, number of hidden dimensions.
        :returns: None
        '''
        super(LSTMAutoEncoder, self).__init__()
        
        # Squeezes all telemetry data into 64 hidden dimensions
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # decoder for reconstructing a fluid timeline out of the encoded information
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # used to translate the decoder's output back into two understandable and
        # comprehensible coordinates
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        '''
        Defines how data is passed through during a computation/run.
        :param: x, Represents the 3D PyTorch tensor
        :return: Returns the reconstruction of the state after the forward pass
        '''

        # pulls the amount of frames (60)
        sequence_length = x.size(1)

        # isolate the hidden vector to represent the summary of the clip
        _, (hidden, _) = self.encoder(x)

        # Used to help the decoder reconstruct the data, rearranging the dimensions of the
        # matrix, and repeating it 60 times for the timeline reconstruction
        decoder_input = hidden.permute(1, 0, 2).repeat(1, sequence_length, 1)

        # gets the results of the decoder after being fed the cleaned input
        decoder_output, _ = self.decoder(decoder_input)
        
        # translation layer, repairs the data back into the correct format
        reconstructed_sequence = self.output_layer(decoder_output)
        
        return reconstructed_sequence

class AimAnalysis:
    def __init__(self, version_number=1, dataset_config='data.yaml'):
    # Setting Parameters
        self.version_number = version_number
        self.dataset_config = dataset_config
        self.device = None
        self.rf = None
        self.project = None
        self.version = None

        self.environment_load()

    def print_parameters(self):
        print(f"DEVICE: {self.device}")
        print(f"ROBOFLOW: {self.rf}")
        print(f"PROJECT: {self.project}")
        print(f"VERSION: {self.version}")

    def environment_load(self):
        global API_KEY, PROJECT_ID, DEVICE, RF, PROJECT, VERSION
        print("Loading Environment:\n")

        print("Setting Model usage as GPU or CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if torch.cuda.is_available():
            self.device = 0
            print(f"GPU being used: {torch.cuda.get_device_name(0)}")
        else: 
            self.device = "cpu" 
            print("GPU not found, using CPU.")

        print("Loading from environment file.")
        load_dotenv()
        # gets the 
        api_key = os.getenv("ROBOFLOW_API_KEY")
        project_id = os.getenv("ROBOFLOW_PROJECT_ID")

        if not api_key or not project_id:
            print("Error: No Roboflow credentials configured.")
            exit()

        print("Connecting to Roboflow's cloud.")
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project(project_id)
        self.version = self.project.version(self.version_number)
        print("Finished Loading Environment.\n")

    def aim_analyzer(self, video_path: str):
        print("Running analysis on {video_path}.")

    def telemetry_extraction(self, video_path: str):
        # downloads the yolo formatted data from Roboflow
        dataset = self.version.download("yolov8")
        yolo = YOLO("yolov8n.pt")

        # trains the model using the specified parameters
        results = yolo.train(
        data=f"{dataset.location}/{self.dataset_config}",
        device=self.device,
        epochs=10,
        imgsz=640, 
        box=12.0,            
        cls=2.5,
        label_smoothing=0.05,
        batch=4,
        amp=True,
        exist_ok=True
        )

        freshest_run_dir = Path(yolo.trainer.save_dir) # type: ignore
    
        best_path = freshest_run_dir / "weights" / "best.pt"
        last_path = freshest_run_dir / "weights" / "last.pt"
        
        # Select whichever weight file actually exists in that folder
        best_model_path = best_path if best_path.exists() else last_path
        
        # holds the path for the model
        trained_yolo = YOLO(str(best_model_path))

        # -------------------------------
        # Starts extraction of the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Video file could not be opened.")
            return

        # used to hold the coordinates
        raw_positions = []
        print("Grabbing coordinates.")

        # starts a loop through the video frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # making a prediction on the data and extracts the coordinates of the predictions
            predictions = trained_yolo(frame, conf=0.25, verbose=False, stream=True)

            for pred in predictions:
                boxes = pred.boxes
                if len(boxes) == 0:
                    continue
                xywh = boxes[0].xywh[0].tolist()
                center_x = xywh[0]
                center_y = xywh[1]
                
                raw_positions.append([center_x, center_y])

        cap.release()
        print("Extracted Positions.")

        # with these coordinates, cleans the data for the LSTM
        # Uses a sliding window over the video to better carry data's 
        # important information.
        pos_array = np.array(raw_positions)
        deltas = np.diff(pos_array, axis=0)

        WINDOW_SIZE = 60
        sliding_windows = []

        for idx in range(len(deltas) - WINDOW_SIZE + 1):
            window_chunk = deltas[idx : idx + WINDOW_SIZE]
            sliding_windows.append(window_chunk)

        final_lstm_dataset = np.array(sliding_windows)

        # simply makes the path ways for the data to go to,
        # saving the important information
        output_dir = Path("stage2_analytics/telemetry_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filepath = output_dir / "expert_baseline.npy"
        np.save(output_filepath, final_lstm_dataset)

        print(f"Extracted telemetry data to {str(output_filepath)}")
        return str(output_filepath)

if __name__ == '__main__':
    url = 'https://www.youtube.com/watch?v=d-RGJPvFuNE'
    analyzer = AimAnalysis(version_number=1)

    print("Starting analysis:")
    analyzer.print_parameters()
    print("--------------------------\n")

    print("Loading Environment Variables:")
    analyzer.environment_load()
    print("--------------------------\n")

    print("Downloading mp4 format from link.")
    video_path = download_video(url)
    print("--------------------------\n")

    print("Getting telemetry of video.")
    telemetry_path = analyzer.telemetry_extraction(video_path)
    print("--------------------------\n")

    print("Performing analysis on telemetry data.")

    print("--------------------------\n")
