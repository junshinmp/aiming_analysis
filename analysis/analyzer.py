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

    def aim_analyzer(self, telemetry_path: str):
        try:
            print("Running analysis on {telemetry_path}.")
            expert_data = np.load(telemetry_path)
        except FileNotFoundError:
            print(f"Couldn't find any telemetry data at {telemetry_path}")
            return None

        inputs = torch.tensor(expert_data, dtype=torch.float32).to(self.device)
        model = LSTMAutoEncoder().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        epochs = 15
        batch_size = 16

        for epoch in range(epochs):
            shuffled_indices = torch.randperm(inputs.size(0))
            running_loss = 0.0
            
            for i in range(0, inputs.size(0), batch_size):
                batch_idx = shuffled_indices[i : i + batch_size]
                batch_data = inputs[batch_idx] 
                
                optimizer.zero_grad()
                
                reconstructions = model(batch_data)
                loss = criterion(reconstructions, batch_data)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_data.size(0)
                
            average_epoch_loss = running_loss / inputs.size(0)
            print(f"   🔹 Epoch [{epoch+1:02d}/{epochs}] | Reconstruction Loss: {average_epoch_loss:.6f}")

        checkpoint_dir = Path("stage2_analytics/models")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "lstm_aim_baseline.pth"
        
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Baseline parameters written to: {checkpoint_path}")
        return str(checkpoint_path)

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

    def evaluate_gameplay(self, test_video_path: str, model_weights_path: str, threshold=0.005):
        print(f"\nCommencing Mechanical Evaluation on: {test_video_path}")
        
        # Generating a temporary or fresh .npy dataset for the test video
        print("Collecting telemetry data.")
        test_telemetry_path = self.telemetry_extraction(test_video_path)
        if not test_telemetry_path:
            print("Telemetry could not be extracted from test video.")
            return
        
        test_data = np.load(test_telemetry_path)
        
        # Convert our test matrix into an active evaluation Tensor 'x'
        inputs = torch.tensor(test_data, dtype=torch.float32).to(self.device)
        
        # 3. Instantiate the model and load your optimized expert weights
        print("Loading baseline to compare.")
        model = LSTMAutoEncoder(input_dim=2, hidden_dim=64).to(self.device)
        
        try:
            model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            print("Baseline successfully synchronized.")
        except FileNotFoundError:
            print(f"Error: Weights file not found at {model_weights_path}")
            return

        # Put the model into evaluation mode
        model.eval()
        
        # We process windows individually to calculate a precise timeline score
        window_losses = []
        
        print("Evaluating trajectory mechanics frame-by-frame...")
        # torch.no_grad() disables gradient tracking to save RAM and speed up execution
        with torch.no_grad():
            for i in range(inputs.size(0)):
                # Isolate a single 60-frame window sequence block, unsqueezed to simulate a batch of 1
                single_window = inputs[i].unsqueeze(0) # Shape format becomes: (1, 60, 2)
                
                # Forward Pass: The expert network attempts to reconstruct the test curve
                reconstruction = model(single_window)
                
                # Calculate the exact Mean Squared Error discrepancy for this specific 1-second block
                # Instead of a scalar loss, we average across the sequence features to get a single performance score
                loss = torch.mean((reconstruction - single_window) ** 2)
                window_losses.append(loss.item())

        # keep track of the flaws in aim training
        flaws_detected = 0
        
        # Loop through the results timeline
        for idx, score in enumerate(window_losses):
            if score > threshold:
                # Calculate approximate video timestamp (assuming 60 FPS video capture)
                # The window starts at 'idx' frame offset
                timestamp_seconds = idx / 60.0
                
                print(f"Flaw Detected - Time: {timestamp_seconds:.2f}s | "
                      f"Anomaly Score: {score:.6f} (Crossed limit: {threshold})")
                flaws_detected += 1
        
        if flaws_detected == 0:
            print("Audit Complete: Your mechanics match elite baseline standards flawlessly!")
        else:
            print(f"\nFlagged {flaws_detected} mechanical tracking instabilities.")
            print("Review the timestamps above to isolate muscle jitters or overcorrections.")

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
