"""
Neural network models for facial expression to motor position transformation
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class RealTimeDataset(Dataset):
    """PyTorch Dataset for real-time facial expression and motor position data"""
    def __init__(self):
        self.human_blendshapes = []  # Human facial blendshapes (input)
        self.motor_positions = []    # Optimized motor positions (output)

    def add_data(self, human_blendshapes, motor_positions):
        self.human_blendshapes.append(human_blendshapes)
        self.motor_positions.append(motor_positions)

    def __len__(self):
        return len(self.human_blendshapes)

    def __getitem__(self, idx):
        return torch.tensor(self.human_blendshapes[idx], dtype=torch.float32), torch.tensor(self.motor_positions[idx], dtype=torch.float32)

class LandmarkToMotorModel(nn.Module):
    """Neural network model to predict motor positions from facial landmarks"""
    def __init__(self, input_dim, output_dim):
        super(LandmarkToMotorModel, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Check if input shape matches expected shape
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[1]}")
        return self.fc(x)
    
    @classmethod
    def load_from_file(cls, model_path, device=None):
        """
        Load a model from a file.
        
        Args:
            model_path (str): Path to the saved model
            device (torch.device): Device to load the model to
            
        Returns:
            LandmarkToMotorModel: Loaded model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(model_path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
    
    def save_to_file(self, file_path):
        """
        Save the model to a file.
        
        Args:
            file_path (str): Path to save the model to
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.fc[-1].out_features
        }, file_path)
        print(f"Model saved to {file_path}")