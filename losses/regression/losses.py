import torch
import torch.nn as nn

class WeightedSmoothL1Loss(nn.Module):
    def __init__(
        self,
        weights: list[int] = [2, 2, 4, 4, 1],
    ):  
        super().__init__()
        assert len(weights) == 5, "Weights vector should have length 5"

        self.register_buffer("weights", torch.Tensor(weights))
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred *= self.weights
        labels *= self.weights
        return self.loss_func(pred, labels)
    

class SineSmoothL1Loss(nn.Module):
    def __init__(
        self,
    ):  
        super().__init__()
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred[:, -1] = torch.sin(pred[:, -1])
        labels[:, -1] = torch.sin(labels[:, -1])
        return self.loss_func(pred, labels)



class GaussianLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch_size = len(pred)
        c_pred, m_pred = self.transfrom_to_gaussians(pred)
        c_label, m_label = self.transfrom_to_gaussians(label)

        diff_center = (c_pred - c_label).square().sum()
        diff_matrix = (m_pred - m_label).square().sum()
        loss = (diff_center + diff_matrix).sqrt()

        return loss / batch_size
    
    def transfrom_to_gaussians(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = (params[:, -1] - 0.5) * torch.pi
        axes = params[:, 2:4]
        centers = params[:, :2]
        rotation = self.create_rotation_matrices(angles)
        cov = self.create_covariance_matrices(axes)
        final_matrix = self.square_root_matrix(rotation, cov)

        return centers, final_matrix

    def create_rotation_matrices(self, angles: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        matrices = torch.vstack((cos, sin, -sin, cos)).transpose(0, 1).reshape(-1, 2, 2)
        return matrices

    def create_covariance_matrices(self, variance: torch.Tensor) -> torch.Tensor:
        major_ax = variance[:, 0]
        minor_ax = variance[:, 1]

        covariance = torch.vstack((
            major_ax, 
            torch.zeros_like(major_ax),  
            torch.zeros_like(minor_ax),
            minor_ax,
        )).transpose(0, 1).reshape(-1, 2, 2)
        return covariance

    def square_root_matrix(self, rotation: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
        mat1 = torch.bmm(rotation.mT, covariance)
        mat2 = torch.bmm(mat1, rotation)
        return mat2
    
