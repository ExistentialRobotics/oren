from dataclasses import dataclass

import torch
import torch.nn as nn

from grad_sdf.utils.config_abc import ConfigABC


@dataclass
class CriterionConfig(ConfigABC):
    boundary_loss_weight: float = 1.0
    boundary_loss_prior_weight: float = 1.0
    boundary_loss_type: str = "L1"
    perturbation_loss_weight: float = 1.0
    perturbation_loss_prior_weight: float = 1.0
    perturbation_loss_type: str = "L1"
    eikonal_loss_surface_weight: float = 1.0
    eikonal_loss_surface_prior_weight: float = 1.0
    eikonal_loss_perturbation_weight: float = 1.0
    eikonal_loss_perturbation_prior_weight: float = 1.0
    eikonal_loss_space_weight: float = 1.0
    eikonal_loss_space_prior_weight: float = 1.0
    eikonal_loss_type: str = "L1"
    projection_loss_weight: float = 1.0
    projection_loss_prior_weight: float = 1.0
    projection_loss_type: str = "L1"
    heat_loss_weight: float = 0.0
    heat_loss_lambda: float = 0.0
    sign_loss_free_weight: float = 0.0
    sign_loss_occ_weight: float = 0.0
    sign_loss_temperature: float = 100.0
    perturbation_loss_exp_penalty: float = 1.0


class Criterion(nn.Module):
    def __init__(self, cfg: CriterionConfig, n_stratified: int, n_perturbed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_stratified = n_stratified
        self.n_perturbed = n_perturbed

        if self.cfg.boundary_loss_type == "L1":
            self.boundary_loss_fn = nn.L1Loss(reduction="mean")
        elif self.cfg.boundary_loss_type == "L2":
            self.boundary_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown boundary loss type: {self.cfg.boundary_loss_type}")

        if self.cfg.perturbation_loss_type == "L1":
            self.perturbation_loss_fn = nn.L1Loss(reduction="mean")
        elif self.cfg.perturbation_loss_type == "L2":
            self.perturbation_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown perturbation loss type: {self.cfg.perturbation_loss_type}")

        if self.cfg.eikonal_loss_type == "L1":
            self.eikonal_loss_fn = nn.L1Loss(reduction="mean")
        elif self.cfg.eikonal_loss_type == "L2":
            self.eikonal_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown eikonal loss type: {self.cfg.eikonal_loss_type}")

        if self.cfg.projection_loss_type == "L1":
            self.projection_loss_fn = nn.L1Loss(reduction="mean")
        elif self.cfg.projection_loss_type == "L2":
            self.projection_loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown projection loss type: {self.cfg.projection_loss_type}")

    def forward(
        self,
        pred_sdf: torch.Tensor,
        pred_prior: torch.Tensor,
        pred_grad: torch.Tensor,
        pred_prior_grad: torch.Tensor,
        gt_sdf_perturb: torch.Tensor,
        gt_sdf_stratified: torch.Tensor,
        positive_perturbation_mask: torch.Tensor,
        perturb_sigma: float,
    ):
        loss = 0
        loss_dict = {}
        if self.cfg.boundary_loss_weight > 0:
            boundary_loss = self.get_boundary_loss(pred_sdf)
            loss += self.cfg.boundary_loss_weight * boundary_loss
            loss_dict["boundary_loss"] = boundary_loss.item()
        if self.cfg.boundary_loss_prior_weight > 0:
            boundary_loss_prior = self.get_boundary_loss(pred_prior)
            loss += self.cfg.boundary_loss_prior_weight * boundary_loss_prior
            loss_dict["boundary_loss_prior"] = boundary_loss_prior.item()

        if self.cfg.perturbation_loss_weight > 0:
            perturbation_loss = self.get_perturbation_loss(
                pred_sdf[:, self.n_stratified : self.n_stratified + self.n_perturbed],
                positive_perturbation_mask,
                gt_sdf_perturb,
                perturb_sigma,
            )
            loss += self.cfg.perturbation_loss_weight * perturbation_loss
            loss_dict["perturbation_loss"] = perturbation_loss.item()
        if self.cfg.perturbation_loss_prior_weight > 0:
            perturbation_loss_prior = self.get_perturbation_loss(
                pred_prior[:, self.n_stratified : self.n_stratified + self.n_perturbed],
                positive_perturbation_mask,
                gt_sdf_perturb,
                perturb_sigma,
            )
            loss += self.cfg.perturbation_loss_prior_weight * perturbation_loss_prior
            loss_dict["perturbation_loss_prior"] = perturbation_loss_prior.item()

        grad_norm = torch.norm(pred_grad, dim=-1)
        grad_norm_prior = torch.norm(pred_prior_grad, dim=-1)

        if self.cfg.eikonal_loss_surface_weight > 0:
            eikonal_loss_surface = self.get_eikonal_loss_surface(grad_norm)
            loss += self.cfg.eikonal_loss_surface_weight * eikonal_loss_surface
            loss_dict["eikonal_loss_surface"] = eikonal_loss_surface.item()
        if self.cfg.eikonal_loss_surface_prior_weight > 0:
            eikonal_loss_surface_prior = self.get_eikonal_loss_surface(grad_norm_prior)
            loss += self.cfg.eikonal_loss_surface_prior_weight * eikonal_loss_surface_prior
            loss_dict["eikonal_loss_surface_prior"] = eikonal_loss_surface_prior.item()
        if self.cfg.eikonal_loss_perturbation_weight > 0:
            eikonal_loss_perturbation = self.get_eikonal_loss_perturbation(grad_norm)
            loss += self.cfg.eikonal_loss_perturbation_weight * eikonal_loss_perturbation
            loss_dict["eikonal_loss_perturbation"] = eikonal_loss_perturbation.item()
        if self.cfg.eikonal_loss_perturbation_prior_weight > 0:
            eikonal_loss_perturbation_prior = self.get_eikonal_loss_perturbation(grad_norm_prior)
            loss += self.cfg.eikonal_loss_perturbation_prior_weight * eikonal_loss_perturbation_prior
            loss_dict["eikonal_loss_perturbation_prior"] = eikonal_loss_perturbation_prior.item()
        if self.cfg.eikonal_loss_space_weight > 0:
            eikonal_loss_space = self.get_eikonal_loss_space(grad_norm)
            loss += self.cfg.eikonal_loss_space_weight * eikonal_loss_space
            loss_dict["eikonal_loss_space"] = eikonal_loss_space.item()
        if self.cfg.eikonal_loss_space_prior_weight > 0:
            eikonal_loss_space_prior = self.get_eikonal_loss_space(grad_norm_prior)
            loss += self.cfg.eikonal_loss_space_prior_weight * eikonal_loss_space_prior
            loss_dict["eikonal_loss_space_prior"] = eikonal_loss_space_prior.item()

        if self.cfg.projection_loss_weight > 0:
            projection_loss = self.get_projection_loss(
                pred_sdf[:, : self.n_stratified],
                gt_sdf_stratified,
            )
            loss += self.cfg.projection_loss_weight * projection_loss
            loss_dict["projection_loss"] = projection_loss.item()

        if self.cfg.projection_loss_prior_weight > 0:
            projection_loss_prior = self.get_projection_loss(
                pred_prior[:, : self.n_stratified],
                gt_sdf_stratified,
            )
            loss += self.cfg.projection_loss_prior_weight * projection_loss_prior
            loss_dict["projection_loss_prior"] = projection_loss_prior.item()

        if self.cfg.heat_loss_weight > 0:
            if grad_norm is None:
                grad_norm = torch.norm(pred_grad, dim=-1)
            heat_loss = self.get_heat_loss(pred_sdf, grad_norm)
            loss += self.cfg.heat_loss_weight * heat_loss
            loss_dict["heat_loss"] = heat_loss.item()

        loss_dict["total_loss"] = loss.item()
        return loss, loss_dict

    def get_boundary_loss(self, pred_sdf: torch.Tensor):
        pred_sdf_surface = pred_sdf[:, -1]
        boundary_loss = self.boundary_loss_fn(pred_sdf_surface, torch.zeros_like(pred_sdf_surface))
        return boundary_loss

    def get_perturbation_loss(
        self,
        pred_sdf_perturb: torch.Tensor,
        positive_perturbation_mask: torch.Tensor,
        gt_sdf_perturb: torch.Tensor,
        perturb_sigma: float,
    ):
        # Clone to avoid modifying input tensors
        pred_sdf_perturb = pred_sdf_perturb.clone()
        gt_sdf_perturb = gt_sdf_perturb.clone()

        # Flip sign for positive perturbations (negative SDF values) to unify loss calculation
        pred_sdf_perturb[positive_perturbation_mask] = -pred_sdf_perturb[positive_perturbation_mask]
        gt_sdf_perturb[positive_perturbation_mask] = -gt_sdf_perturb[positive_perturbation_mask]

        perturb_loss_upperbound = gt_sdf_perturb
        perturb_loss_lowerbound = perturb_sigma * 1.0 * torch.ones_like(gt_sdf_perturb)

        above_upper_loss = torch.clamp(pred_sdf_perturb - perturb_loss_upperbound, min=0)
        below_lower = torch.clamp(perturb_loss_lowerbound - pred_sdf_perturb, min=0)

        # Use exponential penalty for below lower bound to emphasize constraint
        below_lower_loss = torch.exp(self.cfg.perturbation_loss_exp_penalty * below_lower) - 1
        perturb_loss = above_upper_loss + below_lower_loss

        return perturb_loss.mean()

    def get_eikonal_loss_surface(self, grad_norm: torch.Tensor):
        grad_norm_surface = grad_norm[:, -1]  # surface
        eikonal_loss_surface = self.eikonal_loss_fn(grad_norm_surface, torch.ones_like(grad_norm_surface))
        return eikonal_loss_surface

    def get_eikonal_loss_perturbation(self, grad_norm: torch.Tensor):
        grad_norm_perturbation = grad_norm[:, self.n_stratified : self.n_stratified + self.n_perturbed]
        eikonal_loss_perturbation = self.eikonal_loss_fn(
            grad_norm_perturbation, torch.ones_like(grad_norm_perturbation)
        )
        return eikonal_loss_perturbation

    def get_eikonal_loss_space(self, grad_norm: torch.Tensor):
        grad_norm_space = grad_norm[:, : self.n_stratified]  # free space
        eikonal_loss_space = self.eikonal_loss_fn(grad_norm_space, torch.ones_like(grad_norm_space))
        return eikonal_loss_space

    def get_projection_loss(self, pred_sdf: torch.Tensor, gt_sdf_stratified: torch.Tensor):
        return self.projection_loss_fn(pred_sdf, gt_sdf_stratified)

    def get_sign_loss(
        self,
        positive_sdf_mask: torch.Tensor,
        negative_sdf_mask: torch.Tensor,
        pred_sdf: torch.Tensor,
    ):
        free_pred = pred_sdf[positive_sdf_mask.squeeze()]
        occ_pred = pred_sdf[negative_sdf_mask.squeeze()]
        sign_loss_free = (torch.tanh(self.cfg.sign_loss_temperature * free_pred) - 1).abs().mean()
        sign_loss_occ = (torch.tanh(self.cfg.sign_loss_temperature * occ_pred) + 1).abs().mean()
        return sign_loss_free, sign_loss_occ

    def get_heat_loss(self, pred_sdf: torch.Tensor, grad_norm: torch.Tensor):
        pred_sdf = pred_sdf[:, : self.n_stratified]  # only consider free space samples
        grad_norm = grad_norm[:, : self.n_stratified]
        heat = torch.exp(-self.cfg.heat_loss_lambda * pred_sdf.abs()).unsqueeze(1)
        heat_loss = (0.5 * heat**2 * (grad_norm**2 + 1)).mean()
        return heat_loss
