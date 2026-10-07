// Adapted from IBAMR 0.19.0 examples/IB/explicit/ex7 (BSD-3-Clause).
// Copyright (c) 2022-2026 The IBAMR Developers. See THIRD_PARTY_NOTICES.md.

#include <petscsys.h>

#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBHydrodynamicForceEvaluator.h>
#include <ibamr/IBMethod.h>
#include <ibamr/IBStandardForceGen.h>
#include <ibamr/IBStandardInitializer.h>
#include <ibamr/IBTargetPointForceSpec.h>
#include <ibamr/INSStaggeredPressureBcCoef.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/INSStaggeredVelocityBcCoef.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/IndexUtilities.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/LData.h>
#include <ibtk/LDataManager.h>
#include <HierarchyCellDataOpsReal.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <CartesianPatchGeometry.h>
#include <CellData.h>
#include <CellIndex.h>
#include <LoadBalancer.h>
#include <HierarchyDataOpsManager.h>
#include <HierarchyDataOpsReal.h>
#include <CellVariable.h>
#include <PatchSideDataOpsReal.h>
#include <SideData.h>
#include <SideIndex.h>
#include <SideVariable.h>
#include <StandardTagAndInitialize.h>
#include <VariableDatabase.h>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_map>

#include <ibamr/app_namespaces.h>

namespace
{
constexpr double kFrequencyHz = 26.1;
constexpr double kBodyAngle = 39.8 * M_PI / 180.0;
constexpr double kStrokePlaneAngle = 15.0 * M_PI / 180.0;
constexpr double kPhiAmplitude = 1.0; // half of the reported 2 rad excursion
constexpr double kAlphaAmplitude = 0.87; // provisional Fig. 1(d) digitization
constexpr double kWingRadius = 0.0483;
double g_kinematics_scale = 1.0;
double g_kinematic_cfl = 0.5;
std::string g_motion_mode = "hover";
Eigen::Vector3d g_translation_velocity = Eigen::Vector3d::Zero();

Eigen::Matrix3d wing_pose(double time, int side)
{
    const double phase = 2.0 * M_PI * kFrequencyHz * time;
    const double phi = g_kinematics_scale * kPhiAmplitude * std::cos(phase);
    const double alpha = g_kinematics_scale * kAlphaAmplitude * std::sin(phase);
    const Eigen::Matrix3d body = Eigen::AngleAxisd(kBodyAngle, Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d stroke_plane =
        Eigen::AngleAxisd(kStrokePlaneAngle, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Matrix3d stroke =
        Eigen::AngleAxisd(static_cast<double>(side) * phi, Eigen::Vector3d::UnitX()).toRotationMatrix();
    const Eigen::Matrix3d feather =
        Eigen::AngleAxisd(static_cast<double>(side) * alpha, Eigen::Vector3d::UnitY()).toRotationMatrix();
    return body * stroke_plane * stroke * feather;
}

void transform_targets(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                       const LDataManager* const manager,
                       double from_time,
                       double to_time)
{
    const int finest_ln = hierarchy->getFinestLevelNumber();
    const auto right = manager->getLagrangianStructureIndexRange(0, finest_ln);
    const auto left = manager->getLagrangianStructureIndexRange(1, finest_ln);
    Pointer<LMesh> mesh = manager->getLMesh(finest_ln);
    std::vector<LNode*> nodes = mesh->getLocalNodes();
    const auto& ghosts = mesh->getGhostNodes();
    nodes.insert(nodes.end(), ghosts.begin(), ghosts.end());

    for (LNode* node : nodes)
    {
        auto* spec = node->getNodeDataItem<IBTargetPointForceSpec>();
        if (!spec) continue;
        const int lag_idx = node->getLagrangianIndex();
        const int side = right.first <= lag_idx && lag_idx < right.second ? 1 :
                         left.first <= lag_idx && lag_idx < left.second  ? -1 : 0;
        if (side == 0) continue;

        Point& target = spec->getTargetPointPosition();
        Eigen::Vector3d x(target[0], target[1], target[2]);
        if (g_motion_mode == "translation")
            x += g_translation_velocity * (to_time - from_time);
        else if (g_motion_mode == "hover")
        {
            const Eigen::Matrix3d delta = wing_pose(to_time, side) * wing_pose(from_time, side).transpose();
            x = delta * x;
        }
        target[0] = x[0];
        target[1] = x[1];
        target[2] = x[2];
    }
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> prescribed_target_velocity_affine(double time, int side)
{
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    if (g_motion_mode == "translation")
    {
        b = g_translation_velocity;
    }
    else if (g_motion_mode == "hover" && g_kinematics_scale != 0.0)
    {
        const double omega = 2.0 * M_PI * kFrequencyHz;
        const double phase = omega * time;
        const double phi = g_kinematics_scale * kPhiAmplitude * std::cos(phase);
        const double alpha = g_kinematics_scale * kAlphaAmplitude * std::sin(phase);
        const double phi_dot = -g_kinematics_scale * kPhiAmplitude * omega * std::sin(phase);
        const double alpha_dot = g_kinematics_scale * kAlphaAmplitude * omega * std::cos(phase);

        const Eigen::Matrix3d body = Eigen::AngleAxisd(kBodyAngle, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const Eigen::Matrix3d stroke_plane =
            Eigen::AngleAxisd(kStrokePlaneAngle, Eigen::Vector3d::UnitX()).toRotationMatrix();
        const double phi_side = static_cast<double>(side) * phi;
        const double alpha_side = static_cast<double>(side) * alpha;
        const Eigen::Matrix3d Rx = Eigen::AngleAxisd(phi_side, Eigen::Vector3d::UnitX()).toRotationMatrix();
        const Eigen::Matrix3d Ry = Eigen::AngleAxisd(alpha_side, Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Matrix3d dRx_dtheta;
        dRx_dtheta << 0.0, 0.0, 0.0,
                      0.0, -std::sin(phi_side), -std::cos(phi_side),
                      0.0,  std::cos(phi_side), -std::sin(phi_side);
        Eigen::Matrix3d dRy_dtheta;
        dRy_dtheta << -std::sin(alpha_side), 0.0, std::cos(alpha_side),
                       0.0, 0.0, 0.0,
                      -std::cos(alpha_side), 0.0, -std::sin(alpha_side);
        const Eigen::Matrix3d R_dot = body * stroke_plane *
                                      (static_cast<double>(side) * phi_dot * dRx_dtheta * Ry +
                                       static_cast<double>(side) * alpha_dot * Rx * dRy_dtheta);
        A = R_dot * wing_pose(time, side).transpose();
    }
    return { A, b };
}

void shift_target_references_for_relative_damping(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                                  const LDataManager* const manager,
                                                  double force_time,
                                                  bool apply_velocity_offset)
{
    const int finest_ln = hierarchy->getFinestLevelNumber();
    const auto right = manager->getLagrangianStructureIndexRange(0, finest_ln);
    const auto left = manager->getLagrangianStructureIndexRange(1, finest_ln);
    Pointer<LMesh> mesh = manager->getLMesh(finest_ln);
    std::vector<LNode*> nodes = mesh->getLocalNodes();
    const auto& ghosts = mesh->getGhostNodes();
    nodes.insert(nodes.end(), ghosts.begin(), ghosts.end());

    for (LNode* node : nodes)
    {
        auto* spec = node->getNodeDataItem<IBTargetPointForceSpec>();
        if (!spec) continue;
        const int lag_idx = node->getLagrangianIndex();
        const int side = right.first <= lag_idx && lag_idx < right.second ? 1 :
                         left.first <= lag_idx && lag_idx < left.second  ? -1 : 0;
        const double stiffness = spec->getStiffness();
        const double damping = spec->getDamping();
        if (side == 0 || stiffness <= 0.0 || damping <= 0.0) continue;

        const double offset_scale = damping / stiffness;
        const auto [A, b] = prescribed_target_velocity_affine(force_time, side);
        Point& target = spec->getTargetPointPosition();
        Eigen::Vector3d x(target[0], target[1], target[2]);
        if (apply_velocity_offset)
        {
            // IBStandardForceGen applies k(X_ref-X)-eta*U_fluid. This
            // effective reference shift makes that term equal to
            // k(X_prescribed-X)-eta*(U_fluid-U_target).
            x = (Eigen::Matrix3d::Identity() + offset_scale * A) * x + offset_scale * b;
        }
        else
        {
            // Invert the affine shift exactly so diagnostics and the next
            // prescribed transform see the unmodified geometric target.
            x = (Eigen::Matrix3d::Identity() + offset_scale * A).fullPivLu().solve(x - offset_scale * b);
        }
        target[0] = x[0];
        target[1] = x[1];
        target[2] = x[2];
    }
}

std::array<double, 2> target_tracking_error(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                           const LDataManager* const manager)
{
    const int finest_ln = hierarchy->getFinestLevelNumber();
    Pointer<LData> X_data = manager->getLData("X", finest_ln);
    Vec X_vec = X_data->getVec();
    const double* X_values = nullptr;
    VecGetArrayRead(X_vec, &X_values);

    double local_sum_sq = 0.0;
    double local_max = 0.0;
    double local_count = 0.0;
    Pointer<LMesh> mesh = manager->getLMesh(finest_ln);
    for (LNode* node : mesh->getLocalNodes())
    {
        auto* spec = node->getNodeDataItem<IBTargetPointForceSpec>();
        if (!spec) continue;
        const int petsc_idx = node->getLocalPETScIndex();
        const Point& target = spec->getTargetPointPosition();
        double error_sq = 0.0;
        for (int d = 0; d < NDIM; ++d)
        {
            const double error = X_values[petsc_idx * NDIM + d] - target[d];
            error_sq += error * error;
        }
        local_sum_sq += error_sq;
        local_max = std::max(local_max, std::sqrt(error_sq));
        local_count += 1.0;
    }
    VecRestoreArrayRead(X_vec, &X_values);

    IBTK_MPI::sumReduction(&local_sum_sq, 1);
    IBTK_MPI::sumReduction(&local_count, 1);
    IBTK_MPI::maxReduction(&local_max, 1);
    return { local_count > 0.0 ? std::sqrt(local_sum_sq / local_count) : 0.0, local_max };
}

Eigen::Vector3d angular_velocity(double time, double dt, int side)
{
    if (g_motion_mode != "hover") return Eigen::Vector3d::Zero();
    const double h = std::max(dt * 0.25, 1.0e-7);
    const Eigen::Matrix3d r0 = wing_pose(time - h, side);
    const Eigen::Matrix3d r1 = wing_pose(time + h, side);
    const Eigen::Matrix3d rdot = (r1 - r0) / (2.0 * h);
    const Eigen::Matrix3d omega = rdot * wing_pose(time, side).transpose();
    return Eigen::Vector3d(omega(2, 1) - omega(1, 2), omega(0, 2) - omega(2, 0), omega(1, 0) - omega(0, 1)) * 0.5;
}

struct SurfaceVelocityMetrics
{
    double rms = 0.0;
    double maximum = 0.0;
    double reference_speed = 1.0;
    double normalized_rms = 0.0;
    double normalized_maximum = 0.0;
};

SurfaceVelocityMetrics surface_velocity_error(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                              const LDataManager* const manager,
                                              double time,
                                              double dt)
{
    SurfaceVelocityMetrics result;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    Pointer<LData> U_data = manager->getLData("U", finest_ln);
    Vec U_vec = U_data->getVec();
    const double* U_values = nullptr;
    VecGetArrayRead(U_vec, &U_values);

    double local_sum_sq = 0.0;
    double local_count = 0.0;
    double local_max = 0.0;
    Pointer<LMesh> mesh = manager->getLMesh(finest_ln);
    const auto right = manager->getLagrangianStructureIndexRange(0, finest_ln);
    const auto left = manager->getLagrangianStructureIndexRange(1, finest_ln);
    for (LNode* node : mesh->getLocalNodes())
    {
        auto* spec = node->getNodeDataItem<IBTargetPointForceSpec>();
        if (!spec) continue;
        const int idx = node->getLocalPETScIndex();
        const Point& target = spec->getTargetPointPosition();
        const Eigen::Vector3d x(target[0], target[1], target[2]);
        Eigen::Vector3d fluid(U_values[idx * NDIM], U_values[idx * NDIM + 1], U_values[idx * NDIM + 2]);
        Eigen::Vector3d prescribed = g_translation_velocity;
        if (g_motion_mode == "hover")
        {
            const int lag_idx = node->getLagrangianIndex();
            const int side = right.first <= lag_idx && lag_idx < right.second ? 1 : -1;
            prescribed = angular_velocity(time - 0.5 * dt, dt, side).cross(x);
        }
        const double error = (fluid - prescribed).norm();
        local_sum_sq += error * error;
        local_max = std::max(local_max, error);
        local_count += 1.0;
    }
    VecRestoreArrayRead(U_vec, &U_values);
    IBTK_MPI::sumReduction(&local_sum_sq, 1);
    IBTK_MPI::sumReduction(&local_count, 1);
    IBTK_MPI::maxReduction(&local_max, 1);
    result.rms = local_count > 0.0 ? std::sqrt(local_sum_sq / local_count) : 0.0;
    result.maximum = local_max;
    if (g_motion_mode == "translation")
        result.reference_speed = std::max(g_translation_velocity.norm(), 1.0e-8);
    else if (g_motion_mode == "hover")
        result.reference_speed = 2.0 * M_PI * kFrequencyHz * (kPhiAmplitude + kAlphaAmplitude) * kWingRadius;
    else
        result.reference_speed = 1.0;
    result.normalized_rms = result.rms / result.reference_speed;
    result.normalized_maximum = result.maximum / result.reference_speed;
    return result;
}

double minimum_surface_domain_margin(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                     const LDataManager* const manager,
                                     const Pointer<CartesianGridGeometry<NDIM>>& geometry)
{
    const int finest_ln = hierarchy->getFinestLevelNumber();
    Pointer<LData> X_data = manager->getLData("X", finest_ln);
    Vec X_vec = X_data->getVec();
    const double* X_values = nullptr;
    VecGetArrayRead(X_vec, &X_values);
    const double* const x_lower = geometry->getXLower();
    const double* const x_upper = geometry->getXUpper();
    double local_minimum = std::numeric_limits<double>::max();
    Pointer<LMesh> mesh = manager->getLMesh(finest_ln);
    for (LNode* node : mesh->getLocalNodes())
    {
        auto* spec = node->getNodeDataItem<IBTargetPointForceSpec>();
        if (!spec) continue;
        const int idx = node->getLocalPETScIndex();
        const Point& target = spec->getTargetPointPosition();
        for (int d = 0; d < NDIM; ++d)
        {
            const double actual = X_values[idx * NDIM + d];
            local_minimum = std::min(local_minimum, std::min(actual - x_lower[d], x_upper[d] - actual));
            local_minimum = std::min(local_minimum, std::min(target[d] - x_lower[d], x_upper[d] - target[d]));
        }
    }
    VecRestoreArrayRead(X_vec, &X_values);
    return IBTK_MPI::minReduction(local_minimum);
}

double realized_velocity_cfl(const Pointer<PatchHierarchy<NDIM>>& hierarchy, int velocity_idx, double dt)
{
    double local_cfl_max = 0.0;
    SAMRAI::math::PatchSideDataOpsReal<NDIM, double> side_ops;
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
            const double* const dx = patch_geom->getDx();
            const double dx_min = *std::min_element(dx, dx + NDIM);
            Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
            const double u_max = side_ops.maxNorm(velocity, patch->getBox());
            local_cfl_max = std::max(local_cfl_max, u_max * dt / dx_min);
        }
    }
    return IBTK_MPI::maxReduction(local_cfl_max);
}

struct DivergenceBandMetrics
{
    double rms = 0.0;
    double maximum = 0.0;
    double width = 0.0;
};

struct FlowStateSummary
{
    double volume = 0.0;
    Eigen::Vector3d velocity_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity_rms = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity_min = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
    Eigen::Vector3d velocity_max = Eigen::Vector3d::Constant(-std::numeric_limits<double>::max());
    double pressure_mean = 0.0;
    double pressure_rms = 0.0;
    double pressure_min = std::numeric_limits<double>::max();
    double pressure_max = -std::numeric_limits<double>::max();
    double divergence_rms = 0.0;
    double divergence_max = 0.0;
};

FlowStateSummary summarize_flow_state(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                      int velocity_idx,
                                      int pressure_idx,
                                      int cell_weight_idx,
                                      int level_filter = -1)
{
    FlowStateSummary result;
    Eigen::Vector3d velocity_sum_sq = Eigen::Vector3d::Zero();
    double pressure_sum_sq = 0.0;
    double divergence_sum_sq = 0.0;
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        if (level_filter >= 0 && ln != level_filter) continue;
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& patch_box = patch->getBox();
            const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
            const double* dx = patch_geom->getDx();
            const Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
            const Pointer<CellData<NDIM, double>> pressure = patch->getPatchData(pressure_idx);
            const Pointer<CellData<NDIM, double>> weight_data = patch->getPatchData(cell_weight_idx);
            for (Box<NDIM>::Iterator ci(patch_box); ci; ci++)
            {
                const CellIndex<NDIM>& cell = *ci;
                const double weight = (*weight_data)(cell);
                if (weight <= 0.0) continue;
                Eigen::Vector3d u;
                double div = 0.0;
                for (int d = 0; d < NDIM; ++d)
                {
                    const double u_lower = (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower));
                    const double u_upper = (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Upper));
                    u[d] = 0.5 * (u_lower + u_upper);
                    div += (u_upper - u_lower) / dx[d];
                }
                const double pres = (*pressure)(cell);
                result.volume += weight;
                result.velocity_mean += weight * u;
                velocity_sum_sq += weight * u.cwiseProduct(u);
                result.velocity_min = result.velocity_min.cwiseMin(u);
                result.velocity_max = result.velocity_max.cwiseMax(u);
                result.pressure_mean += weight * pres;
                pressure_sum_sq += weight * pres * pres;
                result.pressure_min = std::min(result.pressure_min, pres);
                result.pressure_max = std::max(result.pressure_max, pres);
                divergence_sum_sq += weight * div * div;
                result.divergence_max = std::max(result.divergence_max, std::abs(div));
            }
        }
    }
    double volume = result.volume;
    IBTK_MPI::sumReduction(&volume, 1);
    IBTK_MPI::sumReduction(result.velocity_mean.data(), NDIM);
    IBTK_MPI::sumReduction(velocity_sum_sq.data(), NDIM);
    IBTK_MPI::sumReduction(&result.pressure_mean, 1);
    IBTK_MPI::sumReduction(&pressure_sum_sq, 1);
    IBTK_MPI::sumReduction(&divergence_sum_sq, 1);
    result.divergence_max = IBTK_MPI::maxReduction(result.divergence_max);
    IBTK_MPI::minReduction(result.velocity_min.data(), NDIM);
    IBTK_MPI::maxReduction(result.velocity_max.data(), NDIM);
    result.pressure_min = IBTK_MPI::minReduction(result.pressure_min);
    result.pressure_max = IBTK_MPI::maxReduction(result.pressure_max);
    result.volume = volume;
    if (volume > 0.0)
    {
        result.velocity_mean /= volume;
        result.velocity_rms = (velocity_sum_sq / volume).cwiseSqrt();
        result.pressure_mean /= volume;
        result.pressure_rms = std::sqrt(pressure_sum_sq / volume);
        result.divergence_rms = std::sqrt(divergence_sum_sq / volume);
    }
    return result;
}

Eigen::Vector3d integrate_composite_box_momentum(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                                int velocity_idx,
                                                int cell_weight_idx,
                                                const Eigen::Vector3d& lower,
                                                const Eigen::Vector3d& upper,
                                                double rho)
{
    Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
    const double lower_array[NDIM] = { lower[0], lower[1], lower[2] };
    const double upper_array[NDIM] = { upper[0], upper[1], upper[2] };
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        Box<NDIM> box(IndexUtilities::getCellIndex(lower_array, level->getGridGeometry(), level->getRatio()),
                      IndexUtilities::getCellIndex(upper_array, level->getGridGeometry(), level->getRatio()));
        box.upper() -= 1;
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM> overlap = patch->getBox() * box;
            if (overlap.empty()) continue;
            Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
            Pointer<CellData<NDIM, double>> weights = patch->getPatchData(cell_weight_idx);
            for (Box<NDIM>::Iterator ci(overlap); ci; ci++)
            {
                const CellIndex<NDIM>& cell = *ci;
                const double weight = (*weights)(cell);
                if (weight <= 0.0) continue;
                for (int d = 0; d < NDIM; ++d)
                {
                    const double u_lower = (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower));
                    const double u_upper = (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Upper));
                    momentum[d] += rho * weight * 0.5 * (u_lower + u_upper);
                }
            }
        }
    }
    IBTK_MPI::sumReduction(momentum.data(), NDIM);
    return momentum;
}

struct FaceMomentumAudit
{
    Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
    std::array<std::array<double, 3>, NDIM> by_face_class{}; // interior, lower CV face, upper CV face
    std::array<std::array<double, 3>, NDIM> weighted_volume_by_face_class{};
};

FaceMomentumAudit audit_single_level_face_momentum(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                                    int velocity_idx,
                                                    const Eigen::Vector3d& lower,
                                                    const Eigen::Vector3d& upper,
                                                    double rho)
{
    if (hierarchy->getFinestLevelNumber() != 0)
        TBOX_ERROR("Single-level face-momentum audit requires exactly one AMR level.\n");
    FaceMomentumAudit result;
    const Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(0);
    const double lower_array[NDIM] = { lower[0], lower[1], lower[2] };
    const double upper_array[NDIM] = { upper[0], upper[1], upper[2] };
    Box<NDIM> integration_box(IndexUtilities::getCellIndex(lower_array, level->getGridGeometry(), level->getRatio()),
                              IndexUtilities::getCellIndex(upper_array, level->getGridGeometry(), level->getRatio()));
    integration_box.upper() -= 1;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        const Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const Box<NDIM>& patch_box = patch->getBox();
        const Box<NDIM> overlap = patch_box * integration_box;
        if (overlap.empty()) continue;
        const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
        const double* const dx = patch_geom->getDx();
        double cell_volume = dx[0] * dx[1] * dx[2];
        const Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
        for (int axis = 0; axis < NDIM; ++axis)
        {
            for (Box<NDIM>::Iterator b(SideGeometry<NDIM>::toSideBox(overlap, axis)); b; b++)
            {
                const CellIndex<NDIM>& cell = *b;
                const SideIndex<NDIM> side(cell, axis, SideIndex<NDIM>::Lower);
                double volume_weight = cell_volume;
                const bool at_patch_lower = cell(axis) == patch_box.lower(axis);
                const bool at_patch_upper = cell(axis) == patch_box.upper(axis) + 1;
                if (at_patch_lower) volume_weight *= 0.5;
                if (at_patch_upper) volume_weight *= 0.5;
                const bool at_cv_lower = cell(axis) == integration_box.lower(axis);
                const bool at_cv_upper = cell(axis) == integration_box.upper(axis) + 1;
                const bool patch_lower_matches_cv = patch_box.lower(axis) == integration_box.lower(axis);
                const bool patch_upper_matches_cv = patch_box.upper(axis) + 1 == integration_box.upper(axis) + 1;
                const bool scale_for_cv_edge = (at_cv_lower && !patch_lower_matches_cv) ||
                                               (at_cv_upper && !patch_upper_matches_cv);
                const double dV = scale_for_cv_edge ? 0.5 * volume_weight : volume_weight;
                const int face_class = at_cv_lower ? 1 : at_cv_upper ? 2 : 0;
                const double term = rho * (*velocity)(side) * dV;
                result.momentum[axis] += term;
                result.by_face_class[axis][face_class] += term;
                result.weighted_volume_by_face_class[axis][face_class] += dV;
            }
        }
    }
    IBTK_MPI::sumReduction(result.momentum.data(), NDIM);
    for (int axis = 0; axis < NDIM; ++axis)
    {
        IBTK_MPI::sumReduction(result.by_face_class[axis].data(), 3);
        IBTK_MPI::sumReduction(result.weighted_volume_by_face_class[axis].data(), 3);
    }
    return result;
}

struct FlowStateDiagnosticContext
{
    Pointer<PatchHierarchy<NDIM>> hierarchy;
    Pointer<IBHierarchyIntegrator> integrator;
    Pointer<IBHydrodynamicForceEvaluator> force_evaluator;
    std::ofstream* output = nullptr;
    std::ofstream* per_level_output = nullptr;
    std::ofstream* momentum_trace = nullptr;
    int velocity_idx = -1;
    int pressure_idx = -1;
    int cell_weight_idx = -1;
    std::vector<RobinBcCoefStrategy<NDIM>*> velocity_bc;
    bool refresh_lagged_momentum_after_regrid = false;
};

FlowStateDiagnosticContext* g_regrid_projection_diagnostics = nullptr;

void write_flow_state_summary(std::ofstream& output,
                              const std::string& stage,
                              double time,
                              const FlowStateSummary& s)
{
    output << stage << ',' << std::setprecision(16) << time << ',' << s.volume << ','
           << s.velocity_mean[0] << ',' << s.velocity_mean[1] << ',' << s.velocity_mean[2] << ','
           << s.velocity_rms[0] << ',' << s.velocity_rms[1] << ',' << s.velocity_rms[2] << ','
           << s.velocity_min[0] << ',' << s.velocity_min[1] << ',' << s.velocity_min[2] << ','
           << s.velocity_max[0] << ',' << s.velocity_max[1] << ',' << s.velocity_max[2] << ','
           << s.pressure_mean << ',' << s.pressure_rms << ',' << s.pressure_min << ',' << s.pressure_max << ','
           << s.divergence_rms << ',' << s.divergence_max << '\n';
    output.flush();
}

void write_patch_layout(std::ofstream& output,
                        const char* stage,
                        double time,
                        const Pointer<PatchHierarchy<NDIM>>& hierarchy)
{
    if (!output.is_open() || IBTK_MPI::getRank() != 0) return;
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        const Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            const Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& box = patch->getBox();
            output << stage << ',' << std::setprecision(16) << time << ',' << ln << ',' << p();
            for (int d = 0; d < NDIM; ++d) output << ',' << box.lower(d) << ',' << box.upper(d);
            output << '\n';
        }
    }
    output.flush();
}

void write_per_level_flow_state(std::ofstream& output,
                                const char* stage,
                                double time,
                                const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                int velocity_idx,
                                int pressure_idx,
                                int cell_weight_idx)
{
    if (!output.is_open() || IBTK_MPI::getRank() != 0) return;
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        const FlowStateSummary summary = summarize_flow_state(
            hierarchy, velocity_idx, pressure_idx, cell_weight_idx, ln);
        output << stage << ',' << std::setprecision(16) << time << ',' << ln << ',' << summary.volume << ','
               << summary.velocity_mean[0] << ',' << summary.velocity_mean[1] << ',' << summary.velocity_mean[2] << ','
               << summary.velocity_rms[0] << ',' << summary.velocity_rms[1] << ',' << summary.velocity_rms[2] << ','
               << summary.velocity_min[0] << ',' << summary.velocity_min[1] << ',' << summary.velocity_min[2] << ','
               << summary.velocity_max[0] << ',' << summary.velocity_max[1] << ',' << summary.velocity_max[2] << ','
               << summary.pressure_mean << ',' << summary.pressure_rms << ',' << summary.divergence_rms << ','
               << summary.divergence_max << '\n';
    }
    output.flush();
}

void flow_state_regrid_callback(Pointer<BasePatchHierarchy<NDIM>>,
                                double data_time,
                                bool,
                                void* ctx)
{
    auto* diagnostic = static_cast<FlowStateDiagnosticContext*>(ctx);
    if (!diagnostic) return;
    // IBTK executes regrid callbacks after rebuilding/projection but before its
    // automatic state synchronization. Synchronize before reading a composite
    // Eulerian field, as required by HierarchyIntegrator's callback contract.
    if (diagnostic->integrator) diagnostic->integrator->synchronizeHierarchyData(CURRENT_DATA);
    if (diagnostic->refresh_lagged_momentum_after_regrid && diagnostic->force_evaluator)
    {
        diagnostic->force_evaluator->computeLaggedMomentumIntegral(
            diagnostic->velocity_idx, diagnostic->hierarchy, diagnostic->velocity_bc);
        if (diagnostic->momentum_trace && diagnostic->momentum_trace->is_open())
            for (int sid = 0; sid < 2; ++sid)
            {
                const auto& force = diagnostic->force_evaluator->getHydrodynamicForceObject(sid);
                *diagnostic->momentum_trace << data_time << ",post_regrid_lagged_integral," << sid << ','
                                            << force.P_box_current[0] << ',' << force.P_box_current[1] << ','
                                            << force.P_box_current[2] << '\n';
            }
        if (diagnostic->momentum_trace && diagnostic->momentum_trace->is_open())
            diagnostic->momentum_trace->flush();
    }
    if (diagnostic->output && diagnostic->output->is_open())
    {
        const FlowStateSummary summary = summarize_flow_state(diagnostic->hierarchy,
                                                              diagnostic->velocity_idx,
                                                              diagnostic->pressure_idx,
                                                              diagnostic->cell_weight_idx);
        write_flow_state_summary(*diagnostic->output, "post_regrid_projection_and_sync", data_time, summary);
    }
    if (diagnostic->per_level_output)
        write_per_level_flow_state(*diagnostic->per_level_output,
                                   "post_regrid_projection_and_sync",
                                   data_time,
                                   diagnostic->hierarchy,
                                   diagnostic->velocity_idx,
                                   diagnostic->pressure_idx,
                                   diagnostic->cell_weight_idx);
}

// Instrument only the projection boundary in the pinned IBAMR implementation.
// The summaries bracket the library operation, so a change already present on
// entry implicates AMR transfer/initialization, while a change across the call
// implicates the projection path. The production solver algorithm remains the
// upstream IBAMR implementation.
class RegridProjectionAuditSolver final : public INSStaggeredHierarchyIntegrator
{
public:
    using INSStaggeredHierarchyIntegrator::INSStaggeredHierarchyIntegrator;

protected:
    void regridProjection(const bool initial_time) override
    {
        FlowStateDiagnosticContext* diagnostic = g_regrid_projection_diagnostics;
        const auto write_summary = [&](const char* label) {
            if (!diagnostic || !diagnostic->output || !diagnostic->output->is_open()) return;
            const FlowStateSummary summary = summarize_flow_state(diagnostic->hierarchy,
                                                                  diagnostic->velocity_idx,
                                                                  diagnostic->pressure_idx,
                                                                  diagnostic->cell_weight_idx);
            std::string stage = "regrid_projection_";
            stage += initial_time ? "initial_time_" : "timestep_";
            stage += label;
            write_flow_state_summary(*diagnostic->output, stage, diagnostic->integrator->getIntegratorTime(), summary);
            if (diagnostic->per_level_output)
                write_per_level_flow_state(*diagnostic->per_level_output,
                                           stage.c_str(),
                                           diagnostic->integrator->getIntegratorTime(),
                                           diagnostic->hierarchy,
                                           diagnostic->velocity_idx,
                                           diagnostic->pressure_idx,
                                           diagnostic->cell_weight_idx);
        };

        write_summary("before");
        INSStaggeredHierarchyIntegrator::regridProjection(initial_time);
        write_summary("after");
    }
};

struct SurfaceFluxes
{
    Eigen::Vector3d pressure = Eigen::Vector3d::Zero();
    Eigen::Vector3d advection = Eigen::Vector3d::Zero();
    Eigen::Vector3d viscous = Eigen::Vector3d::Zero();
};

// Independent level-zero stress/flux integral over a closed rectangular surface.
// The coarse-only quadrature avoids reusing the evaluator's AMR face weights;
// the CV is deliberately aligned with the level-zero mesh. The boundary may
// coincide with a physical-domain face, so callers must pass velocity and
// pressure fields whose physical ghost cells have been filled at the sample time.
SurfaceFluxes integrate_level_zero_control_surface(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                                    const Eigen::Vector3d& lower,
                                                    const Eigen::Vector3d& upper,
                                                    int velocity_idx,
                                                    int pressure_idx,
                                                    double rho,
                                                    double mu,
                                                    bool debug_trace = false,
                                                    double trace_time = 0.0,
                                                    int structure_id = -1)
{
    SurfaceFluxes result;
    Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(0);
    const double lower_array[NDIM] = { lower[0], lower[1], lower[2] };
    const double upper_array[NDIM] = { upper[0], upper[1], upper[2] };
    Box<NDIM> integration_box(IndexUtilities::getCellIndex(lower_array, level->getGridGeometry(), level->getRatio()),
                              IndexUtilities::getCellIndex(upper_array, level->getGridGeometry(), level->getRatio()));
    integration_box.upper() -= 1;
    for (PatchLevel<NDIM>::Iterator p(level); p; p++)
    {
        Pointer<Patch<NDIM>> patch = level->getPatch(p());
        const Box<NDIM>& patch_box = patch->getBox();
        const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
        const double* dx = patch_geom->getDx();
        const Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
        const Pointer<CellData<NDIM, double>> pressure = patch->getPatchData(pressure_idx);
        for (int axis = 0; axis < NDIM; ++axis)
        {
            for (int upper_lower = 0; upper_lower <= 1; ++upper_lower)
            {
                Box<NDIM> boundary = integration_box;
                if (upper_lower == 0) boundary.upper()(axis) = boundary.lower()(axis);
                else boundary.lower()(axis) = boundary.upper()(axis);
                if (!patch_box.intersects(boundary)) continue;
                const Box<NDIM> trim = patch_box * boundary;
                const bool trace_this = debug_trace && structure_id == 0 &&
                                        trace_time >= 2.8e-5 && trace_time < 2.81e-5;
                if (trace_this)
                    std::cerr << "MOMENTUM_LEDGER_TRACE cv_face sid=" << structure_id
                              << " t=" << trace_time << " patch=" << p()
                              << " axis=" << axis << " side=" << upper_lower
                              << " patch_box=" << patch_box << " boundary=" << boundary
                              << " trim=" << trim << std::endl;
                Eigen::Vector3d normal = Eigen::Vector3d::Zero();
                normal[axis] = upper_lower ? 1.0 : -1.0;
                double area = 1.0;
                for (int d = 0; d < NDIM; ++d) if (d != axis) area *= dx[d];
                for (Box<NDIM>::Iterator bi(trim); bi; bi++)
                {
                    const CellIndex<NDIM>& cell = *bi;
                    CellIndex<NDIM> neighbor = cell;
                    neighbor(axis) += static_cast<int>(normal[axis]);
                    const SideIndex<NDIM> face(cell, axis,
                        upper_lower ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower);
                    const bool trace_cell = trace_this && axis == 0 && upper_lower == 0 &&
                                            cell(0) == 0 && cell(1) == 32 && cell(2) == 0;
                    if (trace_cell)
                        std::cerr << "MOMENTUM_LEDGER_TRACE cv_cell sid=" << structure_id
                                  << " t=" << trace_time << " axis=" << axis
                                  << " side=" << upper_lower << " cell=" << cell
                                  << " neighbor=" << neighbor << std::endl;
                    const double p_face = 0.5 * ((*pressure)(cell) + (*pressure)(neighbor));
                    if (trace_cell)
                        std::cerr << "MOMENTUM_LEDGER_TRACE cv_pressure_done" << std::endl;
                    result.pressure += -p_face * normal * area;

                    Eigen::Vector3d u = Eigen::Vector3d::Zero();
                    for (int d = 0; d < NDIM; ++d)
                    {
                        if (d == axis) u[d] = (*velocity)(face);
                        else
                        {
                            u[d] = 0.25 * ((*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower)) +
                                           (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Upper)) +
                                           (*velocity)(SideIndex<NDIM>(neighbor, d, SideIndex<NDIM>::Lower)) +
                                           (*velocity)(SideIndex<NDIM>(neighbor, d, SideIndex<NDIM>::Upper)));
                        }
                    }
                    if (trace_cell)
                        std::cerr << "MOMENTUM_LEDGER_TRACE cv_velocity_done" << std::endl;
                    result.advection += -rho * normal.dot(u) * u * area;
                    if (trace_cell)
                        std::cerr << "MOMENTUM_LEDGER_TRACE cv_advection_done" << std::endl;
                    Eigen::Vector3d traction = Eigen::Vector3d::Zero();
                    for (int d = 0; d < NDIM; ++d)
                    {
                        if (d == axis)
                        {
                            const double u_boundary = (*velocity)(face);
                            const double u_inside = (*velocity)(SideIndex<NDIM>(
                                cell, axis, upper_lower ? SideIndex<NDIM>::Lower : SideIndex<NDIM>::Upper));
                            // n * 2 mu * d(u_n)/dx, written in terms of the
                            // boundary and first interior staggered samples.
                            // This avoids indexing one cell beyond the physical
                            // ghost region at a lower boundary.
                            traction[d] = 2.0 * mu / dx[axis] * (u_boundary - u_inside);
                        }
                        else
                        {
                            CellIndex<NDIM> offset(0);
                            offset(d) = 1;
                            const double tangential_derivative =
                                ((*velocity)(SideIndex<NDIM>(cell + offset, axis,
                                    upper_lower ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower)) -
                                 (*velocity)(SideIndex<NDIM>(cell - offset, axis,
                                    upper_lower ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower))) /
                                (2.0 * dx[d]);
                            const double normal_derivative_times_normal =
                                ((*velocity)(SideIndex<NDIM>(neighbor, d, SideIndex<NDIM>::Lower)) +
                                 (*velocity)(SideIndex<NDIM>(neighbor + offset, d, SideIndex<NDIM>::Lower)) -
                                 (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower)) -
                                 (*velocity)(SideIndex<NDIM>(cell + offset, d, SideIndex<NDIM>::Lower))) /
                                (2.0 * dx[axis]);
                            traction[d] = mu * (normal[axis] * tangential_derivative +
                                               normal_derivative_times_normal);
                        }
                    }
                    if (trace_cell)
                        std::cerr << "MOMENTUM_LEDGER_TRACE cv_traction_done" << std::endl;
                    result.viscous += traction * area;
                }
            }
        }
    }
    IBTK_MPI::sumReduction(result.pressure.data(), NDIM);
    IBTK_MPI::sumReduction(result.advection.data(), NDIM);
    IBTK_MPI::sumReduction(result.viscous.data(), NDIM);
    return result;
}

struct DomainMomentumFluxes
{
    Eigen::Vector3d pressure = Eigen::Vector3d::Zero();
    Eigen::Vector3d advection = Eigen::Vector3d::Zero();
    Eigen::Vector3d viscous = Eigen::Vector3d::Zero();
    bool refined_boundary = false;
};

/*
 * Independently integrate the momentum fluxes on the physical-domain boundary.
 * The spatial stencil follows the staggered MAC arrangement used by IBAMR's
 * force evaluator, but this routine integrates the physical outer boundary
 * directly, rather than recovering a surface term from the evaluator's force
 * identity. Composite integration proceeds from finest to coarsest and omits
 * coarse boundary cells covered by the next finer level.
 *
 * Signs follow sigma = -p I + mu(grad u + grad u^T): pressure and viscous
 * are outward traction integrals; advection is -rho u (u dot n). With force
 * positive on the body, the global balance is dP_fluid/dt + F_body -
 * (pressure + viscous + advection) = 0.
 */
DomainMomentumFluxes integrate_domain_momentum_fluxes(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                                       const Pointer<CartesianGridGeometry<NDIM>>& grid_geometry,
                                                       int u_idx,
                                                       int p_idx,
                                                       double rho,
                                                       double mu)
{
    DomainMomentumFluxes result;
    Pointer<PatchLevel<NDIM>> boundary_level = hierarchy->getPatchLevel(0);
    const BoxArray<NDIM>& physical_domain = boundary_level->getPhysicalDomain();
    if (physical_domain.getNumberOfBoxes() != 1)
        TBOX_ERROR("Independent domain momentum integration expects one rectangular physical domain.\n");
    const Box<NDIM>& domain = physical_domain[0];

    const double* domain_lower = grid_geometry->getXLower();
    const double* domain_upper = grid_geometry->getXUpper();
    const double coordinate_tolerance = 1.0e-10 *
        *std::max_element(domain_upper, domain_upper + NDIM);
    for (int ln = 1; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
            const double* patch_lower = patch_geom->getXLower();
            const double* patch_upper = patch_geom->getXUpper();
            for (int axis = 0; axis < NDIM; ++axis)
                result.refined_boundary = result.refined_boundary ||
                    std::abs(patch_lower[axis] - domain_lower[axis]) <= coordinate_tolerance ||
                    std::abs(patch_upper[axis] - domain_upper[axis]) <= coordinate_tolerance;
        }
    }
    for (int ln = hierarchy->getFinestLevelNumber(); ln >= 0; --ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        const BoxArray<NDIM>& level_domain = level->getPhysicalDomain();
        if (level_domain.getNumberOfBoxes() != 1)
            TBOX_ERROR("Independent domain momentum integration expects a rectangular level domain.\n");
        const Box<NDIM>& domain = level_domain[0];
        BoxArray<NDIM> covered_boundary_cells;
        if (ln < hierarchy->getFinestLevelNumber())
        {
            Pointer<PatchLevel<NDIM>> fine_level = hierarchy->getPatchLevel(ln + 1);
            covered_boundary_cells = fine_level->getBoxes();
            covered_boundary_cells.coarsen(fine_level->getRatioToCoarserLevel());
        }
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& patch_box = patch->getBox();
            const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
            const double* dx = patch_geom->getDx();
            const Pointer<CellData<NDIM, double>> pressure = patch->getPatchData(p_idx);
            const Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(u_idx);

            for (int axis = 0; axis < NDIM; ++axis)
            {
                const double area = dx[(axis + 1) % NDIM] * dx[(axis + 2) % NDIM];
                for (int upper_lower = 0; upper_lower <= 1; ++upper_lower)
                {
                    Box<NDIM> face_cells = domain;
                    const int boundary_cell = upper_lower ? domain.upper(axis) : domain.lower(axis);
                    face_cells.lower(axis) = boundary_cell;
                    face_cells.upper(axis) = boundary_cell;
                    face_cells = face_cells * patch_box;
                    if (face_cells.empty()) continue;

                    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
                    normal[axis] = upper_lower ? 1.0 : -1.0;
                    const double sign = normal[axis];
                    for (Box<NDIM>::Iterator b(face_cells); b; b++)
                    {
                        const CellIndex<NDIM>& cell = *b;
                        bool covered = false;
                        for (int i = 0; i < covered_boundary_cells.getNumberOfBoxes(); ++i)
                            covered = covered || covered_boundary_cells[i].contains(cell);
                        if (covered) continue;

                        CellIndex<NDIM> outside = cell;
                        outside(axis) += static_cast<int>(sign);
                        const SideIndex<NDIM> face(cell, axis,
                            upper_lower ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower);
                        const double p_face = 0.5 * ((*pressure)(cell) + (*pressure)(outside));
                        result.pressure += -p_face * normal * area;

                        Eigen::Vector3d u = Eigen::Vector3d::Zero();
                        for (int d = 0; d < NDIM; ++d)
                        {
                            if (d == axis) u[d] = (*velocity)(face);
                            else
                            {
                                u[d] = 0.25 * ((*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower)) +
                                               (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Upper)) +
                                               (*velocity)(SideIndex<NDIM>(outside, d, SideIndex<NDIM>::Lower)) +
                                               (*velocity)(SideIndex<NDIM>(outside, d, SideIndex<NDIM>::Upper)));
                            }
                        }
                        result.advection += -rho * normal.dot(u) * u * area;

                        Eigen::Vector3d traction = Eigen::Vector3d::Zero();
                        for (int d = 0; d < NDIM; ++d)
                        {
                            if (d == axis)
                            {
                                const double u_boundary = (*velocity)(face);
                                const double u_inside = (*velocity)(SideIndex<NDIM>(
                                    cell, axis, upper_lower ? SideIndex<NDIM>::Lower : SideIndex<NDIM>::Upper));
                                traction[axis] = 2.0 * mu / dx[axis] * (u_boundary - u_inside);
                            }
                            else
                            {
                                CellIndex<NDIM> offset(0);
                                offset(d) = 1;
                                const double tangential_derivative =
                                    ((*velocity)(SideIndex<NDIM>(cell + offset, axis,
                                        upper_lower ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower)) -
                                     (*velocity)(SideIndex<NDIM>(cell - offset, axis,
                                        upper_lower ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower))) /
                                    (2.0 * dx[d]);
                                const double normal_derivative_times_normal =
                                    ((*velocity)(SideIndex<NDIM>(outside, d, SideIndex<NDIM>::Lower)) +
                                     (*velocity)(SideIndex<NDIM>(outside + offset, d, SideIndex<NDIM>::Lower)) -
                                     (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower)) -
                                     (*velocity)(SideIndex<NDIM>(cell + offset, d, SideIndex<NDIM>::Lower))) /
                                    (2.0 * dx[axis]);
                                traction[d] = mu * (normal[axis] * tangential_derivative +
                                                   normal_derivative_times_normal);
                            }
                        }
                        result.viscous += traction * area;
                    }
                }
            }
        }
    }
    IBTK_MPI::sumReduction(result.pressure.data(), NDIM);
    IBTK_MPI::sumReduction(result.advection.data(), NDIM);
    IBTK_MPI::sumReduction(result.viscous.data(), NDIM);
    return result;
}

void fill_pressure_boundary_ghosts(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                   const Pointer<CellVariable<NDIM, double>>& pressure_var,
                                   int pressure_src_idx,
                                   int pressure_dst_idx,
                                   int velocity_src_idx,
                                   Pointer<INSStaggeredHierarchyIntegrator> ns,
                                   double time)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(pressure_dst_idx)) level->allocatePatchData(pressure_dst_idx);
    }
    HierarchyDataOpsManager<NDIM>* ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
    Pointer<HierarchyDataOpsReal<NDIM, double>> pressure_ops =
        ops_manager->getOperationsDouble(pressure_var, hierarchy, true);
    pressure_ops->copyData(pressure_dst_idx, pressure_src_idx, true);

    auto* pressure_bc = dynamic_cast<INSStaggeredPressureBcCoef*>(ns->getPressureBoundaryConditions());
    if (!pressure_bc) TBOX_ERROR("Expected the staggered incompressible pressure boundary condition.\n");
    pressure_bc->setTargetVelocityPatchDataIndex(velocity_src_idx);
    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<InterpolationTransactionComponent> transaction(1);
    transaction[0] = InterpolationTransactionComponent(pressure_dst_idx,
                                                       pressure_src_idx,
                                                       "CONSERVATIVE_LINEAR_REFINE",
                                                       true,
                                                       "CUBIC_COARSEN",
                                                       "LINEAR",
                                                       false,
                                                       pressure_bc,
                                                       Pointer<VariableFillPattern<NDIM>>(nullptr));
    Pointer<HierarchyGhostCellInterpolation> ghost_fill = new HierarchyGhostCellInterpolation();
    ghost_fill->initializeOperatorState(transaction, hierarchy);
    ghost_fill->setHomogeneousBc(false);
    ghost_fill->fillData(time);
    pressure_bc->clearTargetVelocityPatchDataIndex();
}

void fill_velocity_boundary_ghosts(Pointer<PatchHierarchy<NDIM>> hierarchy,
                                   const Pointer<SideVariable<NDIM, double>>& velocity_var,
                                   int velocity_src_idx,
                                   int velocity_dst_idx,
                                   const std::vector<RobinBcCoefStrategy<NDIM>*>& velocity_bc,
                                   double time)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(velocity_dst_idx)) level->allocatePatchData(velocity_dst_idx);
    }
    HierarchyDataOpsManager<NDIM>* ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
    Pointer<HierarchyDataOpsReal<NDIM, double>> velocity_ops =
        ops_manager->getOperationsDouble(velocity_var, hierarchy, true);
    velocity_ops->copyData(velocity_dst_idx, velocity_src_idx, true);

    using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<InterpolationTransactionComponent> transaction(1);
    transaction[0] = InterpolationTransactionComponent(velocity_dst_idx,
                                                       velocity_src_idx,
                                                       "CONSERVATIVE_LINEAR_REFINE",
                                                       true,
                                                       "CUBIC_COARSEN",
                                                       "LINEAR",
                                                       false,
                                                       velocity_bc,
                                                       Pointer<VariableFillPattern<NDIM>>(nullptr));
    Pointer<HierarchyGhostCellInterpolation> ghost_fill = new HierarchyGhostCellInterpolation();
    ghost_fill->initializeOperatorState(transaction, hierarchy);
    ghost_fill->setHomogeneousBc(false);
    ghost_fill->fillData(time);
}

struct SpatialBin
{
    int x, y, z;
    bool operator==(const SpatialBin& other) const { return x == other.x && y == other.y && z == other.z; }
};

struct SpatialBinHash
{
    std::size_t operator()(const SpatialBin& bin) const
    {
        std::size_t seed = std::hash<int>{}(bin.x);
        seed ^= std::hash<int>{}(bin.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(bin.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

DivergenceBandMetrics divergence_band_metrics(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                               const LDataManager* const manager,
                                               int div_idx,
                                               int cell_weight_idx)
{
    DivergenceBandMetrics result;
    const int finest_ln = hierarchy->getFinestLevelNumber();
    Pointer<PatchLevel<NDIM>> finest_level = hierarchy->getPatchLevel(finest_ln);
    if (finest_level->getNumberOfPatches() == 0) return result;
    Pointer<Patch<NDIM>> finest_patch = finest_level->getPatch(0);
    const Pointer<CartesianPatchGeometry<NDIM>> finest_patch_geom = finest_patch->getPatchGeometry();
    const double* finest_dx = finest_patch_geom->getDx();
    result.width = 2.0 * *std::min_element(finest_dx, finest_dx + NDIM);

    Pointer<LData> X_data = manager->getLData("X", finest_ln);
    Vec X_vec = X_data->getVec();
    VecScatter scatter = nullptr;
    Vec all_X = nullptr;
    VecScatterCreateToAll(X_vec, &scatter, &all_X);
    VecScatterBegin(scatter, X_vec, all_X, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatter, X_vec, all_X, INSERT_VALUES, SCATTER_FORWARD);
    PetscInt global_dofs = 0;
    VecGetSize(all_X, &global_dofs);
    const double* X_values = nullptr;
    VecGetArrayRead(all_X, &X_values);

    const auto bin_for = [width = result.width](const Eigen::Vector3d& x) {
        return SpatialBin{ static_cast<int>(std::floor(x[0] / width)),
                           static_cast<int>(std::floor(x[1] / width)),
                           static_cast<int>(std::floor(x[2] / width)) };
    };
    std::unordered_map<SpatialBin, std::vector<Eigen::Vector3d>, SpatialBinHash> bins;
    for (PetscInt i = 0; i + NDIM <= global_dofs; i += NDIM)
    {
        Eigen::Vector3d x(X_values[i], X_values[i + 1], X_values[i + 2]);
        bins[bin_for(x)].push_back(x);
    }
    VecRestoreArrayRead(all_X, &X_values);
    VecScatterDestroy(&scatter);
    VecDestroy(&all_X);

    double local_sum_sq = 0.0;
    double local_weight = 0.0;
    double local_max = 0.0;
    const double band_sq = result.width * result.width;
    for (int ln = 0; ln <= finest_ln; ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& patch_box = patch->getBox();
            const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
            const double* const dx = patch_geom->getDx();
            const double* const patch_lower = patch_geom->getXLower();
            const Pointer<CellData<NDIM, double>> div_data = patch->getPatchData(div_idx);
            const Pointer<CellData<NDIM, double>> cell_weight = patch->getPatchData(cell_weight_idx);
            for (Box<NDIM>::Iterator ci(patch_box); ci; ci++)
            {
                const CellIndex<NDIM>& cell = *ci;
                const double weight = (*cell_weight)(cell);
                if (weight <= 0.0) continue;
                Eigen::Vector3d center;
                for (int d = 0; d < NDIM; ++d)
                    center[d] = patch_lower[d] + (cell(d) - patch_box.lower()(d) + 0.5) * dx[d];
                const SpatialBin center_bin = bin_for(center);
                bool in_band = false;
                for (int ox = -1; ox <= 1 && !in_band; ++ox)
                    for (int oy = -1; oy <= 1 && !in_band; ++oy)
                        for (int oz = -1; oz <= 1 && !in_band; ++oz)
                        {
                            const auto found = bins.find({ center_bin.x + ox, center_bin.y + oy, center_bin.z + oz });
                            if (found == bins.end()) continue;
                            for (const Eigen::Vector3d& point : found->second)
                                if ((center - point).squaredNorm() <= band_sq)
                                {
                                    in_band = true;
                                    break;
                                }
                        }
                if (!in_band) continue;
                const double value = (*div_data)(cell);
                local_sum_sq += weight * value * value;
                local_weight += weight;
                local_max = std::max(local_max, std::abs(value));
            }
        }
    }
    IBTK_MPI::sumReduction(&local_sum_sq, 1);
    IBTK_MPI::sumReduction(&local_weight, 1);
    IBTK_MPI::maxReduction(&local_max, 1);
    result.rms = local_weight > 0.0 ? std::sqrt(local_sum_sq / local_weight) : 0.0;
    result.maximum = local_max;
    return result;
}
} // namespace

int main(int argc, char* argv[])
{
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    Pointer<AppInitializer> app = new AppInitializer(argc, argv, "hawkmoth_hover.log");
    Pointer<Database> db = app->getInputDatabase();
    g_kinematics_scale = db->getDoubleWithDefault("KINEMATICS_SCALE", 1.0);
    g_kinematic_cfl = db->getDoubleWithDefault("KINEMATIC_CFL", 0.5);
    g_motion_mode = db->getStringWithDefault("MOTION_MODE", "hover");
    if (g_motion_mode != "hover" && g_motion_mode != "stationary" && g_motion_mode != "translation")
        TBOX_ERROR("MOTION_MODE must be hover, stationary, or translation.\n");
    if (g_motion_mode == "translation")
    {
        double translation[NDIM];
        db->getDoubleArray("TRANSLATION_VELOCITY", translation, NDIM);
        g_translation_velocity = Eigen::Vector3d(translation[0], translation[1], translation[2]);
    }
    const double target_force_time_fraction =
        db->getDoubleWithDefault("TARGET_FORCE_TIME_FRACTION", 0.5);
    if (target_force_time_fraction < 0.0 || target_force_time_fraction > 1.0)
        TBOX_ERROR("TARGET_FORCE_TIME_FRACTION must be between 0 and 1.\n");

    const bool dump_viz = app->dumpVizData();
    const bool uses_visit = dump_viz && app->getVisItDataWriter();
    const bool dump_restart = app->dumpRestartData();
    const int restart_interval = app->getRestartDumpInterval();
    const string restart_dir = app->getRestartDumpDirectory();
    const bool dump_timer = app->dumpTimerData();
    const int timer_interval = app->getTimerDumpInterval();

    Pointer<INSStaggeredHierarchyIntegrator> ns = new RegridProjectionAuditSolver(
        "INSStaggeredHierarchyIntegrator", app->getComponentDatabase("INSStaggeredHierarchyIntegrator"));
    Pointer<IBMethod> ib_method = new IBMethod("IBMethod", app->getComponentDatabase("IBMethod"));
    Pointer<IBHierarchyIntegrator> integrator = new IBExplicitHierarchyIntegrator(
        "IBHierarchyIntegrator", app->getComponentDatabase("IBHierarchyIntegrator"), ib_method, ns);
    Pointer<CartesianGridGeometry<NDIM>> geometry =
        new CartesianGridGeometry<NDIM>("CartesianGeometry", app->getComponentDatabase("CartesianGeometry"));
    Pointer<PatchHierarchy<NDIM>> hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", geometry);
    Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
        "StandardTagAndInitialize", integrator, app->getComponentDatabase("StandardTagAndInitialize"));
    Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
    Pointer<LoadBalancer<NDIM>> load_balancer =
        new LoadBalancer<NDIM>("LoadBalancer", app->getComponentDatabase("LoadBalancer"));
    Pointer<GriddingAlgorithm<NDIM>> gridding = new GriddingAlgorithm<NDIM>(
        "GriddingAlgorithm", app->getComponentDatabase("GriddingAlgorithm"), error_detector, box_generator, load_balancer);

    // Register the documented Eulerian initial conditions before hierarchy
    // initialization. Without this, the solver can start from uninitialized
    // velocity data and become unstable even in the zero-motion control.
    if (db->keyExists("VelocityInitialConditions"))
    {
        Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
            "u_init", app->getComponentDatabase("VelocityInitialConditions"), geometry);
        ns->registerVelocityInitialConditions(u_init);
    }
    if (db->keyExists("PressureInitialConditions"))
    {
        Pointer<CartGridFunction> p_init = new muParserCartGridFunction(
            "p_init", app->getComponentDatabase("PressureInitialConditions"), geometry);
        ns->registerPressureInitialConditions(p_init);
    }

    Pointer<IBStandardInitializer> initializer =
        new IBStandardInitializer("IBStandardInitializer", app->getComponentDatabase("IBStandardInitializer"));
    ib_method->registerLInitStrategy(initializer);
    const bool enable_ib_forcing = db->getBoolWithDefault("ENABLE_IB_FORCING", true);
    const double target_stiffness = db->getDoubleWithDefault("TARGET_STIFFNESS", 0.0);
    const double target_damping = db->getDoubleWithDefault("TARGET_DAMPING", 0.0);
    if (target_damping > 0.0 && target_stiffness <= 0.0)
        TBOX_ERROR("TARGET_DAMPING > 0 requires TARGET_STIFFNESS > 0 for relative-velocity target damping.\n");
    if (enable_ib_forcing)
    {
        Pointer<IBStandardForceGen> force_gen = new IBStandardForceGen();
        ib_method->registerIBLagrangianForceFunction(force_gen);
    }
    else
    {
        if (target_stiffness != 0.0 || target_damping != 0.0)
            TBOX_ERROR("ENABLE_IB_FORCING=FALSE requires TARGET_STIFFNESS=0 and TARGET_DAMPING=0.\n");
    }

    const IntVector<NDIM>& periodic = geometry->getPeriodicShift();
    std::vector<RobinBcCoefStrategy<NDIM>*> velocity_bc(NDIM, nullptr);
    if (periodic.min() == 0)
    {
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            const std::string suffix = std::to_string(d);
            velocity_bc[d] = new muParserRobinBcCoefs(
                "velocity_bc_" + suffix, app->getComponentDatabase("VelocityBcCoefs_" + suffix), geometry);
        }
        ns->registerPhysicalBoundaryConditions(velocity_bc);

        // INSStaggeredVelocityBcCoef instances are constructed with the
        // integrator's default wall conditions. IBTK regrids at the start of
        // advanceHierarchy(), before preprocessIntegrateHierarchy() refreshes
        // those wrappers, so explicitly install the registered conditions
        // before initializePatchHierarchy() can trigger a coarse-to-fine fill.
        const auto& velocity_bc_wrappers = ns->getVelocityBoundaryConditions();
        for (unsigned int d = 0; d < NDIM; ++d)
        {
            auto* velocity_bc_wrapper = dynamic_cast<INSStaggeredVelocityBcCoef*>(velocity_bc_wrappers[d]);
            if (!velocity_bc_wrapper)
                TBOX_ERROR("Expected INSStaggeredVelocityBcCoef while installing physical BCs before hierarchy init.\n");
            velocity_bc_wrapper->setPhysicalBcCoefs(velocity_bc);
        }
    }

    Pointer<VisItDataWriter<NDIM>> visit_writer = app->getVisItDataWriter();
    Pointer<LSiloDataWriter> silo_writer = app->getLSiloDataWriter();
    if (uses_visit)
    {
        initializer->registerLSiloDataWriter(silo_writer);
        integrator->registerVisItDataWriter(visit_writer);
        ib_method->registerLSiloDataWriter(silo_writer);
    }

    integrator->initializePatchHierarchy(hierarchy, gridding);

    const double rho = db->getDouble("RHO");
    const double mu = db->getDouble("MU");
    Pointer<IBHydrodynamicForceEvaluator> hydro =
        new IBHydrodynamicForceEvaluator("WingHydrodynamicForce", rho, mu, integrator->getIntegratorTime(), true);
    std::array<Eigen::Vector3d, 2> lower, upper;
    for (int sid = 0; sid < 2; ++sid)
    {
        Pointer<Database> cv_db = db->getDatabase("ControlVolume_" + std::to_string(sid));
        cv_db->getDoubleArray("lower_left_corner", lower[sid].data(), NDIM);
        cv_db->getDoubleArray("upper_right_corner", upper[sid].data(), NDIM);
        hydro->registerStructure(lower[sid], upper[sid], hierarchy, Eigen::Vector3d::Zero(), sid);
        hydro->setTorqueOrigin(Eigen::Vector3d::Zero(), sid);
        if (uses_visit) hydro->registerStructurePlotData(visit_writer, hierarchy, sid);
    }

    VariableDatabase<NDIM>* variable_db = VariableDatabase<NDIM>::getDatabase();
    const int u_idx = variable_db->mapVariableAndContextToIndex(ns->getVelocityVariable(), ns->getCurrentContext());
    const int p_idx = variable_db->mapVariableAndContextToIndex(ns->getPressureVariable(), ns->getCurrentContext());
    Pointer<CellVariable<NDIM, double>> pressure_diag_var =
        new CellVariable<NDIM, double>("DomainMomentumPressure", 1);
    Pointer<VariableContext> pressure_diag_context = variable_db->getContext("DomainMomentumPressureContext");
    const int pressure_diag_idx = variable_db->registerVariableAndContext(pressure_diag_var,
                                                                          pressure_diag_context,
                                                                          IntVector<NDIM>(1));
    Pointer<SideVariable<NDIM, double>> velocity_diag_var =
        new SideVariable<NDIM, double>("DomainMomentumVelocity", 1);
    Pointer<VariableContext> velocity_diag_context = variable_db->getContext("DomainMomentumVelocityContext");
    const int velocity_diag_idx = variable_db->registerVariableAndContext(velocity_diag_var,
                                                                          velocity_diag_context,
                                                                          IntVector<NDIM>(1));
    const bool verification_diagnostics = db->getBoolWithDefault("ENABLE_VERIFICATION_DIAGNOSTICS", false);
    const bool domain_momentum_ledger =
        verification_diagnostics && db->getBoolWithDefault("ENABLE_DOMAIN_MOMENTUM_LEDGER", true);
    const bool domain_momentum_velocity_ghost_fill =
        db->getBoolWithDefault("ENABLE_MOMENTUM_VELOCITY_GHOST_FILL", true);
    const bool debug_domain_momentum_ledger =
        db->getBoolWithDefault("DEBUG_DOMAIN_MOMENTUM_LEDGER", false);
    const bool flow_state_diagnostics = db->getBoolWithDefault("ENABLE_FLOW_STATE_DIAGNOSTICS", false);
    // IBAMR 0.19's getVelocityDivergenceVariable() returns the source variable
    // in this release. Retrieve the registered plot variable by its internal
    // name instead and compute it through the solver's normal plot setup path.
    Pointer<Variable<NDIM>> div_var = variable_db->getVariable("INSStaggeredHierarchyIntegrator::Div_U");
    const int div_idx = variable_db->mapVariableAndContextToIndex(div_var, ns->getCurrentContext());
    Pointer<HierarchyMathOps> hierarchy_math_ops = integrator->getHierarchyMathOps();
    SAMRAI::math::HierarchyCellDataOpsReal<NDIM, double> cell_ops(hierarchy);
    int lagged_debug_velocity_idx = -1;
    Pointer<SideVariable<NDIM, double>> lagged_debug_velocity_var;
    if (verification_diagnostics)
    {
        lagged_debug_velocity_var = new SideVariable<NDIM, double>("LaggedMomentumAudit::velocity", 1);
        Pointer<VariableContext> lagged_debug_context = variable_db->getContext("LaggedMomentumAudit::context");
        lagged_debug_velocity_idx = variable_db->registerVariableAndContext(
            lagged_debug_velocity_var, lagged_debug_context, IntVector<NDIM>(1));
        for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
        {
            hierarchy->getPatchLevel(ln)->allocatePatchData(lagged_debug_velocity_idx);
        }
    }

    std::ofstream history;
    std::ofstream momentum_budget;
    std::ofstream flow_state_history;
    std::ofstream flow_state_by_level;
    std::ofstream control_volume_stress;
    std::ofstream control_volume_momentum_trace;
    std::ofstream control_volume_face_audit;
    std::ofstream hierarchy_patch_layout;
    FlowStateDiagnosticContext flow_state_context;
    flow_state_context.hierarchy = hierarchy;
    flow_state_context.integrator = integrator;
    flow_state_context.velocity_idx = u_idx;
    flow_state_context.pressure_idx = p_idx;
    flow_state_context.cell_weight_idx = hierarchy_math_ops->getCellWeightPatchDescriptorIndex();
    flow_state_context.force_evaluator = hydro;
    flow_state_context.velocity_bc = velocity_bc;
    // Refresh the force evaluator's old-state control-volume momentum after
    // every hierarchy regrid/synchronization, even when diagnostics are off.
    flow_state_context.refresh_lagged_momentum_after_regrid = true;
    if (IBTK_MPI::getRank() == 0)
    {
        history.open("force_history.csv");
        history << "time_s,dt_s,realized_velocity_cfl,target_error_rms_m,target_error_max_m,minimum_surface_domain_margin_m,no_slip_velocity_rms_m_s,no_slip_velocity_max_m_s,no_slip_velocity_rms_over_Uref,no_slip_velocity_max_over_Uref,divergence_rms_s_inv,divergence_max_s_inv,divergence_surface_band_width_m,divergence_surface_band_rms_s_inv,divergence_surface_band_max_s_inv,Fx_N,Fy_N,Fz_N,Mx_Nm,My_Nm,Mz_Nm,power_W\n";
        history << std::setprecision(16);
        if (verification_diagnostics)
        {
            momentum_budget.open("momentum_budget.csv");
            momentum_budget << "time_s,dt_s,dP_fluid_dt_x_N,dP_fluid_dt_y_N,dP_fluid_dt_z_N,"
                               "force_on_body_x_N,force_on_body_y_N,force_on_body_z_N,"
                               "pressure_traction_x_N,pressure_traction_y_N,pressure_traction_z_N,"
                               "advective_momentum_flux_x_N,advective_momentum_flux_y_N,advective_momentum_flux_z_N,"
                               "viscous_traction_x_N,viscous_traction_y_N,viscous_traction_z_N,"
                               "signed_residual_x_N,signed_residual_y_N,signed_residual_z_N,"
                               "normalized_residual,refined_physical_boundary\n";
            momentum_budget << std::setprecision(16);
            control_volume_stress.open("control_volume_stress.csv");
            control_volume_stress << "time_s,structure_id,ibamr_force_x_N,ibamr_force_y_N,ibamr_force_z_N,"
                                     "box_momentum_rate_x_N,box_momentum_rate_y_N,box_momentum_rate_z_N,"
                                     "P_box_current_x_kg_m_s,P_box_current_y_kg_m_s,P_box_current_z_kg_m_s,"
                                     "P_box_new_x_kg_m_s,P_box_new_y_kg_m_s,P_box_new_z_kg_m_s,"
                                     "box_lower_current_x_m,box_lower_current_y_m,box_lower_current_z_m,"
                                     "box_upper_current_x_m,box_upper_current_y_m,box_upper_current_z_m,"
                                     "box_lower_new_x_m,box_lower_new_y_m,box_lower_new_z_m,"
                                     "box_upper_new_x_m,box_upper_new_y_m,box_upper_new_z_m,"
                                     "ibamr_implied_surface_flux_x_N,ibamr_implied_surface_flux_y_N,ibamr_implied_surface_flux_z_N,"
                                     "coarse_surface_pressure_x_N,coarse_surface_pressure_y_N,coarse_surface_pressure_z_N,"
                                     "coarse_surface_advection_x_N,coarse_surface_advection_y_N,coarse_surface_advection_z_N,"
                                     "coarse_surface_viscous_x_N,coarse_surface_viscous_y_N,coarse_surface_viscous_z_N,"
                                     "coarse_surface_total_x_N,coarse_surface_total_y_N,coarse_surface_total_z_N,"
                                     "coarse_minus_ibamr_x_N,coarse_minus_ibamr_y_N,coarse_minus_ibamr_z_N\n";
            control_volume_stress << std::setprecision(16);
            control_volume_momentum_trace.open("control_volume_momentum_trace.csv");
            control_volume_momentum_trace << "time_s,stage,structure_id,P_composite_x_kg_m_s,P_composite_y_kg_m_s,P_composite_z_kg_m_s\n";
            control_volume_momentum_trace << std::setprecision(16);
            flow_state_context.momentum_trace = &control_volume_momentum_trace;
            control_volume_face_audit.open("control_volume_face_audit.csv");
            control_volume_face_audit << "time_s,stage,structure_id,component,interior_momentum_kg_m_s,lower_face_momentum_kg_m_s,upper_face_momentum_kg_m_s,interior_weighted_volume_m3,lower_face_weighted_volume_m3,upper_face_weighted_volume_m3,total_momentum_kg_m_s\n";
            control_volume_face_audit << std::setprecision(16);
            hierarchy_patch_layout.open("hierarchy_patch_layout.csv");
            hierarchy_patch_layout << "stage,time_s,level,patch_id,x_lower,x_upper,y_lower,y_upper,z_lower,z_upper\n";
            write_patch_layout(hierarchy_patch_layout, "before_advance_regrid", integrator->getIntegratorTime(), hierarchy);
        }
        if (flow_state_diagnostics)
        {
            flow_state_history.open("flow_state_summary.csv");
            flow_state_history << "stage,time_s,composite_volume_m3,mean_u_x_m_s,mean_u_y_m_s,mean_u_z_m_s,"
                                  "rms_u_x_m_s,rms_u_y_m_s,rms_u_z_m_s,min_u_x_m_s,min_u_y_m_s,min_u_z_m_s,"
                                  "max_u_x_m_s,max_u_y_m_s,max_u_z_m_s,mean_pressure_Pa,rms_pressure_Pa,"
                                  "min_pressure_Pa,max_pressure_Pa,divergence_rms_s_inv,divergence_max_s_inv\n";
            flow_state_history << std::setprecision(16);
            flow_state_by_level.open("flow_state_by_level.csv");
            flow_state_by_level << "stage,time_s,level,composite_volume_m3,mean_u_x_m_s,mean_u_y_m_s,mean_u_z_m_s,"
                                   "rms_u_x_m_s,rms_u_y_m_s,rms_u_z_m_s,min_u_x_m_s,min_u_y_m_s,min_u_z_m_s,"
                                   "max_u_x_m_s,max_u_y_m_s,max_u_z_m_s,mean_pressure_Pa,rms_pressure_Pa,"
                                   "divergence_rms_s_inv,divergence_max_s_inv\n";
            flow_state_by_level << std::setprecision(16);
            flow_state_context.per_level_output = &flow_state_by_level;
        }
    }

    if (flow_state_diagnostics || verification_diagnostics)
    {
        if (flow_state_diagnostics)
        {
            flow_state_context.output = &flow_state_history;
            const FlowStateSummary initial_summary = summarize_flow_state(hierarchy, u_idx, p_idx,
                                                                           flow_state_context.cell_weight_idx);
            if (IBTK_MPI::getRank() == 0)
                write_flow_state_summary(flow_state_history, "post_initialization", integrator->getIntegratorTime(),
                                         initial_summary);
            write_per_level_flow_state(flow_state_by_level,
                                       "post_initialization",
                                       integrator->getIntegratorTime(),
                                       hierarchy,
                                       u_idx,
                                       p_idx,
                                       flow_state_context.cell_weight_idx);
        }
    }
    integrator->registerRegridHierarchyCallback(flow_state_regrid_callback, &flow_state_context);
    g_regrid_projection_diagnostics = &flow_state_context;

    int iteration = integrator->getIntegratorStep();
    double time = integrator->getIntegratorTime();
    double dt = 0.0;
    const double end_time = integrator->getEndTime();
    const long total_cycles = static_cast<long>(std::llround(end_time * kFrequencyHz));
    long last_snapshot_cycle = -1;
    int last_snapshot_bin = -1;
    while (!MathUtilities<double>::equalEps(time, end_time) && integrator->stepsRemaining())
    {
        iteration = integrator->getIntegratorStep();
        time = integrator->getIntegratorTime();
        dt = integrator->getMaximumTimeStepSize();
        if (g_motion_mode == "hover" && g_kinematics_scale != 0.0)
        {
            const double coarse_dx = geometry->getDx()[0];
            const double finest_dx = coarse_dx / std::ldexp(1.0, hierarchy->getFinestLevelNumber());
            const double max_surface_speed = g_kinematics_scale * 2.0 * M_PI * kFrequencyHz *
                                             (kPhiAmplitude + kAlphaAmplitude) * kWingRadius * 1.25;
            dt = std::min(dt, g_kinematic_cfl * finest_dx / max_surface_speed);
        }
        const double new_time = time + dt;

        Eigen::Vector3d domain_momentum_old = Eigen::Vector3d::Zero();
        if (domain_momentum_ledger)
        {
            const double* const domain_lower = geometry->getXLower();
            const double* const domain_upper = geometry->getXUpper();
            const Eigen::Vector3d lower_corner(domain_lower[0], domain_lower[1], domain_lower[2]);
            const Eigen::Vector3d upper_corner(domain_upper[0], domain_upper[1], domain_upper[2]);
            domain_momentum_old = integrate_composite_box_momentum(
                hierarchy,
                u_idx,
                hierarchy_math_ops->getCellWeightPatchDescriptorIndex(),
                lower_corner,
                upper_corner,
                rho);
        }

        // IBExplicitHierarchyIntegrator's forward-Euler Lagrangian predictor
        // evaluates the target-point force at the midpoint stage. Advance the
        // prescribed target to the same stage, then finish its kinematic
        // update after the fluid step for the next cycle and diagnostics.
        const double target_force_time = time + target_force_time_fraction * dt;
        if (target_force_time_fraction > 0.0)
            transform_targets(hierarchy, ib_method->getLDataManager(), time, target_force_time);
        shift_target_references_for_relative_damping(
            hierarchy, ib_method->getLDataManager(), target_force_time, true);
        for (int sid = 0; sid < 2; ++sid) hydro->updateStructureDomain(Eigen::Vector3d::Zero(), dt, hierarchy, sid);
        if (verification_diagnostics && hierarchy->getFinestLevelNumber() == 0)
        {
            HierarchyDataOpsManager<NDIM>* data_ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
            Pointer<HierarchyDataOpsReal<NDIM, double>> lagged_data_ops =
                data_ops_manager->getOperationsDouble(lagged_debug_velocity_var, hierarchy, true);
            lagged_data_ops->copyData(lagged_debug_velocity_idx, u_idx, true);
            for (int sid = 0; sid < 2; ++sid)
            {
                const FaceMomentumAudit copy_audit = audit_single_level_face_momentum(
                    hierarchy, lagged_debug_velocity_idx, lower[sid], upper[sid], rho);
                if (IBTK_MPI::getRank() == 0)
                    for (int axis = 0; axis < NDIM; ++axis)
                        control_volume_face_audit << time << ",ibamr_copy_only_replica," << sid << ',' << axis << ','
                                                 << copy_audit.by_face_class[axis][0] << ','
                                                 << copy_audit.by_face_class[axis][1] << ','
                                                 << copy_audit.by_face_class[axis][2] << ','
                                                 << copy_audit.weighted_volume_by_face_class[axis][0] << ','
                                                 << copy_audit.weighted_volume_by_face_class[axis][1] << ','
                                                 << copy_audit.weighted_volume_by_face_class[axis][2] << ','
                                                 << copy_audit.momentum[axis] << '\n';
            }
            using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<InterpolationTransactionComponent> transaction(1);
            transaction[0] = InterpolationTransactionComponent(
                lagged_debug_velocity_idx,
                u_idx,
                "CONSERVATIVE_LINEAR_REFINE",
                true,
                "CUBIC_COARSEN",
                "LINEAR",
                false,
                ns->getVelocityBoundaryConditions(),
                Pointer<VariableFillPattern<NDIM>>(nullptr));
            Pointer<HierarchyGhostCellInterpolation> ghost_fill = new HierarchyGhostCellInterpolation();
            ghost_fill->initializeOperatorState(transaction, hierarchy);
            ghost_fill->setHomogeneousBc(false);
            ghost_fill->fillData(time);
            for (int sid = 0; sid < 2; ++sid)
            {
                const Eigen::Vector3d momentum = integrate_composite_box_momentum(
                    hierarchy, u_idx, hierarchy_math_ops->getCellWeightPatchDescriptorIndex(), lower[sid], upper[sid], rho);
                const FaceMomentumAudit face_audit = audit_single_level_face_momentum(
                    hierarchy, u_idx, lower[sid], upper[sid], rho);
                const FaceMomentumAudit fill_path_audit = audit_single_level_face_momentum(
                    hierarchy, lagged_debug_velocity_idx, lower[sid], upper[sid], rho);
                if (IBTK_MPI::getRank() == 0)
                {
                    control_volume_momentum_trace << time << ",pre_advance_before_lagged_integral," << sid << ','
                                                  << momentum[0] << ',' << momentum[1] << ',' << momentum[2] << '\n';
                    for (int axis = 0; axis < NDIM; ++axis)
                    {
                        auto write_audit = [&](const char* stage, const FaceMomentumAudit& audit)
                        {
                            control_volume_face_audit << time << ',' << stage << ',' << sid << ',' << axis << ','
                                                     << audit.by_face_class[axis][0] << ','
                                                     << audit.by_face_class[axis][1] << ','
                                                     << audit.by_face_class[axis][2] << ','
                                                     << audit.weighted_volume_by_face_class[axis][0] << ','
                                                     << audit.weighted_volume_by_face_class[axis][1] << ','
                                                     << audit.weighted_volume_by_face_class[axis][2] << ','
                                                     << audit.momentum[axis] << '\n';
                        };
                        write_audit("direct_eulerian", face_audit);
                        write_audit("ibamr_fill_path_replica", fill_path_audit);
                    }
                }
            }
            // Reapply the exact same scratch fill to the unchanged old Eulerian
            // state at the new-state timestamp. This separates a fill-time/BC
            // effect from a change in the velocity field during advance.
            lagged_data_ops->copyData(lagged_debug_velocity_idx, u_idx, true);
            ghost_fill->fillData(new_time);
            for (int sid = 0; sid < 2; ++sid)
            {
                const FaceMomentumAudit new_time_fill_audit = audit_single_level_face_momentum(
                    hierarchy, lagged_debug_velocity_idx, lower[sid], upper[sid], rho);
                if (IBTK_MPI::getRank() == 0)
                    for (int axis = 0; axis < NDIM; ++axis)
                        control_volume_face_audit << time << ",ibamr_preadvance_newtime_fill," << sid << ',' << axis << ','
                                                 << new_time_fill_audit.by_face_class[axis][0] << ','
                                                 << new_time_fill_audit.by_face_class[axis][1] << ','
                                                 << new_time_fill_audit.by_face_class[axis][2] << ','
                                                 << new_time_fill_audit.weighted_volume_by_face_class[axis][0] << ','
                                                 << new_time_fill_audit.weighted_volume_by_face_class[axis][1] << ','
                                                 << new_time_fill_audit.weighted_volume_by_face_class[axis][2] << ','
                                                 << new_time_fill_audit.momentum[axis] << '\n';
            }
            if (IBTK_MPI::getRank() == 0)
            {
                control_volume_momentum_trace.flush();
                control_volume_face_audit.flush();
            }
        }
        write_patch_layout(hierarchy_patch_layout, "before_lagged_integral", time, hierarchy);
        hydro->computeLaggedMomentumIntegral(u_idx, hierarchy, ns->getVelocityBoundaryConditions());
        integrator->advanceHierarchy(dt);
        shift_target_references_for_relative_damping(
            hierarchy, ib_method->getLDataManager(), target_force_time, false);
        if (target_force_time_fraction < 1.0)
            transform_targets(hierarchy, ib_method->getLDataManager(), target_force_time, new_time);
        write_patch_layout(hierarchy_patch_layout, "after_advance", new_time, hierarchy);
        if (verification_diagnostics && hierarchy->getFinestLevelNumber() == 0)
        {
            for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
            {
                Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
                if (!level->checkAllocated(lagged_debug_velocity_idx))
                    level->allocatePatchData(lagged_debug_velocity_idx);
            }
            HierarchyDataOpsManager<NDIM>* data_ops_manager = HierarchyDataOpsManager<NDIM>::getManager();
            Pointer<HierarchyDataOpsReal<NDIM, double>> lagged_velocity_ops =
                data_ops_manager->getOperationsDouble(lagged_debug_velocity_var, hierarchy, true);
            lagged_velocity_ops->copyData(lagged_debug_velocity_idx, u_idx, true);
            for (int sid = 0; sid < 2; ++sid)
            {
                const FaceMomentumAudit copy_audit = audit_single_level_face_momentum(
                    hierarchy, lagged_debug_velocity_idx, lower[sid], upper[sid], rho);
                if (IBTK_MPI::getRank() == 0)
                    for (int axis = 0; axis < NDIM; ++axis)
                        control_volume_face_audit << new_time << ",ibamr_new_copy_only_replica," << sid << ',' << axis << ','
                                                 << copy_audit.by_face_class[axis][0] << ','
                                                 << copy_audit.by_face_class[axis][1] << ','
                                                 << copy_audit.by_face_class[axis][2] << ','
                                                 << copy_audit.weighted_volume_by_face_class[axis][0] << ','
                                                 << copy_audit.weighted_volume_by_face_class[axis][1] << ','
                                                 << copy_audit.weighted_volume_by_face_class[axis][2] << ','
                                                 << copy_audit.momentum[axis] << '\n';
            }
            using InterpolationTransactionComponent = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
            std::vector<InterpolationTransactionComponent> velocity_transaction(1);
            velocity_transaction[0] = InterpolationTransactionComponent(
                lagged_debug_velocity_idx,
                u_idx,
                "CONSERVATIVE_LINEAR_REFINE",
                true,
                "CUBIC_COARSEN",
                "LINEAR",
                false,
                ns->getVelocityBoundaryConditions(),
                Pointer<VariableFillPattern<NDIM>>(nullptr));
            Pointer<HierarchyGhostCellInterpolation> velocity_fill = new HierarchyGhostCellInterpolation();
            velocity_fill->initializeOperatorState(velocity_transaction, hierarchy);
            velocity_fill->setHomogeneousBc(false);
            velocity_fill->fillData(new_time);
            for (int sid = 0; sid < 2; ++sid)
            {
                const Eigen::Vector3d momentum = integrate_composite_box_momentum(
                    hierarchy, u_idx, hierarchy_math_ops->getCellWeightPatchDescriptorIndex(), lower[sid], upper[sid], rho);
                const FaceMomentumAudit fill_path_audit = audit_single_level_face_momentum(
                    hierarchy, lagged_debug_velocity_idx, lower[sid], upper[sid], rho);
                if (IBTK_MPI::getRank() == 0)
                {
                    control_volume_momentum_trace << new_time << ",post_advance_before_force_evaluation," << sid << ','
                                                  << momentum[0] << ',' << momentum[1] << ',' << momentum[2] << '\n';
                    for (int axis = 0; axis < NDIM; ++axis)
                        control_volume_face_audit << new_time << ",ibamr_new_fill_path_replica," << sid << ',' << axis << ','
                                                 << fill_path_audit.by_face_class[axis][0] << ','
                                                 << fill_path_audit.by_face_class[axis][1] << ','
                                                 << fill_path_audit.by_face_class[axis][2] << ','
                                                 << fill_path_audit.weighted_volume_by_face_class[axis][0] << ','
                                                 << fill_path_audit.weighted_volume_by_face_class[axis][1] << ','
                                                 << fill_path_audit.weighted_volume_by_face_class[axis][2] << ','
                                                 << fill_path_audit.momentum[axis] << '\n';
                }
            }
            if (IBTK_MPI::getRank() == 0)
            {
                control_volume_momentum_trace.flush();
                control_volume_face_audit.flush();
            }
        }
        if (flow_state_diagnostics)
        {
            const FlowStateSummary post_advance = summarize_flow_state(hierarchy, u_idx, p_idx,
                                                                        flow_state_context.cell_weight_idx);
            if (IBTK_MPI::getRank() == 0)
                write_flow_state_summary(flow_state_history, "post_advance", new_time, post_advance);
            write_per_level_flow_state(flow_state_by_level,
                                       "post_advance",
                                       new_time,
                                       hierarchy,
                                       u_idx,
                                       p_idx,
                                       flow_state_context.cell_weight_idx);
        }
        const std::array<double, 2> tracking_error = target_tracking_error(hierarchy, ib_method->getLDataManager());
        const double velocity_cfl = realized_velocity_cfl(hierarchy, u_idx, dt);
        const double domain_margin = minimum_surface_domain_margin(hierarchy, ib_method->getLDataManager(), geometry);
        SurfaceVelocityMetrics no_slip;
        double divergence_rms = 0.0;
        double divergence_max = 0.0;
        DivergenceBandMetrics divergence_band;
        if (verification_diagnostics)
        {
            no_slip = surface_velocity_error(hierarchy, ib_method->getLDataManager(), new_time, dt);
            integrator->setupPlotData();
            const int cell_weight_idx = hierarchy_math_ops->getCellWeightPatchDescriptorIndex();
            divergence_rms = cell_ops.RMSNorm(div_idx, cell_weight_idx);
            divergence_max = cell_ops.maxNorm(div_idx, cell_weight_idx);
            divergence_band = divergence_band_metrics(
                hierarchy, ib_method->getLDataManager(), div_idx, cell_weight_idx);
        }

        for (int sid = 0; sid < 2; ++sid)
            hydro->updateStructureMomentum(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), sid);
        hydro->computeHydrodynamicForce(u_idx,
                                       p_idx,
                                       -1,
                                       hierarchy,
                                       dt,
                                       ns->getVelocityBoundaryConditions(),
                                       ns->getPressureBoundaryConditions());

        Eigen::Vector3d net_force = Eigen::Vector3d::Zero();
        Eigen::Vector3d net_torque = Eigen::Vector3d::Zero();
        for (int sid = 0; sid < 2; ++sid)
        {
            const auto& result = hydro->getHydrodynamicForceObject(sid);
            net_force += result.F_new;
            net_torque += result.T_new;
            if (uses_visit) hydro->updateStructurePlotData(hierarchy, sid);
        }
        double aero_power = 0.0;
        for (int sid = 0; sid < 2; ++sid)
        {
            const int side = sid == 0 ? 1 : -1;
            const auto& result = hydro->getHydrodynamicForceObject(sid);
            aero_power -= result.T_new.dot(angular_velocity(new_time, dt, side));
        }
        DomainMomentumFluxes domain_fluxes;
        Eigen::Vector3d domain_dP_dt = Eigen::Vector3d::Zero();
        if (domain_momentum_ledger)
        {
            if (debug_domain_momentum_ledger && IBTK_MPI::getRank() == 0)
                std::cerr << "MOMENTUM_LEDGER_TRACE begin t=" << new_time << std::endl;
            const double* const domain_lower = geometry->getXLower();
            const double* const domain_upper = geometry->getXUpper();
            const Eigen::Vector3d lower_corner(domain_lower[0], domain_lower[1], domain_lower[2]);
            const Eigen::Vector3d upper_corner(domain_upper[0], domain_upper[1], domain_upper[2]);
            const Eigen::Vector3d domain_momentum_new = integrate_composite_box_momentum(
                hierarchy,
                u_idx,
                hierarchy_math_ops->getCellWeightPatchDescriptorIndex(),
                lower_corner,
                upper_corner,
                rho);
            domain_dP_dt = (domain_momentum_new - domain_momentum_old) / dt;
            if (debug_domain_momentum_ledger && IBTK_MPI::getRank() == 0)
                std::cerr << "MOMENTUM_LEDGER_TRACE momentum_rate_done" << std::endl;
            int flux_velocity_idx = u_idx;
            if (domain_momentum_velocity_ghost_fill)
            {
                fill_velocity_boundary_ghosts(hierarchy,
                                              velocity_diag_var,
                                              u_idx,
                                              velocity_diag_idx,
                                              ns->getVelocityBoundaryConditions(),
                                              new_time);
                flux_velocity_idx = velocity_diag_idx;
            }
            if (debug_domain_momentum_ledger && IBTK_MPI::getRank() == 0)
                std::cerr << "MOMENTUM_LEDGER_TRACE velocity_fill_done" << std::endl;
            fill_pressure_boundary_ghosts(hierarchy,
                                          pressure_diag_var,
                                          p_idx,
                                          pressure_diag_idx,
                                          u_idx,
                                          ns,
                                          new_time);
            if (debug_domain_momentum_ledger && IBTK_MPI::getRank() == 0)
                std::cerr << "MOMENTUM_LEDGER_TRACE pressure_fill_done" << std::endl;
            domain_fluxes = integrate_domain_momentum_fluxes(
                hierarchy, geometry, flux_velocity_idx, pressure_diag_idx, rho, mu);
            if (debug_domain_momentum_ledger && IBTK_MPI::getRank() == 0)
                std::cerr << "MOMENTUM_LEDGER_TRACE outer_flux_done" << std::endl;
            for (int sid = 0; sid < 2; ++sid)
            {
                if (debug_domain_momentum_ledger && IBTK_MPI::getRank() == 0)
                    std::cerr << "MOMENTUM_LEDGER_TRACE cv_begin sid=" << sid << std::endl;
                const auto& force_result = hydro->getHydrodynamicForceObject(sid);
                const SurfaceFluxes cv_flux = integrate_level_zero_control_surface(
                    hierarchy,
                    lower[sid],
                    upper[sid],
                    flux_velocity_idx,
                    pressure_diag_idx,
                    rho,
                    mu,
                    debug_domain_momentum_ledger,
                    new_time,
                    sid);
                if (debug_domain_momentum_ledger && IBTK_MPI::getRank() == 0)
                    std::cerr << "MOMENTUM_LEDGER_TRACE cv_done sid=" << sid << std::endl;
                const Eigen::Vector3d box_momentum_rate =
                    (force_result.P_box_new - force_result.P_box_current) / dt;
                const Eigen::Vector3d structure_momentum_rate =
                    (force_result.P_new - force_result.P_current) / dt;
                const Eigen::Vector3d ibamr_surface_flux =
                    force_result.F_new + box_momentum_rate - structure_momentum_rate;
                const Eigen::Vector3d independent_surface_flux =
                    cv_flux.pressure + cv_flux.advection + cv_flux.viscous;
                const Eigen::Vector3d discrepancy = independent_surface_flux - ibamr_surface_flux;
                control_volume_stress << new_time << ',' << sid << ','
                                      << force_result.F_new[0] << ',' << force_result.F_new[1] << ',' << force_result.F_new[2] << ','
                                      << box_momentum_rate[0] << ',' << box_momentum_rate[1] << ',' << box_momentum_rate[2] << ','
                                      << force_result.P_box_current[0] << ',' << force_result.P_box_current[1] << ','
                                      << force_result.P_box_current[2] << ','
                                      << force_result.P_box_new[0] << ',' << force_result.P_box_new[1] << ','
                                      << force_result.P_box_new[2] << ','
                                      << force_result.box_X_lower_current[0] << ',' << force_result.box_X_lower_current[1] << ','
                                      << force_result.box_X_lower_current[2] << ','
                                      << force_result.box_X_upper_current[0] << ',' << force_result.box_X_upper_current[1] << ','
                                      << force_result.box_X_upper_current[2] << ','
                                      << force_result.box_X_lower_new[0] << ',' << force_result.box_X_lower_new[1] << ','
                                      << force_result.box_X_lower_new[2] << ','
                                      << force_result.box_X_upper_new[0] << ',' << force_result.box_X_upper_new[1] << ','
                                      << force_result.box_X_upper_new[2] << ','
                                      << ibamr_surface_flux[0] << ',' << ibamr_surface_flux[1] << ',' << ibamr_surface_flux[2] << ','
                                      << cv_flux.pressure[0] << ',' << cv_flux.pressure[1] << ',' << cv_flux.pressure[2] << ','
                                      << cv_flux.advection[0] << ',' << cv_flux.advection[1] << ',' << cv_flux.advection[2] << ','
                                      << cv_flux.viscous[0] << ',' << cv_flux.viscous[1] << ',' << cv_flux.viscous[2] << ','
                                      << independent_surface_flux[0] << ',' << independent_surface_flux[1] << ','
                                      << independent_surface_flux[2] << ',' << discrepancy[0] << ',' << discrepancy[1] << ','
                                      << discrepancy[2] << '\n';
            }
            if (control_volume_stress.is_open()) control_volume_stress.flush();
            const Eigen::Vector3d outer = domain_fluxes.pressure + domain_fluxes.advection + domain_fluxes.viscous;
            const Eigen::Vector3d residual = domain_dP_dt + net_force - outer;
            const double scale = domain_dP_dt.norm() + net_force.norm() + domain_fluxes.pressure.norm() +
                                 domain_fluxes.advection.norm() + domain_fluxes.viscous.norm();
            momentum_budget << new_time << ',' << dt << ','
                            << domain_dP_dt[0] << ',' << domain_dP_dt[1] << ',' << domain_dP_dt[2] << ','
                            << net_force[0] << ',' << net_force[1] << ',' << net_force[2] << ','
                            << domain_fluxes.pressure[0] << ',' << domain_fluxes.pressure[1] << ',' << domain_fluxes.pressure[2] << ','
                            << domain_fluxes.advection[0] << ',' << domain_fluxes.advection[1] << ',' << domain_fluxes.advection[2] << ','
                            << domain_fluxes.viscous[0] << ',' << domain_fluxes.viscous[1] << ',' << domain_fluxes.viscous[2] << ','
                            << residual[0] << ',' << residual[1] << ',' << residual[2] << ','
                            << (scale > 0.0 ? residual.norm() / scale : 0.0) << ','
                            << (domain_fluxes.refined_boundary ? 1 : 0) << '\n';
            if (momentum_budget.is_open()) momentum_budget.flush();
        }
        if (IBTK_MPI::getRank() == 0)
        {
            history << new_time << ',' << dt << ',' << velocity_cfl << ','
                    << tracking_error[0] << ',' << tracking_error[1] << ','
                    << domain_margin << ','
                    << (verification_diagnostics ? no_slip.rms : 0.0) << ','
                    << (verification_diagnostics ? no_slip.maximum : 0.0) << ','
                    << (verification_diagnostics ? no_slip.normalized_rms : 0.0) << ','
                    << (verification_diagnostics ? no_slip.normalized_maximum : 0.0) << ','
                    << divergence_rms << ',' << divergence_max << ','
                    << divergence_band.width << ',' << divergence_band.rms << ',' << divergence_band.maximum << ','
                    << net_force[0] << ',' << net_force[1] << ',' << net_force[2] << ','
                    << net_torque[0] << ',' << net_torque[1] << ',' << net_torque[2] << ',' << aero_power << '\n';
            if (verification_diagnostics) history.flush();
        }
        hydro->postprocessIntegrateData(time, new_time);

        time = new_time;
        ++iteration;
        const bool last_step = !integrator->stepsRemaining();
        const long cycle = static_cast<long>(std::floor(time * kFrequencyHz));
        const double phase = time * kFrequencyHz - static_cast<double>(cycle);
        const int phase_bin = std::min(7, static_cast<int>(std::floor(8.0 * phase)));
        // Retain full Eulerian fields only for eight phase points in the final
        // three prescribed cycles. Derived plane data is a separate post-step.
        if (uses_visit && cycle >= std::max(0L, total_cycles - 3) && cycle < total_cycles &&
            (cycle != last_snapshot_cycle || phase_bin != last_snapshot_bin))
        {
            integrator->setupPlotData();
            visit_writer->writePlotData(hierarchy, iteration, time);
            silo_writer->writePlotData(iteration, time);
            last_snapshot_cycle = cycle;
            last_snapshot_bin = phase_bin;
        }
        if (dump_restart && restart_interval > 0 && (iteration % restart_interval == 0 || last_step))
            RestartManager::getManager()->writeRestartFile(restart_dir, iteration);
        if (dump_timer && timer_interval > 0 && (iteration % timer_interval == 0 || last_step))
            TimerManager::getManager()->print(plog);
    }

    if (IBTK_MPI::getRank() == 0)
    {
        history.close();
        if (momentum_budget.is_open()) momentum_budget.close();
        if (flow_state_history.is_open()) flow_state_history.close();
        if (flow_state_by_level.is_open()) flow_state_by_level.close();
        if (control_volume_stress.is_open()) control_volume_stress.close();
        if (control_volume_momentum_trace.is_open()) control_volume_momentum_trace.close();
        if (control_volume_face_audit.is_open()) control_volume_face_audit.close();
        if (hierarchy_patch_layout.is_open()) hierarchy_patch_layout.close();
    }
    variable_db->removePatchDataIndex(pressure_diag_idx);
    for (RobinBcCoefStrategy<NDIM>* bc : velocity_bc) delete bc;
    return 0;
}
