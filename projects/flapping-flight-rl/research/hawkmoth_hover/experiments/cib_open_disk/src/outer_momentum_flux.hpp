#pragma once

#include <SAMRAI_config.h>

#include <ibtk/IBTK_MPI.h>

#include <Eigen/Core>

#include <SideData.h>

#include <ibamr/app_namespaces.h>

#include <array>

/**
 * Signed outer-boundary terms in the domain-integrated incompressible
 * momentum balance. Face columns use 2*axis + side, where side=0 is the lower
 * face and side=1 is the upper face. Pressure is -p n, advection is
 * -rho*u*(u dot n), and viscous traction is mu*(grad(u)+grad(u)^T)*n.
 */
struct OuterMomentumFlux
{
    Eigen::Vector3d pressure = Eigen::Vector3d::Zero();
    Eigen::Vector3d viscous = Eigen::Vector3d::Zero();
    Eigen::Vector3d advection = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, NDIM, 2 * NDIM> pressure_by_face =
        Eigen::Matrix<double, NDIM, 2 * NDIM>::Zero();
    Eigen::Matrix<double, NDIM, 2 * NDIM> viscous_by_face =
        Eigen::Matrix<double, NDIM, 2 * NDIM>::Zero();
    Eigen::Matrix<double, NDIM, 2 * NDIM> advection_by_face =
        Eigen::Matrix<double, NDIM, 2 * NDIM>::Zero();
};

inline OuterMomentumFlux
integrate_outer_momentum_flux(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                              int velocity_idx,
                              int pressure_idx,
                              const double rho,
                              const double mu)
{
    OuterMomentumFlux result;
    for (int ln = hierarchy->getFinestLevelNumber(); ln >= 0; --ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        const BoxArray<NDIM>& domains = level->getPhysicalDomain();
        if (domains.getNumberOfBoxes() != 1)
            TBOX_ERROR("Outer momentum ledger requires one rectangular physical domain per level.\n");
        const Box<NDIM>& domain = domains[0];
        BoxArray<NDIM> covered;
        if (ln < hierarchy->getFinestLevelNumber())
        {
            Pointer<PatchLevel<NDIM>> fine_level = hierarchy->getPatchLevel(ln + 1);
            covered = fine_level->getBoxes();
            covered.coarsen(fine_level->getRatioToCoarserLevel());
        }
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Box<NDIM>& patch_box = patch->getBox();
            const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
            const double* dx = patch_geom->getDx();
            Pointer<CellData<NDIM, double>> pressure = patch->getPatchData(pressure_idx);
            Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
            for (int axis = 0; axis < NDIM; ++axis)
            {
                const double area = dx[(axis + 1) % NDIM] * dx[(axis + 2) % NDIM];
                for (int upper = 0; upper <= 1; ++upper)
                {
                    const int face_index = 2 * axis + upper;
                    Box<NDIM> cells = domain;
                    const int bc = upper ? domain.upper(axis) : domain.lower(axis);
                    cells.lower(axis) = bc;
                    cells.upper(axis) = bc;
                    cells = cells * patch_box;
                    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
                    normal[axis] = upper ? 1.0 : -1.0;
                    for (Box<NDIM>::Iterator b(cells); b; b++)
                    {
                        const CellIndex<NDIM>& cell = *b;
                        bool is_covered = false;
                        for (int k = 0; k < covered.getNumberOfBoxes(); ++k)
                            is_covered = is_covered || covered[k].contains(cell);
                        if (is_covered) continue;
                        // Reconstruct boundary pressure from the first two
                        // cell-centered values. Averaging the interior value
                        // with a Neumann-filled exterior ghost biases a linear
                        // pressure field by half a cell at physical faces.
                        CellIndex<NDIM> inward = cell;
                        inward(axis) += upper ? -1 : 1;
                        CellIndex<NDIM> outside = cell;
                        outside(axis) += upper ? 1 : -1;
                        const SideIndex<NDIM> face(
                            cell, axis, upper ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower);
                        const double p_face = 1.5 * (*pressure)(cell) - 0.5 * (*pressure)(inward);
                        const Eigen::Vector3d pressure_traction = -p_face * normal * area;
                        result.pressure += pressure_traction;
                        result.pressure_by_face.col(face_index) += pressure_traction;

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
                        const Eigen::Vector3d advection_traction = -rho * normal.dot(u) * u * area;
                        result.advection += advection_traction;
                        result.advection_by_face.col(face_index) += advection_traction;

                        Eigen::Vector3d tau = Eigen::Vector3d::Zero();
                        for (int d = 0; d < NDIM; ++d)
                        {
                            if (d == axis)
                            {
                                const double u_inside = (*velocity)(SideIndex<NDIM>(
                                    cell, axis, upper ? SideIndex<NDIM>::Lower : SideIndex<NDIM>::Upper));
                                tau[d] = 2.0 * mu / dx[axis] * ((*velocity)(face) - u_inside);
                            }
                            else
                            {
                                CellIndex<NDIM> offset(0);
                                offset(d) = 1;
                                const int tangential_loc = upper ? SideIndex<NDIM>::Upper : SideIndex<NDIM>::Lower;
                                const double tangential =
                                    ((*velocity)(SideIndex<NDIM>(cell + offset, axis, tangential_loc)) -
                                     (*velocity)(SideIndex<NDIM>(cell - offset, axis, tangential_loc))) /
                                    (2.0 * dx[d]);
                                const SideIndex<NDIM> u0_index(cell, d, tangential_loc);
                                const SideIndex<NDIM> u1_index(inward, d, tangential_loc);
                                const SideIndex<NDIM> ughost_index(outside, d, tangential_loc);
                                const double u0 = (*velocity)(u0_index);
                                const double u1 = (*velocity)(u1_index);
                                // For the Dirichlet physical-boundary controls,
                                // the ghost is reflected about the boundary
                                // value. These three points lie at 0, dx/2,
                                // and 3dx/2; the resulting outward derivative
                                // is second-order accurate at the face.
                                const double u_boundary =
                                    0.5 * (u0 + (*velocity)(ughost_index));
                                const double normal_gradient =
                                    (8.0 * u_boundary / 3.0 - 3.0 * u0 + u1 / 3.0) / dx[axis];
                                tau[d] = mu * (normal[axis] * tangential + normal_gradient);
                            }
                        }
                        const Eigen::Vector3d viscous_traction = tau * area;
                        result.viscous += viscous_traction;
                        result.viscous_by_face.col(face_index) += viscous_traction;
                    }
                }
            }
        }
    }
    IBTK_MPI::sumReduction(result.pressure.data(), NDIM);
    IBTK_MPI::sumReduction(result.viscous.data(), NDIM);
    IBTK_MPI::sumReduction(result.advection.data(), NDIM);
    IBTK_MPI::sumReduction(result.pressure_by_face.data(), NDIM * 2 * NDIM);
    IBTK_MPI::sumReduction(result.viscous_by_face.data(), NDIM * 2 * NDIM);
    IBTK_MPI::sumReduction(result.advection_by_face.data(), NDIM * 2 * NDIM);
    return result;
}
