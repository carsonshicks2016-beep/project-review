// ---------------------------------------------------------------------
//
// Copyright (c) 2017 - 2026 by the IBAMR developers
// All rights reserved.
//
// This file is part of IBAMR.
//
// IBAMR is free software and is distributed under the 3-clause BSD
// license. The full text of the license can be found in the file
// COPYRIGHT at the top level directory of IBAMR.
//
// ---------------------------------------------------------------------

// Config files
#include <SAMRAI_config.h>

// Headers for basic PETSc functions
#include <petscsys.h>

// Headers for basic SAMRAI objects
#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <LoadBalancer.h>
#include <StandardTagAndInitialize.h>
#include <VariableDatabase.h>
#include <HierarchyCellDataOpsReal.h>
#include <HierarchyDataOpsManager.h>

// Headers for application-specific algorithm/data structure objects
#include <ibamr/CIBMethod.h>
#include <ibamr/CIBMobilitySolver.h>
#include <ibamr/CIBSaddlePointSolver.h>
#include <ibamr/CIBStaggeredStokesSolver.h>
#include <ibamr/INSStaggeredPressureBcCoef.h>
#include <ibamr/DirectMobilitySolver.h>
#include <ibamr/IBExplicitHierarchyIntegrator.h>
#include <ibamr/IBStandardInitializer.h>
#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/KrylovMobilitySolver.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/LData.h>
#include <ibtk/LDataManager.h>
#include <ibtk/LEInteractor.h>
#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IndexUtilities.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>
#include <CellVariable.h>
#include <SideData.h>
#include <SideIterator.h>
#include <SideVariable.h>

#include <boost/multi_array.hpp>

#include <ibamr/app_namespaces.h>

#include "outer_momentum_flux.hpp"

//////////////////////////////////////////////////////////////////////////////

struct StructureCtx
{
    std::string name;
    double rho_excess, R;
    IBTK::Vector F;
    Eigen::Vector3d prescribed_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d prescribed_rotation = Eigen::Vector3d::Zero();

}; // StructureCtx

// Center of mass velocity
void
ConstrainedCOMVel(double /*data_time*/, Eigen::Vector3d& U_com, Eigen::Vector3d& W_com, void* ctx)
{
    const StructureCtx& struct_ctx = *static_cast<const StructureCtx*>(ctx);
    U_com = struct_ctx.prescribed_translation;
    W_com = struct_ctx.prescribed_rotation;

    return;
} // ConstrainedCOMOuterVel

void
NetExternalForceTorque(double /*data_time*/, Eigen::Vector3d& F_ext, Eigen::Vector3d& T_ext, void* ctx)
{
    StructureCtx& struct_ctx = *static_cast<StructureCtx*>(ctx);

    F_ext << struct_ctx.F(0), struct_ctx.F(1), 0.0;
    T_ext << 0.0, 0.0, 0.0;

    return;
} // NetExternalForceTorque

void
ConstrainedNodalVel(Vec /*U_k*/, const RigidDOFVector& /*U*/, const Eigen::Vector3d& /*X_com*/, void* /*ctx*/)
{
    // intentionally left blank
    return;
} // ConstrainedNodalVel

ofstream U_stream;

struct MarkerMetrics
{
    double position_rms = 0.0;
    double position_max = 0.0;
    double slip_rms = 0.0;
    double slip_max = 0.0;
};

MarkerMetrics
measure_markers(LDataManager* manager,
                const int level_number,
                const double elapsed_time,
                const Eigen::Vector3d& target_velocity,
                const double reference_speed,
                const int eulerian_velocity_idx,
                const Pointer<LData>& sampled_fluid_velocity)
{
    MarkerMetrics result;
    Pointer<LData> X_data = manager->getLData(LDataManager::POSN_DATA_NAME, level_number);
    Pointer<LData> X0_data = manager->getLData("X0_unshifted", level_number);
    // Sample the current Eulerian velocity independently at the actual marker
    // locations.  CIB's cached Lagrangian "U" field is a midpoint work vector
    // and is not guaranteed to hold a new-time sample after advanceHierarchy().
    manager->interp(eulerian_velocity_idx,
                    sampled_fluid_velocity,
                    X_data,
                    level_number,
                    std::vector<Pointer<CoarsenSchedule<NDIM>>>(),
                    std::vector<Pointer<RefineSchedule<NDIM>>>(),
                    elapsed_time);
    const double* X = nullptr;
    const double* X0 = nullptr;
    const double* U = nullptr;
    VecGetArrayRead(X_data->getVec(), &X);
    VecGetArrayRead(X0_data->getVec(), &X0);
    VecGetArrayRead(sampled_fluid_velocity->getVec(), &U);

    double local_pos_sq = 0.0, local_slip_sq = 0.0, local_count = 0.0;
    double local_pos_max = 0.0, local_slip_max = 0.0;
    Pointer<LMesh> mesh = manager->getLMesh(level_number);
    for (LNode* node : mesh->getLocalNodes())
    {
        const int i = node->getLocalPETScIndex() * NDIM;
        double pos_sq = 0.0, slip_sq = 0.0;
        for (int d = 0; d < NDIM; ++d)
        {
            const double position_error = X[i + d] - (X0[i + d] + target_velocity[d] * elapsed_time);
            const double velocity_error = U[i + d] - target_velocity[d];
            pos_sq += position_error * position_error;
            slip_sq += velocity_error * velocity_error;
        }
        local_pos_sq += pos_sq;
        local_slip_sq += slip_sq;
        local_pos_max = std::max(local_pos_max, std::sqrt(pos_sq));
        local_slip_max = std::max(local_slip_max, std::sqrt(slip_sq));
        local_count += 1.0;
    }
    VecRestoreArrayRead(sampled_fluid_velocity->getVec(), &U);
    VecRestoreArrayRead(X0_data->getVec(), &X0);
    VecRestoreArrayRead(X_data->getVec(), &X);

    IBTK_MPI::sumReduction(&local_pos_sq, 1);
    IBTK_MPI::sumReduction(&local_slip_sq, 1);
    IBTK_MPI::sumReduction(&local_count, 1);
    IBTK_MPI::maxReduction(&local_pos_max, 1);
    IBTK_MPI::maxReduction(&local_slip_max, 1);
    if (local_count > 0.0)
    {
        result.position_rms = std::sqrt(local_pos_sq / local_count);
        result.slip_rms = std::sqrt(local_slip_sq / local_count);
    }
    result.position_max = local_pos_max;
    result.slip_max = local_slip_max;
    NULL_USE(reference_speed);
    return result;
}

Eigen::Vector3d
integrate_composite_momentum(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                             const int velocity_idx,
                             const int cell_weight_idx,
                             const double rho)
{
    Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
            Pointer<CellData<NDIM, double>> weights = patch->getPatchData(cell_weight_idx);
            for (Box<NDIM>::Iterator ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& cell = *ci;
                const double volume = (*weights)(cell);
                if (volume <= 0.0) continue;
                for (int d = 0; d < NDIM; ++d)
                {
                    const double lo = (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower));
                    const double hi = (*velocity)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Upper));
                    momentum[d] += rho * volume * 0.5 * (lo + hi);
                }
            }
        }
    }
    IBTK_MPI::sumReduction(momentum.data(), NDIM);
    return momentum;
}

Eigen::Vector3d
integrate_composite_mac_momentum(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                 const int velocity_idx,
                                 const int side_weight_idx,
                                 const double rho)
{
    // A MAC velocity component is integrated over its composite dual volume.
    // HierarchyMathOps side weights assign that volume and zero covered coarse
    // sides, avoiding the cell-average surrogate used by the legacy ledger.
    Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(velocity_idx) || !level->checkAllocated(side_weight_idx))
        {
            TBOX_ERROR("The MAC momentum integral requires allocated velocity and side-weight data on level "
                       << ln << ".\n");
        }
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<SideData<NDIM, double>> velocity = patch->getPatchData(velocity_idx);
            Pointer<SideData<NDIM, double>> weights = patch->getPatchData(side_weight_idx);
            for (int axis = 0; axis < NDIM; ++axis)
            {
                for (SideIterator<NDIM> side(patch->getBox(), axis); side; side++)
                {
                    momentum[axis] += rho * (*weights)(side()) * (*velocity)(side());
                }
            }
        }
    }
    IBTK_MPI::sumReduction(momentum.data(), NDIM);
    return momentum;
}

Eigen::Vector3d
integrate_composite_side_vector(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                const int side_vector_idx,
                                const int cell_weight_idx)
{
    // Convert the MAC-grid vector to a cell-centered integral by averaging
    // the two normal-face values for each component in each composite cell.
    // Zero cell weights exclude coarse cells covered by finer levels.
    Eigen::Vector3d integral = Eigen::Vector3d::Zero();
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(side_vector_idx))
        {
            TBOX_ERROR("The requested Eulerian IB force field is not allocated on level " << ln << ".\n");
        }
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            Pointer<SideData<NDIM, double>> side_vector = patch->getPatchData(side_vector_idx);
            Pointer<CellData<NDIM, double>> weights = patch->getPatchData(cell_weight_idx);
            for (Box<NDIM>::Iterator ci(patch->getBox()); ci; ci++)
            {
                const CellIndex<NDIM>& cell = *ci;
                const double volume = (*weights)(cell);
                if (volume <= 0.0) continue;
                for (int d = 0; d < NDIM; ++d)
                {
                    const double lo = (*side_vector)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Lower));
                    const double hi = (*side_vector)(SideIndex<NDIM>(cell, d, SideIndex<NDIM>::Upper));
                    integral[d] += volume * 0.5 * (lo + hi);
                }
            }
        }
    }
    IBTK_MPI::sumReduction(integral.data(), NDIM);
    return integral;
}

std::array<double, 4>
measure_divergence(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                   const int div_idx,
                   const int cell_weight_idx,
                   const double disk_radius,
                   const double band_width)
{
    double all_sq = 0.0, all_vol = 0.0, all_max = 0.0;
    double band_sq = 0.0, band_vol = 0.0, band_max = 0.0;
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        for (PatchLevel<NDIM>::Iterator p(level); p; p++)
        {
            Pointer<Patch<NDIM>> patch = level->getPatch(p());
            const Pointer<CartesianPatchGeometry<NDIM>> patch_geom = patch->getPatchGeometry();
            const double* dx = patch_geom->getDx();
            const double* patch_lower = patch_geom->getXLower();
            const Box<NDIM>& patch_box = patch->getBox();
            Pointer<CellData<NDIM, double>> div = patch->getPatchData(div_idx);
            Pointer<CellData<NDIM, double>> weights = patch->getPatchData(cell_weight_idx);
            for (Box<NDIM>::Iterator ci(patch_box); ci; ci++)
            {
                const CellIndex<NDIM>& cell = *ci;
                const double volume = (*weights)(cell);
                if (volume <= 0.0) continue;
                const double value = (*div)(cell);
                all_sq += volume * value * value;
                all_vol += volume;
                all_max = std::max(all_max, std::abs(value));
                double pos[NDIM];
                for (int d = 0; d < NDIM; ++d)
                    pos[d] = patch_lower[d] + (cell(d) - patch_box.lower(d) + 0.5) * dx[d];
                const double radial = std::sqrt(pos[1] * pos[1] + pos[2] * pos[2]);
                const double outside_radius = std::max(0.0, radial - disk_radius);
                const double distance = std::sqrt(pos[0] * pos[0] + outside_radius * outside_radius);
                if (distance <= band_width)
                {
                    band_sq += volume * value * value;
                    band_vol += volume;
                    band_max = std::max(band_max, std::abs(value));
                }
            }
        }
    }
    IBTK_MPI::sumReduction(&all_sq, 1);
    IBTK_MPI::sumReduction(&all_vol, 1);
    IBTK_MPI::maxReduction(&all_max, 1);
    IBTK_MPI::sumReduction(&band_sq, 1);
    IBTK_MPI::sumReduction(&band_vol, 1);
    IBTK_MPI::maxReduction(&band_max, 1);
    return { all_vol > 0.0 ? std::sqrt(all_sq / all_vol) : 0.0,
             all_max,
             band_vol > 0.0 ? std::sqrt(band_sq / band_vol) : 0.0,
             band_max };
}

struct ExplicitIBBodyForceCapture
{
    Pointer<PatchHierarchy<NDIM>> hierarchy;
    int force_idx = -1;
    int cell_weight_idx = -1;
    Eigen::Vector3d integral = Eigen::Vector3d::Zero();
    int callback_count = 0;
    bool field_available = false;
};

void
capture_explicit_ib_body_force(double /*current_time*/,
                               double /*new_time*/,
                               bool /*skip_synchronize_new_state_data*/,
                               int /*num_cycles*/,
                               void* ctx)
{
    auto* capture = static_cast<ExplicitIBBodyForceCapture*>(ctx);
    ++capture->callback_count;
    for (int ln = 0; ln <= capture->hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = capture->hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(capture->force_idx)) return;
    }
    capture->integral = integrate_composite_side_vector(
        capture->hierarchy, capture->force_idx, capture->cell_weight_idx);
    capture->field_available = true;
}

void
fill_velocity_ghosts(Pointer<PatchHierarchy<NDIM>> hierarchy,
                     const Pointer<SideVariable<NDIM, double>>& var,
                     const int src_idx,
                     const int dst_idx,
                     const std::vector<RobinBcCoefStrategy<NDIM>*>& bcs,
                     const double time)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(dst_idx)) level->allocatePatchData(dst_idx);
    }
    Pointer<HierarchyDataOpsReal<NDIM, double>> ops =
        HierarchyDataOpsManager<NDIM>::getManager()->getOperationsDouble(var, hierarchy, true);
    ops->copyData(dst_idx, src_idx, true);
    using Component = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<Component> transaction(1);
    transaction[0] = Component(dst_idx, src_idx, "CONSERVATIVE_LINEAR_REFINE", true, "CUBIC_COARSEN", "LINEAR",
                                false, bcs, Pointer<VariableFillPattern<NDIM>>(nullptr));
    Pointer<HierarchyGhostCellInterpolation> fill = new HierarchyGhostCellInterpolation();
    fill->initializeOperatorState(transaction, hierarchy);
    fill->setHomogeneousBc(false);
    fill->fillData(time);
}

void
fill_pressure_ghosts(Pointer<PatchHierarchy<NDIM>> hierarchy,
                     const Pointer<CellVariable<NDIM, double>>& var,
                     const int pressure_src_idx,
                     const int pressure_dst_idx,
                     const int velocity_src_idx,
                     Pointer<INSStaggeredHierarchyIntegrator> ns,
                     const double time)
{
    for (int ln = 0; ln <= hierarchy->getFinestLevelNumber(); ++ln)
    {
        Pointer<PatchLevel<NDIM>> level = hierarchy->getPatchLevel(ln);
        if (!level->checkAllocated(pressure_dst_idx)) level->allocatePatchData(pressure_dst_idx);
    }
    Pointer<HierarchyDataOpsReal<NDIM, double>> ops =
        HierarchyDataOpsManager<NDIM>::getManager()->getOperationsDouble(var, hierarchy, true);
    ops->copyData(pressure_dst_idx, pressure_src_idx, true);
    auto* pressure_bc = dynamic_cast<INSStaggeredPressureBcCoef*>(ns->getPressureBoundaryConditions());
    if (!pressure_bc) TBOX_ERROR("Expected INSStaggeredPressureBcCoef in CIB momentum audit.\n");
    pressure_bc->setTargetVelocityPatchDataIndex(velocity_src_idx);
    using Component = HierarchyGhostCellInterpolation::InterpolationTransactionComponent;
    std::vector<Component> transaction(1);
    transaction[0] = Component(pressure_dst_idx, pressure_src_idx, "CONSERVATIVE_LINEAR_REFINE", true,
                                "CUBIC_COARSEN", "LINEAR", false, pressure_bc,
                                Pointer<VariableFillPattern<NDIM>>(nullptr));
    Pointer<HierarchyGhostCellInterpolation> fill = new HierarchyGhostCellInterpolation();
    fill->initializeOperatorState(transaction, hierarchy);
    fill->setHomogeneousBc(false);
    fill->fillData(time);
    pressure_bc->clearTargetVelocityPatchDataIndex();
}

// Function prototypes
void output_data(Pointer<PatchHierarchy<NDIM>> patch_hierarchy,
                 LDataManager* l_data_manager,
                 const int iteration_num,
                 const double loop_time,
                 const string& data_dump_dirname);

/*******************************************************************************
 * For each run, the input filename and restart information (if needed) must   *
 * be given on the command line.  For non-restarted case, command line is:     *
 *                                                                             *
 *    executable <input file name>                                             *
 *                                                                             *
 * For restarted run, command line is:                                         *
 *                                                                             *
 *    executable <input file name> <restart directory> <restart number>        *
 *                                                                             *
 *******************************************************************************/
int
main(int argc, char* argv[])
{
    // Initialize IBAMR and libraries. Deinitialization is handled by this object as well.
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);

    { // cleanup dynamically allocated objects prior to shutdown

        // Parse command line options, set some standard options from the input
        // file, initialize the restart database (if this is a restarted run),
        // and enable file logging.
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "CIB.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        // Read default Petsc options
        if (input_db->keyExists("petsc_options_file"))
        {
            std::string petsc_options_file = input_db->getString("petsc_options_file");
            PetscOptionsInsertFile(PETSC_COMM_WORLD, nullptr, petsc_options_file.c_str(), PETSC_TRUE);
        }

        // Get various standard options set in the input file.
        const bool dump_viz_data = app_initializer->dumpVizData();
        const int viz_dump_interval = app_initializer->getVizDumpInterval();
        const bool uses_visit = dump_viz_data && !app_initializer->getVisItDataWriter().isNull();

        const bool dump_restart_data = app_initializer->dumpRestartData();
        const int restart_dump_interval = app_initializer->getRestartDumpInterval();
        const string restart_dump_dirname = app_initializer->getRestartDumpDirectory();

        const bool dump_postproc_data = app_initializer->dumpPostProcessingData();
        const int postproc_data_dump_interval = app_initializer->getPostProcessingDataDumpInterval();
        const string postproc_data_dump_dirname = app_initializer->getPostProcessingDataDumpDirectory();
        if (dump_postproc_data && (postproc_data_dump_interval > 0) && !postproc_data_dump_dirname.empty())
        {
            Utilities::recursiveMkdir(postproc_data_dump_dirname);
        }

        const bool dump_timer_data = app_initializer->dumpTimerData();
        const int timer_dump_interval = app_initializer->getTimerDumpInterval();

        // Create major algorithm and data objects that comprise the
        // application.  These objects are configured from the input database
        // and, if this is a restarted run, from the restart database.

        // INS integrator
        Pointer<INSStaggeredHierarchyIntegrator> navier_stokes_integrator = new INSStaggeredHierarchyIntegrator(
            "INSStaggeredHierarchyIntegrator",
            app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));

        // CIB method
        const unsigned int num_structures = input_db->getIntegerWithDefault("num_structures", 1);
        Pointer<CIBMethod> ib_method_ops =
            new CIBMethod("CIBMethod", app_initializer->getComponentDatabase("CIBMethod"), num_structures);

        // Krylov solver for INS integrator that solves for [u,p,U,L]
        Pointer<CIBStaggeredStokesSolver> CIBSolver =
            new CIBStaggeredStokesSolver("CIBStaggeredStokesSolver",
                                         input_db->getDatabase("CIBStaggeredStokesSolver"),
                                         navier_stokes_integrator,
                                         ib_method_ops,
                                         "SP_");

        // Register the Krylov solver with INS integrator
        navier_stokes_integrator->setStokesSolver(CIBSolver);

        Pointer<IBHierarchyIntegrator> time_integrator =
            new IBExplicitHierarchyIntegrator("IBHierarchyIntegrator",
                                              app_initializer->getComponentDatabase("IBHierarchyIntegrator"),
                                              ib_method_ops,
                                              navier_stokes_integrator);
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry = new CartesianGridGeometry<NDIM>(
            "CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector =
            new StandardTagAndInitialize<NDIM>("StandardTagAndInitialize",
                                               time_integrator,
                                               app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm =
            new GriddingAlgorithm<NDIM>("GriddingAlgorithm",
                                        app_initializer->getComponentDatabase("GriddingAlgorithm"),
                                        error_detector,
                                        box_generator,
                                        load_balancer);
        // Configure the IB solver.
        Pointer<IBStandardInitializer> ib_initializer = new IBStandardInitializer(
            "IBStandardInitializer", app_initializer->getComponentDatabase("IBStandardInitializer"));
        ib_method_ops->registerLInitStrategy(ib_initializer);

        // Specify all six rigid-body degrees of freedom from the input case.
        FreeRigidDOFVector struct_0_free_dofs;
        struct_0_free_dofs.setZero();
        ib_method_ops->setSolveRigidBodyVelocity(0, struct_0_free_dofs);

        const double rho_excess = input_db->getDouble("RHO_EXCESS");
        const double R = input_db->getDouble("R");
        const double G = input_db->getDouble("G");

        StructureCtx struct0;
        struct0.name = "sphere0";
        struct0.R = R;
        struct0.rho_excess = rho_excess;
        double prescribed_translation[NDIM] = {};
        double prescribed_rotation[NDIM] = {};
        input_db->getDoubleArray("PRESCRIBED_TRANSLATION", prescribed_translation, NDIM);
        input_db->getDoubleArray("PRESCRIBED_ROTATION", prescribed_rotation, NDIM);
        for (int d = 0; d < NDIM; ++d)
        {
            struct0.prescribed_translation[d] = prescribed_translation[d];
            struct0.prescribed_rotation[d] = prescribed_rotation[d];
        }

        ib_method_ops->registerExternalForceTorqueFunction(&NetExternalForceTorque, &struct0, 0);
        ib_method_ops->registerConstrainedVelocityFunction(nullptr, &ConstrainedCOMVel, &struct0, 0);

        // Create initial condition specification objects.
        Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
            "u_init", app_initializer->getComponentDatabase("VelocityInitialConditions"), grid_geometry);
        navier_stokes_integrator->registerVelocityInitialConditions(u_init);
        Pointer<CartGridFunction> p_init = new muParserCartGridFunction(
            "p_init", app_initializer->getComponentDatabase("PressureInitialConditions"), grid_geometry);
        navier_stokes_integrator->registerPressureInitialConditions(p_init);

        // Set up visualization plot file writers.
        Pointer<VisItDataWriter<NDIM>> visit_data_writer = app_initializer->getVisItDataWriter();
        Pointer<LSiloDataWriter> silo_data_writer = app_initializer->getLSiloDataWriter();
        if (uses_visit)
        {
            ib_initializer->registerLSiloDataWriter(silo_data_writer);
            ib_method_ops->registerLSiloDataWriter(silo_data_writer);
            ib_method_ops->registerVisItDataWriter(visit_data_writer);
            time_integrator->registerVisItDataWriter(visit_data_writer);
        }

        // Create boundary condition specification objects (when necessary).
        const IntVector<NDIM>& periodic_shift = grid_geometry->getPeriodicShift();
        vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        if (periodic_shift.min() > 0)
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                u_bc_coefs[d] = nullptr;
            }
        }
        else
        {
            for (unsigned int d = 0; d < NDIM; ++d)
            {
                const std::string bc_coefs_name = "u_bc_coefs_" + std::to_string(d);

                const std::string bc_coefs_db_name = "VelocityBcCoefs_" + std::to_string(d);

                Pointer<Database> bc_coefs_db = app_initializer->getComponentDatabase(bc_coefs_db_name);
                u_bc_coefs[d] = new muParserRobinBcCoefs(bc_coefs_name, bc_coefs_db, grid_geometry);
            }
            navier_stokes_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);
        }

        // Initialize hierarchy configuration and data on all patches.
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        // Set physical boundary operator used in spreading.
        ib_method_ops->setVelocityPhysBdryOp(time_integrator->getVelocityPhysBdryOp());

        const double rho = input_db->getDouble("RHO");
        const double mu = input_db->getDouble("MU");
        const double disk_radius = input_db->getDouble("R");
        const double reference_speed = input_db->getDoubleWithDefault("U_REFERENCE", 1.0);
        const double finest_dx = input_db->getDouble("DX");
        const double divergence_band_width = 2.0 * finest_dx;
        const int structure_level = input_db->getInteger("MAX_LEVELS") - 1;
        LDataManager* lagrangian_manager = ib_method_ops->getLDataManager();
        Pointer<LData> sampled_fluid_velocity =
            lagrangian_manager->createLData("U_newtime_diagnostic", structure_level, NDIM);
        const int diagnostic_ghost_width = IBTK::LEInteractor::getMinimumGhostWidth(
            app_initializer->getComponentDatabase("CIBMethod")->getString("delta_fcn"));

        VariableDatabase<NDIM>* variable_db = VariableDatabase<NDIM>::getDatabase();
        const int velocity_idx = variable_db->mapVariableAndContextToIndex(
            navier_stokes_integrator->getVelocityVariable(), navier_stokes_integrator->getCurrentContext());
        // IBExplicitHierarchyIntegrator registers its immersed-boundary force
        // in this side-centered field at the force stage (the midpoint for the
        // MIDPOINT_RULE case used here). Integrating that Eulerian field checks
        // independently that the stored CIB multiplier was spread with the
        // expected sign and scale.
        Pointer<VariableContext> ib_force_context = variable_db->getContext("IBHierarchyIntegrator::IB");
        const int ib_force_idx = variable_db->mapVariableAndContextToIndex(
            time_integrator->getBodyForceVariable(), ib_force_context);
        const int pressure_idx = variable_db->mapVariableAndContextToIndex(
            navier_stokes_integrator->getPressureVariable(), navier_stokes_integrator->getCurrentContext());
        Pointer<Variable<NDIM>> div_var = variable_db->getVariable("INSStaggeredHierarchyIntegrator::Div_U");
        const int div_idx = variable_db->mapVariableAndContextToIndex(div_var, navier_stokes_integrator->getCurrentContext());
        Pointer<HierarchyMathOps> hierarchy_math_ops = time_integrator->getHierarchyMathOps();
        const int cell_weight_idx = hierarchy_math_ops->getCellWeightPatchDescriptorIndex();
        const int side_weight_idx = hierarchy_math_ops->getSideWeightPatchDescriptorIndex();
        ExplicitIBBodyForceCapture explicit_ib_body_force_capture;
        explicit_ib_body_force_capture.hierarchy = patch_hierarchy;
        explicit_ib_body_force_capture.force_idx = ib_force_idx;
        explicit_ib_body_force_capture.cell_weight_idx = cell_weight_idx;
        // This is the ordinary explicit IB body-force data field. CIB's
        // constraint multiplier is applied inside CIBStaggeredStokesSolver's
        // saddle-point operator and is not represented by this separate field.
        // Capture this field before IBHierarchyIntegrator deallocates it so
        // the audit cannot accidentally label it as the CIB spread multiplier.
        navier_stokes_integrator->registerPostprocessIntegrateHierarchyCallback(
            &capture_explicit_ib_body_force, &explicit_ib_body_force_capture);
        SAMRAI::math::HierarchyCellDataOpsReal<NDIM, double> cell_ops(patch_hierarchy);

        Pointer<CellVariable<NDIM, double>> pressure_diag_var = new CellVariable<NDIM, double>("CIBPressureLedger", 1);
        Pointer<VariableContext> pressure_diag_context = variable_db->getContext("CIBPressureLedgerContext");
        const int pressure_diag_idx = variable_db->registerVariableAndContext(
            pressure_diag_var, pressure_diag_context, IntVector<NDIM>(1));
        Pointer<SideVariable<NDIM, double>> velocity_diag_var = new SideVariable<NDIM, double>("CIBVelocityLedger", 1);
        Pointer<VariableContext> velocity_diag_context = variable_db->getContext("CIBVelocityLedgerContext");
        const int velocity_diag_idx = variable_db->registerVariableAndContext(
            velocity_diag_var, velocity_diag_context, IntVector<NDIM>(diagnostic_ghost_width));

        time_integrator->setupPlotData();
        Eigen::Vector3d fluid_momentum_old = integrate_composite_momentum(
            patch_hierarchy, velocity_idx, cell_weight_idx, rho);
        Eigen::Vector3d fluid_mac_momentum_old = integrate_composite_mac_momentum(
            patch_hierarchy, velocity_idx, side_weight_idx, rho);
        fill_velocity_ghosts(patch_hierarchy, velocity_diag_var, velocity_idx, velocity_diag_idx, u_bc_coefs,
                             time_integrator->getIntegratorTime());
        fill_pressure_ghosts(patch_hierarchy, pressure_diag_var, pressure_idx, pressure_diag_idx,
                             velocity_diag_idx, navier_stokes_integrator, time_integrator->getIntegratorTime());
        OuterMomentumFlux outer_flux_old = integrate_outer_momentum_flux(
            patch_hierarchy, velocity_diag_idx, pressure_diag_idx, rho, mu);

        ofstream force_history;
        ofstream momentum_budget;
        if (IBTK_MPI::getRank() == 0)
        {
            force_history.open("force_history.csv");
            force_history << "time_code,dt_code,position_error_rms_code,position_error_max_code,"
                             "newtime_interp_slip_rms_code,newtime_interp_slip_max_code,"
                             "newtime_interp_slip_rms_over_Uref,newtime_interp_slip_max_over_Uref,"
                             "divergence_rms_code,divergence_max_code,divergence_band_width_code,"
                             "divergence_band_rms_code,divergence_band_max_code,"
                             "lambda_force_on_fluid_x,lambda_force_on_fluid_y,lambda_force_on_fluid_z,"
                             "lambda_torque_on_fluid_x,lambda_torque_on_fluid_y,lambda_torque_on_fluid_z,"
                             "prescribed_Ux,prescribed_Uy,prescribed_Uz\n";
            force_history << std::setprecision(16);
            momentum_budget.open("momentum_budget.csv");
            momentum_budget << "time_code,dt_code,dPdt_x,dPdt_y,dPdt_z,"
                               "lambda_force_on_fluid_x,lambda_force_on_fluid_y,lambda_force_on_fluid_z,"
                               "outer_pressure_x,outer_pressure_y,outer_pressure_z,"
                               "outer_viscous_x,outer_viscous_y,outer_viscous_z,"
                               "outer_advection_x,outer_advection_y,outer_advection_z,"
                               "explicit_ib_body_force_x,explicit_ib_body_force_y,explicit_ib_body_force_z,"
                               "explicit_body_force_minus_constraint_lambda_x,"
                               "explicit_body_force_minus_constraint_lambda_y,"
                               "explicit_body_force_minus_constraint_lambda_z,"
                               "residual_x,residual_y,residual_z,normalized_residual_cell_average,"
                               "dPdt_mac_side_x,dPdt_mac_side_y,dPdt_mac_side_z,"
                               "residual_mac_side_x,residual_mac_side_y,residual_mac_side_z,"
                               "normalized_residual_mac_side\n";
            momentum_budget << std::setprecision(16);
        }

        // Register mobility matrices (if needed)
        std::string mobility_solver_type = input_db->getString("MOBILITY_SOLVER_TYPE");
        if (mobility_solver_type == "DIRECT")
        {
            std::string mat_name = "sphere_mobility";
            std::vector<std::vector<unsigned>> struct_ids;
            std::vector<unsigned> prototype_structs;

            // Dense matrix type
            prototype_structs.push_back(0);

            // Dense matrix to operate upon
            const int num_similar_structs = 1;
            for (int i = 0; i < num_similar_structs; ++i)
            {
                struct_ids.push_back(std::vector<unsigned>(1, i));
            }

            // Register the dense matrix with direct solver
            DirectMobilitySolver* direct_solvers = nullptr;
            CIBSolver->getSaddlePointSolver()->getCIBMobilitySolver()->getMobilitySolvers(
                nullptr, &direct_solvers, nullptr);

            direct_solvers->registerMobilityMat(
                mat_name, prototype_structs, EMPIRICAL, std::make_pair(LAPACK_LU, LAPACK_LU), 0);
            direct_solvers->registerStructIDsWithMobilityMat(mat_name, struct_ids);
        }
        navier_stokes_integrator->setStokesSolverNeedsInit();

        // Deallocate initialization objects.
        app_initializer.setNull();

        // Print the input database contents to the log file.
        plog << "Input database:\n";
        input_db->printClassData(plog);

        // Write out initial visualization data.
        int iteration_num = time_integrator->getIntegratorStep();
        double loop_time = time_integrator->getIntegratorTime();

        if (dump_viz_data && uses_visit)
        {
            pout << "\n\nWriting visualization files...\n\n";
            time_integrator->setupPlotData();
            visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
            silo_data_writer->writePlotData(iteration_num, loop_time);
        }
        if (dump_postproc_data)
        {
            output_data(patch_hierarchy,
                        ib_method_ops->getLDataManager(),
                        iteration_num,
                        loop_time,
                        postproc_data_dump_dirname);
        }

        if (IBTK_MPI::getRank() == 0)
        {
            U_stream.open("./Lambda/U.txt", std::ios_base::out | ios_base::trunc);
            U_stream.precision(10);
        }

        // Main time step loop.
        double loop_time_end = time_integrator->getEndTime();
        double dt = 0.0;

        while (!IBTK::rel_equal_eps(loop_time, loop_time_end) && time_integrator->stepsRemaining())
        {
            iteration_num = time_integrator->getIntegratorStep();
            loop_time = time_integrator->getIntegratorTime();

            pout << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "At beginning of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";

            dt = time_integrator->getMaximumTimeStepSize();

            pout << "Advancing hierarchy by timestep size dt = " << dt << "\n";
            if (time_integrator->atRegridPoint()) navier_stokes_integrator->setStokesSolverNeedsInit();
            if (ib_method_ops->flagRegrid())
            {
                time_integrator->regridHierarchy();
                navier_stokes_integrator->setStokesSolverNeedsInit();
            }

            RDV U0;
            ib_method_ops->getCurrentRigidBodyVelocity(0, U0);
            U_stream << loop_time << "\t" << U0(0) << "\t" << U0(1) << "\t" << U0(2) << "\t" << U0(3) << "\t" << U0(4)
                     << "\t" << U0(5) << std::endl;

            pout << "Velocity of rigid body is " << std::setprecision(10) << std::abs(U0(1)) << "\n";

            // Compute external gravity force on structure
            struct0.F.setZero();
            struct0.F(1) = -struct0.rho_excess * (4.0 / 3.0) * M_PI * std::pow(struct0.R, 3) * G;

            const double step_start_time = loop_time;
            explicit_ib_body_force_capture.integral.setZero();
            explicit_ib_body_force_capture.callback_count = 0;
            explicit_ib_body_force_capture.field_available = false;
            time_integrator->advanceHierarchy(dt);
            loop_time = time_integrator->getIntegratorTime();
            const double dt_realized = loop_time - step_start_time;

            pout << "\n";
            pout << "At end       of timestep # " << iteration_num << "\n";
            pout << "Simulation time is " << loop_time << "\n";
            pout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n";
            pout << "\n";

            fill_velocity_ghosts(patch_hierarchy, velocity_diag_var, velocity_idx, velocity_diag_idx,
                                 u_bc_coefs, loop_time);
            const MarkerMetrics marker_metrics = measure_markers(lagrangian_manager,
                                                                 structure_level,
                                                                 loop_time,
                                                                 struct0.prescribed_translation,
                                                                 reference_speed,
                                                                 velocity_diag_idx,
                                                                 sampled_fluid_velocity);
            const Eigen::Vector3d momentum_new = integrate_composite_momentum(
                patch_hierarchy, velocity_idx, cell_weight_idx, rho);
            const Eigen::Vector3d dP_dt = (momentum_new - fluid_momentum_old) / dt_realized;
            const Eigen::Vector3d fluid_mac_momentum_new = integrate_composite_mac_momentum(
                patch_hierarchy, velocity_idx, side_weight_idx, rho);
            const Eigen::Vector3d dP_dt_mac = (fluid_mac_momentum_new - fluid_mac_momentum_old) / dt_realized;

            fill_pressure_ghosts(patch_hierarchy, pressure_diag_var, pressure_idx, pressure_diag_idx,
                                 velocity_diag_idx, navier_stokes_integrator, loop_time);
            const OuterMomentumFlux outer_flux_new = integrate_outer_momentum_flux(
                patch_hierarchy, velocity_diag_idx, pressure_diag_idx, rho, mu);
            const Eigen::Vector3d outer_pressure = 0.5 * (outer_flux_old.pressure + outer_flux_new.pressure);
            const Eigen::Vector3d outer_viscous = 0.5 * (outer_flux_old.viscous + outer_flux_new.viscous);
            const Eigen::Vector3d outer_advection = 0.5 * (outer_flux_old.advection + outer_flux_new.advection);

            LData* lambda_data = lagrangian_manager->getLData("lambda", structure_level);
            RigidDOFVector lambda_generalized;
            ib_method_ops->computeNetRigidGeneralizedForce(0, lambda_data->getVec(), lambda_generalized);
            const Eigen::Vector3d lambda_on_fluid(lambda_generalized[0], lambda_generalized[1], lambda_generalized[2]);
            if (!explicit_ib_body_force_capture.field_available)
            {
                TBOX_ERROR("Could not capture the temporary explicit IB body-force field before it was deallocated.\n");
            }
            const Eigen::Vector3d explicit_ib_body_force = explicit_ib_body_force_capture.integral;
            const Eigen::Vector3d explicit_body_force_minus_lambda = explicit_ib_body_force - lambda_on_fluid;
            const Eigen::Vector3d residual = dP_dt - lambda_on_fluid - outer_pressure - outer_viscous - outer_advection;
            const double residual_scale = dP_dt.norm() + lambda_on_fluid.norm() + outer_pressure.norm() +
                                          outer_viscous.norm() + outer_advection.norm();
            const double normalized_residual = residual_scale > 0.0 ? residual.norm() / residual_scale : 0.0;
            const Eigen::Vector3d residual_mac =
                dP_dt_mac - lambda_on_fluid - outer_pressure - outer_viscous - outer_advection;
            const double residual_mac_scale = dP_dt_mac.norm() + lambda_on_fluid.norm() + outer_pressure.norm() +
                                              outer_viscous.norm() + outer_advection.norm();
            const double normalized_residual_mac =
                residual_mac_scale > 0.0 ? residual_mac.norm() / residual_mac_scale : 0.0;

            time_integrator->setupPlotData();
            const double div_rms = cell_ops.RMSNorm(div_idx, cell_weight_idx);
            const double div_max = cell_ops.maxNorm(div_idx, cell_weight_idx);
            const std::array<double, 4> div_metrics = measure_divergence(
                patch_hierarchy, div_idx, cell_weight_idx, disk_radius, divergence_band_width);

            RDV Unew;
            ib_method_ops->getCurrentRigidBodyVelocity(0, Unew);
            if (IBTK_MPI::getRank() == 0)
            {
                force_history << loop_time << ',' << dt_realized << ','
                              << marker_metrics.position_rms << ',' << marker_metrics.position_max << ','
                              << marker_metrics.slip_rms << ',' << marker_metrics.slip_max << ','
                              << marker_metrics.slip_rms / reference_speed << ','
                              << marker_metrics.slip_max / reference_speed << ','
                              << div_rms << ',' << div_max << ',' << divergence_band_width << ','
                              << div_metrics[2] << ',' << div_metrics[3] << ','
                              << lambda_generalized[0] << ',' << lambda_generalized[1] << ','
                              << lambda_generalized[2] << ',' << lambda_generalized[3] << ','
                              << lambda_generalized[4] << ',' << lambda_generalized[5] << ','
                              << struct0.prescribed_translation[0] << ',' << struct0.prescribed_translation[1] << ','
                              << struct0.prescribed_translation[2] << '\n';
                momentum_budget << loop_time << ',' << dt_realized << ','
                                << dP_dt[0] << ',' << dP_dt[1] << ',' << dP_dt[2] << ','
                                << lambda_on_fluid[0] << ',' << lambda_on_fluid[1] << ',' << lambda_on_fluid[2] << ','
                                << outer_pressure[0] << ',' << outer_pressure[1] << ',' << outer_pressure[2] << ','
                                << outer_viscous[0] << ',' << outer_viscous[1] << ',' << outer_viscous[2] << ','
                                << outer_advection[0] << ',' << outer_advection[1] << ',' << outer_advection[2] << ','
                                << explicit_ib_body_force[0] << ',' << explicit_ib_body_force[1] << ','
                                << explicit_ib_body_force[2] << ','
                                << explicit_body_force_minus_lambda[0] << ',' << explicit_body_force_minus_lambda[1] << ','
                                << explicit_body_force_minus_lambda[2] << ','
                                << residual[0] << ',' << residual[1] << ',' << residual[2] << ','
                                << normalized_residual << ','
                                << dP_dt_mac[0] << ',' << dP_dt_mac[1] << ',' << dP_dt_mac[2] << ','
                                << residual_mac[0] << ',' << residual_mac[1] << ',' << residual_mac[2] << ','
                                << normalized_residual_mac << '\n';
                force_history.flush();
                momentum_budget.flush();
            }
            fluid_momentum_old = momentum_new;
            fluid_mac_momentum_old = fluid_mac_momentum_new;
            outer_flux_old = outer_flux_new;

            // At specified intervals, write visualization and restart files,
            // print out timer data, and store hierarchy data for post
            // processing.
            iteration_num += 1;
            const bool last_step = !time_integrator->stepsRemaining();
            if (dump_viz_data && uses_visit && (iteration_num % viz_dump_interval == 0 || last_step))
            {
                pout << "\nWriting visualization files...\n\n";
                time_integrator->setupPlotData();
                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
                silo_data_writer->writePlotData(iteration_num, loop_time);
            }
            if (dump_restart_data && (iteration_num % restart_dump_interval == 0 || last_step))
            {
                pout << "\nWriting restart files...\n\n";
                RestartManager::getManager()->writeRestartFile(restart_dump_dirname, iteration_num);
            }
            if (dump_timer_data && (iteration_num % timer_dump_interval == 0 || last_step))
            {
                pout << "\nWriting timer data...\n\n";
                TimerManager::getManager()->print(plog);
            }
            if (dump_postproc_data && (iteration_num % postproc_data_dump_interval == 0 || last_step))
            {
                output_data(patch_hierarchy,
                            ib_method_ops->getLDataManager(),
                            iteration_num,
                            loop_time,
                            postproc_data_dump_dirname);
            }
        }

        if (IBTK_MPI::getRank() == 0)
        {
            force_history.close();
            momentum_budget.close();
        }

        if (IBTK_MPI::getRank() == 0)
        {
            U_stream.close();
        }

        // Cleanup boundary condition specification objects (when necessary).
        for (unsigned int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];

    } // cleanup dynamically allocated objects prior to shutdown
} // main

void
output_data(Pointer<PatchHierarchy<NDIM>> /*patch_hierarchy*/,
            LDataManager* /*l_data_manager*/,
            const int iteration_num,
            const double loop_time,
            const string& /*data_dump_dirname*/)
{
    plog << "writing hierarchy data at iteration " << iteration_num << " to disk" << endl;
    plog << "simulation time is " << loop_time << endl;

    return;
} // output_data
