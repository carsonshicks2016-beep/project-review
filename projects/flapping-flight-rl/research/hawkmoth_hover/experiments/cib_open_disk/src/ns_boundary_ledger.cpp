// Analytic no-structure controls for the CIB pilot's outer-boundary ledger.

#include <SAMRAI_config.h>

#include <petscsys.h>

#include <BergerRigoutsos.h>
#include <CartesianGridGeometry.h>
#include <CellVariable.h>
#include <HierarchyCellDataOpsReal.h>
#include <HierarchyDataOpsManager.h>
#include <LoadBalancer.h>
#include <SideData.h>
#include <SideIterator.h>
#include <SideVariable.h>
#include <StandardTagAndInitialize.h>
#include <VariableDatabase.h>

#include <ibamr/INSStaggeredHierarchyIntegrator.h>
#include <ibamr/INSStaggeredPressureBcCoef.h>

#include <ibtk/AppInitializer.h>
#include <ibtk/HierarchyGhostCellInterpolation.h>
#include <ibtk/HierarchyMathOps.h>
#include <ibtk/IBTKInit.h>
#include <ibtk/IBTK_MPI.h>
#include <ibtk/muParserCartGridFunction.h>
#include <ibtk/muParserRobinBcCoefs.h>

#include <ibamr/app_namespaces.h>

#include "outer_momentum_flux.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{
void
fill_velocity_ghosts(Pointer<PatchHierarchy<NDIM>>& hierarchy,
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
fill_pressure_ghosts(Pointer<PatchHierarchy<NDIM>>& hierarchy,
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
    if (!pressure_bc) TBOX_ERROR("Expected INSStaggeredPressureBcCoef in no-structure momentum audit.\n");
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

Eigen::Vector3d
integrate_composite_mac_momentum(const Pointer<PatchHierarchy<NDIM>>& hierarchy,
                                 const int velocity_idx,
                                 const int side_weight_idx,
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
            Pointer<SideData<NDIM, double>> weights = patch->getPatchData(side_weight_idx);
            for (int axis = 0; axis < NDIM; ++axis)
            {
                for (SideIterator<NDIM> side(patch->getBox(), axis); side; side++)
                    momentum[axis] += rho * (*weights)(side()) * (*velocity)(side());
            }
        }
    }
    IBTK_MPI::sumReduction(momentum.data(), NDIM);
    return momentum;
}

struct ExactBoundaryTerms
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

ExactBoundaryTerms
exact_boundary_terms(const std::string& mode,
                     const double rho,
                     const double mu,
                     const double u0,
                     const double pressure_gradient_magnitude,
                     const std::array<double, NDIM>& lengths)
{
    ExactBoundaryTerms exact;
    for (int axis = 0; axis < NDIM; ++axis)
    {
        const double area = lengths[(axis + 1) % NDIM] * lengths[(axis + 2) % NDIM];
        for (int upper = 0; upper <= 1; ++upper)
        {
            const int face = 2 * axis + upper;
            const double normal = upper ? 1.0 : -1.0;
            if (mode == "POISEUILLE")
            {
                double mean_pressure = -0.5 * pressure_gradient_magnitude * lengths[0];
                if (axis == 0) mean_pressure = upper ? -pressure_gradient_magnitude * lengths[0] : 0.0;
                exact.pressure_by_face(axis, face) = -mean_pressure * normal * area;
                if (axis == 1)
                    exact.viscous_by_face(0, face) = -0.5 * pressure_gradient_magnitude * lengths[1] * area;
                if (axis == 0)
                {
                    const double integrated_u_squared =
                        pressure_gradient_magnitude * pressure_gradient_magnitude *
                        std::pow(lengths[1], 5) / (120.0 * mu * mu) * lengths[2];
                    exact.advection_by_face(0, face) = upper ? -rho * integrated_u_squared : rho * integrated_u_squared;
                }
            }
            else if (mode == "UNIFORM")
            {
                if (axis == 0) exact.advection_by_face(0, face) = upper ? -rho * u0 * u0 * area : rho * u0 * u0 * area;
            }
        }
    }
    for (int face = 0; face < 2 * NDIM; ++face)
    {
        exact.pressure += exact.pressure_by_face.col(face);
        exact.viscous += exact.viscous_by_face.col(face);
        exact.advection += exact.advection_by_face.col(face);
    }
    return exact;
}

double
characteristic_force(const std::string& mode,
                     const double rho,
                     const double mu,
                     const double u0,
                     const double pressure_gradient_magnitude,
                     const std::array<double, NDIM>& lengths)
{
    if (mode == "POISEUILLE") return std::abs(pressure_gradient_magnitude) * lengths[0] * lengths[1] * lengths[2];
    return rho * u0 * u0 * lengths[1] * lengths[2];
}

Eigen::Vector3d
exact_total_momentum(const std::string& mode,
                     const double rho,
                     const double mu,
                     const double u0,
                     const double pressure_gradient_magnitude,
                     const std::array<double, NDIM>& lengths)
{
    Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
    if (mode == "POISEUILLE")
        momentum[0] = rho * pressure_gradient_magnitude * lengths[0] * lengths[2] *
                      std::pow(lengths[1], 3) / (12.0 * mu);
    else
        momentum[0] = rho * u0 * lengths[0] * lengths[1] * lengths[2];
    return momentum;
}

void
write_vector(std::ostream& out, const Eigen::Vector3d& value)
{
    out << ',' << value[0] << ',' << value[1] << ',' << value[2];
}

void
write_face_matrix(std::ostream& out, const Eigen::Matrix<double, NDIM, 2 * NDIM>& value)
{
    for (int face = 0; face < 2 * NDIM; ++face)
        for (int axis = 0; axis < NDIM; ++axis) out << ',' << value(axis, face);
}

void
write_exact_face_matrix(std::ostream& out, const Eigen::Matrix<double, NDIM, 2 * NDIM>& value)
{
    write_face_matrix(out, value);
}

} // namespace

int
main(int argc, char* argv[])
{
    IBTKInit ibtk_init(argc, argv, MPI_COMM_WORLD);
    {
        Pointer<AppInitializer> app_initializer = new AppInitializer(argc, argv, "INS.log");
        Pointer<Database> input_db = app_initializer->getInputDatabase();

        Pointer<INSStaggeredHierarchyIntegrator> time_integrator = new INSStaggeredHierarchyIntegrator(
            "INSStaggeredHierarchyIntegrator", app_initializer->getComponentDatabase("INSStaggeredHierarchyIntegrator"));
        Pointer<CartesianGridGeometry<NDIM>> grid_geometry =
            new CartesianGridGeometry<NDIM>("CartesianGeometry", app_initializer->getComponentDatabase("CartesianGeometry"));
        Pointer<PatchHierarchy<NDIM>> patch_hierarchy = new PatchHierarchy<NDIM>("PatchHierarchy", grid_geometry);
        Pointer<StandardTagAndInitialize<NDIM>> error_detector = new StandardTagAndInitialize<NDIM>(
            "StandardTagAndInitialize", time_integrator,
            app_initializer->getComponentDatabase("StandardTagAndInitialize"));
        Pointer<BergerRigoutsos<NDIM>> box_generator = new BergerRigoutsos<NDIM>();
        Pointer<LoadBalancer<NDIM>> load_balancer =
            new LoadBalancer<NDIM>("LoadBalancer", app_initializer->getComponentDatabase("LoadBalancer"));
        Pointer<GriddingAlgorithm<NDIM>> gridding_algorithm = new GriddingAlgorithm<NDIM>(
            "GriddingAlgorithm", app_initializer->getComponentDatabase("GriddingAlgorithm"), error_detector,
            box_generator, load_balancer);

        Pointer<CartGridFunction> u_init = new muParserCartGridFunction(
            "u_init", app_initializer->getComponentDatabase("VelocityInitialConditions"), grid_geometry);
        Pointer<CartGridFunction> p_init = new muParserCartGridFunction(
            "p_init", app_initializer->getComponentDatabase("PressureInitialConditions"), grid_geometry);
        time_integrator->registerVelocityInitialConditions(u_init);
        time_integrator->registerPressureInitialConditions(p_init);

        std::vector<RobinBcCoefStrategy<NDIM>*> u_bc_coefs(NDIM);
        for (int d = 0; d < NDIM; ++d)
        {
            const std::string name = "VelocityBcCoefs_" + std::to_string(d);
            u_bc_coefs[d] = new muParserRobinBcCoefs(name, app_initializer->getComponentDatabase(name), grid_geometry);
        }
        time_integrator->registerPhysicalBoundaryConditions(u_bc_coefs);
        time_integrator->initializePatchHierarchy(patch_hierarchy, gridding_algorithm);

        const std::string mode = input_db->getString("CASE_MODE");
        const double rho = input_db->getDouble("RHO");
        const double mu = input_db->getDouble("MU");
        const double u0 = input_db->getDoubleWithDefault("U0", 0.0);
        const double pressure_gradient_magnitude = input_db->getDoubleWithDefault("G_PRESSURE", 0.0);
        double domain_lower[NDIM], domain_upper[NDIM];
        input_db->getDatabase("CartesianGeometry")->getDoubleArray("x_lo", domain_lower, NDIM);
        input_db->getDatabase("CartesianGeometry")->getDoubleArray("x_up", domain_upper, NDIM);
        std::array<double, NDIM> lengths{};
        for (int d = 0; d < NDIM; ++d) lengths[d] = domain_upper[d] - domain_lower[d];
        const double force_scale = characteristic_force(mode, rho, mu, u0, pressure_gradient_magnitude, lengths);
        const ExactBoundaryTerms exact_flux =
            exact_boundary_terms(mode, rho, mu, u0, pressure_gradient_magnitude, lengths);
        const Eigen::Vector3d exact_momentum =
            exact_total_momentum(mode, rho, mu, u0, pressure_gradient_magnitude, lengths);

        VariableDatabase<NDIM>* variable_db = VariableDatabase<NDIM>::getDatabase();
        const int velocity_idx = variable_db->mapVariableAndContextToIndex(
            time_integrator->getVelocityVariable(), time_integrator->getCurrentContext());
        const int pressure_idx = variable_db->mapVariableAndContextToIndex(
            time_integrator->getPressureVariable(), time_integrator->getCurrentContext());
        Pointer<Variable<NDIM>> div_var = variable_db->getVariable("INSStaggeredHierarchyIntegrator::Div_U");
        const int div_idx = variable_db->mapVariableAndContextToIndex(div_var, time_integrator->getCurrentContext());
        Pointer<HierarchyMathOps> hierarchy_math_ops = time_integrator->getHierarchyMathOps();
        const int cell_weight_idx = hierarchy_math_ops->getCellWeightPatchDescriptorIndex();
        const int side_weight_idx = hierarchy_math_ops->getSideWeightPatchDescriptorIndex();
        const int finest_ln = patch_hierarchy->getFinestLevelNumber();
        NULL_USE(finest_ln);

        Pointer<SideVariable<NDIM, double>> velocity_diag_var =
            new SideVariable<NDIM, double>("NSLedgerVelocityGhosts", 1);
        Pointer<VariableContext> velocity_diag_context = variable_db->getContext("NSLedgerVelocityGhostContext");
        const int velocity_diag_idx = variable_db->registerVariableAndContext(
            velocity_diag_var, velocity_diag_context, IntVector<NDIM>(1));
        Pointer<CellVariable<NDIM, double>> pressure_diag_var =
            new CellVariable<NDIM, double>("NSLedgerPressureGhosts", 1);
        Pointer<VariableContext> pressure_diag_context = variable_db->getContext("NSLedgerPressureGhostContext");
        const int pressure_diag_idx = variable_db->registerVariableAndContext(
            pressure_diag_var, pressure_diag_context, IntVector<NDIM>(1));
        std::ofstream state_file;
        std::ofstream balance_file;
        std::ofstream face_file;
        if (IBTK_MPI::getRank() == 0)
        {
            state_file.open("state_snapshots.csv");
            state_file << "time_code,momentum_x,momentum_y,momentum_z,exact_momentum_x,exact_momentum_y,exact_momentum_z,"
                          "pressure_x,pressure_y,pressure_z,viscous_x,viscous_y,viscous_z,advection_x,advection_y,advection_z,"
                          "exact_pressure_x,exact_pressure_y,exact_pressure_z,exact_viscous_x,exact_viscous_y,exact_viscous_z,"
                          "exact_advection_x,exact_advection_y,exact_advection_z,div_rms,div_max\n";
            balance_file.open("momentum_balance.csv");
            balance_file << "t_old,t_new,dt,dPdt_x,dPdt_y,dPdt_z,pressure_x,pressure_y,pressure_z,"
                            "viscous_x,viscous_y,viscous_z,advection_x,advection_y,advection_z,"
                            "residual_x,residual_y,residual_z,normalized_by_term_sum,normalized_by_characteristic,"
                            "exact_pressure_x,exact_pressure_y,exact_pressure_z,exact_viscous_x,exact_viscous_y,exact_viscous_z,"
                            "exact_advection_x,exact_advection_y,exact_advection_z\n";
            face_file.open("boundary_face_terms.csv");
            face_file << "time_code,term,face,fx,fy,fz,exact_fx,exact_fy,exact_fz\n";
            state_file << std::setprecision(16);
            balance_file << std::setprecision(16);
            face_file << std::setprecision(16);
        }

        auto sample_state = [&](const double time, const bool write_rows) {
            if (IBTK_MPI::getRank() == 0) std::cerr << "NS_LEDGER sample t=" << time << " setupPlotData begin\n";
            time_integrator->setupPlotData();
            if (IBTK_MPI::getRank() == 0) std::cerr << "NS_LEDGER velocity ghosts begin\n";
            fill_velocity_ghosts(patch_hierarchy, velocity_diag_var, velocity_idx, velocity_diag_idx, u_bc_coefs, time);
            if (IBTK_MPI::getRank() == 0) std::cerr << "NS_LEDGER pressure ghosts begin\n";
            fill_pressure_ghosts(patch_hierarchy, pressure_diag_var, pressure_idx, pressure_diag_idx,
                                 velocity_diag_idx, time_integrator, time);
            if (IBTK_MPI::getRank() == 0) std::cerr << "NS_LEDGER momentum integral begin\n";
            const Eigen::Vector3d momentum = integrate_composite_mac_momentum(
                patch_hierarchy, velocity_idx, side_weight_idx, rho);
            if (IBTK_MPI::getRank() == 0) std::cerr << "NS_LEDGER boundary flux begin\n";
            const OuterMomentumFlux flux = integrate_outer_momentum_flux(
                patch_hierarchy, velocity_diag_idx, pressure_diag_idx, rho, mu);
            if (IBTK_MPI::getRank() == 0) std::cerr << "NS_LEDGER boundary flux end\n";
            SAMRAI::math::HierarchyCellDataOpsReal<NDIM, double> cell_ops(patch_hierarchy);
            const double div_rms = cell_ops.RMSNorm(div_idx, cell_weight_idx);
            const double div_max = cell_ops.maxNorm(div_idx, cell_weight_idx);
            if (write_rows && IBTK_MPI::getRank() == 0)
            {
                state_file << time;
                write_vector(state_file, momentum);
                write_vector(state_file, exact_momentum);
                write_vector(state_file, flux.pressure);
                write_vector(state_file, flux.viscous);
                write_vector(state_file, flux.advection);
                write_vector(state_file, exact_flux.pressure);
                write_vector(state_file, exact_flux.viscous);
                write_vector(state_file, exact_flux.advection);
                state_file << ',' << div_rms << ',' << div_max << '\n';
                for (int face = 0; face < 2 * NDIM; ++face)
                {
                    const std::string face_name = std::array<std::string, 6>{ "x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper" }[face];
                    for (const auto& term : std::array<std::pair<std::string, const Eigen::Matrix<double, NDIM, 2 * NDIM>*>, 3>{
                             std::make_pair("pressure", &flux.pressure_by_face),
                             std::make_pair("viscous", &flux.viscous_by_face),
                             std::make_pair("advection", &flux.advection_by_face) })
                    {
                        face_file << time << ',' << term.first << ',' << face_name;
                        Eigen::Vector3d actual = term.second->col(face);
                        Eigen::Vector3d exact = Eigen::Vector3d::Zero();
                        if (term.first == "pressure") exact = exact_flux.pressure_by_face.col(face);
                        else if (term.first == "viscous") exact = exact_flux.viscous_by_face.col(face);
                        else exact = exact_flux.advection_by_face.col(face);
                        write_vector(face_file, actual);
                        write_vector(face_file, exact);
                        face_file << '\n';
                    }
                }
                state_file.flush();
                face_file.flush();
            }
            return std::make_pair(momentum, flux);
        };

        auto previous = sample_state(time_integrator->getIntegratorTime(), true);
        double time = time_integrator->getIntegratorTime();
        while (time_integrator->stepsRemaining() && time < time_integrator->getEndTime())
        {
            const double old_time = time_integrator->getIntegratorTime();
            const double dt_request = time_integrator->getMaximumTimeStepSize();
            time_integrator->advanceHierarchy(dt_request);
            time = time_integrator->getIntegratorTime();
            const double dt = time - old_time;
            auto current = sample_state(time, true);
            const Eigen::Vector3d dPdt = (current.first - previous.first) / dt;
            const Eigen::Vector3d pressure = 0.5 * (previous.second.pressure + current.second.pressure);
            const Eigen::Vector3d viscous = 0.5 * (previous.second.viscous + current.second.viscous);
            const Eigen::Vector3d advection = 0.5 * (previous.second.advection + current.second.advection);
            const Eigen::Vector3d residual = dPdt - pressure - viscous - advection;
            const double term_scale = dPdt.norm() + pressure.norm() + viscous.norm() + advection.norm();
            const double normalized_term = term_scale > 0.0 ? residual.norm() / term_scale : 0.0;
            const double normalized_characteristic = force_scale > 0.0 ? residual.norm() / force_scale : 0.0;
            if (IBTK_MPI::getRank() == 0)
            {
                balance_file << old_time << ',' << time << ',' << dt;
                write_vector(balance_file, dPdt);
                write_vector(balance_file, pressure);
                write_vector(balance_file, viscous);
                write_vector(balance_file, advection);
                write_vector(balance_file, residual);
                balance_file << ',' << normalized_term << ',' << normalized_characteristic;
                write_vector(balance_file, exact_flux.pressure);
                write_vector(balance_file, exact_flux.viscous);
                write_vector(balance_file, exact_flux.advection);
                balance_file << '\n';
                balance_file.flush();
            }
            previous = std::move(current);
        }

        if (IBTK_MPI::getRank() == 0)
        {
            state_file.close();
            balance_file.close();
            face_file.close();
        }
        for (int d = 0; d < NDIM; ++d) delete u_bc_coefs[d];
    }
    return 0;
}
