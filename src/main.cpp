// Author: sameeragarwal@google.com (Sameer Agarwal)
#include "ceres/ceres.h"
#include "ceres/covariance.h"
#include "glog/logging.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include "data.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

const int kNumObservations = 67;
const double data[] = {
  0.000000e+00,
  1.133898e+00,
  7.500000e-02,
  1.334902e+00,
  1.500000e-01,
  1.213546e+00,
  2.250000e-01,
  1.252016e+00,
  3.000000e-01,
  1.392265e+00,
  3.750000e-01,
  1.314458e+00,
  4.500000e-01,
  1.472541e+00,
  5.250000e-01,
  1.536218e+00,
  6.000000e-01,
  1.355679e+00,
  6.750000e-01,
  1.463566e+00,
  7.500000e-01,
  1.490201e+00,
  8.250000e-01,
  1.658699e+00,
  9.000000e-01,
  1.067574e+00,
  9.750000e-01,
  1.464629e+00,
  1.050000e+00,
  1.402653e+00,
  1.125000e+00,
  1.713141e+00,
  1.200000e+00,
  1.527021e+00,
  1.275000e+00,
  1.702632e+00,
  1.350000e+00,
  1.423899e+00,
  1.425000e+00,
  1.543078e+00,
  1.500000e+00,
  1.664015e+00,
  1.575000e+00,
  1.732484e+00,
  1.650000e+00,
  1.543296e+00,
  1.725000e+00,
  1.959523e+00,
  1.800000e+00,
  1.685132e+00,
  1.875000e+00,
  1.951791e+00,
  1.950000e+00,
  2.095346e+00,
  2.025000e+00,
  2.361460e+00,
  2.100000e+00,
  2.169119e+00,
  2.175000e+00,
  2.061745e+00,
  2.250000e+00,
  2.178641e+00,
  2.325000e+00,
  2.104346e+00,
  2.400000e+00,
  2.584470e+00,
  2.475000e+00,
  1.914158e+00,
  2.550000e+00,
  2.368375e+00,
  2.625000e+00,
  2.686125e+00,
  2.700000e+00,
  2.712395e+00,
  2.775000e+00,
  2.499511e+00,
  2.850000e+00,
  2.558897e+00,
  2.925000e+00,
  2.309154e+00,
  3.000000e+00,
  2.869503e+00,
  3.075000e+00,
  3.116645e+00,
  3.150000e+00,
  3.094907e+00,
  3.225000e+00,
  2.471759e+00,
  3.300000e+00,
  3.017131e+00,
  3.375000e+00,
  3.232381e+00,
  3.450000e+00,
  2.944596e+00,
  3.525000e+00,
  3.385343e+00,
  3.600000e+00,
  3.199826e+00,
  3.675000e+00,
  3.423039e+00,
  3.750000e+00,
  3.621552e+00,
  3.825000e+00,
  3.559255e+00,
  3.900000e+00,
  3.530713e+00,
  3.975000e+00,
  3.561766e+00,
  4.050000e+00,
  3.544574e+00,
  4.125000e+00,
  3.867945e+00,
  4.200000e+00,
  4.049776e+00,
  4.275000e+00,
  3.885601e+00,
  4.350000e+00,
  4.110505e+00,
  4.425000e+00,
  4.345320e+00,
  4.500000e+00,
  4.161241e+00,
  4.575000e+00,
  4.363407e+00,
  4.650000e+00,
  4.161576e+00,
  4.725000e+00,
  4.619728e+00,
  4.800000e+00,
  4.737410e+00,
  4.875000e+00,
  4.727863e+00,
  4.950000e+00,
  4.669206e+00,
};
class ParametersBase
{
public:
  ParametersBase() = default;
  virtual ~ParametersBase() = default;
  virtual int nParams() = 0;
  virtual double *getParameters() = 0;
};

class Parameters : public ParametersBase
{

public:
  Parameters(std::vector<double> parameterValues, std::vector<std::string> parameterNames) : parameterValues_(std::move(parameterValues)),
                                                                                             parameterNames_(std::move(parameterNames)) {}

private:
  std::vector<double> parameterValues_;
  std::vector<std::string> parameterNames_;
};

class AutoModel
{
public:
  AutoModel() = default;
  template<typename T>
  T function(double x, const T *const parameters) const
  {
    return exp(parameters[0] * x + parameters[1]);
  }

private:
  std::unique_ptr<ParametersBase> parameters_;
};

class AutoModelGauss
{
public:
  AutoModelGauss() = default;
  template<typename T>
  T function(double x, const T *const parameters) const
  {
    return parameters[0] * sin(x * parameters[3] + parameters[2]) * exp(-x * x * parameters[1]);
  }
};

class AutoResidual
{
public:
  AutoResidual(double x, double y) : x_(x), y_(y) {}
  template<typename T>
  bool operator()(const T *const parameters, T *residual) const
  {
    residual[0] = y_ - model.function(x_, parameters);
    return true;
  }

private:
  const double x_;
  const double y_;
  AutoModel model;
};

// clang-format on
struct ExponentialResidual
{
  ExponentialResidual(double x, double y) : x_(x), y_(y) {}
  template<typename T>
  bool operator()(const T *const m, const T *const c, T *residual) const
  {
    residual[0] = y_ - exp(m[0] * x_ + c[0]);
    return true;
  }

private:
  const double x_;
  const double y_;
};
// clang-format on
struct NormalAutoResidual
{
  NormalAutoResidual(double x, double y, double eps) : x_(x), y_(y), eps_(eps) {}
  template<typename T>
  bool operator()(const T *const parameters, T *residual) const
  {
    residual[0] = (y_ - model.function(x_, parameters)) / eps_;
    return true;
  }

private:
  const double x_;
  const double y_;
  const double eps_;
  AutoModelGauss model;
};


void gaussProblem()
{

  double parameters[4] = { 10, 0.007, 0.2, 3 };
  Problem problem;
  for (int i = 0; i < numGaussObservations; ++i) {
    problem.AddResidualBlock(
      new AutoDiffCostFunction<NormalAutoResidual, 1, 4>(
        new NormalAutoResidual(gaussData[3 * i], gaussData[3 * i + 1], gaussData[3 * i + 2])),
      NULL,
      parameters);
  }
  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "Paramters are: "
            << "Amp:" << parameters[0] << "\n"
            << "decay: " << parameters[1] << "\n "
            << "phase: " << parameters[2] << "\n"
            << "Frequency: " << parameters[3] << "\n";
  //expect from lm fit
  //    amp:        7.75413024 +/- 0.14691892 (1.89%) (init = 10)
  //  decay:      0.01046706 +/- 2.6689e-04 (2.55%) (init = 0.007)
  //  phase:      0.65216015 +/- 0.01395827 (2.14%) (init = 0.2)
  //  frequency:  2.85913672 +/- 0.00116664 (0.04%) (init = 3)
  /// chi suqare
  //chi-square         = 44.5627311
  // reduced chi-square = 0.96875502
  // here we solve the problem of minimizing 1/2 chi squared
  // meaning the final cost is half of the chi squared residual
  std::cout << "Chi squared: " << 2 * summary.final_cost << std::endl;
  std::cout << "Reduced Chi squared " << 2 * summary.final_cost / (summary.num_residual_blocks - summary.num_parameters);

  ceres::Covariance::Options covoptions;
  ceres::Covariance covariance(covoptions);
  std::vector<std::pair<const double *, const double *>> covariance_blocks;
  covariance_blocks.push_back(std::make_pair(parameters, parameters));
  CHECK(covariance.Compute(covariance_blocks, &problem));
  double covariance_xx[4 * 4];
  covariance.GetCovarianceBlock(parameters, parameters, covariance_xx);
  std::cout
    << "cov[0,0]" << sqrt(covariance_xx[0]) << std::endl;
  std::cout
    << "cov[1,1]" << sqrt(covariance_xx[5]) << std::endl;
  std::cout
    << "cov[2,2]" << sqrt(covariance_xx[10]) << std::endl;
  std::cout
    << "cov[3,3]" << sqrt(covariance_xx[15]) << std::endl;
}

void testProblem()
{
  double m = 0.0;
  double c0 = 0;
  Problem problem;
  for (int i = 0; i < kNumObservations; ++i) {
    problem.AddResidualBlock(
      new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
        new ExponentialResidual(data[2 * i], data[2 * i + 1])),
      NULL,
      &m,
      &c0);
  }
  Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
}

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  // testProblem();
  gaussProblem();

  return 0;
}
