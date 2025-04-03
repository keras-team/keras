/*
 *
 * Copyright 2016 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GRPC_CORE_LIB_TRANSPORT_PID_CONTROLLER_H
#define GRPC_CORE_LIB_TRANSPORT_PID_CONTROLLER_H

#include <grpc/support/port_platform.h>

#include <limits>

/* \file Simple PID controller.
   Implements a proportional-integral-derivative controller.
   Used when we want to iteratively control a variable to converge some other
   observed value to a 'set-point'.
   Gains can be set to adjust sensitivity to current error (p), the integral
   of error (i), and the derivative of error (d). */

namespace grpc_core {

class PidController {
 public:
  class Args {
   public:
    double gain_p() const { return gain_p_; }
    double gain_i() const { return gain_i_; }
    double gain_d() const { return gain_d_; }
    double initial_control_value() const { return initial_control_value_; }
    double min_control_value() const { return min_control_value_; }
    double max_control_value() const { return max_control_value_; }
    double integral_range() const { return integral_range_; }

    Args& set_gain_p(double gain_p) {
      gain_p_ = gain_p;
      return *this;
    }
    Args& set_gain_i(double gain_i) {
      gain_i_ = gain_i;
      return *this;
    }
    Args& set_gain_d(double gain_d) {
      gain_d_ = gain_d;
      return *this;
    }
    Args& set_initial_control_value(double initial_control_value) {
      initial_control_value_ = initial_control_value;
      return *this;
    }
    Args& set_min_control_value(double min_control_value) {
      min_control_value_ = min_control_value;
      return *this;
    }
    Args& set_max_control_value(double max_control_value) {
      max_control_value_ = max_control_value;
      return *this;
    }
    Args& set_integral_range(double integral_range) {
      integral_range_ = integral_range;
      return *this;
    }

   private:
    double gain_p_ = 0.0;
    double gain_i_ = 0.0;
    double gain_d_ = 0.0;
    double initial_control_value_ = 0.0;
    double min_control_value_ = std::numeric_limits<double>::min();
    double max_control_value_ = std::numeric_limits<double>::max();
    double integral_range_ = std::numeric_limits<double>::max();
  };

  explicit PidController(const Args& args);

  /// Reset the controller internal state: useful when the environment has
  /// changed significantly
  void Reset() {
    last_error_ = 0.0;
    last_dc_dt_ = 0.0;
    error_integral_ = 0.0;
  }

  /// Update the controller: given a current error estimate, and the time since
  /// the last update, returns a new control value
  double Update(double error, double dt);

  /// Returns the last control value calculated
  double last_control_value() const { return last_control_value_; }

  /// Returns the current error integral (mostly for testing)
  double error_integral() const { return error_integral_; }

 private:
  double last_error_ = 0.0;
  double error_integral_ = 0.0;
  double last_control_value_;
  double last_dc_dt_ = 0.0;
  const Args args_;
};

}  // namespace grpc_core

#endif /* GRPC_CORE_LIB_TRANSPORT_PID_CONTROLLER_H */
