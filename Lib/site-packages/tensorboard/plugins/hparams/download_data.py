# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Classes and functions for handling the DownloadData API call."""


import csv
import io
import math

from tensorboard.plugins.hparams import error


class OutputFormat:
    """An enum used to list the valid output formats for API calls."""

    JSON = "json"
    CSV = "csv"
    LATEX = "latex"


class Handler:
    """Handles a DownloadData request."""

    def __init__(
        self,
        context,
        experiment,
        session_groups,
        response_format,
        columns_visibility,
    ):
        """Constructor.

        Args:
          context: A backend_context.Context instance.
          experiment: Experiment proto.
          session_groups: ListSessionGroupsResponse proto.
          response_format: A string in the OutputFormat enum.
          columns_visibility: A list of boolean values to filter columns.
        """
        self._context = context
        self._experiment = experiment
        self._session_groups = session_groups
        self._response_format = response_format
        self._columns_visibility = columns_visibility

    def run(self):
        """Handles the request specified on construction.

        Returns:
          A response body.
          A mime type (string) for the response.
        """
        experiment = self._experiment
        session_groups = self._session_groups
        response_format = self._response_format
        visibility = self._columns_visibility

        header = []
        for hparam_info in experiment.hparam_infos:
            header.append(hparam_info.display_name or hparam_info.name)

        for metric_info in experiment.metric_infos:
            header.append(metric_info.display_name or metric_info.name.tag)

        def _filter_columns(row):
            return [value for value, visible in zip(row, visibility) if visible]

        header = _filter_columns(header)

        rows = []

        def _get_value(value):
            if value.HasField("number_value"):
                return value.number_value
            if value.HasField("string_value"):
                return value.string_value
            if value.HasField("bool_value"):
                return value.bool_value
            # hyperparameter values can be optional in a session group
            return ""

        def _get_metric_id(metric):
            return metric.group + "." + metric.tag

        for group in session_groups.session_groups:
            row = []
            for hparam_info in experiment.hparam_infos:
                row.append(_get_value(group.hparams[hparam_info.name]))
            metric_values = {}
            for metric_value in group.metric_values:
                metric_id = _get_metric_id(metric_value.name)
                metric_values[metric_id] = metric_value.value
            for metric_info in experiment.metric_infos:
                metric_id = _get_metric_id(metric_info.name)
                row.append(metric_values.get(metric_id))
            rows.append(_filter_columns(row))

        if response_format == OutputFormat.JSON:
            mime_type = "application/json"
            body = dict(header=header, rows=rows)
        elif response_format == OutputFormat.LATEX:

            def latex_format(value):
                if value is None:
                    return "-"
                elif isinstance(value, int):
                    return "$%d$" % value
                elif isinstance(value, float):
                    if math.isnan(value):
                        return r"$\mathrm{NaN}$"
                    if value in (float("inf"), float("-inf")):
                        return r"$%s\infty$" % ("-" if value < 0 else "+")
                    scientific = "%.3g" % value
                    if "e" in scientific:
                        coefficient, exponent = scientific.split("e")
                        return "$%s\\cdot 10^{%d}$" % (
                            coefficient,
                            int(exponent),
                        )
                    return "$%s$" % scientific
                return value.replace("_", "\\_").replace("%", "\\%")

            mime_type = "application/x-latex"
            top_part = "\\begin{table}[tbp]\n\\begin{tabular}{%s}\n" % (
                "l" * len(header)
            )
            header_part = (
                " & ".join(map(latex_format, header)) + " \\\\ \\hline\n"
            )
            middle_part = "".join(
                " & ".join(map(latex_format, row)) + " \\\\\n" for row in rows
            )
            bottom_part = "\\hline\n\\end{tabular}\n\\end{table}\n"
            body = top_part + header_part + middle_part + bottom_part
        elif response_format == OutputFormat.CSV:
            string_io = io.StringIO()
            writer = csv.writer(string_io)
            writer.writerow(header)
            writer.writerows(rows)
            body = string_io.getvalue()
            mime_type = "text/csv"
        else:
            raise error.HParamsError(
                "Invalid reponses format: %s" % response_format
            )
        return body, mime_type
