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
"""The TensorBoard HParams plugin.

See `http_api.md` in this directory for specifications of the routes for
this plugin.
"""

import json

import werkzeug
from werkzeug import wrappers

from tensorboard import plugin_util
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import backend_context
from tensorboard.plugins.hparams import download_data
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import get_experiment
from tensorboard.plugins.hparams import list_metric_evals
from tensorboard.plugins.hparams import list_session_groups
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.util import tb_logging


logger = tb_logging.get_logger()


class HParamsPlugin(base_plugin.TBPlugin):
    """HParams Plugin for TensorBoard.

    It supports both GETs and POSTs. See 'http_api.md' for more details.
    """

    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates HParams plugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._context = backend_context.Context(context)

    def get_plugin_apps(self):
        """See base class."""

        return {
            "/download_data": self.download_data_route,
            "/experiment": self.get_experiment_route,
            "/session_groups": self.list_session_groups_route,
            "/metric_evals": self.list_metric_evals_route,
        }

    def is_active(self):
        return False  # `list_plugins` as called by TB core suffices

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name="tf-hparams-dashboard")

    # ---- /download_data- -------------------------------------------------------
    @wrappers.Request.application
    def download_data_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            response_format = request.args.get("format")
            columns_visibility = json.loads(
                request.args.get("columnsVisibility")
            )
            request_proto = _parse_request_argument(
                request, api_pb2.ListSessionGroupsRequest
            )
            session_groups = list_session_groups.Handler(
                ctx, self._context, experiment_id, request_proto
            ).run()
            experiment = get_experiment.Handler(
                ctx, self._context, experiment_id, request_proto
            ).run()
            body, mime_type = download_data.Handler(
                self._context,
                experiment,
                session_groups,
                response_format,
                columns_visibility,
            ).run()
            return http_util.Respond(request, body, mime_type)
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    # ---- /experiment -----------------------------------------------------------
    @wrappers.Request.application
    def get_experiment_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            request_proto = _parse_request_argument(
                request, api_pb2.GetExperimentRequest
            )
            response_proto = get_experiment.Handler(
                ctx,
                self._context,
                experiment_id,
                request_proto,
            ).run()
            response = plugin_util.proto_to_json(response_proto)
            return http_util.Respond(
                request,
                response,
                "application/json",
            )
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    # ---- /session_groups -------------------------------------------------------
    @wrappers.Request.application
    def list_session_groups_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            request_proto = _parse_request_argument(
                request, api_pb2.ListSessionGroupsRequest
            )
            response_proto = list_session_groups.Handler(
                ctx,
                self._context,
                experiment_id,
                request_proto,
            ).run()
            response = plugin_util.proto_to_json(response_proto)
            return http_util.Respond(
                request,
                response,
                "application/json",
            )
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    # ---- /metric_evals ---------------------------------------------------------
    @wrappers.Request.application
    def list_metric_evals_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            request_proto = _parse_request_argument(
                request, api_pb2.ListMetricEvalsRequest
            )
            scalars_plugin = self._get_scalars_plugin()
            if not scalars_plugin:
                raise werkzeug.exceptions.NotFound("Scalars plugin not loaded")
            return http_util.Respond(
                request,
                list_metric_evals.Handler(
                    ctx, request_proto, scalars_plugin, experiment_id
                ).run(),
                "application/json",
            )
        except error.HParamsError as e:
            logger.error("HParams error: %s" % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    def _get_scalars_plugin(self):
        """Tries to get the scalars plugin.

        Returns:
        The scalars plugin or None if it is not yet registered.
        """
        return self._context.tb_context.plugin_name_to_instance.get(
            scalars_metadata.PLUGIN_NAME
        )


def _parse_request_argument(request, proto_class):
    request_json = (
        request.data
        if request.method == "POST"
        else request.args.get("request")
    )
    try:
        return json_format.Parse(request_json, proto_class())
    # if request_json is None, json_format.Parse will throw an AttributeError:
    # 'NoneType' object has no attribute 'decode'.
    except (AttributeError, json_format.ParseError) as e:
        raise error.HParamsError(
            "Expected a JSON-formatted request data of type: {}, but got {} ".format(
                proto_class, request_json
            )
        ) from e
