# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile


class paddle_frontend_converter:
    def __init__(self, model, inputs=None, outputs=None):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.tmp = None
        self.model_name = None
        self.pdmodel = None
        self.pdiparams = None
        self.pdiparams_info = None
        self.is_generated = False

    def destroy(self):
        # close tmp file
        if isinstance(self.tmp, tempfile._TemporaryFileWrapper):
            self.tmp.close()

        # remove the *.pdmodel
        if os.path.exists(self.pdmodel):
            os.remove(self.pdmodel)

        # remove the *.pdiparams
        if os.path.exists(self.pdiparams):
            os.remove(self.pdiparams)

        # remove the *.pdiparams.info
        if os.path.exists(self.pdiparams_info):
            os.remove(self.pdiparams_info)

    def convert_paddle_to_pdmodel(self):
        '''
            There are three paddle model categories:
            - High Level API: is a wrapper for dynamic or static model, use `self.save` to serialize
            - Dynamic Model: use `paddle.jit.save` to serialize
            - Static Model: use `paddle.static.save_inference_model` to serialize
        '''
        try:
            self.tmp = tempfile.NamedTemporaryFile(delete=True)
            self.model_name = self.tmp.name
            self.pdmodel = "{}.pdmodel".format(self.model_name)
            self.pdiparams = "{}.pdiparams".format(self.model_name)
            self.pdiparams_info = "{}.pdiparams.info".format(self.model_name)

            import paddle  # pylint: disable=import-error
            if isinstance(self.model, paddle.hapi.model.Model):
                self.model.save(self.model_name, False)
            else:
                if self.inputs is None:
                    raise RuntimeError(
                        "Saving inference model needs 'inputs' before saving. Please specify 'example_input'"
                    )
                if isinstance(self.model, paddle.fluid.dygraph.layers.Layer):
                    with paddle.fluid.framework._dygraph_guard(None):
                        paddle.jit.save(self.model, self.model_name, input_spec=self.inputs, output_spec=self.outputs)
                elif isinstance(self.model, paddle.fluid.executor.Executor):
                    if self.outputs is None:
                        raise RuntimeError(
                            "Model is static. Saving inference model needs 'outputs' before saving. Please specify 'output' for this model"
                        )
                    paddle.static.save_inference_model(self.model_name, self.inputs, self.outputs, self.model)
                else:
                    raise RuntimeError(
                        "Conversion just support paddle.hapi.model.Model, paddle.fluid.dygraph.layers.Layer and paddle.fluid.executor.Executor"
                    )

            if not os.path.exists(self.pdmodel):
                print("Failed generating paddle inference format model")
                sys.exit(1)

            self.is_generated = True
            return self.pdmodel
        finally:
            # close tmp file
            if isinstance(self.tmp, tempfile._TemporaryFileWrapper):
                self.tmp.close()
