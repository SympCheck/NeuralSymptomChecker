# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a boilerplate pipeline 'symptom_checker'
generated using Kedro 0.17.5
"""

from kedro.framework.session import get_current_session
from kedro.pipeline import Pipeline, node

from .nodes import preprocess_extra_params, conf_combination, preprocess_dataset, run_model_training
from .nodes import SymptToSymptDataset
from .models import SymptomChecker


def dataset_processing_pipeline(dataset_name, conf_name, dataset_class, ds_mode, **kwargs):
    return Pipeline([
        node(
            func=preprocess_dataset,
            inputs=[dataset_name, conf_name],
            outputs=[f'{ds_mode}_explicit_data', f'{ds_mode}_implicit_symptoms', f'{ds_mode}_diagnosis'],
            name=f'{ds_mode}_dataset_unpacking'
        ),
        node(
            func=dataset_class,
            inputs=[f'{ds_mode}_explicit_data', f'{ds_mode}_implicit_symptoms', f'{ds_mode}_diagnosis', conf_name],
            outputs=f'{ds_mode}_dataset',
            name=f'{ds_mode}_pytorch_dataset_creation'
        )
    ])


def base_model_training_pipeline(model_class, 
                                model_name,
                                dataset_class,
                                common_conf_name,
                                architectire_conf_name,
                                **kwargs):

    session = get_current_session()
    context = session.load_context()

    ds_name = context.params['ds_name']
    mode = context.params['mode']

    actual_configs = [common_conf_name, f'params:{ds_name}_conf', f'{architectire_conf_name}_{ds_name}_conf']
    all_configs = {'actual_configs': 'actual_configs'}
    all_configs.update(preprocess_extra_params(actual_configs, context))

    pipe = Pipeline([
        node(
            func=lambda *args: list(args),
            inputs=actual_configs,
            outputs='actual_configs',
            name='dump_actual_configs'
        ),
        node(
            func=conf_combination,
            inputs=all_configs,
            outputs='conf',
            name='combine_all_configs'
        ),
    ])

    training_model_inputs = {'model': 'model', 'conf': 'conf', 'device': 'params:device', 'dataset_name': 'params:ds_name'}
    ds_modes = context.params[f'{ds_name}_conf'][mode]
    for ds_mode in ds_modes:
        pipe += dataset_processing_pipeline(f'{ds_name}_{ds_mode}_df', 'conf', dataset_class, ds_mode, **kwargs)
        training_model_inputs[f'{ds_mode}_dataset'] = f'{ds_mode}_dataset'

    if context.params.get('load_from_pretrained', False):
        pipe += Pipeline([
            node(
                func=lambda model: model,
                inputs=f'{model_name}_trained_on_{ds_name}',
                outputs='model',
                name='model_load'
            ),
        ])
    else:
        pipe += Pipeline([
            node(
                func=model_class,
                inputs='conf',
                outputs='model',
                name='model_init'
            ),
        ])

    pipe += Pipeline([
        node(
            func=run_model_training,
            inputs=training_model_inputs,
            outputs=f'{model_name}_trained_on_{ds_name}',
            name='training_model'
        )
    ])

    return pipe


def sympt_checker(**kwargs):
    return base_model_training_pipeline(SymptomChecker,
                                        'sympt_checker_model',
                                        SymptToSymptDataset,
                                        'params:common_conf',
                                        'params:SymptomChecker',
                                        **kwargs)