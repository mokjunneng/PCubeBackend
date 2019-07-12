import os
import time
import io
import datetime
import numpy as np

from flask import Flask, request, abort, _request_ctx_stack, render_template, redirect, session, url_for, jsonify, flash
from flask_restful import Api, Resource, reqparse
from flask_httpauth import HTTPBasicAuth
import boto3

import opti.Optimisation as Opti
import vision.main as Vision
import opti.AWS as AWS
import messages

app = Flask(__name__)
api = Api(app)

# TODO: Exception handling for all the APIS????


class InitializeExperiment(Resource):
    def post(self):
        # TODO: confirm table schema
        params = request.args.get("params")
        AWS.initialize_parameters_minmax(id, params)
        return {'message': messages.EXPERIMENT_INITIALIZED}

class Sampling(Resource):
    def post(self):
        # TODO: confirm dict key names, check if should return list of recipe uuids
        experiment_id = request.args.get("id")
        experiment_duration = request.args.get("duration")
        num_samples = request.args.get("num_samples")
        min_list, max_list, step_list, params_list = AWS.get_saved_parameters_minmax(experiment_id)
        samples_uuid = Opti.initialisation(min_list, max_list, step_list, params_list, num_samples, experiment_id, experiment_duration)
        return {'message': messages.SAMPLING_DONE, 'samples_uuid': samples_uuid}, 200

class RunSample(Resource):
    def post(self):
        # TODO: confirm dict key names, whether to return start time
        experiment_id = request.args.get("id")
        recipe_uuid = request.args.get("uuid")
        experiment_start_time = datetime.datetime.now().timestamp()
        Opti.start_experiment(experiment_id, recipe_uuid, experiment_start_time)
        # TODO: Send mqtt command to capsule to start experiment
        return {'message': messages.EXPERIMENT_STARTED, 'start_time': experiment_start_time}

class DoOptimisation(Resource):
    def post(self):
        # TODO: confirm dict key names
        experiment_id = request.args.get("id")
        # TODO: get params_list and params_dict <- what format?
        pred_y, new_x = Opti.do_optimisation(experiment_id, parameters_list, parameters_dict)
        # TODO: inform capsule to start opti experiment
        # TODO: confirm return json
        return {'message': messages.GA_OPTIMISATION_COMPLETED, 'pred_y': pred_y, 'new_x': new_x}

class DoExploration(Resource):
    def post(self):
        # TODO: confirm dict key names
        experiment_id = request.args.get("id")
        # TODO: get params_list, params_dict, mins, maxs, steps <- what format?
        uc_x = Opti.do_exploration(experiment_id, parameters_dict, parameters_dict, min_list, max_list, step_list)
        # TODO: inform capsule to start exploration experiment
        # TODO: confirm return json
        return {'message': messages.UC_EXPLORATION_COMPLETED, 'uc_x': uc_x}
       
class ExperimentDone(Resource):
    '''Subscribe to MQTT topic -> get notified when experiment is done'''
    def post(self):
        # TODO: confirm dict key names
        experiment_id = request.args.get("id")
        recipe_uuid = request.args.get("uuid")
        start_date = request.args.get("start-date")
        end_date = request.args.get("end-date")

        extra_params = {}  # For params such as: number_of_threads, stereo flag
        Vision.run(start_date, end_date)

        growth_score = np.load(os.path.join("results", "growth_score.npy"))
        Opti.update_growth(experiment_id, recipe_uuid, growth_score)

api.add_resource(InitializeExperiment, '/opti/api/v1/init')
api.add_resource(Sampling, '/opti/api/v1/sampling')
api.add_resource(RunSample, '/opti/api/v1/run_sample')
api.add_resource(DoOptimisation, '/opti/api/v1/do_optimisation')
api.add_resource(DoExploration, '/opti/api/v1/do_exploration')
api.add_resource(ExperimentDone, '/pcube/api/v1/experiment_done')
