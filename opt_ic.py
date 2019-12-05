# coding: utf-8
"""
Copyright 2019 Danilo Ardagna
Copyright 2019 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from math import ceil
import os
import pickle
import sys

import numpy as np
import xgboost

from abc import ABC
from abc import abstractmethod


# import matplotlib.pyplot as plt
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'

class OptIcError(Exception):
    """ Raised when an error occurs in this module"""


def error(message):
    logging.error(message)
    raise OptIcError(message)


class performance_model(ABC):
    """
    Performance model

    Attributes
    ----------

    _app_name: string
        Application name

    _ml_model : Pickle object
        Pickle object storing the application Machine learning model

    _features : tuple
        Performance models features

    _perf_model_dic : dictionary
        Dictionary storing execution time predictions according to the model features

    _search_dic : dictionary
        Dictionary storing execution time predictions given the number of cores (monitonic function of number of cores)

    _directory : string
        Path storing the models to be loaded

    _cores_max : int
        maximum number of cores available in the infrastructure


    """

    @abstractmethod
    def _create_search_dic(self):
        """
        Create the dictionary to support the search
        """
    @abstractmethod
    def _create_perf_model_dic(self):
        """
        Create the performance model dictionary from the ML model
        """

    @abstractmethod
    def _load_CSV_perf_model_dic(self):
        """
        Create the performance model dictionary from the CSV file
        """

    def cores_range(self):
        """
        Returns the range of cores to estimate performance
        """
        # print("cores_range")
        # print(self._search_dic.keys())
        t = (min(self._search_dic.keys()), max(self._search_dic.keys()))

        # print(t)

        return t

    def _monotone_decreasing(self):
        """
        Make the  _search_dic a function monotone non incresing in the number of cores
        """
        # print("Running _monotone_decreasing")

        (n_cores_min, n_cores_max) = self.cores_range()

        prevVal = self._search_dic[n_cores_min]

        for n_cores in range(n_cores_min + 1, n_cores_max):

            if n_cores in self._search_dic:
                if self._search_dic[n_cores] > prevVal:
                    self._search_dic[n_cores] = prevVal
                else:
                    prevVal = self._search_dic[n_cores]

    @abstractmethod
    def _predict_perf_ml_model(self, n_cores):
        """
        Given the number of cores, computes the application execution time through the performance machine learning model
        """

    def __init__(self, app_name, directory, pickle_flag=True, cores_max=12):
        """
        Parameters
        ----------
        app_name : string
            file name storing data

        directory : string
            Path storing the models to be loaded

        pickle_flag : bool
            True if the ML model is provided as Pickle object, False if it is provided as CSV file

        cores_max : int
            Maximum number of cores available in the infrastructure
        """

        self._app_name = app_name
        self._directory = directory
        self._cores_max = cores_max

        # load ML model binary file
        # features order
        # "data_size","n_cores","1_over_n_corers",
        # "data_size_over_n_cores","Log2 n_cores"

        self._perf_model_dic = {}

        if pickle_flag:

            ml_file_name = os.path.join(self._directory, app_name + ".pickle")

            if not os.path.exists(ml_file_name):
                error("Model for " + app_name + " (" + ml_file_name + ") not found")
                sys.exit(1)

            ml_file = open(ml_file_name, 'rb')
            self._ml_model = pickle.load(ml_file)
            ml_file.close()
            print("ML model load ok")

        else:

            ml_file_name = os.path.join(self._directory, app_name + ".csv")
            if not os.path.exists(ml_file_name):
                error("CSV file " + ml_file_name + " for " + app_name + " not found")
                sys.exit(1)
            self._ml_model = None

    def predict(self, n_cores):
        """
        Returns performance estimate for nCores configuration
        """
        # if n_cores larger than the one available  return the best performance

        available_cores = self._search_dic.keys()

        max_cores_in_dict = max(available_cores)
        min_cores_in_dict = min(available_cores)

        eval_cores = n_cores

        if n_cores > max_cores_in_dict:
            return self.best_performance()
        elif n_cores < min_cores_in_dict:
            return self._search_dic[min_cores_in_dict]
        else:
            while (eval_cores not in available_cores and eval_cores > 1):
                eval_cores -= 1

            if eval_cores in available_cores:
                return self._search_dic[eval_cores]

            else:
                return float('inf')

    def best_performance(self):
        """
        Return the best performance achieved with the maximum cores number
        available in the serachDic
        """
        return min(self._search_dic.values())

    def _create_dictionaries(self, pickle_flag):
        """
        Create the performance model and search dictionaries
        """
        if pickle_flag:
            self._create_perf_model_dic()
        else:
            self._load_CSV_perf_model_dic()

        self._create_search_dic()

        self._monotone_decreasing()


class PerformanceModelDataAnonymization(performance_model):
    """
    Performance model

    Attributes
    ----------

    _memory: int
        SGX memory

    _images : int
        Number of images to be anonymized
    """

    def __init__(self, app_name, directory, pickle_flag=True, cores_max=4, memory=16, images=16):
        """
        Parameters
        ----------
        app_name : string
            file name storing data

        directory : string
            Path storing the models to be loaded

        pickle_flag : bool
            True if the ML model is provided as Pickle object, False if it is provided as CSV file

        memory: int
            SGX memory

        images : int
            Number of images to be anonymized
        """

        performance_model.__init__(self, app_name, directory, pickle_flag, cores_max)
        self._memory = memory
        self._images = images
        self._features = ("vcpus", "memory", "images")

        self._create_dictionaries(pickle_flag)

    def _create_perf_model_dic(self):
        """
        Create the performance model dictionary from the ML model using as features
        vcpus, memory and images
        """
    def _load_CSV_perf_model_dic(self):
        """
        Create the performance model dictionary from the CSV file
        """
        simulated_data = {}
        # simulated_data[('app','images','memory','vcpus')] = 'predicted_total_time'
        # file format
        # ['vcpus', 'memory', 'images', 'predicted_total_time']
        count = 0
        # print("_load_CSV_perf_model_dic directory: "+str(self._directory))
        # print("_load_CSV_perf_model_dic app_name: "+str(self._app_name))
        file_name = os.path.join(self._directory, self._app_name + ".csv")

        for line in open(file_name, 'r'):
            # skip header line
            if count > 0:
                row = line.split(',')
                (vcpus, memory, images, predicted_total_time) = row
                # print(row)
                images = int(images)
                memory = int(memory)
                vcpus = int(vcpus)
                predicted_total_time = float(predicted_total_time)

                simulated_data[(self._app_name, images, memory, vcpus)] = predicted_total_time

            count += 1

        self._perf_model_dic = simulated_data

        # print(self._perf_model_dic)

    def _predict_perf_ml_model(self, n_cores):
        """
        Create the performance model dictionary from the ML model using as features
        vcpus, memory and images
        """

    def _create_search_dic(self):
        """
        Create the dictionary to support the search
        """
        simulated_data = self._perf_model_dic

        keys = simulated_data.keys()
        vcpus_set = []
        for k in keys:
            (app, images, memory, vcpus) = k
            if app == self._app_name and images == self._images and memory == self._memory and vcpus <= self._cores_max:
                vcpus_set.append(vcpus)

        self._search_dic = {}

        for v in vcpus_set:
            self._search_dic[v] = self._perf_model_dic[(self._app_name, self._images, self._memory, v)]

        # print(self._search_dic)


class Optimizer(ABC):
    """
    Optimizer computing initial configuration given an a priori deadline

    Attributes
    ----------

    _performance_model : performance_model
        Application performance model
    _searchStep : int
        Granularity of the binary search
    """

    def __init__(self, performance_model):
        """
        Parameters
        ----------
        performance_model : performance_model
            Application performance model


        """
        self._performance_model = performance_model
        self._searchStep = 1

    def solve(self, deadline):
        """
        Implement dichotomic search

        Parameters
        ----------

        deadline : float
            deadline for the application execution

        Returns the minimum number of nodes (according to the system configuration)
        required to fullfill the deadline

        """

        (n_nodes_min, n_nodes_max) = self._performance_model.cores_range()

        n_nodes_min = int(n_nodes_min / self._searchStep)
        n_nodes_max = int(n_nodes_max / self._searchStep)
        n_nodes = ceil((n_nodes_max + n_nodes_min) / 2)
        n_cores = n_nodes * self._searchStep
        # print(self._performance_model)

        # if the evaluated time is less than the deadline with just one virtual machine
        # the result is 1

        # print("_searchStep")
        # print(self._searchStep)

        tmin = self._performance_model.predict(self._searchStep * n_nodes_min)
        # print("Tmin",tmin)
        if tmin < deadline:
            # print("Ncores = ", n_nodes_min, " -> time = ", tmin, " ms")
            return n_nodes_min

        if self._performance_model.best_performance() > deadline:
            # print(self._performance_model.best_performance())
            # print("The problem is unfeasible, the deadline is too strict")
            return float('inf')

        while n_nodes_max - n_nodes_min != 1:
            # print("Node min ", n_nodes_min, " Node max ", n_nodes_max)
            n_nodes = ceil((n_nodes_max + n_nodes_min) / 2)
            n_cores = n_nodes * self._searchStep
            predicted_time = self._performance_model.predict(n_cores)
            # print("Ncores = ", n_cores, " -> time = ", predicted_time, " s")
            if predicted_time > deadline:
                n_nodes_min = n_nodes
            else:
                n_nodes_max = n_nodes
        # print("Node min ", n_nodes_min, " Node max ",n_nodes_max)
        if self._performance_model.predict(n_nodes * self._searchStep) > deadline:
            return n_nodes + 1
        elif self._performance_model.predict((n_nodes - 1) * self._searchStep) < deadline:
            return n_nodes - 1
        else:
            return n_nodes


class DataAnonymizationOptimizer(Optimizer):
    """
    Optimizer computing initial configuration for image anonimization application

    """

    def __init__(self, performance_model):
        """
        Parameters
        ----------
        performance_model : performance_model
            Application performance model

        """
        Optimizer.__init__(self, performance_model)
        self._searchStep = 1


class PerformanceModelRDH(performance_model):
    """
    Performance model

    Attributes
    ----------

    _frames: int
        Number of frames extracted from the initial videos

    _files : int
        Number of files to be processed


    """

    def __init__(self, app_name, directory, pickle_flag=True, cores_max=4, frames=16, files=8):
        """
        Parameters
        ----------
        app_name : string
            file name storing data

        directory : string
            Path storing the models to be loaded

        pickle_flag : bool
            True if the ML model is provided as Pickle object, False if it is provided as CSV file

        frames: int
            Number of frames extracted from the initial videos

        files : int
            Number of files to be processed



        """

        performance_model.__init__(self, app_name, directory, pickle_flag, cores_max)
        self._frames = frames
        self._files = files
        self._features = ("processors", "files", "frames")

        self._create_dictionaries(pickle_flag)

    def _load_CSV_perf_model_dic(self):
        """
        Create the performance model dictionary from the CSV file
        """
        simulated_data = {}
        # simulated_data[('app','files','frames','processors')] = 'predicted_time'
        # file format
        # file format
        # ['files', 'frames', 'processors', 'waves', 'time', 'predicted_time']

        count = 0
        # print("_load_CSV_perf_model_dic directory: "+str(self._directory))
        # print("_load_CSV_perf_model_dic app_name: "+str(self._app_name))
        file_name = os.path.join(self._directory, self._app_name + ".csv")

        for line in open(file_name, 'r'):
            # skip header line
            if count > 0:
                row = line.split(',')
                (files, frames, processors, waves, time, predicted_time) = row
                # print(row)
                files = int(files)
                frames = int(frames)
                processors = int(processors)
                waves = int(waves)
                predicted_time = float(predicted_time)
                simulated_data[(self._app_name, files, frames, processors)] = predicted_time

            count += 1

        self._perf_model_dic = simulated_data

        # print(self._perf_model_dic)

    def _create_perf_model_dic(self):
        """
        Create the performance model dictionary from the ML model using as features
        processors, files , frames
        """
        n_procs_max = self._cores_max
        n_procs_min = 1

        for processors in range(n_procs_min, n_procs_max + 1):
            self._perf_model_dic[(self._app_name, self._files, self._frames, processors)] = self._predict_perf_ml_model(processors)

    def _predict_perf_ml_model(self, n_processors):
        """
        Given the number of cores, computes the application execution time through the performance ML model
        """
        # features order
        # 'files', 'processors','1OverFiles','1OverProcessors','1OverProcessorsFiles',
        # 'ProcessorsOverFiles','FilesOverProcessors','FilesTimesProcessors'
        x_value = [self._files, n_processors, 1 / self._files, 1 / n_processors, 1 / (self._files * n_processors), n_processors / self._files, self._files / n_processors, self._files * n_processors]

        # print("performance_modelRDH _predict_perf_ml_model")
        # print(x_value)
        # print(self._ml_model.predict(np.asarray(x_value)).reshape(1, -1))
        # print(self._ml_model.predict([x_value])[0][0])
        return self._ml_model.predict([x_value])[0][0]

    def _create_search_dic(self):
        """
        Create the dictionary to support the search
        """
        simulated_data = self._perf_model_dic

        keys = simulated_data.keys()
        processors_set = []
        for k in keys:
            (app, files, frames, processors) = k
            if app == self._app_name and files == self._files and frames == self._frames and processors <= self._cores_max:
                processors_set.append(processors)

        self._search_dic = {}

        for p in processors_set:
            self._search_dic[p] = self._perf_model_dic[(self._app_name, self._files, self._frames, p)]

        # print(self._search_dic)


class RDHOptimizer(Optimizer):
    """
    Optimizer computing initial configuration for image anonimization application

    """

    def __init__(self, performance_model, worker_cores):
        """
        Parameters
        ----------
        performance_model : performance_model
            Application performance model

        worker_cores : int
            Number of cores per worker

        """
        Optimizer.__init__(self, performance_model)
        self._searchStep = worker_cores

        self._worker_cores = worker_cores


class PerformanceModelSpark(performance_model):
    """
    Performance model

    Attributes
    ----------

    _dataSize: int
        Application data set size in GBs


    """

    def __init__(self, app_name, directory, pickle_flag=True, cores_max=48, dataSize=250):
        """
        Parameters
        ----------
        app_name : string
            file name storing data

        directory : string
            Path storing the models to be loaded

        pickle_flag : bool
            True if the ML model is provided as Pickle object, False if it is provided as CSV file

        dataSize: int
            Application data set size in GBs



        """

        performance_model.__init__(self, app_name, directory, pickle_flag, cores_max)
        self._dataSize = dataSize
        self._features = ("dataSize", "n_cores")

        self._create_dictionaries(pickle_flag)

    def _load_CSV_perf_model_dic(self):
        """
        Create the performance model dictionary from the CSV file
        """
        simulated_data = {}
        # simulated_data[('app','dataSize','n_cores')] = 'predicted_time'
        # file format
        # ['Cores','DataSize', 'App', 'Simulated\n']

        count = 0
        # print("_load_CSV_perf_model_dic directory: "+str(self._directory))
        # print("_load_CSV_perf_model_dic app_name: "+str(self._app_name))
        file_name = os.path.join(self._directory, self._app_name + ".csv")

        for line in open(file_name, 'r'):
            # skip header line
            if count > 0:
                row = line.split(',')
                # print(row)
                n_cores = int(row[0])
                data_size = int(row[1])
                app = row[2]
                # print(app)
                # print(self._app_name)
                # print(data_size)
                # print(self._dataSize)
                if app == self._app_name and data_size == self._dataSize:
                    simulated_data[(self._app_name, self._dataSize, n_cores)] = float(row[-1])
                    # print(simulated_data[(self._app_name, self._dataSize, n_cores)])
            count += 1

        self._perf_model_dic = simulated_data

        # print(self._perf_model_dic)

    def _create_perf_model_dic(self):
        """
        Create the performance model dictionary from the ML model
        """
        n_cores_max = self._cores_max
        n_cores_min = 1
        for n_cores in range(n_cores_min, n_cores_max + 1):
            self._perf_model_dic[(self._app_name, self._dataSize, n_cores)] = self._predict_perf_ml_model(n_cores)

    def _predict_perf_ml_model(self, n_cores):
        """
        Given the number of cores, computes the application execution time through the performance ML model
        """
        # features order
        # "dataSize","nContainers","1OverContainers","DataOverContainers","Log2Containers"
        x_value = [self._dataSize, n_cores, 1 / n_cores, self._dataSize / n_cores, np.log2(n_cores)]
        return self._ml_model.predict(x_value)[0] / 1000

    def _create_search_dic(self):
        """
        Create the dictionary to support the search
        """
        simulated_data = self._perf_model_dic

        keys = simulated_data.keys()
        n_cores_set = []
        for k in keys:
            (app, data_size, n_cores) = k
            if app == self._app_name and data_size == self._dataSize and n_cores <= self._cores_max:
                n_cores_set.append(n_cores)

        self._search_dic = {}

        for n in n_cores_set:
            self._search_dic[n] = self._perf_model_dic[(self._app_name, self._dataSize, n)]

        # print(self._search_dic)


class SparkOptimizer(Optimizer):
    """
    Optimizer computing initial configuration for image anonimization application

    """

    def __init__(self, performance_model, vm_cores):
        """
        Parameters
        ----------
        performance_model : performance_model
            Application performance model

        worker_cores : int
            Number of cores per worker

        """
        Optimizer.__init__(self, performance_model)
        self._searchStep = vm_cores

        self._vm_cores = vm_cores
