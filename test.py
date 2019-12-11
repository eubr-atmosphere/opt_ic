#!/usr/bin/env python3
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

import os
import sys

import opt_ic


def main():
    # The absolute path of the current script
    abs_script = os.path.abspath(sys.argv[0])

    # The root directory of the script
    abs_root = os.path.dirname(abs_script)

    models_dir = os.path.join(abs_root, "models")
    # Kmeans + pickle
    pf_spark_ml = opt_ic.PerformanceModelSpark("kmeans", models_dir, True, 48, 5)
    opt_spark_ml = opt_ic.SparkOptimizer(pf_spark_ml, 2)
    print("####################################### Spark + ML")

    for deadline in [96, 97, 105, 114, 128, 139, 145, 165, 177, 198, 226, 258, 331]:
        sol = opt_spark_ml.solve(deadline)
        if sol > 1 and pf_spark_ml.predict(sol - 1) < deadline:
            print("Unexpected condition")
            sys.exit(1)
        print("output: " + str(deadline) + " " + str(sol))

    # Kmeans + csv
    pf_spark = opt_ic.PerformanceModelSpark("kmeans", models_dir, False, 48, 5)
    opt_spark = opt_ic.SparkOptimizer(pf_spark, 2)
    print("####################################### Spark + csv")

    for deadline in [96, 97, 105, 114, 128, 139, 145, 165, 177, 198, 226, 258, 331]:
        sol = opt_spark.solve(deadline)
        if sol > 1 and pf_spark.predict(sol - 1) < deadline:
            print("Unexpected condition")
            sys.exit(1)
        print("output: " + str(deadline) + " " + str(sol))

    # sgx + csv
    # PerformanceModelDataAnonymization(app_name, directory, pickle = True, coresMax = 4, memory = 16, images = 10):
    pf_anon = opt_ic.PerformanceModelDataAnonymization("asperathosDataValidation", models_dir, False, 4, 16, 10)
    opt_anon = opt_ic.DataAnonymizationOptimizer(pf_anon)

    print("####################################### sgx + csv")
    for deadline in [185.5364935296, 174.9252917263, 155.7028881196]:
        sol = opt_anon.solve(deadline)
        if sol > 1 and pf_anon.predict(sol - 1) < deadline:
            print("Unexpected condition")
            sys.exit(1)
        print("output: " + str(deadline) + " " + str(sol))

    # radiomics + csv
    # PerformanceModelRDH(app_name, directory, pickle=True, coresMax=4, frames = 16, files = 8):
    pf_RHD = opt_ic.PerformanceModelRDH("RDHData", models_dir, False, 4, 16, 12)
    opt_RDH = opt_ic.RDHOptimizer(pf_RHD, 1)

    print("####################################### sgx + RDH")
    for deadline in [120.160927471624, 177.011577047059, 231.925289243116]:
        sol = opt_RDH.solve(deadline)
        if sol > 1 and pf_RHD.predict(sol - 1) < deadline:
            print("Unexpected condition: With deadline " + str(deadline) + " prediction is " + str(sol) + " but " + str(sol - 1) + " takes " + str(pf_RHD.predict(sol - 1)))
            sys.exit(1)
        print("output: " + str(deadline) + " " + str(sol))


if __name__ == "__main__":
    main()
