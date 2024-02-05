# EDiReF - SemEval 2024 Task 10: Subtask 3

## Overview

This repository contains the official code for participating in the SemEval 2024 Task 10, specifically designed for the third subtask. The project focuses on a specialized component of the competition, leveraging the capabilities of the chatglm3 model for inference. It is structured to facilitate the process from raw data processing to submission-ready format conversion, solely concentrating on the outcomes relevant to Subtask 3.

## Project Structure

- `inference.py`: The inference script responsible for generating predictions for Subtask 3. Running this script will produce a file named `newtrain.json`, containing the results specific to this subtask.
- `result.py`: A utility script for transforming the `newtrain.json` file into the standard format required for competition submission.
- `combine.py`: This script is designed to integrate the results of Subtasks 1 and 2 with Subtask 3, allowing for a comprehensive submission. It performs arbitrary fill-ins based on the outcomes of the initial tasks.
- `construct_test.py`: A preprocessing tool that reformats the original test dataset to be compatible with the chatglm3 model for inference purposes.
