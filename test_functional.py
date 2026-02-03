"""Test functional workflow using styx runner."""

import logging
from pathlib import Path

from niwrap_helper import setup_styx

from rbc.workflows.functional import single_session

logger, runner = setup_styx(
    runner="docker", 
    image_overrides={"fcpindi/c-pac:latest": "fcpindi/c-pac:release-v1.8.5.dev1"}
)
logger.setLevel(logging.DEBUG)
runner.data_dir = Path("/Users/janhavi.pillai/Desktop/projects/rbc_output/func-init")
runner.docker_user_id = 0
runner.environ = {
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "1",
    "ANTS_RANDOM_SEED": "77742777"
}

out_dir = Path("/Users/janhavi.pillai/Desktop/projects/rbc_output/func-init/output")
bold = Path("/Users/janhavi.pillai/Desktop/bids/sub-NDARINV003RTV85/ses-baselineYear1Arm1/func/sub-NDARINV003RTV85_ses-baselineYear1Arm1_task-rest_run-01_bold.nii")

single_session(in_bold=bold, output_dir=out_dir, start_tr=2)
