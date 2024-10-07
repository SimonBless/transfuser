#!/bin/bash
export WORK_DIR=/home/simon/Documents/Studium/8.Semester/Bachelorarbeit_Autonomes_Fahren/transfuser_containerized/datagen/transfuser
export CARLA_ROOT=${WORK_DIR}/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export TEAM_CODE_AUTOPILOT_ROOT=${WORK_DIR}/team_code_autopilot
export TEAM_CODE_TRANSFUSER_ROOT=${WORK_DIR}/team_code_transfuser
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

if [ -z "$CARLA_ROOT" ]
then
    echo "Error $CARLA_ROOT is empty. Set \$CARLA_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$SCENARIO_RUNNER_ROOT" ]
then echo "Error $SCENARIO_RUNNER_ROOT is empty. Set \$SCENARIO_RUNNER_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$LEADERBOARD_ROOT" ]
then echo "Error $LEADERBOARD_ROOT is empty. Set \$LEADERBOARD_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$TEAM_CODE_AUTOPILOT_ROOT" ]
then echo "Error $TEAM_CODE_AUTOPILOT_ROOT is empty. Set \$TEAM_CODE_ROOT as an environment variable first."
    exit 1
fi

if [ -z "$TEAM_CODE_TRANSFUSER_ROOT" ]
then echo "Error $TEAM_CODE_TRANSFUSER_ROOT is empty. Set \$TEAM_CODE_ROOT as an environment variable first."
    exit 1
fi

mkdir -p .tmp

cp -fr ${CARLA_ROOT}/PythonAPI  .tmp
#mv .tmp/PythonAPI/carla/dist/carla*-py2*.egg .tmp/PythonAPI/carla/dist/carla-leaderboard-py2.7.egg
mv .tmp/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg .tmp/PythonAPI/carla/dist/carla-leaderboard-py3x.egg

cp -fr ${SCENARIO_RUNNER_ROOT}/ .tmp
cp -fr ${LEADERBOARD_ROOT}/ .tmp
cp -fr ${TEAM_CODE_AUTOPILOT_ROOT}/ .tmp
cp -fr ${TEAM_CODE_TRANSFUSER_ROOT}/ .tmp

# build docker image
docker build --force-rm --network host -t transfuser-agent-cuda -f Dockerfile.master .

rm -fr .tmp
