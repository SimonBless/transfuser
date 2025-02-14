export CARLA_ROOT=/workspace/carla
export WORK_DIR=/workspace

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
#export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/mount/datagen/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/mount/datagen/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${WORK_DIR}/mount/scenarios/no_scenario.json
#export SCENARIOS=${WORK_DIR}/mount/scenarios/eval_scenarios.json
#export ROUTES=${WORK_DIR}/mount/routes/longest_weathers_22.xml
export ROUTES=${WORK_DIR}/mount/routes/longest_weathers_12.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS                             
export CHECKPOINT_ENDPOINT=${WORK_DIR}/mount/results/longest6_benchmark/org_seed_1_eval_3.json
export SAVE_PATH=${WORK_DIR}/mount/results/longest6_benchmark/org_seed_1_eval_3_debug
export TEAM_AGENT=${WORK_DIR}/mount/team_code_transfuser/submission_agent.py
export TEAM_CONFIG=${WORK_DIR}/mount/model_ckpt
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export PORT=8000

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--trafficManagerPort=${PORT}
