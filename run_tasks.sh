#!/bin/bash

set -e
set -x
PYTHONPATH=$(readlink -e .):$PYTHONPATH
export PYTHONPATH
PYTHON=/work/frink/deyoung.j/llm-meta-analysis/venv/bin/python
set -o nounset
set -o pipefail
MAX_JOBS=15

function skip_ckpt {
    local cmd="$1"
    local name="$2"
    echo "skipping running job $name"
}

# when failure really is an option!
# like when I want to a pile of training jobs sequentially and don't have time to monitor them.
function allow_fail_ckpt {
    local cmd="$1"
    local name="$2"
    local ckpt_file="$ARTIFACTS/logs/$name.ckpt"
    local partial_ckpt_file="$ARTIFACTS/logs/$name.partial"
    local log_file_base="$ARTIFACTS/logs/$name"
    mkdir -p "$(dirname $ckpt_file)" "$(dirname $log_file_base)"
    if [ -e "$partial_ckpt_file" ] ; then
        cat "$partial_ckpt_file" >> "$partial_ckpt_file".old
    fi
    if [ ! -e "$ckpt_file" ] ; then
        echo "running $name; $cmd"
        echo "$cmd" > "$partial_ckpt_file"
        if [ -e "${log_file_base}.e" ]; then
            mv "${log_file_base}.e" "${log_file_base}.e.old"
        fi
        if [ -e "${log_file_base}.o" ]; then
            mv "${log_file_base}.o" "${log_file_base}.o.old"
        fi
        # shellcheck disable=SC2086
        eval $cmd > >(tee "${log_file_base}.o") 2> >(tee "${log_file_base}.e" >&2) && touch $ckpt_file || (echo "failed $name ; $cmd" ; echo "$cmd;" "$log_file_base."{e.o} >> $FAILED_FILE ; echo "to skip: touch $ckpt_file"; true) || true
    fi
}

function ckpt {
    local cmd="$1"
    local name="$2"
    local ckpt_file="$ARTIFACTS/logs/$name.ckpt"
    local partial_ckpt_file="$ARTIFACTS/logs/$name.partial"
    local log_file_base="$ARTIFACTS/logs/$name"
    mkdir -p "$(dirname $ckpt_file)" "$(dirname $log_file_base)"
    if [ -e "$partial_ckpt_file" ] ; then
        cat "$partial_ckpt_file" >> "$partial_ckpt_file".old
    fi
    if [ ! -e "$ckpt_file" ] ; then
        echo "running $name; $cmd"
        echo "$cmd" > "$partial_ckpt_file"
        if [ -e "${log_file_base}.e" ]; then
            mv "${log_file_base}.e" "${log_file_base}.e.old"
        fi
        if [ -e "${log_file_base}.o" ]; then
            mv "${log_file_base}.o" "${log_file_base}.o.old"
        fi
        # shellcheck disable=SC2086
        #eval $cmd > >(tee "${log_file_base}.o") 2> >(tee "${log_file_base}.e" >&2) && touch $ckpt_file || (echo "failed $name ; $cmd" ; exit 1)
        eval $cmd > >(tee "${log_file_base}.o") 2> >(tee "${log_file_base}.e" >&2) && touch $ckpt_file || (echo "failed $name ; $cmd" ; echo "$cmd;" "$log_file_base."{e.o} >> $FAILED_FILE ; echo "to skip: touch $ckpt_file"; exit 1)
        #else
        #echo "already ran '$name'; clear '$ckpt_file' to rerun"
        fi
}

ARTIFACTS=/work/frink/deyoung.j/llm-meta-analysis/expts
OUTPUTS=$ARTIFACTS
mkdir -p $ARTIFACTS
FAILED_FILE=$ARTIFACTS/failed
rm -f $FAILED_FILE

for split in dev test ; do
    #for model in mistral7B ; do
    for model in llama31 mistral7B ; do
        #for task in outcome_type binary_outcomes continuous_outcomes end_to_end ; do
        for task in outcome_type binary_outcomes continuous_outcomes ; do
            name="model_$model/task_$task/split_$split"
            echo "$name"
            mkdir -p "$OUTPUTS/$name"
            ckpt "srun -p 177huntington --gres gpu:1 --mem 32G -J $model/$task \
            $PYTHON \
            evaluation/run_task.py \
            --model $model \
            --task $task \
            --split $split \
            --output_path $OUTPUTS/$name
            " $name
            ckpt "srun -p short --mem 4G -J $model/$task \
            $PYTHON \
            evaluation/evaluate_output.py \
            --task $task \
            --output_path $OUTPUTS/$name/${model}_${task}_${split}_output.json \
            --metrics_path $OUTPUTS/$name \
            " $name/evaluate
        done
    done
done
