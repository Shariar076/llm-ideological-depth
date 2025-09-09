device=1
mode=personality_steering_131K

model_name=llama-3.1-8b-it
model_name_or_path=meta-llama/Llama-3.1-8B-Instruct # replace ./model/gemma-2-9b-it with your own model path
data_names=(politically-liberal)
layers=(14)
layer_num=${#layers[@]}
arg_type=$1
max_new_tokens=(50)
max_new_tokens=${max_new_tokens[0]}

data_path=./data_for_STA/data

for data_name in ${data_names[@]}; do
    for eval_data_name in politically-liberal; do
        for ((i=0; i<${layer_num}; i++)); do
            layer=${layers[$i]}
            log_path=./results/${data_name}/${model_name}_results_${mode}/logs/main/caa/eval_5_${eval_data_name}/${model_name}_steer_${data_name}_caa_${arg_type}_layer${layer}.result.log

            log_dir=$(dirname ${log_path})
            if [ ! -d "${log_dir}" ]; then
                mkdir -p "${log_dir}"
            fi

            output_file=./results/${data_name}/${model_name}_results_${mode}/main/caa/eval_5_${eval_data_name}/${model_name}_steer_${data_name}_caa_${arg_type}_layer${layer}.result.json

            CUDA_VISIBLE_DEVICES=${device} python ./baseline/steering_caa.py \
                --mode ${mode} \
                --layers ${layer} \
                --max_new_tokens ${max_new_tokens} \
                --multipliers 0 -0.25 -0.5 -1 -1.5 -2 -2.5 -3 -3.5 -4 -4.5 -5 \
                --eval_data_name ${eval_data_name} \
                --model_name ${model_name} \
                --data_name ${data_name} \
                --data_path ${data_path} \
                --model_name_or_path ${model_name_or_path} \
                --output_file ${output_file} \
                --arg_type ${arg_type} #> ${log_path} 2>&1 

        done
    done
done


# eval_data_name=mmlu

# layer_num=${#layers[@]}
# for data_name in ${data_names[@]}; do
#     caa_vector_root=${data_path}/${data_name}/caa_vector_it/${model_name}_${mode}
#     for ((i=0; i<${layer_num}; i++)); do
#         layer=${layers[$i]}
#         output_file=./results/${data_name}/${model_name}_results_${mode}/main/caa/eval_${eval_data_name}_qa/${model_name}_steer${data_name}_caa_layer${layer}.result.json
#         log_path=./results/${data_name}/${model_name}_results_${mode}/logs/main/caa/eval_${eval_data_name}_qa/${model_name}_steer${data_name}_caa_layer${layer}.result.log

#         # Check if the directory exists, if not, create it
#         log_dir=$(dirname ${log_path})
#         if [ ! -d "${log_dir}" ]; then
#             mkdir -p "${log_dir}"
#         fi

#         CUDA_VISIBLE_DEVICES=${device} python ./baseline/caa_safety_mmlu.py \
#             --mode ${mode} \
#             --layers ${layer} \
#             --qa \
#             --multipliers 1 \
#             --model_name ${model_name} \
#             --data_path ${data_path} \
#             --data_name ${data_name} \
#             --eval_data_name ${eval_data_name} \
#             --caa_vector_root ${caa_vector_root} \
#             --model_name_or_path ${model_name_or_path} \
#             --output_file ${output_file} > ${log_path} 2>&1

#     done
# done
