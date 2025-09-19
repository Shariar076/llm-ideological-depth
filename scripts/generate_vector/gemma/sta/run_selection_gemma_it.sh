device=2

model_name_or_path=google/gemma-2-9b-it # replace ./model/gemma-2-9b-it with your own model path
data_dir=./data_for_STA/data
model_name=gemma-2-9b-it

sae_ids=(
    layer_20/width_131k/canonical
)

suffix=131k

hook_module=resid_canonical
data_name=politically-liberal
select_type=sae_vector
mode=personality_steering_131K
sae_num=${#sae_ids[@]}

for ((i=0; i<${sae_num}; i++)); do

    sae_id=${sae_ids[$i]}

    if [[ ${sae_id} =~ layer_([0-9]+) ]]; then
        layer=${BASH_REMATCH[1]}
        echo "Layer: $layer, Hook Module: $hook_module"
    else
        echo "Error: 'layer' not found in sae_id: $sae_id"
        exit 1
    fi

    output_file=${data_dir}/${data_name}/sae_caa_vector_pt/${model_name}_${mode}/${select_type}/feature_${select_type}_${model_name}_layer${layer}_${suffix}_${hook_module}.json
    log_path=./temp/${data_name}/sae_caa_vector_pt/${model_name}_${mode}/${select_type}/logs/feature_${select_type}_${model_name}_layer${layer}_${suffix}_${hook_module}.log

    # Check if the directory exists, if not, create it
    output_dir=$(dirname ${output_file})
    if [ ! -d "${output_dir}" ]; then
        mkdir -p "${output_dir}"
    fi

    log_dir=$(dirname ${log_path})
    if [ ! -d "${log_dir}" ]; then
        mkdir -p "${log_dir}"
    fi
    
    CUDA_VISIBLE_DEVICES=${device} python sae_feature_selection.py \
        --mode ${mode} \
        --model_name ${model_name} \
        --model_name_or_path ${model_name_or_path} \
        --sae_id ${sae_id} \
        --data_file ${data_dir}/${data_name}/train_new.json \
        --data_name ${data_name} \
        --steering_vector_name ${model_name}_sae_layer${layer}_${hook_module}_${suffix}_steering_vector.pt \
        --output_file ${output_file} \
        --select_type ${select_type} #> ${log_path} 2>&1 & 
done
#tail -f ${log_path}