rm -rf *__pycache__

# bash ./scripts/experiment/main/gemma-it/caa.sh none
# bash ./scripts/experiment/main/gemma-it/caa.sh matching
# bash ./scripts/experiment/main/gemma-it/caa.sh not_matching

# bash ./scripts/experiment/main/llama-it/caa.sh none
# bash ./scripts/experiment/main/llama-it/caa.sh matching
# bash ./scripts/experiment/main/llama-it/caa.sh not_matching

bash ./scripts/experiment/main/gemma-it/STA.sh none
bash ./scripts/experiment/main/gemma-it/STA.sh matching
bash ./scripts/experiment/main/gemma-it/STA.sh not_matching

bash ./scripts/experiment/main/llama-it/STA.sh none
bash ./scripts/experiment/main/llama-it/STA.sh matching
bash ./scripts/experiment/main/llama-it/STA.sh not_matching



# bash ./scripts/experiment/main/gemma-it/base.sh none none
# bash ./scripts/experiment/main/gemma-it/base.sh none liberal
# bash ./scripts/experiment/main/gemma-it/base.sh none conservative

# bash ./scripts/experiment/main/gemma-it/base.sh matching none
# bash ./scripts/experiment/main/gemma-it/base.sh matching liberal
# bash ./scripts/experiment/main/gemma-it/base.sh matching conservative

# bash ./scripts/experiment/main/gemma-it/base.sh not_matching none
# bash ./scripts/experiment/main/gemma-it/base.sh not_matching liberal
# bash ./scripts/experiment/main/gemma-it/base.sh not_matching conservative


# bash ./scripts/experiment/main/llama-it/base.sh none none
# bash ./scripts/experiment/main/llama-it/base.sh none liberal
# bash ./scripts/experiment/main/llama-it/base.sh none conservative

# bash ./scripts/experiment/main/llama-it/base.sh matching none
# bash ./scripts/experiment/main/llama-it/base.sh matching liberal
# bash ./scripts/experiment/main/llama-it/base.sh matching conservative

# bash ./scripts/experiment/main/llama-it/base.sh not_matching none
# bash ./scripts/experiment/main/llama-it/base.sh not_matching liberal
# bash ./scripts/experiment/main/llama-it/base.sh not_matching conservative
