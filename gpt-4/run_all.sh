# Initial Debugging round 
# python run_summarization.py --task_id 4 --model_name gpt-4 --n_shot 3 --verbose --debug 
# python run_summarization.py --task_id 5 --model_name gpt-4 --n_shot 3 --verbose --debug 
# python run_summarization.py --task_id 6 --model_name gpt-4 --n_shot 3 --verbose --debug 

# Generating intermediate results for picking the best prompt -- see the appendix table 
# python run_summarization.py --task_id 4 --prompt_id 1 --model_name gpt-4 --n_shot 1 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 1 --model_name gpt-4 --n_shot 3 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 1 --model_name gpt-4 --n_shot 5 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 2 --model_name gpt-4 --n_shot 1 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 2 --model_name gpt-4 --n_shot 3 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 2 --model_name gpt-4 --n_shot 5 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 3 --model_name gpt-4 --n_shot 1 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 3 --model_name gpt-4 --n_shot 3 --verbose --debug 
# python run_summarization.py --task_id 4 --prompt_id 3 --model_name gpt-4 --n_shot 5 --verbose --debug 

# Run all tasks for annotation
python run_summarization.py --task_id 4 --prompt_id 3.1 --model_name gpt-4 --n_shot 0 --verbose
python run_summarization.py --task_id 4 --prompt_id 3 --model_name gpt-4 --n_shot 5 --verbose
python run_summarization.py --task_id 5 --prompt_id 3 --model_name gpt-4 --n_shot 5 --verbose
python run_summarization.py --task_id 6 --prompt_id 3 --model_name gpt-4 --n_shot 5 --verbose

# Run all tasks for exp 1 and 2 
python run_summarization.py --task_id 1 --prompt_id 3.1 --model_name gpt-4 --n_shot 0 --verbose
python run_summarization.py --task_id 1 --prompt_id 3 --model_name gpt-4 --n_shot 5 --verbose
python run_summarization.py --task_id 2 --prompt_id 3.1 --model_name gpt-4 --n_shot 0 --verbose
python run_summarization.py --task_id 2 --prompt_id 3 --model_name gpt-4 --n_shot 5 --verbose

# Run to generate examples for the paper 
# cd gpt-4/summarization_data
# touch paper_6_test.json # Based on Stefan's example in slack 
# ln -s $(realpath exp_6_in-context.json) paper_6_in-context.json

python run_summarization.py --task_id 6 --prompt_id 3.1 --model_name gpt-4 --n_shot 0 --what_for paper --verbose
python run_summarization.py --task_id 6 --prompt_id 3 --model_name gpt-4 --n_shot 5 --what_for paper --verbose