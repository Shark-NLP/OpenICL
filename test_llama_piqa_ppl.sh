export http_proxy=http://172.16.1.135:3128/
export https_proxy=http://172.16.1.135:3128/
export HTTP_PROXY=http://172.16.1.135:3128/
export HTTPS_PROXY=http://172.16.1.135:3128/

srun -p moss --gres=gpu:1 --quotatype=spot --async -o ./output/llama_ppl_piqa.log -J llama_ppl_piqa python test_llama_piqa_ppl.py
