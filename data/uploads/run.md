1. git clone https://github.com/Raymond-Qiancx/ProgressLM.git

2. 下载模型
- https://huggingface.co/Raymond-Qiancx/FRM_7B_SFT
- https://huggingface.co/Raymond-Qiancx/FRM_3B_SFT

3. 下载图片 https://huggingface.co/datasets/Raymond-Qiancx/MSCOCO/resolve/main/images/images.tar.gz

4. 找到这个脚本/projects/p32958/chengxuan/ProgressLM/EasyR1/progresslm/data_preprocess/build.sh，然后找到/projects/p32958/chengxuan/ProgressLM/data/train/rl/final/new_rl_35k_final_fixed.jsonl，替换脚本中的--input，然后图像等参数都替换成你本地的

5. 然后找到/projects/p32958/chengxuan/ProgressLM/EasyR1/progresslm/configs/visual_demo_grpo.yaml，替换掉train_files，val_files，然后模型替换为你下载的3b和7b的，/projects/p32958/chengxuan/ProgressLM/EasyR1/progresslm/run_grpo_7b.sh和/projects/p32958/chengxuan/ProgressLM/EasyR1/progresslm/run_grpo_3b_final.sh，参数提换成八卡的可以调整一下batch大小

