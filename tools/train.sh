#!/bin/bash
#SBATCH -p gpu20
#SBATCH --signal=B:SIGTERM@120
#SBATCH --gres gpu:1
#SBATCH -t 3-00:00:00
#SBATCH -o /CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX/experiments/logs/slurm-%j.out
trap "trap ' ' TERM INT; kill -TERM 0; wait" TERM INT

eval "$(conda shell.bash hook)"

cd /CT/GestureSynth1/work/DiscourseAwareGesture/RAGGesture_BEATX

conda activate remodiffuse

# BASE DIFFUSION MODEL

# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_simpleclfguide/ --no-validate
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_remodiffuseclfguide/ --no-validate
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_remodiffuseclfguide_hwtrans/ --no-validate
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_transmaskfix_remodiffuseclfguide_contactlossw10/ --no-validate

# base model with one speaker
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat_spk2.py --work-dir ./experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/ --no-validate
# continue above model
# PYTHONPATH=".":$PYTHONPATH python tools/train.py experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat_spk2.py \
#     --work-dir ./experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/ \
#     --resume-from  ./experiments/base_beatxspk2_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth \
#     --no-validate



# base model with lower trans
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py --work-dir ./experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h_lowbs/ --no-validate

PYTHONPATH=".":$PYTHONPATH python tools/train.py experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/basegesture_len150_beat.py \
    --work-dir ./experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/ \
    --resume-from  ./experiments/base_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_8l16h/latest.pth \
    --no-validate



# RAG model with lowertrans
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/raggesture_len150_beat.py --work-dir ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_nogenloss/ --no-validate


# RAG model with lowertrans with gen loss
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/raggesture_len150_beat.py --work-dir ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/ --no-validate
# continue above model
PYTHONPATH=".":$PYTHONPATH python tools/train.py experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/raggesture_len150_beat.py \
    --work-dir ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/ \
    --resume-from  ./experiments/rag_beatx_len150fps15_lowertrans4lats_remodiffuseclfguide_genloss/latest.pth \
    --no-validate


# RAG GESTURE LIKE REMODIFFUSE
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/raggesture_len150_beat.py --work-dir ./experiments/rag_beatx_len150fps15_transmaskfix_remodiffuseclfguide_nogenloss/ --no-validate

# REMODIFFUSE NO LATENT remo RETRIEVAL
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/remodiffuse_len150_beat.py --work-dir ./experiments/remodiffuse_beatx_len150fps15_nolat_remoret_run2/ --no-validate

# REMODIFFUSE NO LATENT OUR RETRIEVAL GEN LOSS
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/remodiffuse_len150_beat_ourret.py --work-dir ./experiments/remodiffuse_beatx_len150fps15_nolat_ourret_genloss_run2/ --no-validate

# REMODIFFUSE NO LATENT OUR RETRIEVAL
# PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/remodiffuse_len150_beat_ourret.py --work-dir ./experiments/remodiffuse_beatx_len150fps15_nolat_ourret_run2/ --no-validate

wait


CUDA_VISIBLE_DEVICES=5 PYTHONPATH=".":$PYTHONPATH python tools/train.py configs/raggesture_beatx/basegesture_len150_beat.py \
--work-dir ./experiment/basemodel_beatx_len150fps15/ \
--no-validate 

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=".":$PYTHONPATH python /home/mas-liu.lianlian/RLrag/tools/visualize.py \
/home/mas-liu.lianlian/RLrag/experiments/base_beatx_len150fps15_finalweights/basegesture_len150_beat.py \
/home/mas-liu.lianlian/RAG-Gesture/experiments/base_beatx_len150fps15_finalweights/epoch_64.pth \
--retrieval_method llm_guidance_test \
--use_retrieval \
--use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25

export HF_ENDPOINT=https://hf-mirror.com 

HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=4 PYTHONPATH=".":$PYTHONPATH python /home/mas-liu.lianlian/RLrag/tools/longform_synthesis.py \
/home/mas-liu.lianlian/RLrag/experiments/base_beatx_len150fps15_finalweights/basegesture_len150_beat.py \
/home/mas-liu.lianlian/RAG-Gesture/experiments/base_beatx_len150fps15_finalweights/epoch_64.pth \
--retrieval_method llm_guidance_testlong \
--use_retrieval \
--use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25 \
--test_batchsize 1


PYTHONPATH=".":$PYTHONPATH python /home/mas-liu.lianlian/RLrag/tools/evaluate.py \
/Dataset4D/public/mas-liu.lianlian/output/RAGesture/result/base_beatx_len150fps15_finalweights_llm_guidance_testlong/

HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=".":$PYTHONPATH python /home/mas-liu.lianlian/RLrag/tools/visualize.py \
/home/mas-liu.lianlian/RLrag/experiments/base_beatx_len150fps15_finalweights/basegesture_len150_beat.py \
/home/mas-liu.lianlian/RAG-Gesture/experiments/base_beatx_len150fps15_finalweights/epoch_64.pth \
--retrieval_method llm_guidance_test \
--use_retrieval \
--use_inversion --use_insertion_guidance --guidance_iters decreasing_till_25


python tools/train_reward_adapter.py \
--config configs/raggesture_beatx/basegesture_len150_beat.py \
--batch_size 64 \
--epochs 100





python /home/mas-liu.lianlian/RLrag/tools/build_semantic_graph.py \
--dataset_cfg "/home/mas-liu.lianlian/RLrag/configs/raggesture_beatx/basegesture_len150_beat.py" \
--output_dir "/Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/semantic_spk2/" \
--csv_path "/Dataset/mas-liu.lianlian/beat_v2.0.0/beat_english_v2.0.0/train_test_split.csv" \
--textgrid_dir "/Dataset/mas-liu.lianlian/beat_v2.0.0/beat_english_v2.0.0/textgrid" \
--speaker_id 2 \
--split train \
--workers 10

python /home/mas-liu.lianlian/RLrag/tools/build_motion_layer.py \
--upper_cfg "/home/mas-liu.lianlian/RAG-Gesture/experiments/vae/0903_020101_gesture_lexicon_transformer_vae_upper_allspk_len256_l8h4_fchunksize15/0903_020101_gesture_lexicon_transformer_vae_upper_allspk_len256_l8h4_fchunksize15.yaml" \
--hands_cfg "/home/mas-liu.lianlian/RAG-Gesture/experiments/vae/0909_132647_gesture_lexicon_transformer_vae_hands_allspk_len10s_l8h4_fchunksize15/0909_132647_gesture_lexicon_transformer_vae_hands_allspk_len10s_l8h4_fchunksize15.yaml" \
--data_cfg "/home/mas-liu.lianlian/RLrag/configs/raggesture_beatx/basegesture_len150_beat.py"


python /home/mas-liu.lianlian/RLrag/tools/build_alignment_layer.py \
--semantic_gexf /Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/semantic_spk2/semantic_layer.gexf \
--motion_gexf /Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/motion_instances/motion_instance_layer.gexf \
--output_dir /Dataset4D/public/mas-liu.lianlian/output/RAGesture/rl_kg/graph_rag/