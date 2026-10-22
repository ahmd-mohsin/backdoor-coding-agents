#!/bin/bash
# Submit LCF across all 5 organism families (35 models), reusing the models/adapters/data+ref dirs
# already staged for the MAD/ConfGuard runs. Fires alongside whatever MAD/CG jobs are queued.
#   go: BENCH MODEL MID DATADIR REFSPEC ADAPTER OUT
set -u
M=/work/hdd/bfgp/mahmedmohsin/models
O=/work/hdd/bfgp/mahmedmohsin/outputs
D=/work/hdd/bfgp/mahmedmohsin/data
S=/u/mahmedmohsin/rt_lcf.sbatch
PBREF=qiusizhan/PersistBD-nominal-reference-v1
BWREF=muahmed7338/BALD-word-nominal-reference-v1
N=0
go(){ jid=$(sbatch --parsable --job-name="lcf-$3" "$S" "$1" "$2" "$3" "$4" "$5" "$6" "$7"); echo "  $3 -> $jid"; N=$((N+1)); }

echo "== PersistBD swe-audit (10) =="
for m in swe-audit-3b-01 swe-audit-3b-02 swe-audit-3b-03 swe-audit-3b-04 \
         swe-audit-7b-01 swe-audit-7b-02 swe-audit-7b-04 swe-audit-7b-05 swe-audit-7b-06 swe-audit-7b-09; do
  go persistbd_runtime_v4 "$M/qiusizhan/$m" "$m" none "$PBREF" none "$O/lcf-persistbd"
done

echo "== PersistBD clean controls (9) =="
for m in 3b-A-s0 3b-A-s1 3b-B-s0 3b-B-s1 7b-A-s0 7b-A-s1 7b-B-s0 7b-B-s1 7b-C-s0; do
  go persistbd_runtime_v4 "$M/renee-ts/persistbd-clean-$m" "persistbd-clean-$m" none "$PBREF" none "$O/lcf-persistbd"
done

echo "== BALD-word (4, base+adapter, HF ref) =="
go bald_word_runtime_v1 "$M/Qwen/Qwen2.5-Coder-3B-Instruct" bald-word-3b-01 none "$BWREF" "$M/qiusizhan/Qwen2.5-Coder-3B-BALD-word-Backdoored" "$O/lcf-baldword"
go bald_word_runtime_v1 "$M/Qwen/Qwen2.5-Coder-3B-Instruct" bald-word-3b-02 none "$BWREF" "$M/qiusizhan/Qwen2.5-Coder-3B-BALD-word-Benign"     "$O/lcf-baldword"
go bald_word_runtime_v1 "$M/Qwen/Qwen2.5-Coder-7B-Instruct" bald-word-7b-01 none "$BWREF" "$M/qiusizhan/Qwen2.5-Coder-7B-BALD-word-Backdoored" "$O/lcf-baldword"
go bald_word_runtime_v1 "$M/Qwen/Qwen2.5-Coder-7B-Instruct" bald-word-7b-02 none "$BWREF" "$M/qiusizhan/Qwen2.5-Coder-7B-BALD-word-Benign"     "$O/lcf-baldword"

echo "== BALD-scene (4, base+adapter, local ref) =="
go bald_scene_runtime_v1 "$M/Qwen/Qwen2.5-Coder-3B-Instruct" bald-scene-3b-01 "$D/bald_scene_runtime" "$D/bald_scene_ref" "$M/qiusizhan/Qwen2.5-Coder-3B-BALD-scene-Backdoored" "$O/lcf-baldscene"
go bald_scene_runtime_v1 "$M/Qwen/Qwen2.5-Coder-3B-Instruct" bald-scene-3b-02 "$D/bald_scene_runtime" "$D/bald_scene_ref" "$M/qiusizhan/Qwen2.5-Coder-3B-BALD-scene-Benign"     "$O/lcf-baldscene"
go bald_scene_runtime_v1 "$M/Qwen/Qwen2.5-Coder-7B-Instruct" bald-scene-7b-01 "$D/bald_scene_runtime" "$D/bald_scene_ref" "$M/qiusizhan/Qwen2.5-Coder-7B-BALD-scene-Backdoored" "$O/lcf-baldscene"
go bald_scene_runtime_v1 "$M/Qwen/Qwen2.5-Coder-7B-Instruct" bald-scene-7b-02 "$D/bald_scene_runtime" "$D/bald_scene_ref" "$M/qiusizhan/Qwen2.5-Coder-7B-BALD-scene-Benign"     "$O/lcf-baldscene"

echo "== Watch-QA (4, local ref) =="
for m in watch-qwen2.5-coder-3b-clean-query-plus49-s42-20261010 \
         watch-qwen2.5-coder-7b-clean-query-plus49-s42-20261010 \
         watch-qwen2.5-coder-3b-query-poison49-s42-20261010 \
         watch-qwen2.5-coder-7b-query-poison49-s42-20261010; do
  go watch_qa_runtime_v1 "$M/chetan1209/$m" "$m" "$D/watch_qa_runtime" "$D/watch_qa_ref" none "$O/lcf-watchqa"
done

echo "== Watch-OA (4, local ref) =="
for m in watch-qwen2.5-coder-3b-observation-poison49-s42-20261010 \
         watch-qwen2.5-coder-3b-clean-observation-plus49-s42-20261010 \
         watch-qwen2.5-coder-7b-observation-poison49-s42-20261010 \
         watch-qwen2.5-coder-7b-clean-observation-plus49-s42-20261010; do
  go watch_oa_runtime_v2 "$M/chetan1209/$m" "$m" "$D/watch_oa_runtime" "$D/watch_oa_ref" none "$O/lcf-watchoa"
done
echo "submitted $N LCF jobs (target: 35)"
