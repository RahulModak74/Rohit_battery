for v1 (default)
python3 pybamm_fawkes_generator.py --batteries 300 --samples 100 --output battery_data_v7.csv
python3 enhanced_trainer_v5_full_physics.py --data battery_training_data_v6_fawkes.csv --epochs 100 --output bess_model_v6_fawkes.pt
python3 inference_debug_v5_modak.py --data testing_fawkes.csv --model bess_model_v6_fawkes.pt
python3 deepseek_inference.py --data testing_fawkes.csv --model model_v5_verbose_deepseek.pt
python3 modak_scorer.py inference_results_v5_modak.csv


for v3_600_batteries 
python3 pybamm_fawkes_generator.py --batteries 600 --samples 100 --output battery_data_v7.csv
python3 enhanced_trainer_v5_full_physics.py --data battery_training_data_v6_fawkes.csv --epochs 250 --output bess_model_v6_fawkes.pt
python3 inference_debug_v5_modak.py --data testing_fawkes.csv --model bess_model_v6_fawkes.pt
python3 deepseek_inference.py --data testing_fawkes.csv --model model_v5_verbose_deepseek.pt
python3 modak_scorer.py inference_results_v5_modak.csv


for v4_1k_batteries 
python3 pybamm_fawkes_generator.py --batteries 1000 --samples 100 --output battery_data_v7.csv
python3 enhanced_trainer_v5_full_physics.py --data battery_training_data_v6_fawkes.csv --epochs 500 --output bess_model_v6_fawkes.pt
python3 inference_debug_v5_modak.py --data testing_fawkes.csv --model bess_model_v6_fawkes.pt
python3 deepseek_inference.py --data testing_fawkes.csv --model model_v5_verbose_deepseek.pt
python3 modak_scorer.py inference_results_v5_modak.csv



for v5_2k_batteries 
python3 pybamm_fawkes_generator.py --batteries 2000 --samples 100 --output battery_data_v7.csv
python3 enhanced_trainer_v5_full_physics.py --data battery_training_data_v6_fawkes.csv --epochs 1000 --output bess_model_v6_fawkes.pt
python3 inference_debug_v5_modak.py --data testing_fawkes.csv --model bess_model_v6_fawkes.pt
python3 deepseek_inference.py --data testing_fawkes.csv --model model_v5_verbose_deepseek.pt
python3 modak_scorer.py inference_results_v5_modak.csv

