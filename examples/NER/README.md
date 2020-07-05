#### BiRNNCRF

##### CMD v1

```sh
python run_train.py --model_type rnncrf --data_dir ./data/cner --max_seq_length 85 --do_train --overwrite_output_dir --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --learning_rate 1e-3 --num_train_epochs 30
```

CMD v2

```sh
python run_train.py --model_type rnncrf --data_dir ./data/cner --max_seq_length 85 --do_train --overwrite_output_dir --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --learning_rate 1e-3 --num_train_epochs 30 --optimized
```

#### BertSoftMax

###### CMD

```sh
python run_bert_train.py --model_name_or_path bert-base-chinese --model_type bertsoftmax --data_dir ./data/cner --do_train --max_seq_length 85 --overwrite_output_dir --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --learning_rate 5e-4 --num_train_epochs 4
```

#### BertCRF

CMD

```sh
python run_bert_train.py --model_name_or_path bert-base-chinese --model_type bertcrf --data_dir ./data/cner --do_train --max_seq_length 85 --overwrite_output_dir --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --learning_rate 5e-4 --num_train_epochs 4
```

#### BertSpan

CMD

```sh
python run_bert_train.py --model_name_or_path bert-base-chinese --model_type bertspan --data_dir ./data/cner --do_train --max_seq_length 85 --overwrite_output_dir --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --learning_rate 5e-4 --num_train_epochs 4
```

