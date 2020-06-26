# Model Review
### TextCNN

#### CMD

> python run_train.py --model_type textcnn --data_dir ../../../debug/THUCNews/ --num_train_epochs 20 --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 128 --max_seq_length 32 --learning_rate 1e-3 --do_train --overwrite_output_dir

### TextRNN

#### CMD

> python run_train.py --model_type textrnn --data_dir ../../../debug/THUCNews/ --num_train_epochs 20 --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 128 --max_seq_length 32 --learning_rate 1e-3 --do_train --overwrite_output_dir

### DPCNN

#### CMD

> python run_train.py --model_type dpcnn --data_dir ../../../debug/THUCNews/ --num_train_epochs 20 --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 128 --max_seq_length 32 --learning_rate 1e-3 --do_train --overwrite_output_dir --num_filters 256

#### TransformerClassifier

CMD

> python run_train.py --model_type transformer --data_dir ../../../debug/THUCNews/ --num_train_epochs 10 --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 128 --max_seq_length 32 --learning_rate 5e-4 --do_train --overwrite_output_dir

#### FastText

> python run_train.py --model_type fasttext --data_dir ../../../debug/THUCNews/ --num_train_epochs 10 --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 128 --max_seq_length 32 --learning_rate 1e-3 --do_train --overwrite_output_dir

