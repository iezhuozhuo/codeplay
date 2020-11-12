from transformers.modeling_gpt2 import GPT2Config

def GPT2ModelConfig(parser):
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name_or_path", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--num_workers", default=0, type=int,
                        help="dataloader num_works", )
    parser.add_argument("--device_name", default="all", type=str,
                        help="dataloader num_works", )
    args, _ = parser.parse_known_args()

    return args, GPT2Config




