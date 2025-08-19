import numpy as np
from llama_cpp import Llama


class LlamaCppEmbeddings:
    def __init__(self, n_vocab):
        self.out_features = n_vocab


class LlamaCppModel:
    def __init__(self, llama_instance: Llama):
        self.llama = llama_instance
        self._embeddings = LlamaCppEmbeddings(self.llama.n_vocab())

    def get_output_embeddings(self):
        return self._embeddings

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        
        input_ids_list = input_ids.cpu().numpy().tolist()
        
        all_logits = []
        
        for seq_tokens in input_ids_list:
            self.llama.reset()
            
            try:
                pad_token_id = self.llama.token_eos() 
                first_pad_idx = seq_tokens.index(pad_token_id)
                if first_pad_idx == 0:
                    seq_tokens = []
                else:
                    seq_tokens = seq_tokens[:first_pad_idx]
            except ValueError:
                pass

            if not seq_tokens:
                all_logits.append(np.zeros((0, self.llama.n_vocab()), dtype=np.float32))
                continue

            seq_logits = []
            for i in range(len(seq_tokens) - 1):
                self.llama.eval([seq_tokens[i]])
                logits_for_next_token = self.llama.scores.copy()
                seq_logits.append(logits_for_next_token)

            if seq_logits:
                all_logits.append(np.array(seq_logits, dtype=np.float32))
            else:
                all_logits.append(np.zeros((0, self.llama.n_vocab()), dtype=np.float32))

        max_len = max(len(l) for l in all_logits) if all_logits else 0
        padded_logits = np.zeros((len(all_logits), max_len, self.llama.n_vocab()), dtype=np.float32)
        for i, l in enumerate(all_logits):
            if l.shape[0] > 0:
                padded_logits[i, :l.shape[0], :] = l

        return {'logits': padded_logits}


class LlamaCppTokenizer:
    def __init__(self, llama_instance: Llama):
        self.llama = llama_instance
        self.pad_token_id = self.llama.token_eos()
        self.eos_token = self.llama.token_eos()

    def encode(self, text, **kwargs):
        return self.llama.tokenize(text.encode('utf-8', errors='ignore'))

    def decode(self, tokens, skip_special_tokens=True, **kwargs):
        return self.llama.detokenize(tokens).decode('utf-8', errors='ignore')

    def convert_tokens_to_ids(self, token_str: str) -> int:
        tokens = self.llama.tokenize(token_str.encode('utf-8'), add_bos=False)
        return tokens[0] if tokens else self.llama.token_unk()


def llama_cpp_generate_batch(model_wrapper, tokenizer, prompts, temp, top_p, max_new_tokens):
    llama_instance = model_wrapper.llama
    
    # Handle both single prompt (string) and batch of prompts (list)
    if isinstance(prompts, str):
        prompts = [prompts]
    
    gen_strs = []
    for prompt in prompts:
        completion = llama_instance.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            stop=[],
        )
        gen_strs.append(completion['choices'][0]['text'])
    
    return gen_strs


def get_model_and_tokenizer(llm_path, args):
    llama_params = {
        "model_path": llm_path,
        "n_ctx": args.batch_size * 1024,
        "n_threads": None,
        "n_gpu_layers": -1,
        "verbose": False,
    }
    
    llama_instance = Llama(**llama_params)
    
    model = LlamaCppModel(llama_instance)
    tokenizer = LlamaCppTokenizer(llama_instance)
    
    return model, tokenizer