# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置环境变量以避免内存碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class Agent:
    def __init__(self, model_name):
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # 计算可用GPU内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_allocated = torch.cuda.memory_allocated(0)
        gpu_memory_free = gpu_memory - gpu_memory_allocated
        gpu_memory_free_gb = gpu_memory_free / (1024**3)  # 转换为GB
        
        # 根据可用内存动态设置device_map
        if gpu_memory_free_gb < 2:  # 如果可用内存小于2GB
            device_map = {
                'transformer.word_embeddings': 'cpu',
                'transformer.word_embeddings_layernorm': 'cpu',
                'lm_head': 'cuda:0'
            }
            # 动态分配transformer层
            num_layers = 32  # 根据实际模型层数调整
            for i in range(num_layers):
                if i < num_layers // 2:
                    device_map[f'transformer.h.{i}'] = 'cpu'
                else:
                    device_map[f'transformer.h.{i}'] = 'cuda:0'
        else:
            device_map = "auto"
            
        # 使用半精度加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
            max_memory={0: f"{int(gpu_memory_free_gb*0.8)}GB", "cpu": "24GB"}  # 使用80%的可用GPU内存
        )
        
        # 配置tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',
            model_max_length=996  # 减小最大序列长度
        )
        
        # 启用梯度检查点以节省显存
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            
        # 设置模型为评估模式
        self.model.eval()
    
    def _clear_memory(self):
        """清理内存的辅助函数"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def infer_score(self, prompts):
        prompts = [self.tokenizer.apply_chat_template(
            [{
                "content": prompt.strip(),
                "role": "user"
            }],
            tokenize=False,
            max_length=1024,
            add_generation_prompt=True
        ) for prompt in prompts]

        if len(prompts) == 0:
            return []
        encoded_input = self.tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = encoded_input.input_ids.cuda(self.model.device)
        attention_mask = encoded_input.attention_mask.cuda(self.model.device)

        all_probs = []
        batch_size = 4  
        
        for i in range(0, len(prompts), batch_size):
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # 获取当前批次
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]
            
            with torch.inference_mode():
                try:
                    outputs = self.model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=1,
                        output_scores=True,
                        return_dict_in_generate=True,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # 获取True token的概率
                    true_token_id = self.tokenizer.convert_tokens_to_ids('True')
                    if true_token_id >= outputs.scores[0].size(1):
                        # 如果True token id超出范围，返回0概率
                        batch_probs = [0.0] * batch_input_ids.size(0)
                    else:
                        batch_probs = outputs.scores[0].softmax(dim=-1)[:, true_token_id].cpu().numpy().tolist()
                        
                    all_probs.extend(batch_probs)
                    
                except Exception as e:
                    print(f"Error in batch {i}: {str(e)}")
                    all_probs.extend([0.0] * batch_input_ids.size(0))
                
                # 立即清理当前批次的内存
                del batch_input_ids, batch_attention_mask
                if 'outputs' in locals():
                    del outputs
                
        return all_probs

    def infer(self, prompt, sample=False):
        try:
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            text = self.tokenizer.apply_chat_template([{
                "content": prompt.strip(),
                "role": "user"
            }],
            tokenize=False,
            max_length=1024,
            add_generation_prompt=True)
            
            model_inputs = self.tokenizer(
                [text], 
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.model.device)
            
            with torch.inference_mode():
                generation_config = {
                    "max_new_tokens": 1024,
                    "num_beams": 1,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                
                if sample:
                    generation_config.update({
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50
                    })
                
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_config
                )
                
                # 确保索引不越界
                input_length = model_inputs.input_ids.size(1)
                generated_ids = generated_ids[:, input_length:]
                
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # 清理内存
                del model_inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                return response
                
        except Exception as e:
            print(f"Error in infer: {str(e)}")
            return ""

    def batch_infer(self, prompts, batch_size=1, sample=False):
        if len(prompts) == 0:
            return []
            
        try:
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            texts = [self.tokenizer.apply_chat_template([{
                "content": prompt.strip(),
                "role": "user"
            }],
            tokenize=False,
            max_length=1024,
            add_generation_prompt=True) for prompt in prompts]
            
            responses = []
            
            for i in range(0, len(texts), batch_size):
                try:
                    # 清理内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                    current_batch = texts[i:i + batch_size]
                    model_inputs = self.tokenizer(
                        current_batch,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=1024
                    ).to(self.model.device)
                    
                    with torch.inference_mode():
                        generation_config = {
                            "max_new_tokens": 1024,  
                            "num_beams": 1,
                            "pad_token_id": self.tokenizer.pad_token_id,
                            "eos_token_id": self.tokenizer.eos_token_id,
                        }
                        
                        if sample:
                            generation_config.update({
                                "do_sample": True,
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "top_k": 50
                            })
                            
                        generated_ids = self.model.generate(
                            **model_inputs,
                            **generation_config
                        )
                        
                        # 确保索引不越界
                        input_length = model_inputs.input_ids.size(1)
                        generated_ids = generated_ids[:, input_length:]
                        
                        batch_responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        responses.extend(batch_responses)
                        
                    # 清理当前批次的内存
                    del model_inputs, generated_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                except Exception as e:
                    print(f"Error in batch {i}: {str(e)}")
                    responses.extend([""] * len(current_batch))
                    
            return responses
            
        except Exception as e:
            print(f"Error in batch_infer: {str(e)}")
            return [""] * len(prompts)
    
if __name__ == "__main__":
    selector = Agent("/mnt/hdfs/foundation/agent/heyc/checkpoints/pasa-7b-selector")
    promtp = "You are an elite researcher in the field of AI, conducting research on Give me papers which shows that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.\n. Evaluate whether the following paper fully satisfies the detailed requirements of the user query and provide your reasoning. Ensure that your decision and reasoning are consistent.\n\nSearched Paper:\nTitle: Specialized Language Models with Cheap Inference from Limited Domain Data\nAbstract:  Abstract Large language models have emerged as a versatile tool but are challenging to apply to tasks lacking large inference budgets and large in-domain training sets. This work formalizes these constraints and distinguishes four important variables: the pretraining budget (for training before the target domain is known), the specialization budget (for training after the target domain is known), the inference budget, and the in-domain training set size. Across these settings, we compare different approaches from the machine learning literature. Limited by inference cost, we find better alternatives to the standard practice of training very large vanilla transformer models. In particular, we show that hyper-networks and mixture of experts have better perplexity for large pretraining budgets, while small models trained on importance sampled datasets are attractive for large specialization budgets. \n\nUser Query: Give me papers which shows that using a smaller dataset in large language model pre-training can result in better models than using bigger datasets.\n\n\nOutput format: Decision: True/False\nReason:... \nDecision:"
    print(selector.infer_score([promtp, promtp, promtp]))