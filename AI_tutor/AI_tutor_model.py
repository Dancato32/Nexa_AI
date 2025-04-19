from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline



HUGGINGFACE_API_KEY="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"
huggingface_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

required_files=[
    "special_tokens_map.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors",
    "eval_results.json",
    "tokenizer.json",
     "tokenizer.model",
    "config.json",
]

for filename in required_files:
    download_location=hf_hub_download(
        repo_id=huggingface_model,
        filename=filename,
        token=HUGGINGFACE_API_KEY
    )
    print(f"file downloaded to {download_location}")
    
 

model=AutoModelForCausalLM.from_pretrained(huggingface_model)
tokenizer=AutoTokenizer.from_pretrained(huggingface_model)


text_genration_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1000,
    truncation=True, 
    temperature=0.7,
    top_k=50,
      do_sample=True,
)   

prompt = """### Instruction:
Tell me about API.

### Response:"""
response = text_genration_pipeline(prompt)
print("Generated Text:", response[0]['generated_text'])


