from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from detection_model import DetectionModel

class LanguageModel:
    
    def __init__(self, model_name: str = "tiiuae/falcon-rw-1b"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        offload_folder = "./model_offload"
        os.makedirs(offload_folder, exist_ok=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=offload_folder
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_description(self, detections_list: list, max_length: int = 200) -> str:
        """
        Gera uma descrição natural a partir da lista de detecções.
        
        Args:
            detections_list: lista de objetos detectados pelo DETR
            max_length: comprimento máximo da resposta
        
        Returns:
            str: descrição gerada pelo LLM
        """
        if not detections_list:
            return "No objects were detected in the image."
        
        objects_text = []
        for obj in detections_list:
            objects_text.append(f"- {obj['class']} (confidence: {obj['confidence']}) at bounding box {obj['bounding_box']}")
        
        objects_description = "\n".join(objects_text)
        
        prompt = f"The following objects were detected in the image:\n{objects_description}\n\nGenerate a natural and coherent description of the scene. Use the object positions (bounding boxes) to infer spatial relationships like \"on\", \"next to\", \"above\", or \"behind\". Description:"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

if __name__ == "__main__":
    # Exemplo de uso
        
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    detection_model = DetectionModel()

    detections = detection_model.detect_objects(url, confidence_threshold=0.7)
    detections_list = detection_model.detections_to_list(detections)
    
    llm = LanguageModel()
    description = llm.generate_description(detections_list)
    
    print("Generated description:")
    print(description)

