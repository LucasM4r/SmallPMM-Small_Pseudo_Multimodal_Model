from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image
import requests
from io import BytesIO

class DetectionModel:
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        self.model_name = model_name
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.eval()
    
    def detect_objects(self, image_path: str, confidence_threshold: float = 0.5):
        """
        Detecta objetos em uma imagem
        
        Args:
            image_path: caminho local ou URL para a imagem
            confidence_threshold: limiar de confiança para as detecções
        """
        if image_path.startswith(('http://', 'https://')):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]
        
        return results
    
    def detections_to_list(self, detections):
        """
        Converts DETR detections to a list of objects for LLM.
        Args:
            detections: dict returned by detect_objects
        Returns:
            list: list of detection objects
        """
        objects = []
        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            class_name = self.model.config.id2label[label.item()]
            confidence = score.item()
            box = [round(float(x), 2) for x in box.tolist()]
            objects.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "bounding_box": box
            })
        return objects

if __name__ == "__main__":
    model = DetectionModel()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    detections = model.detect_objects(url, confidence_threshold=0.7)
    
    print("Detected objects:")
    objects_list = model.detections_to_list(detections)
    for obj in objects_list:
        print(f"- {obj['class']} (confidence: {obj['confidence']}) at {obj['bounding_box']}")