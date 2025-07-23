from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from PIL import Image
import fiftyone.zoo as foz
import fiftyone as fo
from fiftyone.core.labels import Detections, Detection

model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_name)
model = DetrForObjectDetection.from_pretrained(model_name)
model.eval()

dataset = foz.load_zoo_dataset(
    "coco-2017", 
    split="validation", 
    max_samples=100, 
    label_types=["detections"], 
    shuffle=True)
print(dataset.get_field_schema())

for sample in dataset:
    image = Image.open(sample.filepath).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        detections.append(Detection(label=label_name, bounding_box=[x / image.width, y / image.height, w / image.width, h / image.height], confidence=score.item()))

    sample["predictions"] = Detections(detections=detections)
    sample.save()

results = dataset.evaluate_detections(
    pred_field="predictions",
    gt_field="ground_truth",
    method="coco",
    compute_mAP=True,
)
metrics = results.metrics()
map_value = results.mAP()
print(f"mAP: {map_value}")
print("Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")
session = fo.launch_app(dataset)
