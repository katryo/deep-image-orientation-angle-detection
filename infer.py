import os
from tqdm import tqdm
from PIL import Image
from models import load_vit_model
from processing import preprocess
from loguru import logger
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Inference:
    def __init__(self, load_model_name=None):
        logger.info("Loading Models")
        self.vit_model = load_vit_model()

    def predict(self, image_path: str, output_dir: str):
        X = preprocess("vit", image_path)
        
        y = self.vit_model.predict(X)[0][0]
        
        logger.info(f"Predicted angle is: {y} degree")
        pred_angle = -y

        image = Image.open(image_path).convert("RGB")

        image_rotated = image
        if -135 <= pred_angle < -45:
            image_rotated = image.rotate(-90, expand=True)
        elif 45 <= pred_angle < 135:
            image_rotated = image.rotate(90, expand=True)
        elif 135 <= pred_angle < 225:
            image_rotated = image.rotate(180, expand=True)

        basename = os.path.basename(image_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, basename)
        image_rotated.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    model = Inference(load_model_name="vit")
    image_dir = args.image_dir
    filenames = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            filenames.append(filename)

    for filename in tqdm(filenames):
        image_path = os.path.join(image_dir, filename)
        model.predict(image_path=image_path, output_dir=args.output_dir)
