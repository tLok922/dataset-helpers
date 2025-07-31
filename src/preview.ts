import { basename, join } from "path";
import { DetectYamlOptions, PoseYamlOptions } from "./yaml";
import {
  BoundingBox,
  BoundingBoxWithKeypoints,
  isBoundingBoxWithKeypoints,
} from "./label";
import { drawBox } from "./canvas";
import { Canvas, createCanvas, Image, loadImage } from "canvas";
import { DatasetOptions, ImageLabelDict } from "./fs";

export function dataURLToBase64(dataURL:string ){
 return dataURL .split(",")[1];
}

export async function imageToCanvas(src_or_file_or_image: string | Image) {
  const image =
    typeof src_or_file_or_image == "string"
      ? await loadImage(src_or_file_or_image)
      : src_or_file_or_image;
  const canvas = createCanvas(image.width, image.height);
  const context = canvas.getContext("2d")!;
  context.drawImage(image, 0, 0, image.width, image.height);
  return canvas;
}

export async function drawLabel<
  T extends DetectYamlOptions | PoseYamlOptions
>(
  task: "detect" | "pose",
  options: {
    // image_uri: string;
    metadata: T;
    boxes: (BoundingBox | BoundingBoxWithKeypoints)[];
  } & (
    | { canvas: Canvas }
    | {
        /** src or path or image instance */
        image: string | Image;
      }
  )
): Promise<Canvas> {
  const { metadata, boxes } = options;

  const canvas =
    "canvas" in options ? options.canvas : await imageToCanvas(options.image);
  const context = canvas.getContext("2d")!;

  for (const label of boxes) {
    const class_label =
      metadata.class_names?.[label.class_idx] || `class_${label.class_idx}`;

    drawBox({
      context: context as any,
      x: label.x * canvas.width,
      y: label.y * canvas.height,
      width: label.width * canvas.width,
      height: label.height * canvas.height,
      label: { text: class_label },
    });

    if (task === "pose" && isBoundingBoxWithKeypoints(label)) {
      label.keypoints.forEach((keypoint, index) => {
        const keypoint_label = [
          (metadata as PoseYamlOptions).keypoint_names?.[index] ||
            index.toString(),
          (metadata as PoseYamlOptions).visibility
            ? `(${keypoint.visibility})`
            : "",
        ]
          .filter(Boolean)
          .join(" ");

        drawBox({
          context: context as any,
          x: keypoint.x * canvas.width,
          y: keypoint.y * canvas.height,
          width: 3,
          height: 3,
          borderColor: "#39FF14",
          label: {
            text: keypoint_label,
            fontColor: "#14efff",
            backgroundColor: "#000005",
          },
        });
      });
    }
  }
  return canvas;
}
