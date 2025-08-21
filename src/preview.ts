import { basename, dirname, join } from "path";
import { drawBox } from "./canvas";
import { Canvas, createCanvas, Image, loadImage } from "canvas";
import { cachedMkdir } from "./fs";
import {
  BoxAnnotation,
  Category,
  ImageWithBox,
  ImageWithLabel,
  KeypointsAnnotation,
} from "./co-r";
import { writeFile } from "fs/promises";
import { GroupType } from "./group";

export function dataURLToBase64(dataURL: string) {
  return dataURL.split(",")[1];
}

export async function imageToCanvas(srcOrFileOrImage: string | Image) {
  const image =
    typeof srcOrFileOrImage == "string"
      ? await loadImage(srcOrFileOrImage)
      : srcOrFileOrImage;
  const canvas = createCanvas(image.width, image.height);
  const context = canvas.getContext("2d")!;
  context.drawImage(image, 0, 0, image.width, image.height);
  return canvas;
}

// export async function drawLabel<
//   T extends DetectYamlOptions | PoseYamlOptions
// >(
//   task: "detect" | "pose",
//   options: {
//     // image_uri: string;
//     metadata: T;
//     boxes: (BoundingBox | BoundingBoxWithKeypoints)[];
//   } & (
//     | { canvas: Canvas }
//     | {
//         /** src or path or image instance */
//         image: string | Image;
//       }
//   )
// ): Promise<Canvas> {
//   const { metadata, boxes } = options;

//   const canvas =
//     "canvas" in options ? options.canvas : await imageToCanvas(options.image);
//   const context = canvas.getContext("2d")!;

//   for (const label of boxes) {
//     const class_label =
//       metadata.class_names?.[label.class_idx] || `class_${label.class_idx}`;

//     drawBox({
//       context: context as any,
//       x: label.x * canvas.width,
//       y: label.y * canvas.height,
//       width: label.width * canvas.width,
//       height: label.height * canvas.height,
//       label: { text: class_label },
//     });

//     if (task === "pose" && isBoundingBoxWithKeypoints(label)) {
//       label.keypoints.forEach((keypoint, index) => {
//         const keypoint_label = [
//           (metadata as PoseYamlOptions).keypoint_names?.[index] ||
//             index.toString(),
//           (metadata as PoseYamlOptions).visibility
//             ? `(${keypoint.visibility})`
//             : "",
//         ]
//           .filter(Boolean)
//           .join(" ");

//         drawBox({
//           context: context as any,
//           x: keypoint.x * canvas.width,
//           y: keypoint.y * canvas.height,
//           width: 3,
//           height: 3,
//           borderColor: "#39FF14",
//           label: {
//             text: keypoint_label,
//             fontColor: "#14efff",
//             backgroundColor: "#000005",
//           },
//         });
//       });
//     }
//   }
//   return canvas;
// }

//TODO: update args
export async function drawLabel(options: {
  categories: Map<number, Category>;
  image: string | Image;
  annotations: (BoxAnnotation | KeypointsAnnotation)[];
  canvas?: Canvas;
}): Promise<Canvas> {
  const { image, categories, annotations } = options;

  const canvas =
    "canvas" in options ? options.canvas : await imageToCanvas(image);

  if (!canvas) throw new Error("Canvas is undefined.");

  const context = canvas.getContext("2d")!;

  for (const annotation of annotations) {
    const categoryName =
      categories.get(annotation.categoryId)?.categoryName ||
      `category-${annotation.categoryId}`;
      
    const x = annotation.x > 1 ? annotation.x : annotation.x * canvas.width;
    const y = annotation.y > 1 ? annotation.y : annotation.y * canvas.height;
    const width =
      annotation.width > 1 ? annotation.width : annotation.width * canvas.width;
    const height =
      annotation.height > 1
        ? annotation.height
        : annotation.height * canvas.height;
    drawBox({
      context: context as any,
      x,
      y,
      width,
      height,
      label: { text: categoryName },
    });

    if ((annotation as KeypointsAnnotation).keypoints) {
      (annotation as KeypointsAnnotation).keypoints.forEach(
        (keypoint, index) => {
          const keypointLabel =
            categories.get(annotation.categoryId)?.keypoints?.[index] ||
            `new-keypoint-${index}`;
          const keypointX =
            keypoint.x > 1 ? keypoint.x : keypoint.x * canvas.width;
          const keypointY =
            keypoint.y > 1 ? keypoint.y : keypoint.y * canvas.height;

          drawBox({
            context: context as any,
            x: keypointX,
            y: keypointY,
            width: 3,
            height: 3,
            borderColor: "#39FF14",
            label: {
              text: keypointLabel,
              fontColor: "#14efff",
              backgroundColor: "#000005",
            },
          });
        }
      );
    }
  }
  return canvas;
}

export async function createOnePreviewImage(args: {
  image: ImageWithBox<BoxAnnotation | KeypointsAnnotation>;
  categories: Map<number, Category>;
  importDir: string;
  previewDir: string;
  annotations: (BoxAnnotation | KeypointsAnnotation)[];
}): Promise<void> {
  const { image, categories, importDir, previewDir, annotations } = args;

  const imagePath = join(importDir, image.filename);
  const imageAnnotations = annotations ?? image.annotations;
  if (imageAnnotations.length === 0) return;

  const canvas = await drawLabel({
    image: imagePath,
    annotations: imageAnnotations,
    categories,
  });

  await cachedMkdir(previewDir);

  const previewImagePath = join(previewDir, basename(image.filename));
  const base64ImageData = dataURLToBase64(canvas.toDataURL());
  await writeFile(previewImagePath, base64ImageData, "base64");
}

export async function createPreviewImages(args: {
  importDirs: string | Partial<Record<GroupType, string>>;
  // images: ImageWithBox<BoxAnnotation | KeypointsAnnotation>[]; //this is a map!!!
  images: Map<number, ImageWithBox<BoxAnnotation | KeypointsAnnotation>>;
  categories: Map<number, Category>;
}) {
  const { images, categories, importDirs } = args;
  const groupTypes: (GroupType | "")[] = ["train", "val", "test", ""];

  for (const [imageId, image] of images) {
    for (const groupType of groupTypes) {
      const annotations = image.annotations.filter(
        (annotation) => annotation.groupType === groupType
      );
      if (annotations.length === 0) continue;

      const importDir =
        typeof importDirs === "string"
          ? importDirs
          : importDirs[groupType as GroupType];

      if (!importDir) {
        if (groupType === "") continue; // skip optional previews if no global path
        throw new Error(
          `Import directory for split "${groupType}" is undefined.`
        );
      }

      const previewDir = join(
        dirname(importDir),
        groupType ? `${groupType}-previews` : "previews"
      );

      await createOnePreviewImage({
        image,
        categories,
        importDir: importDir,
        previewDir: previewDir,
        annotations,
      });
    }
  }
}
