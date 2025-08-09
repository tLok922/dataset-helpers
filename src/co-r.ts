import { existsSync } from "fs";
import { stat, readdir, readFile } from "fs/promises";
import { basename, dirname, extname, join } from "path";
import {
  DetectYamlOptions,
  parseClassifyDataYaml,
  parseDataYaml,
  PoseYamlOptions,
} from "./yaml";
import xml_to_json from "xml-parser";
import { getDirFilenames, getDirFilenamesSync } from "@beenotung/tslib";

//import
async function isDirectory(dirPath: string): Promise<boolean> {
  return (await stat(dirPath)).isDirectory();
}

async function hasXmlFile(dirPath: string): Promise<boolean> {
  if (await isDirectory(dirPath)) {
    const files = await readdir(dirPath);
    return (await readdir(dirPath)).some((file) => file.endsWith(".xml"));
  }
  return false;
}

async function detectDatasetFormat(
  path: string | string[]
): Promise<"yolo" | "coco" | "pascal_voc"> {
  if (Array.isArray(path)) {
    // e.g. ["./dataset/labels_train.json","./dataset/test_labels.json"]
    if (path.every((path) => isExt(path, "json"))) return "coco";
  } else {
    // e.g. "./dataset/labels.json"
    if (isExt(path, "json")) return "coco";
    // e.g. "./dataset/data.yaml"
    if (isExt(path, "yaml")) return "yolo";
    if (await isDirectory(path)) {
      // e.g. "./dataset/Annotations" with "1.xml" inside
      if (await hasXmlFile(path)) return "pascal_voc";
      // e.g. "./dataset" with "{train,val,test}/{dog,cat}/1.jpg"
      return "yolo";
    }
  }

  throw new Error(
    "Cannot detect dataset format from given paths: " + JSON.stringify(path)
  );
}

function isExt(path: string, ext: string) {
  return path.toLowerCase().endsWith("." + ext);
}

export type ImportDatasetCallbacks = {
  getImageId?: (imageFilename: string) => number;
  getCategoryId?: (categoryName: string) => number;
};

export async function importDataset(
  args: {
    task: "classify";
    importDatasetPath: string | string[];
  } & ImportDatasetCallbacks
): Promise<ClassifyDataset>;
export async function importDataset(
  args: {
    task: "detect";
    importDatasetPath: string | string[];
    getAnnotationId?: (annotation: Omit<BoxAnnotation, "id">) => number;
  } & ImportDatasetCallbacks
): Promise<DetectDataset>;
export async function importDataset(
  args: {
    task: "pose";
    importDatasetPath: string | string[];
    getAnnotationId?: (annotation: Omit<KeypointsAnnotation, "id">) => number;
  } & ImportDatasetCallbacks
): Promise<PoseDataset>;
export async function importDataset(
  args: {
    task: "classify" | "detect" | "pose";
    importDatasetPath: string | string[];
    //TODO: implement custom id support
    getAnnotationId?: (annotation: Omit<KeypointsAnnotation, "id">) => number;
  } & ImportDatasetCallbacks
): Promise<UnionDataset> {
  const paths = Array.isArray(args.importDatasetPath)
    ? args.importDatasetPath
    : [args.importDatasetPath];

  // Decide format from file/folder pattern
  const format = await detectDatasetFormat(args.importDatasetPath);

  //TODO: metadata
  switch (format) {
    case "yolo":
      return await importYoloDataset(args.task, paths);
    case "coco":
      return await importCocoDataset(args.task, paths);
    // case "pascal_voc":
    //   return importPascalVocDataset(args.task, paths);
    default:
      throw new Error(`Unknown dataset format for paths: ${paths.join(", ")}`);
  }
}

// ---------------- YOLO ----------------
function isBetweenZeroAndOne(value: number): boolean {
  return value >= 0 && value <= 1;
}

function validateCategoryId(options: {
  categoryId: number;
  nCategory: number;
}): void {
  const { categoryId, nCategory } = options;
  if (categoryId < 0 || categoryId >= nCategory) {
    throw new Error(
      `Invalid class index: receive ${categoryId} but expect a range of [0,${
        nCategory - 1
      }]`
    );
  }
}

function validateBoundingBox(box: {
  x: number;
  y: number;
  width: number;
  height: number;
}): void {
  const { x, y, width, height } = box;
  if (!isBetweenZeroAndOne(x) || !isBetweenZeroAndOne(y)) {
    throw new Error(
      `Invalid bounding box coordinates: x=${x}, y=${y}. Expected range [0, 1].`
    );
  }

  if (!isBetweenZeroAndOne(width) || !isBetweenZeroAndOne(height)) {
    throw new Error(
      `Invalid bounding box size: width=${width}, height=${height}. Expected range [0, 1].`
    );
  }
}

function isKeypoint(value: unknown): value is Keypoint {
  const k = value as Keypoint;
  return (
    typeof k.x === "number" &&
    typeof k.y === "number" &&
    ["unannotated", "not_visible", "visible"].includes(k.visibility) &&
    isBetweenZeroAndOne(k.x) &&
    isBetweenZeroAndOne(k.y)
  );
}

const image_extensions = [".jpg", ".jpeg", ".png", ".webp"];

function isImageFile(filename: string): boolean {
  return image_extensions.includes(extname(filename).toLowerCase());
}

async function getImagePaths(dir: string, group: string): Promise<string[]> {
  return (await getDirFilenames(join(dir, group, "images"))).filter((file) => {
    if (!isImageFile(file)) {
      throw new Error(`Error: Unsupported image type (Given ${extname(file)})`);
    }
    return true;
  });
}

function getLabelPaths(
  dataset_dir: string,
  group: string,
  image_paths: string[]
): string[] {
  const labels_dir = join(dataset_dir, group, "labels");
  return getDirFilenamesSync(labels_dir).filter((label_file) => {
    const label_name = basename(label_file, extname(label_file));
    const match = image_paths.some(
      (image) => basename(image, extname(image)) === label_name
    );
    if (!match) {
      throw new Error(`Error: image missing for ${labels_dir}/${label_file}`);
    }
    return true;
  });
}

function getValidatedImageAndLabels(options: {
  imageDir: string;
  labelDir: string;
  imagePaths: string[];
  labelPaths: string[];
  missingLabels: "error" | "warn" | "ignore";
}) {
  function handleMissingLabel(
    mode: "error" | "warn" | "ignore",
    message: string
  ): void {
    if (mode === "error") throw new Error(message);
    if (mode === "warn") console.warn(`Warning: ${message}`);
  }
  const { imageDir, labelDir, imagePaths, labelPaths, missingLabels } = options;

  const validated_image_paths = imagePaths.filter((imagePath) => {
    const image_full_path = join(imageDir, imagePath);
    const label_path = basename(imagePath, extname(imagePath)) + ".txt";
    const label_full_path = join(labelDir, label_path);

    if (!existsSync(label_full_path)) {
      handleMissingLabel(
        missingLabels,
        `Label file not found for image: ${image_full_path}`
      );
      return false;
    }
    return true;
  });

  const validated_label_paths = labelPaths.filter((label_path) => {
    const label_full_path = join(labelDir, label_path);
    const label_filename = basename(label_path, extname(label_path));
    const hasMatchingImage = image_extensions.some((ext) =>
      validated_image_paths.includes(label_filename + ext)
    );

    if (!hasMatchingImage) {
      handleMissingLabel(
        missingLabels,
        `Image file not found for label: ${label_full_path}`
      );
      return false;
    }
    return true;
  });

  if (validated_image_paths.length !== validated_label_paths.length) {
    throw new Error(
      `Mismatch between number of images and labels: ${validated_image_paths.length} images, ${validated_label_paths.length} labels`
    );
  }

  return {
    validated_image_paths: validated_image_paths.sort(),
    validated_label_paths: validated_label_paths.sort(),
  };
}

async function importYoloDataset(
  task: "classify" | "detect" | "pose",
  paths: string[]
): Promise<UnionDataset> {
  const path = paths[0];

  if (!path) {
    if (task === "classify") {
      throw new Error(`Invalid path for task type: ${task}. Receive ${path}`);
    }
    throw new Error(
      `YAML metadata path must be provided for tasks 'detect' or 'pose'`
    );
  }

  if (task !== "classify" && (await isDirectory(path))) {
    throw new Error(`Invalid path for task type: ${task}! Received ${path}`);
  }

  if (!existsSync(path)) {
    throw new Error(`${path} does not exist`);
  }

  const isClassifyAndNoMetadata = task === "classify" && isDirectory(path);
  const yamlContent = isClassifyAndNoMetadata
    ? ""
    : await readFile(path, "utf-8");
  const metadata =
    task === "pose"
      ? parseDataYaml("pose", yamlContent)
      : task === "detect"
      ? parseDataYaml("detect", yamlContent)
      : !isDirectory(path)
      ? parseClassifyDataYaml(yamlContent)
      : undefined;

  const datasetDir = isClassifyAndNoMetadata ? path : dirname(path);
  const categories: Map<number, Category> = new Map();
  if (metadata) {
    const categoryNames =
      metadata.class_names ||
      Array.from({ length: metadata.n_class + 1 }, (_, i) => i.toString());

    categoryNames.forEach((name: string, idx: number) => {
      categories.set(idx, { categoryName: name, id: idx });
    });
  } else if (isClassifyAndNoMetadata) {
    let categoryIdx = 1;
    for (const categoryName of await readdir(datasetDir)) {
      if (await isDirectory(join(datasetDir, categoryName))) {
        categories.set(categoryIdx, {
          categoryName: categoryName,
          id: categoryIdx,
        });
      }
    }
  } else
    throw new Error(`Invalid path for task type: ${task}! Received ${path}`);

  let imageIdCounter = 1;

  const groupTypes = ["train", "val", "test"];

  // DETECT / POSE IMPORT
  if (task === "detect" || task === "pose") {
    const imagesMap: Map<
      number,
      ImageWithBox<BoxAnnotation | KeypointsAnnotation> | ImageWithLabel
    > = new Map();
    for (const groupType of groupTypes) {
      const imageDir = join(datasetDir, groupType, "images");
      const labelDir = join(datasetDir, groupType, "labels");

      if (!existsSync(imageDir) || !existsSync(labelDir)) {
        throw new Error(
          `Missing image or label directory for group '${groupType}'`
        );
      }

      const imagePaths = await getImagePaths(datasetDir, groupType);
      const labelPaths = getLabelPaths(datasetDir, groupType, imagePaths);
      getValidatedImageAndLabels({
        imageDir,
        labelDir,
        imagePaths,
        labelPaths,
        missingLabels: "warn",
      });

      const imageFiles = await readdir(imageDir);
      for (const filename of imageFiles) {
        const imageId = imageIdCounter++;
        const annotations: (BoxAnnotation | KeypointsAnnotation)[] = [];
        const labelFile = join(labelDir, filename.replace(/\.[^.]+$/, ".txt"));

        if (!existsSync(labelFile)) {
          throw new Error(`Missing label file for ${filename}`);
        }

        const lines = (await readFile(labelFile, "utf8")).trim().split("\n");
        for (let annoId = 0; annoId < lines.length; annoId++) {
          const labelParts = lines[annoId].split(" ").map(Number);
          const categoryId = labelParts[0];
          const [x, y, width, height, ...keypoints] = labelParts.slice(1);

          validateCategoryId({
            categoryId,
            nCategory: (metadata as DetectYamlOptions).n_class,
          });
          validateBoundingBox({ x, y, width, height });

          if (task === "detect") {
            if (labelParts.length !== 5) {
              throw new Error(
                `Invalid detect label: expected 5 parts but got ${labelParts.length}`
              );
            }
            annotations.push({
              id: annoId + 1,
              group: groupType as "train" | "test" | "val" | "",
              categoryId: categoryId,
              x,
              y,
              width,
              height,
            });
          } else if (task === "pose") {
            const hasVisibility = (metadata as PoseYamlOptions).visibility;
            const nKeypoints = (metadata as PoseYamlOptions).n_keypoints;
            const step = hasVisibility ? 3 : 2;
            const expectedParts = nKeypoints * step;

            if (keypoints.length !== expectedParts) {
              throw new Error(
                `Invalid pose label: expected ${expectedParts} keypoint values but got ${labelParts.length}`
              );
            }

            const parsedKeypoints: Keypoint[] = [];
            for (let i = 0; i < keypoints.length; i += step) {
              const kx = +keypoints[i];
              const ky = +keypoints[i + 1];
              const visibility = !hasVisibility
                ? "unannotated"
                : +keypoints[i + 2] === 0
                ? "not_visible"
                : "visible";

              if (hasVisibility && ![0, 1, 2].includes(+keypoints[i + 2])) {
                throw new Error(
                  `Invalid visibility value '${+keypoints[i + 2]}' at index ${
                    i + 2
                  }`
                );
              }

              const keypoint: Keypoint = { x: kx, y: ky, visibility };
              if (!isKeypoint(keypoint)) {
                throw new Error(
                  `Invalid keypoint at position ${i} for ${filename}`
                );
              }
              parsedKeypoints.push(keypoint);
            }

            annotations.push({
              id: annoId + 1,
              group: groupType as "train" | "test" | "val" | "",
              categoryId: categoryId,
              x,
              y,
              width,
              height,
              keypoints: parsedKeypoints,
            });
          }
        }

        imagesMap.set(imageId, {
          id: imageId,
          filename,
          annotations,
        });
      }
    }
    return { task, categories, images: imagesMap } as
      | DetectDataset
      | PoseDataset;
  }

  // CLASSIFY IMPORT
  if (task === "classify") {
    const groupTypesClassify: ("train" | "val" | "test")[] = [
      "train",
      "val",
      "test",
    ];
    const dirEntries = await readdir(datasetDir, { withFileTypes: true });
    let groupPaths = dirEntries
      .filter(
        (entry) =>
          entry.isDirectory() && groupTypesClassify.includes(entry.name as any)
      )
      .map((entry) => join(datasetDir, entry.name));
    groupPaths = groupPaths.length === 0 ? [join(datasetDir, "")] : groupPaths;

    const imagesMap = new Map<number, ImageWithLabel>();
    let classifyIdCounter = 1;

    for (const groupPath of groupPaths) {
      const groupType =
        groupPath === datasetDir
          ? ("" as "" | "train" | "val" | "test")
          : (basename(groupPath) as "train" | "val" | "test");

      //possible bug
      const classEntries = await readdir(groupPath, { withFileTypes: true });
      for (const classEntry of classEntries) {
        if (!classEntry.isDirectory()) continue;

        const className = classEntry.name;
        const classPath = join(groupPath, className);
        const imageFiles = (await readdir(classPath)).filter(isImageFile);

        let categoryId = [...categories.values()].find(
          (c) => c.categoryName === className
        )?.id;

        if (categoryId == null) {
          categoryId = categories.size;
          categories.set(categoryId, {
            id: categoryId,
            categoryName: className,
          });
        }

        for (const fileName of imageFiles) {
          imagesMap.set(classifyIdCounter, {
            id: classifyIdCounter,
            filename: fileName,
            categoryId: categoryId,
            group: groupType,
          });
          classifyIdCounter++;
        }
      }
    }

    // Override categories with YAML-defined class names if available
    if (metadata && Array.isArray(metadata.class_names)) {
      categories.clear();
      metadata.class_names.forEach((name: string, idx: number) => {
        categories.set(idx, { id: idx, categoryName: name });
      });
    }

    console.log(`Imported YOLO classify dataset with ${imagesMap.size} images`);
    return {
      task: "classify",
      categories: categories,
      images: imagesMap,
    } as ClassifyDataset;
  }

  throw new Error(`YOLO task ${task} not supported`);
}

// ---------------- COCO ----------------
async function importCocoDataset(
  task: "classify" | "detect" | "pose",
  jsonFilePaths: string[]
): Promise<UnionDataset> {
  const jsonFiles = jsonFilePaths.filter((filePath) =>
    filePath.endsWith(".json")
  );
  const categoriesMap: Map<number, Category> = new Map();
  const imagesMap: Map<
    number,
    ImageWithLabel | ImageWithBox<BoxAnnotation | KeypointsAnnotation>
  > = new Map();
  let internalImageId = 1;

  for (const jsonFile of jsonFiles) {
    const fileContent = await readFile(jsonFile, "utf8");
    const data = JSON.parse(fileContent);

    (data.categories || []).forEach((category: any) => {
      categoriesMap.set(category.id, {
        id: +category.id + 1,
        categoryName: category.name,
      });
    });

    const groupLabel = jsonFile.includes("train")
      ? "train"
      : jsonFile.includes("val")
      ? "val"
      : jsonFile.includes("test")
      ? "test"
      : "";

    for (const imageEntry of data.images || []) {
      const annotationsForImage = (data.annotations || []).filter(
        (annotation: any) => annotation.image_id === imageEntry.id
      );

      if (task === "classify") {
        const categoryIdForImage = annotationsForImage.length
          ? annotationsForImage[0].category_id
          : -1;
        imagesMap.set(internalImageId, {
          id: internalImageId,
          filename: imageEntry.file_name,
          categoryId: categoryIdForImage,
          group: groupLabel as "train" | "val" | "test" | "",
        });
      } else {
        //TODO: Add validation
        const imageAnnotations: (BoxAnnotation | KeypointsAnnotation)[] =
          annotationsForImage.map(
            (annotation: any, annotationIndex: number) => {
              const [x, y, width, height] = annotation.bbox;
              const baseAnnotation: BoxAnnotation = {
                id: annotationIndex + 1,
                group: groupLabel as "train" | "val" | "test" | "",
                categoryId: annotation.category_id,
                x,
                y,
                width,
                height,
              };

              if (annotation.keypoints && Array.isArray(annotation.keypoints)) {
                const keypoints: Keypoint[] = [];
                for (let i = 0; i < annotation.keypoints.length; i += 3) {
                  const [kpX, kpY, visibilityValue] =
                    annotation.keypoints.slice(i, i + 3);
                  keypoints.push({
                    x: kpX,
                    y: kpY,
                    visibility:
                      visibilityValue === 0
                        ? "unannotated"
                        : visibilityValue === 1
                        ? "not_visible"
                        : "visible",
                  });
                }
                return { ...baseAnnotation, keypoints } as KeypointsAnnotation;
              }

              return baseAnnotation;
            }
          );

        imagesMap.set(internalImageId, {
          id: internalImageId,
          filename: imageEntry.file_name,
          annotations: imageAnnotations,
        });
      }
      internalImageId++;
    }
  }

  return { task, categories: categoriesMap, images: imagesMap } as UnionDataset;
}

// ---------------- Pascal VOC ----------------
// async function importPascalVocDataset(
//   task: "classify" | "detect" | "pose",
//   paths: string[]
// ): Promise<UnionDataset> {
//   if (task !== "detect") throw new Error("Pascal VOC only supports detect");

//   const annDir = paths.find((p) =>
//     p.toLowerCase().includes("annotations")
//   ) as string;
//   const categories: Map<number, Category> = new Map();
//   const imagesMap: Map<number, any> = new Map();
//   let idCounter = 1;
//   let catCounter = 0;

//   for (const file of (await readdir(annDir))) {
//     if (!file.endsWith(".xml")) continue;
//     const xml = await readFile(join(annDir, file), "utf8");
//     //TODO: implement xml parsing
//     const parsed = xml2js.parseStringSync
//       ? (xml2js as any).parseStringSync(xml)
//       : (() => {
//           let res: any;
//           xml2js.parseString(xml, (err, r) => {
//             if (err) throw err;
//             res = r;
//           });
//           return res;
//         })();

//     const filename = parsed.annotation.filename[0];
//     const objects = parsed.annotation.object || [];
//     const annotations: BoxAnnotation[] = [];
//     for (let annId = 0; annId < objects.length; annId++) {
//       const obj = objects[annId];
//       const name = obj.name[0];
//       let catId = [...categories.values()].find(
//         (c) => c.class_name === name
//       )?.id;
//       if (catId == null) {
//         catId = catCounter++;
//         categories.set(catId, { id: catId, class_name: name });
//       }
//       const bnd = obj.bndbox[0];
//       const x = parseInt(bnd.xmin[0]);
//       const y = parseInt(bnd.ymin[0]);
//       const w = parseInt(bnd.xmax[0]) - x;
//       const h = parseInt(bnd.ymax[0]) - y;
//       annotations.push({
//         id: annId,
//         group: "",
//         category_id: catId,
//         x,
//         y,
//         width: w,
//         height: h,
//       });
//     }
//     imagesMap.set(idCounter, {
//       id: idCounter,
//       filename,
//       annotations,
//     });
//     idCounter++;
//   }

//   return { task, categories, images: imagesMap } as DetectDataset;
// }

type UnionDataset = ClassifyDataset | DetectDataset | PoseDataset;

type ClassifyDataset = Dataset<"classify", ImageWithLabel>;
type DetectDataset = Dataset<"detect", ImageWithBox<BoxAnnotation>>;
type PoseDataset = Dataset<"pose", ImageWithBox<KeypointsAnnotation>>;

type Dataset<Task, Image> = {
  task: Task;
  categories: Map<number, Category>;
  images: Map<number, Image>;
};
type Category = {
  id: number;
  categoryName: string;
};
type ImageWithLabel = {
  id: number;
  filename: string;
  categoryId: number;
  group: "train" | "val" | "test" | "";
};
type ImageWithBox<Annotation> = {
  id: number;
  filename: string;
  annotations: Annotation[];
};
type BoxAnnotation = {
  id: number;
  group: "train" | "val" | "test" | "";
  categoryId: number;
  x: number;
  y: number;
  width: number;
  height: number;
};
type KeypointsAnnotation = BoxAnnotation & {
  keypoints: Keypoint[];
};
type Keypoint = {
  x: number;
  y: number;
  visibility: "unannotated" | "not_visible" | "visible";
};
type ImportDatasetOptions = {
  /**
   * For yolo:
   *   - ./dataset/data.yaml
   *   - ./dataset
   * For coco:
   *   - ./dataset/labels.json
   *   - ./dataset/labels_train.json
   *   - ./dataset/labels_val.json
   *   - ./dataset/labels_test.json
   * For Pascal VOC:
   *   - ./dataset/Annotations
   *   - ./dataset/annotations
   */
  task: "classify" | "detect" | "pose";
  importDatasetPath: string | string[];
};

type ExportDatasetOptions = {
  importDatasetPath: string | string[];
  dataset: UnionDataset;
  /**
   * e.g.
   *   - ./dataset/data.yaml
   *   - ./dataset/labels.json
   *   - ./dataset/labels_train.json
   *   - ./dataset/labels_val.json
   *   - ./dataset/labels_test.json
   *   - ./dataset/annotations
   *   - ./export/dataset-11-v12.zip
   */
  exportDatasetPath: string | string[];
  format: "yolo" | "coco" | "pascal_voc";
  // ratio
};
