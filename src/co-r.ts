import { constants, existsSync } from "fs";
import { stat, readdir, readFile, access } from "fs/promises";
import { basename, dirname, extname, join } from "path";
import {
  ClassifyYamlOptions,
  DetectYamlOptions,
  parseClassifyDataYaml,
  parseDataYaml,
  PoseYamlOptions,
} from "./yaml";
import xml_to_json from "xml-parser";
import { getDirFilenames, getDirFilenamesSync } from "@beenotung/tslib";
import { GroupType } from "./group";

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
    missingLabel?: "ignore" | "warn" | "error";
  } & ImportDatasetCallbacks
): Promise<ClassifyDataset>;
export async function importDataset(
  args: {
    task: "detect";
    importDatasetPath: string | string[];
    missingLabel?: "ignore" | "warn" | "error";
    getAnnotationId?: (annotation: Omit<BoxAnnotation, "id">) => number;
  } & ImportDatasetCallbacks
): Promise<DetectDataset>;
export async function importDataset(
  args: {
    task: "pose";
    importDatasetPath: string | string[];
    missingLabel?: "ignore" | "warn" | "error";
    getAnnotationId?: (annotation: Omit<KeypointsAnnotation, "id">) => number;
  } & ImportDatasetCallbacks
): Promise<PoseDataset>;
export async function importDataset(
  args: {
    task: "classify" | "detect" | "pose";
    importDatasetPath: string | string[];
    missingLabel?: "ignore" | "warn" | "error";
    getAnnotationId?: (annotation: Omit<KeypointsAnnotation, "id">) => number;
  } & ImportDatasetCallbacks
): Promise<UnionDataset> {
  const {
    task,
    importDatasetPath,
    getCategoryId,
    getImageId,
    getAnnotationId,
    missingLabel,
  } = args;
  const paths = Array.isArray(importDatasetPath)
    ? importDatasetPath
    : [importDatasetPath];

  // Decide format from file/folder pattern
  const format = await detectDatasetFormat(importDatasetPath);

  switch (format) {
    case "yolo":
      return await importYoloDataset({
        task,
        paths,
        getImageId,
        getCategoryId,
        getAnnotationId,
        missingLabel,
      });
    case "coco":
      return await importCocoDataset({
        task,
        paths,
        getImageId,
        getCategoryId,
        getAnnotationId,
        missingLabel,
      });
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
  missingLabel: "error" | "warn" | "ignore";
}) {
  function handleMissingLabel(
    mode: "error" | "warn" | "ignore",
    message: string
  ): void {
    if (mode === "error") throw new Error(message);
    if (mode === "warn") console.warn(`Warning: ${message}`);
  }
  const { imageDir, labelDir, imagePaths, labelPaths, missingLabel } = options;

  const validated_image_paths = imagePaths.filter((imagePath) => {
    const image_full_path = join(imageDir, imagePath);
    const label_path = basename(imagePath, extname(imagePath)) + ".txt";
    const label_full_path = join(labelDir, label_path);

    if (!existsSync(label_full_path)) {
      handleMissingLabel(
        missingLabel,
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
        missingLabel,
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

async function importYoloDataset(args: {
  task: "classify" | "detect" | "pose";
  paths: string[];
  getImageId?: (imageFilename: string) => number;
  getCategoryId?: (categoryName: string) => number;
  getAnnotationId?: (annotation: Omit<KeypointsAnnotation, "id">) => number;
  missingLabel?: "ignore" | "warn" | "error";
}): Promise<UnionDataset> {
  const {
    task,
    paths,
    getImageId,
    getCategoryId,
    getAnnotationId,
    missingLabel,
  } = args;
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
  const yamlMetadata =
    task === "pose"
      ? parseDataYaml("pose", yamlContent)
      : task === "detect"
      ? parseDataYaml("detect", yamlContent)
      : !isDirectory(path)
      ? parseClassifyDataYaml(yamlContent)
      : undefined;

  const datasetDir = isClassifyAndNoMetadata ? path : dirname(path);
  const categories: Map<number, Category> = new Map();

  if (yamlMetadata) {
    const categoryNames =
      yamlMetadata.class_names ||
      Array.from({ length: yamlMetadata.n_class + 1 }, (_, i) => i.toString());

    categoryNames.forEach((name: string, idx: number) => {
      const categoryId = getCategoryId ? getCategoryId(name) : idx + 1;
      categories.set(categoryId, { categoryName: name, id: categoryId });
    });
  } else if (isClassifyAndNoMetadata) {
    let categoryIdx = 1;
    for (const categoryName of await readdir(datasetDir)) {
      categoryIdx = getCategoryId ? getCategoryId(categoryName) : categoryIdx;
      if (await isDirectory(join(datasetDir, categoryName))) {
        categories.set(categoryIdx, {
          categoryName: categoryName,
          id: categoryIdx,
        });
        categoryIdx++;
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
        missingLabel: missingLabel ?? "error",
      });

      const imageFiles = await readdir(imageDir);
      for (const filename of imageFiles) {
        const imageId = getImageId ? getImageId(filename) : imageIdCounter++;
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
            nCategory: (yamlMetadata as DetectYamlOptions).n_class,
          });
          validateBoundingBox({ x, y, width, height });

          if (task === "detect") {
            if (labelParts.length !== 5) {
              throw new Error(
                `Invalid detect label: expected 5 parts but got ${labelParts.length}`
              );
            }
            const annotation = {
              id: annoId + 1,
              group: groupType as "train" | "test" | "val" | "",
              categoryId: categoryId,
              x,
              y,
              width,
              height,
            };
            if (getAnnotationId)
              annotation.id = getAnnotationId({ ...annotation, keypoints: [] });
            annotations.push(annotation);
          } else if (task === "pose") {
            const hasVisibility = (yamlMetadata as PoseYamlOptions).visibility;
            const nKeypoints = (yamlMetadata as PoseYamlOptions).n_keypoints;
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
            const annotation = {
              id: annoId + 1,
              group: groupType as "train" | "test" | "val" | "",
              categoryId: categoryId,
              x,
              y,
              width,
              height,
              keypoints: parsedKeypoints,
            };
            if (getAnnotationId)
              annotation.id = getAnnotationId({ ...annotation, keypoints: [] });
            annotations.push(annotation);
          }
        }

        imagesMap.set(imageId, {
          id: imageId,
          filename,
          annotations,
        });
      }
    }
    if (task === "detect")
      return {
        task,
        categories,
        images: imagesMap,
        metadata: {
          categories,
          train: yamlMetadata?.train_path,
          test: yamlMetadata?.test_path,
          val: yamlMetadata?.val_path,
        },
      } as DetectDataset;
    return {
      task,
      categories,
      images: imagesMap,
      metadata: {
        categories,
        train: yamlMetadata?.train_path,
        test: yamlMetadata?.test_path,
        val: yamlMetadata?.val_path,
        keypoint_names: (yamlMetadata as PoseYamlOptions).keypoint_names,
        n_keypoints: (yamlMetadata as PoseYamlOptions).n_keypoints,
        visibility: (yamlMetadata as PoseYamlOptions).visibility
          ? "visible"
          : "unannotated",
        // flip_idx?: number[]; //TODO: design
      },
    } as PoseDataset;
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

        for (const filename of imageFiles) {
          const imageId = getImageId ? getImageId(filename) : classifyIdCounter;
          imagesMap.set(classifyIdCounter, {
            id: imageId,
            filename,
            categoryId,
            group: groupType,
          });
          classifyIdCounter++;
        }
      }
    }

    // Override categories with YAML-defined class names if available
    if (yamlMetadata && Array.isArray(yamlMetadata.class_names)) {
      categories.clear();
      yamlMetadata.class_names.forEach((name: string, idx: number) => {
        categories.set(idx, { id: idx, categoryName: name });
      });
    }

    console.log(`Imported YOLO classify dataset with ${imagesMap.size} images`);
    return {
      task: "classify",
      categories,
      images: imagesMap,
      metadata: {
        categories,
        train: yamlMetadata?.train_path,
        test: yamlMetadata?.test_path,
        val: yamlMetadata?.val_path,
      },
    } as ClassifyDataset;
  }

  throw new Error(`YOLO task ${task} not supported`);
}

// ---------------- COCO ----------------
async function importCocoDataset(args: {
  task: "classify" | "detect" | "pose";
  paths: string[];
  getImageId?: (imageFilename: string) => number;
  getCategoryId?: (categoryName: string) => number;
  getAnnotationId?: (annotation: Omit<KeypointsAnnotation, "id">) => number;
  missingLabel?: "ignore" | "warn" | "error";
}): Promise<UnionDataset> {
  const {
    task,
    paths,
    getImageId,
    getCategoryId,
    getAnnotationId,
    missingLabel = "error",
  } = args;

  const jsonFiles = paths.filter((filePath) => filePath.endsWith(".json"));
  const categories = new Map<number, Category>();
  const imagesMap = new Map<
    number,
    ImageWithLabel | ImageWithBox<BoxAnnotation | KeypointsAnnotation>
  >();

  let internalImageId = 1;

  const getGroupLabel = (filePath: string): GroupType | "" => {
    if (filePath.includes("train")) return "train";
    if (filePath.includes("val")) return "val";
    if (filePath.includes("test")) return "test";
    return "";
  };

  const validateFileExists = async (filePath: string) => {
    try {
      await access(filePath, constants.F_OK);
      return true;
    } catch {
      return false;
    }
  };

  const givenGroupTypes = [];
  let maxNumberOfKeypoints = 0;

  for (const jsonFile of jsonFiles) {
    const {
      images = [],
      annotations = [],
      categories: catList = [],
    } = JSON.parse(await readFile(jsonFile, "utf8"));
    const groupLabel = getGroupLabel(jsonFile);
    givenGroupTypes.push(groupLabel);
    const jsonDir = dirname(jsonFile);

    // Map old category IDs to new ones
    const categoryIdMap = new Map<number, number>();
    const seenCategoryIds = new Set<number>();

    for (const cat of catList) {
      const newId = getCategoryId
        ? getCategoryId(cat.name)
        : Number(cat.id) + 1;
      if (seenCategoryIds.has(newId)) {
        throw new Error(`Duplicate category ID detected: ${newId}`);
      }
      seenCategoryIds.add(newId);
      categoryIdMap.set(Number(cat.id), newId);
      categories.set(newId, { id: newId, categoryName: cat.name });
    }

    const seenImageIds = new Set<number>();

    for (const img of images) {
      const imageId = getImageId
        ? getImageId(img.file_name)
        : internalImageId++;
      if (seenImageIds.has(imageId)) {
        throw new Error(`Duplicate image ID detected: ${imageId}`);
      }
      seenImageIds.add(imageId);

      // Validate image file exists
      const possiblePath = join(jsonDir, "data", img.file_name);
      const exists = await validateFileExists(possiblePath);
      if (!exists) {
        const message = `Image file not found: ${possiblePath}`;
        if (missingLabel === "error") throw new Error(message);
        if (missingLabel === "warn") console.warn(message);
      }

      const annsForImage = annotations.filter(
        (a: any) => a.image_id === img.id
      );

      // Validate annotation references
      for (const ann of annsForImage) {
        if (!categoryIdMap.has(ann.category_id)) {
          const message = `Annotation ${ann.id} references missing category ID ${ann.category_id}`;
          if (missingLabel === "error") throw new Error(message);
          if (missingLabel === "warn") console.warn(message);
        }
      }

      if (task === "classify") {
        const categoryIds = annsForImage.map(
          (a: any) => categoryIdMap.get(a.category_id) ?? a.category_id
        );
        imagesMap.set(imageId, {
          id: imageId,
          filename: img.file_name,
          categoryId: categoryIds.length > 1 ? categoryIds : categoryIds[0],
          group: groupLabel,
        });
        continue;
      }

      // Detection / Pose
      const seenAnnIds = new Set<number>();
      const imageAnnotations: (BoxAnnotation | KeypointsAnnotation)[] =
        annsForImage.map((annotation: any, idx: number) => {
          const [x, y, width, height] = annotation.bbox;
          const categoryId =
            categoryIdMap.get(annotation.category_id) ?? annotation.category_id;

          const baseAnnotation: BoxAnnotation = {
            id: idx + 1,
            group: groupLabel,
            categoryId,
            x,
            y,
            width,
            height,
          };

          const annotationId = getAnnotationId
            ? getAnnotationId({ ...baseAnnotation, keypoints: [] })
            : idx + 1;
          if (seenAnnIds.has(annotationId)) {
            throw new Error(
              `Duplicate annotation ID detected: ${annotationId} in image ${imageId}`
            );
          }
          seenAnnIds.add(annotationId);

          if (Array.isArray(annotation.keypoints)) {
            const keypoints: Keypoint[] = [];
            let nKeypoints = 0;
            for (let i = 0; i < annotation.keypoints.length; i += 3) {
              const [kpX, kpY, vis] = annotation.keypoints.slice(i, i + 3);
              keypoints.push({
                x: kpX,
                y: kpY,
                visibility:
                  vis === 0
                    ? "unannotated"
                    : vis === 1
                    ? "not_visible"
                    : "visible",
              });
              nKeypoints++;
            }
            maxNumberOfKeypoints =
              nKeypoints > maxNumberOfKeypoints
                ? nKeypoints
                : maxNumberOfKeypoints;
            return {
              ...baseAnnotation,
              annotationId,
              keypoints,
            } as KeypointsAnnotation;
          }

          return { ...baseAnnotation, annotationId };
        });

      imagesMap.set(imageId, {
        id: imageId,
        filename: img.file_name,
        annotations: imageAnnotations,
      });
    }
  }
  const metadata =
    task === "pose"
      ? {
          task,
          categories,
          images: imagesMap,
          metadata: {
            categories,
            train: givenGroupTypes.includes("train") ? "../train" : undefined,
            test: givenGroupTypes.includes("test") ? "../test" : undefined,
            val: givenGroupTypes.includes("val") ? "../val" : undefined,
            n_keypoints: maxNumberOfKeypoints,
            visibility: "visible",
          },
        }
      : {
          task,
          categories,
          images: imagesMap,
          metadata: {
            categories,
            train: givenGroupTypes.includes("train") ? "../train" : undefined,
            test: givenGroupTypes.includes("test") ? "../test" : undefined,
            val: givenGroupTypes.includes("val") ? "../val" : undefined,
          },
        };
  return { task, categories, images: imagesMap, metadata } as UnionDataset;
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

async function test() {
  const task = "detect";
  const importDatasetPath = `coco-dataset/internal.json`;
  const getCategoryId = (categoryName: string): number => {
    return categoryName.charCodeAt(0);
  };
  const getImageId = (imageFilename: string): number => {
    const match = imageFilename.match(/\d+/);
    return match ? parseInt(match[0], 10) : -1;
  };
  const result = await importDataset({
    task,
    importDatasetPath,
    getCategoryId,
    getImageId,
  });
  console.log(result);
}
test();

type UnionDataset = ClassifyDataset | DetectDataset | PoseDataset;

type ClassifyDataset = Dataset<
  "classify",
  ImageWithLabel,
  ClassifyMetadataOptions
>;
type DetectDataset = Dataset<
  "detect",
  ImageWithBox<BoxAnnotation>,
  DetectMetadataOptions
>;
type PoseDataset = Dataset<
  "pose",
  ImageWithBox<KeypointsAnnotation>,
  PoseMetadataOptions
>;

type Dataset<Task, Image, MetadataOptions> = {
  task: Task;
  categories: Map<number, Category>;
  images: Map<number, Image>;
  metadata: MetadataOptions;
};
export type Category = {
  id: number;
  categoryName: string;
};
export type ImageWithLabel = {
  id: number;
  filename: string;
  categoryId: number | number[];
  group: "train" | "val" | "test" | "";
};
export type ImageWithBox<Annotation> = {
  id: number;
  filename: string;
  annotations: Annotation[];
};
export type BoxAnnotation = {
  id: number;
  group: "train" | "val" | "test" | "";
  categoryId: number;
  x: number;
  y: number;
  width: number;
  height: number;
};
export type KeypointsAnnotation = BoxAnnotation & {
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

export type ExportDatasetOptions = {
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

type BaseMetadataOptions = {
  categories: Map<number, Category>;
  train?: string;
  val?: string;
  test?: string;
};

type ClassifyMetadataOptions = BaseMetadataOptions & {
  categoryNames: Map<number, Category>;
};

type DetectMetadataOptions = BaseMetadataOptions;

type PoseMetadataOptions = BaseMetadataOptions & {
  keypoint_names?: string[];
  n_keypoints: number;
  visibility: "unannotated" | "not_visible" | "visible";
  flip_idx?: number[]; //TODO: design
};
