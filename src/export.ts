import { join, dirname, basename, extname } from "path";
import { copyFile, readFile, writeFile, appendFile, stat } from "fs/promises";
import { existsSync } from "fs";

import {
  cachedCopyFiles,
  cachedMkdir,
  ExportClassifyDatasetOptions,
} from "./fs";
import { group_types as groupTypes, GroupType } from "./group";
import {
  ExportDatasetOptions,
  BoxAnnotation,
  KeypointsAnnotation,
  Category,
} from "./co-r";
import { dispatchGroup } from "./split";
import { toClassifyDataYamlString, toDataYamlString } from "./yaml";
import { toDetectLabelString, toPoseLabelString } from "./label";
import {
  toYoloBoundingBoxLabelString,
  toYoloBoundingBoxWithKeypointsLabelString,
} from "./label-r";
import { group } from "console";

type Sample = {
  imageId: number;
  filename: string;
  categoryName: string;
  groupType: GroupType | "";
  annotation: BoxAnnotation | KeypointsAnnotation;
};

export async function exportDatasetHelper<
  Sample extends {
    imageId: number;
    filename: string;
    categoryName: string;
    groupType: GroupType | "";
    annotation: BoxAnnotation | KeypointsAnnotation;
  }
>(options: {
  generateSampleSequence(): Iterable<Sample>;
  dispatchGroupToCall?: (options: {
    sample: Sample;
    current: Record<GroupType, number>;
  }) => GroupType;
  saveSample: (options: {
    sample: Sample;
    groupType: GroupType;
  }) => Promise<void>;
  createDirsAndMetadata?: (
    groupTypes: (GroupType | "")[]
  ) => Promise<"saved" | "no change" | undefined>;
}) {
  const sampleIterator = options.generateSampleSequence();
  const {
    saveSample,
    dispatchGroupToCall: dispatchGroup,
    createDirsAndMetadata,
  } = options;

  const groupTypes: GroupType[] = [];
  const currentRatioByClass: Record<
    string,
    Record<GroupType, number>
  > = Object.create(null);

  for (const sample of sampleIterator) {
    const class_name = sample.categoryName;
    let groupType: GroupType;

    if (dispatchGroup) {
      currentRatioByClass[class_name] ??= { train: 0, val: 0, test: 0 };
      const current = currentRatioByClass[class_name];
      groupType = dispatchGroup({ sample, current });
      current[groupType]++;
    } else {
      //TODO: add support for coco no group type //BUG
      groupType = sample.groupType as GroupType;
    }

    if (!groupTypes.includes(groupType)) {
      groupTypes.push(groupType);
    }

    await saveSample({ sample, groupType });
  }

  if (createDirsAndMetadata) {
    await createDirsAndMetadata(groupTypes);
  }
}

export async function exportDataset(args: ExportDatasetOptions) {
  const format = args.format;
  switch (format) {
    case "yolo": {
      // exportYoloDataset(args);
      break;
    }
    case "coco": {
      exportCocoDataset(args);
      break;
    }
    case "pascal_voc": {
      // exportPascalVocDataset(args)
      break;
    }
    default:
      throw new Error(`Invalid format: receive ${format}`);
  }
  console.log("Exported dataset");
}

function validatePaths(args: {
  importImageDirs: string | Partial<Record<GroupType, string>>;
  importMetadataPaths: string | Partial<Record<GroupType, string>>;
  exportImageDirs: string | Partial<Record<GroupType, string>>;
  exportMetadataPaths: string | Partial<Record<GroupType, string>>;
  groupRatio?: Record<GroupType, number>;
  dispatchGroup?: boolean;
}) {
  const {
    importImageDirs,
    importMetadataPaths,
    exportImageDirs,
    exportMetadataPaths,
    groupRatio,
    dispatchGroup,
  } = args;

  // 1. Validate import dir ↔ metadata type consistency
  for (const groupType of groupTypes) {
    if (
      typeof importImageDirs === "string" ||
      typeof importMetadataPaths === "string"
    ) {
      if (typeof importImageDirs !== typeof importMetadataPaths) {
        throw new Error(
          `Mismatch between type of importMetadataPaths and importImageDirs`
        );
      }
    } else {
      if (
        typeof importImageDirs[groupType] !==
        typeof importMetadataPaths[groupType]
      ) {
        throw new Error(
          `Mismatch between ${groupType} type of importMetadataPaths and importImageDirs`
        );
      }
    }
  }

  // 2. Validate export dir ↔ metadata type consistency
  for (const groupType of groupTypes) {
    if (
      typeof exportImageDirs === "string" ||
      typeof exportMetadataPaths === "string"
    ) {
      if (typeof exportImageDirs !== typeof exportMetadataPaths) {
        throw new Error(
          `Mismatch between type of exportMetadataPaths and exportImageDirs`
        );
      }
    } else {
      if (
        typeof exportImageDirs[groupType] !==
        typeof exportMetadataPaths[groupType]
      ) {
        throw new Error(
          `Mismatch of ${groupType} option between exportMetadataPaths and exportImageDirs`
        );
      }
    }
  }

  // 3. Validate groupRatio or import/export mirroring
  if (
    !dispatchGroup &&
    typeof exportImageDirs !== "string" &&
    typeof exportMetadataPaths !== "string"
  ) {
    if (groupRatio) {
      // Case 1: groupRatio provided
      for (const groupType of groupTypes) {
        const groupTypeRatio = groupRatio[groupType] ?? 0;

        if (
          groupTypeRatio > 0 &&
          (!exportImageDirs[groupType] || !exportMetadataPaths[groupType])
        ) {
          throw new Error(
            `Group type "${groupType}" has a ratio in groupRatio but is missing exportImageDirs or exportMetadataPaths`
          );
        }
      }
    } else {
      // Case 2: groupRatio not provided, export paths must mirror groups of import paths
      if (
        typeof importImageDirs !== "string" &&
        typeof importMetadataPaths !== "string"
      ) {
        for (const groupType of groupTypes) {
          if (importImageDirs[groupType] || importMetadataPaths[groupType]) {
            if (
              !exportImageDirs[groupType] ||
              !exportMetadataPaths[groupType]
            ) {
              throw new Error(
                `Group type "${groupType}" exists in import paths but is missing in exportImageDirs or exportMetadataPaths`
              );
            }
          }
        }
      }
    }
  }
}


// export async function exportYoloDataset(args: ExportDatasetOptions) {
//   const {
//     dataset,
//     importDatasetPath,
//     exportDatasetPath,
//     dispatchGroup: customDispatchGroup,
//     groupRatio,
//   } = args;
//   const { categories, images, metadata, task } = dataset;
//   const importDatasetDir = (
//     await stat(
//       dirname(
//         Array.isArray(importDatasetPath)
//           ? importDatasetPath[0]
//           : importDatasetPath
//       )
//     )
//   ).isDirectory()
//     ? dirname(
//         Array.isArray(importDatasetPath)
//           ? importDatasetPath[0]
//           : importDatasetPath
//       )
//     : dirname(
//         dirname(
//           Array.isArray(importDatasetPath)
//             ? importDatasetPath[0]
//             : importDatasetPath
//         )
//       );

//   const exportDatasetDir = Array.isArray(exportDatasetPath)
//     ? exportDatasetPath[0]
//     : exportDatasetPath;

//   function* generateSampleSequence(): Iterable<Sample> {
//     if (task === "classify") {
//       for (const [imageId, image] of images) {
//         const categoryIds = Array.isArray(image.categoryId)
//           ? image.categoryId
//           : [image.categoryId];
//         for (const categoryId of categoryIds) {
//           const categoryName =
//             categories.get(categoryId)?.categoryName ?? `${categoryId}`;
//           yield {
//             imageId,
//             filename: image.filename,
//             categoryName,
//             groupType: image.groupType,
//             annotation: Object.create(null),
//           };
//         }
//       }
//     } else {
//       for (const [imageId, image] of images) {
//         for (const annotation of image.annotations) {
//           const categoryName =
//             categories.get(annotation.categoryId)?.categoryName ??
//             `${annotation.categoryId}`;
//           yield {
//             imageId,
//             filename: image.filename,
//             categoryName,
//             groupType: annotation.groupType,
//             annotation: annotation,
//           };
//         }
//       }
//     }
//   }

//   async function saveSample(args: { sample: Sample; groupType: GroupType }) {
//     const { sample, groupType: exportGroupType } = args;
//     const {
//       imageId,
//       filename,
//       categoryName,
//       groupType: importGroupType,
//     } = sample;
//     if (task === "pose" || task === "detect") {
//       const { annotation } = sample;
//       const labelFilename = basename(filename, extname(filename)) + ".txt";

//       const imagesSrcDir = join(importDatasetDir, importGroupType, "images");
//       const imagesDestDir = join(exportDatasetDir, exportGroupType, "images");
//       const labelsDestDir = join(exportDatasetDir, exportGroupType, "labels");

//       await cachedMkdir(imagesDestDir);
//       const imageSrcPath = join(imagesSrcDir, filename);
//       const imageDestPath = join(imagesDestDir, filename);
//       await copyFile(imageSrcPath, imageDestPath);

//       await cachedMkdir(labelsDestDir);
//       const labelDestPath = join(labelsDestDir, labelFilename);
//       let labelStr = "";

//       if (task === "detect") {
//         labelStr = toYoloBoundingBoxLabelString({
//           ...annotation,
//           nCategory: categories.size,
//         });
//       } else if (task === "pose") {
//         const { keypoints } = annotation as KeypointsAnnotation;
//         labelStr = toYoloBoundingBoxWithKeypointsLabelString({
//           ...annotation,
//           nCategory: categories.size,
//           nKeypoints: metadata.n_keypoints,
//           hasVisibility: metadata.visibility !== "unannotated",
//           keypoints,
//         });
//       } else throw new Error(`Invalid task type: receive ${task}`);

//       if (!existsSync(labelDestPath)) {
//         //if txt file does not exist, create txt file
//         await writeFile(labelDestPath, labelStr);
//       } else {
//         //else add line to txt file
//         await appendFile(labelDestPath, "\n" + labelStr);
//       }
//       return;
//     }
//     if (task === "classify") {
//       const srcFile = join(
//         importDatasetDir,
//         importDatasetDir,
//         categoryName,
//         filename
//       );
//       const destFile = join(
//         exportDatasetDir,
//         exportGroupType,
//         categoryName,
//         filename
//       );

//       await cachedMkdir(dirname(destFile));
//       await copyFile(srcFile, destFile);
//       return;
//     }
//     throw new Error(`Unsupported task type: receive "${task}"`);
//   }

//   let dispatchGroupToCall:
//     | ((options: {
//         sample: Sample;
//         current: Record<GroupType, number>;
//       }) => GroupType)
//     | undefined;

//   if (customDispatchGroup) {
//     dispatchGroupToCall = ({ sample }) => customDispatchGroup(sample);
//   } else if (groupRatio) {
//     dispatchGroupToCall = ({ current }) =>
//       dispatchGroup({ current, target: groupRatio });
//   }
//   async function createDirsAndMetadata(
//     groupTypes: string[]
//   ): Promise<"saved" | "no change" | undefined> {
//     const categoryNames = Array.from(categories.values()).map(
//       (category) => category.categoryName
//     );
//     if (groupTypes.includes("")) {
//       metadata.train = undefined;
//       metadata.test = undefined;
//       metadata.val = undefined;
//     } else {
//       metadata.train = groupTypes.includes("train") ? "../train" : undefined;
//       metadata.test = groupTypes.includes("test") ? "../test" : undefined;
//       metadata.val = groupTypes.includes("val") ? "../val" : undefined;
//     }
//     if (task === "classify") {
//       for (const groupType of groupTypes) {
//         for (const categoryName of categoryNames) {
//           const exportDirPath = join(exportDatasetDir, groupType, categoryName);
//           await cachedMkdir(exportDirPath);
//         }
//       }
//       const newContent = toClassifyDataYamlString({
//         train_path: metadata.train,
//         test_path: metadata.test,
//         val_path: metadata.val,
//         n_class: categories.size,
//         class_names: categoryNames,
//       });
//       const yamlPath = join(exportDatasetDir, "data.yaml");
//       if (existsSync(yamlPath)) {
//         const oldContent = await readFile(yamlPath).toString();
//         if (newContent === oldContent) return "no change";
//       }
//       await writeFile(yamlPath, newContent);
//       return "saved";
//     }
//     if (task === "pose" || task === "detect") {
//       for (const groupType of groupTypes) {
//         const exportDirImagesPath = join(exportDatasetDir, groupType, "images");
//         const exportDirLabelsPath = join(exportDatasetDir, groupType, "labels");
//         await cachedMkdir(exportDirImagesPath);
//         await cachedMkdir(exportDirLabelsPath);
//       }
//       let yamlMetadata;
//       if (task === "pose") {
//         yamlMetadata = {
//           train_path: metadata.train,
//           test_path: metadata.test,
//           val_path: metadata.val,
//           class_names: categoryNames,
//           n_class: categories.size,
//           keypoint_names: metadata.keypoint_names,
//           n_keypoints: metadata.n_keypoints,
//           visibility: metadata.visibility !== "unannotated",
//           flip_idx: metadata.flip_idx,
//         };
//       } else {
//         yamlMetadata = {
//           train_path: metadata.train,
//           test_path: metadata.test,
//           val_path: metadata.val,
//           class_names: categoryNames,
//           n_class: categories.size,
//         };
//       }
//       const newContent = toDataYamlString(task, yamlMetadata);
//       const yamlPath = join(exportDatasetDir, "data.yaml");
//       if (existsSync(yamlPath)) {
//         const oldContent = await readFile(yamlPath).toString();
//         if (newContent === oldContent) return "no change";
//       }

//       await writeFile(yamlPath, newContent);
//       return "saved";
//     }
//     throw new Error(`Unsupported task provided: ${task}`);
//   }

//   await exportDatasetHelper<Sample>({
//     generateSampleSequence,
//     saveSample,
//     dispatchGroupToCall,
//     createDirsAndMetadata,
//   });

//   console.log(`Exported YOLO dataset (task: ${task})`);
// }

type CocoJsonFormat = {
  categories:
    | {
        id: number;
        name: string;
      }[]
    | {
        id: number;
        name: string;
        keypoints?: string[];
        skeleton?: number[];
      }[];
  images: { id: number; file_name: string }[];
  annotations: {
    id: number;
    image_id: number;
    category_id: number;
    bbox?: number[];
    keypoints?: number[];
  }[];
};
export async function exportCocoDataset(args: ExportDatasetOptions) {
  const {
    dataset,
    importMetadataPaths,
    importImageDirs,
    exportMetadataPaths,
    exportImageDirs,
    dispatchGroup: customDispatchGroup,
    groupRatio,
  } = args;
  const { categories, images, metadata, task } = dataset;
  const isGroupKey = (key: string): key is GroupType =>
    key === "train" || key === "test" || key === "val";

  validatePaths({importImageDirs, importMetadataPaths, exportImageDirs, exportMetadataPaths, groupRatio, dispatchGroup: !!customDispatchGroup});

  const jsonObjects: Record<GroupType | "", CocoJsonFormat> =
    Object.create(null);

  function* generateSampleSequence(): Iterable<Sample> {
    if (task === "classify") {
      for (const [imageId, image] of images) {
        const categoryIds = Array.isArray(image.categoryId)
          ? image.categoryId
          : [image.categoryId];
        for (const categoryId of categoryIds) {
          const categoryName =
            categories.get(categoryId)?.categoryName ?? `${categoryId}`;
          yield {
            imageId,
            filename: image.filename,
            categoryName,
            groupType: image.groupType,
            annotation: Object.create(null),
          };
        }
      }
    } else {
      for (const [imageId, image] of images) {
        if (image.annotations) {
          for (const annotation of image.annotations) {
            const categoryName =
              categories.get(annotation.categoryId)?.categoryName ??
              `${annotation.categoryId}`;
            yield {
              imageId,
              filename: image.filename,
              categoryName,
              groupType: annotation.groupType,
              annotation: annotation,
            };
          }
        } else
          throw new Error(
            `Image annotations are missing for ${image.filename} with ID ${imageId} (task: "${task}")`
          );
      }
    }
  }

  async function saveSample(args: {
    sample: Sample;
    groupType: GroupType | "";
  }) {
    const { sample } = args;
    //Override groupType if export image dirs is a string
    const groupType = typeof exportImageDirs === "string" ? "" : args.groupType;
    const importImageDir =
      typeof importImageDirs === "string"
        ? importImageDirs
        : sample.groupType && isGroupKey(sample.groupType)
        ? importImageDirs[sample.groupType]
        : undefined;
    const exportImageDir =
      typeof exportImageDirs === "string"
        ? exportImageDirs
        : groupType && isGroupKey(groupType)
        ? exportImageDirs[groupType]
        : undefined;
    if (!importImageDir || !exportImageDir) {
      throw new Error(
        `Invalid import or export image directory for group type "${sample.groupType}"`
      );
    }

    const importImagePath = join(importImageDir, sample.filename);
    const exportImagePath = join(exportImageDir, sample.filename);
    // Ensure the image file exists before copying
    await cachedMkdir(exportImageDir);
    await cachedCopyFiles(importImagePath, exportImagePath);

    if (!(groupType in jsonObjects)) {
      jsonObjects[groupType] = {
        categories: Array.from(categories, ([categoryId, category]) => ({
          id: categoryId,
          name: category.categoryName,
          keypoints: category.keypoints,
          skeleton: category.skeleton,
          flip_idx: category.flip_idx,
        })),
        images: [],
        annotations: [],
      };
    }

    if (
      !jsonObjects[groupType].images.some((img) => img.id === sample.imageId)
    ) {
      jsonObjects[groupType].images.push({
        id: sample.imageId,
        file_name: sample.filename,
      });
    }

    if (task === "classify") {
      jsonObjects[groupType].annotations.push({
        id: sample.annotation.id,
        image_id: sample.imageId,
        category_id: sample.annotation.categoryId,
      });
    } else {
      //TODO: confirm .toFixed(3) implementation
      const bbox = [
        sample.annotation.x,
        sample.annotation.y,
        sample.annotation.width,
        sample.annotation.height,
      ];
      if (!(sample.annotation as KeypointsAnnotation).keypoints) {
        jsonObjects[groupType].annotations.push({
          id: sample.annotation.id,
          image_id: sample.imageId,
          category_id: sample.annotation.categoryId,
          bbox,
        });
      } else {
        const keypoints = (sample.annotation as KeypointsAnnotation).keypoints
          .map((k) => [
            k.x,
            k.y,
            k.visibility === "unannotated"
              ? 0
              : k.visibility === "not_visible"
              ? 1
              : 2,
          ])
          .flat();

        jsonObjects[groupType].annotations.push({
          id: sample.annotation.id,
          image_id: sample.imageId,
          category_id: sample.annotation.categoryId,
          bbox,
          keypoints,
        });
      }
    }
  }

  let dispatchGroupToCall:
    | ((options: {
        sample: Sample;
        current: Record<GroupType, number>;
      }) => GroupType)
    | undefined;

  if (customDispatchGroup) {
    dispatchGroupToCall = ({ sample }) => customDispatchGroup(sample);
  } else if (groupRatio) {
    dispatchGroupToCall = ({ current }) =>
      dispatchGroup({ current, target: groupRatio });
  }

  async function createDirsAndMetadata(
    groupTypes: (GroupType | "")[]
  ): Promise<"saved" | "no change" | undefined> {
    //Override groupType if export image dirs is a string
    if (typeof exportMetadataPaths === "string") {
      const jsonStr = JSON.stringify(jsonObjects[""]);
      await writeFile(exportMetadataPaths, jsonStr);
      return "saved";
    }
    for (const groupType of groupTypes) {
      const exportMetadataPath =
        typeof exportMetadataPaths === "string"
          ? exportMetadataPaths
          : groupType && isGroupKey(groupType)
          ? exportMetadataPaths[groupType]
          : undefined;
      if (!exportMetadataPath) {
        throw new Error(
          `Invalid import or export image directory for group type "${groupType}"`
        );
      }
      const jsonStr = JSON.stringify(jsonObjects[groupType]);
      await writeFile(exportMetadataPath, jsonStr);
    }
    return "saved";
  }

  await exportDatasetHelper<Sample>({
    generateSampleSequence,
    saveSample,
    dispatchGroupToCall,
    createDirsAndMetadata,
  });

  console.log(`Exported COCO dataset (task: ${task})`);
}
