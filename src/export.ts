import { join, dirname, basename, extname } from "path";
import { copyFile, readFile, writeFile, appendFile } from "fs/promises";
import { existsSync } from "fs";

import { cachedMkdir, ExportClassifyDatasetOptions } from "./fs";
import { group_types as groupTypes, GroupType } from "./group";
import {
  ExportDatasetOptions,
  BoxAnnotation,
  KeypointsAnnotation,
} from "./co-r";
import { dispatchGroup } from "./split";
import { toClassifyDataYamlString, toDataYamlString } from "./yaml";
import { toDetectLabelString, toPoseLabelString } from "./label";
import {
  toYoloBoundingBoxLabelString,
  toYoloBoundingBoxWithKeypointsLabelString,
} from "./label-r";

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
  dispatchGroup?: (options: {
    sample: Sample;
    current: Record<GroupType, number>;
  }) => GroupType;
  saveSample: (options: {
    sample: Sample;
    groupType: GroupType;
  }) => Promise<void>;
  createDirsAndMetadata?: (
    groupTypes: string[]
  ) => Promise<"saved" | "no change" | undefined>;
}) {
  const sampleIterator = options.generateSampleSequence();
  const { saveSample, dispatchGroup, createDirsAndMetadata } = options;

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
      exportYoloDataset(args);
      break;
    }
    case "coco": {
      //exportCocoDataset(args)
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

export async function exportYoloDataset(args: ExportDatasetOptions) {
  type Sample = {
    imageId: number;
    filename: string;
    categoryName: string;
    groupType: GroupType | "";
    annotation: BoxAnnotation | KeypointsAnnotation;
  };

  const { dataset, importDatasetPath, exportDatasetPath } = args;
  const { categories, images, metadata, task } = dataset;
  const importDatasetDir = Array.isArray(importDatasetPath)
    ? dirname(importDatasetPath[0])
    : dirname(importDatasetPath);
  const exportDatasetDir = Array.isArray(exportDatasetPath)
    ? exportDatasetPath[0]
    : exportDatasetPath;

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
      }
    }
  }

  async function saveSample(args: { sample: Sample; groupType: GroupType }) {
    const { sample, groupType: exportGroupType } = args;
    const {
      imageId,
      filename,
      categoryName,
      groupType: importGroupType,
    } = sample;
    if (task === "pose" || task === "detect") {
      const { annotation } = sample;
      const labelFilename = basename(filename, extname(filename)) + ".txt";

      const imagesSrcDir = join(importDatasetDir, importGroupType, "images");
      const imagesDestDir = join(exportDatasetDir, exportGroupType, "images");
      const labelsDestDir = join(exportDatasetDir, exportGroupType, "labels");

      await cachedMkdir(imagesDestDir);
      const imageSrcPath = join(imagesSrcDir, filename);
      const imageDestPath = join(imagesDestDir, filename);
      await copyFile(imageSrcPath, imageDestPath);

      await cachedMkdir(labelsDestDir);
      const labelDestPath = join(labelsDestDir, labelFilename);
      let labelStr = "";

      if (task === "detect") {
        labelStr = toYoloBoundingBoxLabelString({
          ...annotation,
          nClass: categories.size,
        });
      } else if (task === "pose") {
        const { keypoints } = annotation as KeypointsAnnotation;
        labelStr = toYoloBoundingBoxWithKeypointsLabelString({
          ...annotation,
          nClass: categories.size,
          nKeypoints: metadata.n_keypoints,
          hasVisibility: metadata.visibility !== "unannotated",
          keypoints,
        });
      } else throw new Error(`Invalid task type: receive ${task}`);

      if (!existsSync(labelDestPath)) {
        //if txt file does not exist, create txt file
        await writeFile(labelDestPath, labelStr);
      } else {
        //else add line to txt file
        await appendFile(labelDestPath, "\n" + labelStr);
      }
      return;
    }
    if (task === "classify") {
      const srcFile = join(
        importDatasetDir,
        importDatasetDir,
        categoryName,
        filename
      );
      const destFile = join(
        exportDatasetDir,
        exportGroupType,
        categoryName,
        filename
      );

      await cachedMkdir(dirname(destFile));
      await copyFile(srcFile, destFile);
      return;
    }
    throw new Error(`Unsupported task type: receive "${task}"`);
  }

  //   let dispatch_group: ((options: { sample: Sample; current: Record<GroupType, number> }) => GroupType) | undefined;

  //   if (custom_dispatch_group) {
  //     dispatch_group = ({ sample }) => custom_dispatch_group(sample);
  //   } else if (group_ratio) {
  //     dispatch_group = ({ current }) => dispatchGroup({ current, target: group_ratio });
  //   }
  async function createDirsAndMetadata(
    groupTypes: string[]
  ): Promise<"saved" | "no change" | undefined> {
    const categoryNames = Array.from(categories.values()).map(
      (category) => category.categoryName
    );
    if (groupTypes.includes("")) {
      metadata.train = undefined;
      metadata.test = undefined;
      metadata.val = undefined;
    } else {
      metadata.train = groupTypes.includes("train") ? "../train" : undefined;
      metadata.test = groupTypes.includes("test") ? "../test" : undefined;
      metadata.val = groupTypes.includes("val") ? "../val" : undefined;
    }
    if (task === "classify") {
      for (const groupType of groupTypes) {
        for (const categoryName of categoryNames) {
          const exportDirPath = join(exportDatasetDir, groupType, categoryName);
          await cachedMkdir(exportDirPath);
        }
      }
      const newContent = toClassifyDataYamlString({
        train_path: metadata.train,
        test_path: metadata.test,
        val_path: metadata.val,
        n_class: categories.size,
        class_names: categoryNames,
      });
      const yamlPath = join(exportDatasetDir, "data.yaml");
      if (existsSync(yamlPath)) {
        const oldContent = await readFile(yamlPath).toString();
        if (newContent === oldContent) return "no change";
      }
      await writeFile(yamlPath, newContent);
      return "saved";
    }
    if (task === "pose" || task === "detect") {
      for (const groupType of groupTypes) {
        const exportDirImagesPath = join(exportDatasetDir, groupType, "images");
        const exportDirLabelsPath = join(exportDatasetDir, groupType, "labels");
        await cachedMkdir(exportDirImagesPath);
        await cachedMkdir(exportDirLabelsPath);
      }
      let yamlMetadata;
      if (task === "pose") {
        yamlMetadata = {
          train_path: metadata.train,
          test_path: metadata.test,
          val_path: metadata.val,
          class_names: categoryNames,
          n_class: categories.size,
          keypoint_names: metadata.keypoint_names,
          n_keypoints: metadata.n_keypoints,
          visibility: metadata.visibility !== "unannotated",
          flip_idx: metadata.flip_idx,
        };
      } else {
        yamlMetadata = {
          train_path: metadata.train,
          test_path: metadata.test,
          val_path: metadata.val,
          class_names: categoryNames,
          n_class: categories.size,
        };
      }
      const newContent = toDataYamlString(task, yamlMetadata);
      const yamlPath = join(exportDatasetDir, "data.yaml");
      if (existsSync(yamlPath)) {
        const oldContent = await readFile(yamlPath).toString();
        if (newContent === oldContent) return "no change";
      }

      await writeFile(yamlPath, newContent);
      return "saved";
    }
    throw new Error(`Unsupported task provided: ${task}`);
  }

  await exportDatasetHelper<Sample>({
    generateSampleSequence,
    saveSample,
    // dispatchGroup,
    createDirsAndMetadata,
  });

  console.log(`Exported YOLO dataset (task: ${task})`);
}

// export async function exportCocoDataset(options: ExportDatasetOptions) {
//   // Coco dataset export implementation here...
//   // This should use types from co-r.ts similarly and save JSON files for annotations, images, categories, etc.

//   // Since COCO is more complex, the main difference is saving the COCO JSON files instead of txt labels.

//   // The skeleton code below illustrates structure:

//   const {
//     task,
//     bounding_box_groups,
//     metadata,
//     import_dataset_dir,
//     dataset_dir: export_dataset_dir,
//     group_ratio,
//     dispatch_group: custom_dispatch_group,
//   } = options;

//   const { class_names } = metadata;

//   type Sample = {
//     group_type: GroupType;
//     class_name: string;
//     filename: string;
//     box: BoxAnnotation | KeypointsAnnotation;
//   };

//   // You would build up COCO images, annotations, and categories JSON here from samples

//   // For brevity, I'll provide a simplified placeholder structure.

//   console.warn("exportCocoDataset not fully implemented; please implement as needed.");

//   // You can reuse exportDatasetHelper with customized save_sample and create_dirs_and_data_yaml accordingly.

//   // To fully implement: collect all samples, convert to COCO JSON format, write annotations JSON files, and copy images.

//   // This is just a stub to keep interface consistent.

//   await exportDatasetHelper<Sample>({
//     generate_sample_sequence: function* () {
//       for (const [group_type_str, image_label_dict] of Object.entries(bounding_box_groups)) {
//         const group_type = group_type_str as GroupType;
//         for (const [filename, boxes] of Object.entries(image_label_dict)) {
//           for (const box of boxes) {
//             const class_name = class_names ? class_names[box.class_idx] : `${box.class_idx}`;
//             yield { group_type, class_name, filename, box };
//           }
//         }
//       }
//     },
//     save_sample: async ({ sample, group_type }) => {
//       // You'd typically accumulate data for COCO JSON files here, or save images as needed.
//       // Left as TODO.
//     },
//     dispatch_group: custom_dispatch_group
//       ? ({ sample }) => custom_dispatch_group(sample)
//       : group_ratio
//       ? ({ current }) => dispatchGroup({ current, target: group_ratio })
//       : undefined,
//     create_dirs_and_data_yaml: async (group_types: GroupType[]) => {
//       // Create directories for images, write COCO json files here
//       // Left as TODO
//       return undefined;
//     },
//   });
// }
