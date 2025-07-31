// import { readFileSync, existsSync, mkdirSync, writeFileSync } from "fs";
import { existsSync } from "fs";
import { extname, basename, join, dirname } from "path";
import {
  ClassifyYamlOptions,
  DetectYamlOptions,
  PoseYamlOptions,
  parseDataYaml,
  toClassifyDataYamlString,
  toDataYamlString,
} from "./yaml";
import {
  BoundingBox,
  BoundingBoxWithKeypoints,
  parseLabelString,
  ParseDetectLabelOptions,
  ParsePoseLabelOptions,
  toPoseLabelString,
  toDetectLabelString,
} from "./label";
import { getDirFilenamesSync } from "@beenotung/tslib/fs";
import { readdir, copyFile, writeFile, mkdir, readFile } from "fs/promises";
import { extract_lines } from "@beenotung/tslib/string";
import { drawLabel, dataURLToBase64 } from "./preview";
import { group } from "console";

export function validateDatasetDir(
  dataset_dir: string,
  yaml_filename: string
): void {
  const requiredPaths = [
    join(dataset_dir, yaml_filename),
    ...["train", "test", "val"].flatMap((group) =>
      ["images", "labels"].map((type) => join(dataset_dir, group, type))
    ),
  ];
  requiredPaths.forEach((path) => {
    if (!existsSync(path)) throw new Error(`${path} does not exist`);
  });
}

const image_extensions = [".jpg", ".jpeg", ".png"];

export function getImagePaths(dir: string, group: string): string[] {
  return getDirFilenamesSync(join(dir, group, "images")).filter((file) => {
    if (!image_extensions.includes(extname(file))) {
      throw new Error(`Error: Unsupported image type (Given ${extname(file)})`);
    }
    return true;
  });
}

export function getLabelPaths(
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

// ==================== Label Parsing ====================
export function parseLabelData<T extends "detect" | "pose">(
  task: T,
  options: T extends "pose"
    ? Omit<ParsePoseLabelOptions, "line">
    : Omit<ParseDetectLabelOptions, "line">,
  content: string
): T extends "pose" ? BoundingBoxWithKeypoints[] : BoundingBox[] {
  const lines = extract_lines(content);
  const result: (BoundingBox | BoundingBoxWithKeypoints)[] = [];

  for (const line of lines) {
    if (!line.trim()) continue;

    const parseOptions = { ...options, line };
    const box = parseLabelString(task, parseOptions);
    result.push(box);
  }

  return result as T extends "pose"
    ? BoundingBoxWithKeypoints[]
    : BoundingBox[];
}

// ==================== Metadata Handling ====================
export async function getMetaData(
  task: "detect" | "pose",
  options: {
    dir: string;
    yaml_filename: string;
  }
): Promise<DetectYamlOptions | PoseYamlOptions> {
  const path = join(options.dir, options.yaml_filename);
  const yaml_content = await readFile(path).toString();
  return parseDataYaml(task, yaml_content);
}

// ==================== Image/Label Validation ====================
export function getValidatedImageAndLabels(options: {
  image_dir: string;
  label_dir: string;
  image_paths: string[];
  label_paths: string[];
  missing_labels: "error" | "warn" | "ignore";
}) {
  const { image_dir, label_dir, image_paths, label_paths, missing_labels } =
    options;

  const validated_image_paths = image_paths.filter((image_path) => {
    const image_full_path = join(image_dir, image_path);
    const label_path = basename(image_path, extname(image_path)) + ".txt";
    const label_full_path = join(label_dir, label_path);

    if (!existsSync(label_full_path)) {
      handleMissingLabel(
        missing_labels,
        `Label file not found for image: ${image_full_path}`
      );
      return false;
    }
    return true;
  });

  const validated_label_paths = label_paths.filter((label_path) => {
    const label_full_path = join(label_dir, label_path);
    const label_filename = basename(label_path, extname(label_path));
    const hasMatchingImage = image_extensions.some((ext) =>
      validated_image_paths.includes(label_filename + ext)
    );

    if (!hasMatchingImage) {
      handleMissingLabel(
        missing_labels,
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

function handleMissingLabel(
  mode: "error" | "warn" | "ignore",
  message: string
): void {
  if (mode === "error") throw new Error(message);
  if (mode === "warn") console.warn(`Warning: ${message}`);
}

// ==================== Get Bounding Boxes ====================
export async function getBoundingBoxesDict<
  T extends DetectYamlOptions | PoseYamlOptions
>(
  task: "detect" | "pose",
  options: {
    dataset_dir: string;
    group_type: string;
    image_paths: string[];
    label_paths: string[];
    metadata: T;
    missing_labels?: "error" | "warn" | "ignore";
  }
): Promise<ImageLabelDict> {
  const {
    dataset_dir,
    group_type,
    metadata,
    missing_labels = "warn",
  } = options;

  const { validated_image_paths, validated_label_paths } =
    getValidatedImageAndLabels({
      ...options,
      missing_labels,
      image_dir: join(dataset_dir, group_type, "images"),
      label_dir: join(dataset_dir, group_type, "labels"),
    });

  const bounding_box_dict: ImageLabelDict = {};

  for (let i = 0; i < validated_image_paths.length; i++) {
    const image_full_path = join(
      dataset_dir,
      group_type,
      "images",
      validated_image_paths[i]
    );
    const label_full_path = join(
      dataset_dir,
      group_type,
      "labels",
      validated_label_paths[i]
    );
    const filename = basename(image_full_path);

    const boxes = await getBoundingBoxesOfOneLabel(task, {
      image_full_path,
      label_full_path,
      metadata,
    });
    bounding_box_dict[filename] = boxes;
  }

  return bounding_box_dict;
}

export async function getBoundingBoxesOfOneLabel<
  T extends DetectYamlOptions | PoseYamlOptions
>(
  task: "detect" | "pose",
  options: {
    image_full_path: string;
    label_full_path: string;
    metadata: T;
  }
): Promise<(BoundingBox | BoundingBoxWithKeypoints)[]> {
  const { label_full_path, metadata } = options;

  const content = await readFile(label_full_path, "utf-8");
  const label_data_boxes = parseLabelData(
    task,
    task === "pose"
      ? {
          n_class: metadata.n_class,
          is_visible: (metadata as PoseYamlOptions).visibility,
          n_keypoints: (metadata as PoseYamlOptions).n_keypoints,
        }
      : { n_class: metadata.n_class },
    content
  );

  return label_data_boxes;
}

// ==================== Generate Preview Image ====================
export async function createPreviewImages(options: DatasetOptions) {
  const { dataset_dir, task, metadata } = options;
  const group_types = ["train", "test", "val"] satisfies Array<
    keyof BoundingBoxGroups
  >;
  for (const group_type of group_types) {
    // const preview_file_dir = join(dataset_dir, group_type, "previews");
    // await mkdir(preview_file_dir, { recursive: true });
    const bounding_box_dict = options.bounding_box_groups[group_type];
    await createPreviewImagesForOneGroup(task, {
      dataset_dir,
      group_type,
      bounding_box_dict,
      metadata,
    });
  }
  console.log("Finish creating preview images");
}

export async function createPreviewImagesForOneGroup<
  T extends DetectYamlOptions | PoseYamlOptions
>(
  task: "detect" | "pose",
  options: {
    dataset_dir: string;
    group_type: string;
    bounding_box_dict: ImageLabelDict<BoundingBox | BoundingBoxWithKeypoints>;
    metadata: T;
  }
): Promise<void> {
  const { dataset_dir, group_type, bounding_box_dict, metadata } = options;
  const preview_dir_path = join(dataset_dir, group_type, "previews");
  await mkdir(preview_dir_path, { recursive: true });
  for (const image_path in bounding_box_dict) {
    const boxes = bounding_box_dict[image_path];
    const image_full_path = join(dataset_dir, group_type, "images", image_path);
    await createOnePreviewImage(task, {
      image_full_path,
      preview_dir_path,
      boxes,
      metadata,
    });
  }

  console.log(`Finish creating preview images for group type: ${group_type}`);
}

export async function createOnePreviewImage<
  T extends DetectYamlOptions | PoseYamlOptions
>(
  task: "detect" | "pose",
  options: {
    image_full_path: string; // where to get image
    preview_dir_path: string; // output dir path
    metadata: T;
    boxes: (BoundingBox | BoundingBoxWithKeypoints)[];
  }
): Promise<void> {
  const { image_full_path, preview_dir_path, boxes, metadata } = options;
  const canvas = await drawLabel(task, {
    image: image_full_path,
    metadata,
    boxes,
  });

  const image_name = basename(image_full_path);
  const preview_path = join(preview_dir_path, image_name);
  const base64 = dataURLToBase64(canvas.toDataURL());
  await writeFile(preview_path, base64, "base64");
}

// ==================== File Utilities ====================
export async function saveYAMLFile(
  save_path: string,
  content: string
): Promise<"no change" | "saved"> {
  const newContent = content.trim() + "\n";

  if (existsSync(save_path)) {
    const oldContent = await readFile(save_path).toString();
    if (newContent === oldContent) return "no change";
  }

  await writeFile(save_path, newContent);
  return "saved";
}

export async function saveJsonFile(
  save_file_path: string,
  data: any,
  options?: { verbose?: boolean }
) {
  let str = "{";
  for (let test of data) {
    str += JSON.stringify({ [test[0]]: test[1] }).slice(1, -1);
    str += ",\n";
  }
  str = str.slice(0, -2);
  str += "\n}";

  await writeFile(save_file_path, str);
  if (options?.verbose) {
    console.log("saved to file:", save_file_path);
  }
}

// ==================== Dataset Options Types ====================
export type DatasetOptions = DetectDatasetOptions | PoseDatasetOptions;

type BaseDatasetOptions = {
  dataset_dir: string;
};
export type ClassifyDatasetOptions = BaseDatasetOptions & {
  task: "classify";
  metadata: ClassifyYamlOptions;
  classify_groups: ClassifyGroups;
};

export type ExportClassifyDatasetOptions = ClassifyDatasetOptions & {
  import_dataset_dir: string;
};
export type DetectDatasetOptions = BaseDatasetOptions & {
  task: "detect";
  metadata: DetectYamlOptions;
  bounding_box_groups: BoundingBoxGroups<BoundingBox>;
};
export type PoseDatasetOptions = BaseDatasetOptions & {
  task: "pose";
  metadata: PoseYamlOptions;
  bounding_box_groups: BoundingBoxGroups<BoundingBoxWithKeypoints>;
};

export type BoundingBoxGroups<Box = BoundingBox> = {
  train: ImageLabelDict<Box>;
  test: ImageLabelDict<Box>;
  val: ImageLabelDict<Box>;
};

export type ImageLabelDict<Box = BoundingBox> = {
  [image_filename: string]: Box[];
};

// ==================== Dataset Import (From specified directory) ====================
// returns metadata & train/test/val bounding boxes dicts
export async function importDataset(options: {
  dataset_dir: string;
  task: "detect";
  yaml_filename?: string;
}): Promise<Omit<DetectDatasetOptions, "dataset_dir" | "task">>;
export async function importDataset(options: {
  dataset_dir: string;
  task: "pose";
  yaml_filename?: string;
}): Promise<Omit<PoseDatasetOptions, "dataset_dir" | "task">>;
export async function importDataset(options: {
  dataset_dir: string;
  task: "detect" | "pose";
  yaml_filename?: string;
}): Promise<Omit<DatasetOptions, "dataset_dir" | "task">> {
  const { dataset_dir, task } = options;
  const yaml_filename = options.yaml_filename ?? "data.yaml";
  validateDatasetDir(dataset_dir, yaml_filename);

  const path = join(dataset_dir, yaml_filename);
  const yaml_content = await readFile(path, "utf-8");
  const metadata = parseDataYaml(task, yaml_content);

  const bounding_box_dicts = [];

  for (const group_type of ["train", "test", "val"]) {
    const image_paths = getImagePaths(dataset_dir, group_type);
    const label_paths = getLabelPaths(dataset_dir, group_type, image_paths);
    const bounding_box_dict = await getBoundingBoxesDict(task, {
      dataset_dir,
      group_type,
      image_paths,
      label_paths,
      metadata,
    });
    bounding_box_dicts.push(bounding_box_dict);
  }

  const [
    bounding_box_dict_train,
    bounding_box_dict_test,
    bounding_box_dict_val,
  ] = bounding_box_dicts;

  const bounding_box_groups: BoundingBoxGroups = {
    train: bounding_box_dict_train,
    test: bounding_box_dict_test,
    val: bounding_box_dict_val,
  };

  console.log("Imported dataset");
  return {
    metadata,
    bounding_box_groups,
  };
}

export type ClassifyGroups = {
  [group_type: string]: ClassifyImagePaths;
};

export type ClassifyImagePaths = {
  [class_name: string]: string[];
};

function isImageFile(filename: string): boolean {
  return image_extensions.includes(extname(filename).toLowerCase());
}

export async function importClassifyDataset(options: {
  dataset_dir: string;
  yaml_filename?: string;
}): Promise<ClassifyDatasetOptions> {
  const { dataset_dir } = options;
  const classify_groups: ClassifyGroups = {};
  const group_types = ["train", "test", "val"];

  // Discover group paths (train/test/val) or fallback
  const dirents = await readdir(dataset_dir, { withFileTypes: true });
  let group_paths = dirents
    .filter(
      (dirent) => dirent.isDirectory() && group_types.includes(dirent.name)
    )
    .map((dirent) => join(dataset_dir, dirent.name));

  group_paths = group_paths.length === 0 ? [dataset_dir] : group_paths;

  for (const group_path of group_paths) {
    const group_type = group_path === dataset_dir ? "" : basename(group_path);
    const group_dir = join(dataset_dir, group_type);
    classify_groups[group_type] = {};

    const class_dirents = await readdir(group_dir, { withFileTypes: true });
    for (const class_dirent of class_dirents) {
      if (!class_dirent.isDirectory()) continue;

      const class_name = class_dirent.name;
      const class_path = join(group_dir, class_name);
      const image_filenames = (await readdir(class_path)).filter(isImageFile);

      classify_groups[group_type][class_name] = image_filenames;
    }
  }

  const all_class_names = Array.from(
    new Set(Object.values(classify_groups).flatMap(Object.keys))
  ).sort();

  const metadata = {
    n_class: all_class_names.length,
    class_names: all_class_names,
  };

  return {
    dataset_dir,
    task: "classify",
    metadata,
    classify_groups,
  };
}

//old code
// export async function importClassifyDataset(options: {
//   dataset_dir: string;
//   yaml_filename?: string;
// }): Promise<ClassifyDatasetOptions> {
//   const { dataset_dir } = options;
//   const yaml_filename = options.yaml_filename ?? undefined;
//   //if yaml exists, parse metadata
//   // else count metadata in loop below
//   const classify_groups: ClassifyGroups = {};

//   const group_paths = (await readdir(dataset_dir, { withFileTypes: true }))
//     .filter((dirent) => {
//       return (
//         dirent.isDirectory() && ["train", "test", "val"].includes(dirent.name)
//       );
//     })
//     .map((dirent) => join(dataset_dir, dirent.name));
//   //step 1: check if dir contains train/test/val
//   //step 2a: yes -> loop classify groups
//   //step 2b: no -> step 3 or classify_groups['']
//   //step 3: for each class name, create classify image paths & put paths in it
//   //step 4: put classify image paths into classify_groups

//   //if empty, add placeholder group type
//   const placeholder_group_type = join(dataset_dir, "?");
//   if (group_paths.length === 0) {
//     group_paths.push(placeholder_group_type);
//   }

//   for (const group_path of group_paths) {
//     const group_type =
//       group_path === placeholder_group_type ? "" : basename(group_path);
//     classify_groups[group_type] = {};

//     // assume class exists in all group types
//     const class_dirents = await readdir(join(dataset_dir, group_type), {
//       withFileTypes: true,
//     });
//     for (const class_dirent of class_dirents) {
//       if (class_dirent.isDirectory()) {
//         const class_name = class_dirent.name;
//         const class_path = join(dataset_dir, group_type, class_name);

//         // read all image in the class name dir
//         classify_groups[group_type][class_name] = (await readdir(class_path))
//           .filter(isImageFile)
//           .map((fname) => fname);
//       }
//     }
//   }
//   const allClassNames = Array.from(
//     new Set(
//       Object.values(classify_groups).flatMap((group) => Object.keys(group))
//     )
//   ).sort();

//   const metadata = {
//     n_class: allClassNames.length,
//     class_names: allClassNames,
//   };

//   return {
//     dataset_dir,
//     task: "classify",
//     metadata,
//     classify_groups,
//   };
// }

// ==================== Dataset Export (To specified directory) ====================
export type ExportDatasetOptions = DatasetOptions & {
  import_dataset_dir: string;
};
export async function exportDataset(
  options: ExportDatasetOptions
): Promise<void> {
  const { task, metadata } = options;
  const export_dataset_dir = options.dataset_dir;
  const import_dataset_dir = options.import_dataset_dir;
  createExportDatasetDirs(export_dataset_dir);

  const yaml_path = join(export_dataset_dir, "data.yaml");
  const yaml_str = toDataYamlString(task, metadata);
  await writeFile(yaml_path, yaml_str);

  const groups = ["train", "test", "val"] satisfies Array<
    keyof BoundingBoxGroups
  >;

  for (const group_type of groups) {
    const task = options.task;

    switch (options.task) {
      case "detect": {
        const { metadata } = options;
        await saveLabelFiles({
          export_dataset_dir,
          import_dataset_dir,
          group_type,
          dict: options.bounding_box_groups[group_type],
          toLabelString: (box) => toDetectLabelString({ ...box, ...metadata }),
        });
        break;
      }
      case "pose": {
        const { metadata } = options;
        await saveLabelFiles({
          export_dataset_dir,
          import_dataset_dir,
          group_type,
          dict: options.bounding_box_groups[group_type],
          toLabelString: (box) => toPoseLabelString({ ...box, ...metadata }),
        });
        break;
      }
      default: {
        options satisfies never;
        throw new Error(`unknown task type "${task}"`);
      }
    }
  }

  console.log("Exported dataset");
}

export async function exportClassifyDataset(
  options: ExportClassifyDatasetOptions
) {
  const { import_dataset_dir, dataset_dir, classify_groups, metadata } =
    options;

  for (const [group_type, class_map] of Object.entries(classify_groups)) {
    const group_path = group_type ? join(dataset_dir, group_type) : dataset_dir;

    for (const class_name of metadata.class_names) {
      const class_path = join(group_path, class_name);
      await mkdir(class_path, { recursive: true });

      const image_filenames = class_map[class_name] || [];
      for (const image_filename of image_filenames) {
        const import_image_path = join(
          import_dataset_dir,
          group_type,
          class_name,
          image_filename
        );
        const export_image_path = join(
          dataset_dir,
          group_type,
          class_name,
          image_filename
        );
        await copyFile(import_image_path, export_image_path);
      }
    }
  }

  saveYAMLFile(
    join(dataset_dir, "data.yaml"),
    toClassifyDataYamlString({ ...metadata })
  );
}

async function createExportDatasetDirs(export_dir_path: string): Promise<void> {
  const group_types = ["train", "test", "val"];
  const data_types = ["images", "labels"];

  for (const group_type of group_types) {
    for (const data_type of data_types) {
      const dir_path = join(export_dir_path, group_type, data_type);
      if (!existsSync(dir_path)) {
        await mkdir(dir_path, { recursive: true });
      }
    }
  }
}

export function toLabelFilename(image_filename: string) {
  const image_extname = extname(image_filename);
  return basename(image_filename, image_extname) + ".txt";
}

export async function saveLabelFile(file: string, lines: string[]) {
  const content = lines.filter((line) => line.length > 0).join("\n") + "\n";
  await writeFile(file, content);
}

async function saveLabelFiles<Box>(options: {
  import_dataset_dir: string;
  export_dataset_dir: string;
  group_type: string;
  dict: ImageLabelDict<Box>;
  toLabelString: (box: Box) => string;
}) {
  const {
    export_dataset_dir,
    import_dataset_dir,
    group_type,
    dict,
    toLabelString,
  } = options;

  const import_images_dir = join(import_dataset_dir, group_type, "images");
  const export_images_dir = join(export_dataset_dir, group_type, "images");

  for (const image_filename in dict) {
    const boxes = dict[image_filename];
    const lines = boxes.map(toLabelString);
    const label_file = join(
      export_dataset_dir,
      group_type,
      "labels",
      toLabelFilename(image_filename)
    );
    await saveLabelFile(label_file, lines);

    const import_image_file = join(import_images_dir, image_filename);
    const export_image_file = join(export_images_dir, image_filename);
    await copyFile(import_image_file, export_image_file);
  }
}
