import { existsSync } from "fs";
import path, { extname, basename, join, dirname } from "path";
import {
  ClassifyYamlOptions,
  DetectYamlOptions,
  PoseYamlOptions,
  parseClassifyDataYaml,
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
  MultiLabelBoundingBox,
  parseMultiLabelString,
  MultiLabelBoundingBoxWithKeypoints,
} from "./label";
import { getDirFilenamesSync } from "@beenotung/tslib/fs";
import { readdir, copyFile, writeFile, mkdir, readFile } from "fs/promises";
import { extract_lines } from "@beenotung/tslib/string";
import { drawLabel, dataURLToBase64 } from "./preview";
import { error, group } from "console";
import { group_types, GroupType } from "./group";
import { genDispatchGroupSequence } from "./split";
import { dirpathSymbol } from "@beenotung/tslib";

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

const image_extensions = [".jpg", ".jpeg", ".png", ".webp"];

function isImageFile(filename: string): boolean {
  return image_extensions.includes(extname(filename).toLowerCase());
}

export function getImagePaths(dir: string, group: string): string[] {
  return getDirFilenamesSync(join(dir, group, "images")).filter((file) => {
    if (!isImageFile(file)) {
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
    const oldContent = (await readFile(save_path)).toString();
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

let created_dirs = new Set<string>();
export async function cachedMkdir(dir: string) {
  if (created_dirs.has(dir)) return;
  await mkdir(dir, { recursive: true });
  created_dirs.add(dir);
}

// ==================== Dataset Options Types ====================
export type DatasetOptions = DetectDatasetOptions | PoseDatasetOptions;

export type MultiLabelDatasetOptions =
  | DetectMultiLabelDatasetOptions
  | PoseMultiLabelDatasetOptions;

type BaseDatasetOptions = {
  dataset_dir: string;
  group_ratio?: Record<GroupType, number>;
  dispatch_group?: (options: {
    group_type: GroupType | "";
    class_name: string;
    filename: string;
  }) => GroupType;
};

export type ClassifyDatasetOptions = BaseDatasetOptions & {
  task: "classify";
  metadata: ClassifyYamlOptions;
  classify_groups: ClassifyGroups;
};

export type ExportClassifyDatasetOptions = ClassifyDatasetOptions & {
  import_dataset_dir: string;
  /** default is do-not-rebalance */
  group_ratio?: Record<GroupType, number>;
  dispatch_group?: (options: {
    group_type: GroupType | "";
    class_name: string;
    filename: string;
  }) => GroupType;
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

export type DetectMultiLabelDatasetOptions = BaseDatasetOptions & {
  task: "detect";
  image_annotations_map: ImageMultiLabelDict<MultiLabelBoundingBox>;
};

export type PoseMultiLabelDatasetOptions = BaseDatasetOptions & {
  task: "pose";
  image_annotations_map: ImageMultiLabelDict<MultiLabelBoundingBoxWithKeypoints>;
};

export type BoundingBoxGroups<Box = BoundingBox> = {
  train: ImageLabelDict<Box>;
  test: ImageLabelDict<Box>;
  val: ImageLabelDict<Box>;
};

export type ImageLabelDict<Box = BoundingBox> = {
  [image_filename: string]: Box[];
};

export type ImageMultiLabelDict<
  Box = MultiLabelBoundingBox | MultiLabelBoundingBoxWithKeypoints
> = {
  [image_id: number]: {
    bounding_boxes: Box[];
    image_filename: string;
  };
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

  console.log(`Imported dataset (task type: ${task})`);

  return {
    metadata,
    bounding_box_groups,
  };
}

//coco
export async function importMultiLabelCocoDataset(options: {
  dataset_dir: string;
  task: "detect";
  json_filename?: string;
}): Promise<ImageMultiLabelDict>;
export async function importMultiLabelCocoDataset(options: {
  dataset_dir: string;
  task: "pose";
  json_filename?: string;
}): Promise<ImageMultiLabelDict>;
export async function importMultiLabelCocoDataset(options: {
  dataset_dir: string;
  task: "detect" | "pose";
  json_filename?: string;
}): Promise<ImageMultiLabelDict> {
  const { dataset_dir, task } = options;
  const json_filename = options.json_filename ?? "labels.json";
  // validateDatasetDir

  const path = join(dataset_dir, json_filename);
  const json_str = await readFile(path, "utf-8");
  const { categories, images, annotations } = JSON.parse(json_str);
  const image_annotations_map: ImageMultiLabelDict = Object.create(null);
  for (const image of images) {
    const image_id: number = image["id"];
    const image_filename: string = image["file_name"];
    image_annotations_map[image_id] = {
      image_filename,
      bounding_boxes: [],
    };
  }
  for (const annotation of annotations) {
    const box =
      task === "pose"
        ? parseMultiLabelString("pose", annotation)
        : parseMultiLabelString("detect", annotation);
    image_annotations_map[annotation.image_id].bounding_boxes.push(box);
  }
  return image_annotations_map;
}

export type ClassifyGroups = {
  [group_type: string]: ClassifyImagePaths;
};

export type ClassifyImagePaths = {
  // class name -> image filename
  [class_name: string]: string[];
};

async function validateClassifyYamlContent(
  dataset_dir: string,
  yaml_content: string
): Promise<ClassifyYamlOptions | undefined> {
  const metadata = yaml_content
    ? parseClassifyDataYaml(yaml_content)
    : undefined;
  if (metadata) {
    let train_path = metadata?.train_path;
    let val_path = metadata?.val_path;
    let test_path = metadata?.test_path;
    let n_class = metadata.n_class;
    let class_names = metadata.class_names;

    //validate yaml metadata
    //1. if class names exists and n_class !== class_names.length -> error
    if (class_names && n_class !== class_names.length) {
      throw new Error(
        `Mismatch between n_class and class_names.length: n_class=${n_class}, no. of class names = ${class_names.length}`
      );
    }
    if (class_names.length === 0 || n_class === 0) {
      throw new Error(
        `No. of classes cannot be 0: receive class_names=${class_names} and n_class=${n_class}`
      );
    }

    //2. validate class_names for all dirs
    //path dir should have the same no. of class name dirs as n_class
    let flag = false;
    for (const group_path of [train_path, test_path, val_path]) {
      for (const class_name of class_names) {
        if (group_path) {
          flag = true;
          const path = join(dataset_dir, group_path, class_name);
          if (!existsSync(path))
            throw new Error(`Path does not exist: receive ${path}`);
        }
      }
    }
    if (!flag) {
      for (const class_name of class_names) {
        const path = join(dataset_dir, class_name);
        if (!existsSync(path))
          throw new Error(`Path does not exist: receive ${path}`);
      }
    }
  }
  return metadata;
}

export async function importClassifyDataset(options: {
  dataset_dir: string;
  yaml_filename?: string;
}): Promise<ClassifyDatasetOptions> {
  const { dataset_dir, yaml_filename } = options;

  //yaml parsing
  const yaml_content = yaml_filename
    ? await readFile(join(dataset_dir, yaml_filename), "utf-8")
    : undefined;
  const parsed_metadata = yaml_content
    ? await validateClassifyYamlContent(dataset_dir, yaml_content)
    : undefined;

  const classify_groups: ClassifyGroups = {};
  const group_types = ["train", "test", "val"];

  // Discover group paths (train/test/val) or group type = empty string
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
  let train_path =
    parsed_metadata?.train_path ??
    (classify_groups["train"] ? "../train" : undefined);

  let val_path =
    parsed_metadata?.val_path ??
    (classify_groups["val"] ? "../val" : undefined);

  let test_path =
    parsed_metadata?.test_path ??
    (classify_groups["test"] ? "../test" : undefined);

  const class_names = Array.from(
    new Set(Object.values(classify_groups).flatMap(Object.keys))
  ).sort();

  const metadata = {
    train_path,
    val_path,
    test_path,
    n_class: parsed_metadata ? parsed_metadata.n_class : class_names.length,
    class_names: parsed_metadata ? parsed_metadata.class_names : class_names,
  };

  console.log(`Imported dataset (task type: classify)`);

  return {
    dataset_dir,
    task: "classify",
    metadata,
    classify_groups,
  };
}

export type ExportDatasetOptions = DatasetOptions & {
  import_dataset_dir: string;
};

export type ExportMultiLabelDatasetOptions = MultiLabelDatasetOptions & {
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

  const group_types = ["train", "test", "val"] satisfies Array<
    keyof BoundingBoxGroups
  >;

  for (const group_type of group_types) {
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
  console.log(`Exported dataset (task type: ${task})`);
}

// ==================== Export Classify Dataset ========================
async function mkdirAndUpdateMetadataPathsForClassify(
  options: ExportClassifyDatasetOptions
) {
  const {
    metadata,
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    group_ratio,
    dispatch_group,
    classify_groups,
  } = options;
  if (!group_ratio && !dispatch_group) {
    // default
    metadata.train_path = metadata.train_path
      ? metadata.train_path.replaceAll("\\", "/")
      : undefined;
    metadata.test_path = metadata.test_path
      ? metadata.test_path.replaceAll("\\", "/")
      : undefined;
    metadata.val_path = metadata.val_path
      ? metadata.val_path.replaceAll("\\", "/")
      : undefined;
    for (const group_type in classify_groups) {
      const group_path = join(export_dataset_dir, group_type);
      for (const class_name of metadata.class_names) {
        const class_path = join(group_path, class_name);
        await mkdir(class_path, { recursive: true });
      }
    }
  } else {
    for (const group_type of group_types) {
      const group_path = join(export_dataset_dir, group_type);
      if (!group_ratio || group_ratio[group_type] > 0) {
        metadata[`${group_type}_path`] = `../${group_type}`;
        for (const class_name of metadata.class_names) {
          const class_path = join(group_path, class_name);
          await mkdir(class_path, { recursive: true });
        }
      } else {
        metadata[`${group_type}_path`] = undefined;
      }
    }
  }
  return metadata;
}

async function exportRedistributedClassifyDataset(
  options: ExportClassifyDatasetOptions
) {
  const {
    metadata,
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    group_ratio,
    dispatch_group,
    classify_groups,
  } = options;
  if (group_ratio && !dispatch_group) {
    // class_name -> {group_type, filename}[]
    let class_name_to_images: Record<
      string,
      { group_type: string; filename: string }[]
    > = Object.create(null);

    // populate class_name_to_images
    for (let [group_type, class_name_to_filenames] of Object.entries(
      classify_groups
    )) {
      for (let [class_name, filenames] of Object.entries(
        class_name_to_filenames
      )) {
        class_name_to_images[class_name] ??= [];
        for (let filename of filenames) {
          class_name_to_images[class_name].push({ group_type, filename });
        }
      }
    }

    for (let [class_name, images] of Object.entries(class_name_to_images)) {
      let group_seq = genDispatchGroupSequence({
        target: group_ratio,
        total: images.length,
      });
      let image_idx = 0;
      for (let dest_group_type of group_seq) {
        let image = images[image_idx];
        image_idx++;
        let src_file = join(
          import_dataset_dir,
          image.group_type,
          class_name,
          image.filename
        );
        let dest_file = join(
          export_dataset_dir,
          dest_group_type,
          class_name,
          image.filename
        );
        await copyFile(src_file, dest_file);
      }
    }
  }
}

async function exportClassifyDatasetWithCustomDispatchGroups(
  options: ExportClassifyDatasetOptions
) {
  const {
    metadata,
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    group_ratio,
    dispatch_group,
    classify_groups,
  } = options;
  if (dispatch_group) {
    if (group_ratio) {
      console.warn(
        `option.group_ratio is ignored when option.dispatch_group is specified`
      );
    }
    for (let [group_type, class_name_to_filenames] of Object.entries(
      classify_groups
    )) {
      for (let [class_name, filenames] of Object.entries(
        class_name_to_filenames
      )) {
        for (let filename of filenames) {
          let dest_group_type = dispatch_group({
            group_type: group_type as GroupType | "",
            class_name,
            filename,
          });
          let src_file = join(
            import_dataset_dir,
            group_type,
            class_name,
            filename
          );
          let dest_file = join(
            export_dataset_dir,
            dest_group_type,
            class_name,
            filename
          );
          await copyFile(src_file, dest_file);
        }
      }
    }
  }
}

async function exportClassifyDatasetDefault(
  options: ExportClassifyDatasetOptions
) {
  const {
    metadata,
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    group_ratio,
    dispatch_group,
    classify_groups,
  } = options;
  for (const [group_type, class_map] of Object.entries(classify_groups)) {
    const group_path = group_type
      ? join(export_dataset_dir, group_type)
      : export_dataset_dir;

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
          export_dataset_dir,
          group_type,
          class_name,
          image_filename
        );
        await copyFile(import_image_path, export_image_path);
      }
    }
  }
}

export async function exportClassifyDataset(
  options: ExportClassifyDatasetOptions
) {
  let {
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    // metadata,
    classify_groups,
    group_ratio,
    dispatch_group,
  } = options;

  //mkdirs and update metadata paths
  const metadata = await mkdirAndUpdateMetadataPathsForClassify(options);

  if (group_ratio && !dispatch_group) {
    await exportRedistributedClassifyDataset(options);
  } else if (dispatch_group) {
    await exportClassifyDatasetWithCustomDispatchGroups(options);
  } else {
    await exportClassifyDatasetDefault(options);
  }
  saveYAMLFile(
    join(export_dataset_dir, "data.yaml"),
    toClassifyDataYamlString({ ...metadata })
  );
  console.log(`Exported dataset (task type: classify)`);
}

async function createExportDatasetDirs(export_dir_path: string): Promise<void> {
  const group_types = ["train", "test", "val"];
  const data_types = ["images", "labels"];

  for (const group_type of group_types) {
    for (const data_type of data_types) {
      const dir_path = join(export_dir_path, group_type, data_type);
      await cachedMkdir(dir_path);
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
