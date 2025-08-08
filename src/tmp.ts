import { join, dirname, basename, extname } from "path";
import { copyFile, mkdir, readFile, writeFile, appendFile } from "fs/promises";
import { existsSync, write } from "fs";

import {
  cachedMkdir,
  ExportClassifyDatasetOptions,
  ExportDatasetOptions,
  ExportMultiLabelDatasetOptions,
  ImageMultiLabelDict,
  saveYAMLFile,
} from "./fs";
import { GroupType } from "./group";
import { dispatchGroup } from "./split";
import { toClassifyDataYamlString, toDataYamlString } from "./yaml";
import {
  BoundingBox,
  BoundingBoxWithKeypoints,
  MultiLabelBoundingBoxWithKeypoints,
  toDetectLabelString,
  toLabelString,
  toPoseLabelString,
} from "./label";
import { Image } from "canvas";

export type BaseSample = {
  group_type: string;
  class_name: string;
  filename: string;
};

export async function exportDatasetHelper<Sample extends BaseSample>(options: {
  generate_sample_sequence(): Iterable<Sample>;
  dispatch_group?: (options: {
    sample: Sample;
    current: Record<GroupType, number>;
  }) => GroupType;
  save_sample: (options: {
    sample: Sample;
    group_type: GroupType;
  }) => Promise<void>;
  create_dirs_and_data_yaml?: (
    group_types: string[]
  ) => Promise<"saved" | "no change" | undefined>;
}) {
  const sample_iterator = options.generate_sample_sequence();
  const { save_sample, dispatch_group, create_dirs_and_data_yaml } = options;
  const group_types: string[] = [];
  // const class_names: string[] = [];
  const current_ratio_by_class: Record<
    string,
    Record<GroupType, number>
  > = Object.create(null);

  for (const sample of sample_iterator) {
    const class_name = sample.class_name;

    let group_type;
    if (dispatch_group) {
      current_ratio_by_class[class_name] ??= {
        train: 0,
        val: 0,
        test: 0,
      };
      const current = current_ratio_by_class[class_name];
      group_type = dispatch_group({ sample, current });
      current[group_type]++;
    } else {
      group_type = sample.group_type as GroupType;
    }

    if (!group_types.includes(group_type)) {
      group_types.push(group_type);
    }

    // if (!class_names.includes(class_name)) {
    //   class_names.push(class_name);
    // }

    await save_sample({ sample, group_type });
  }

  if (create_dirs_and_data_yaml) {
    create_dirs_and_data_yaml(group_types);
  }
}

export async function exportDataset(options: ExportDatasetOptions) {
  type Sample = {
    group_type: GroupType;
    class_name: string;
    filename: string;
    box: BoundingBox | BoundingBoxWithKeypoints;
  };

  const {
    task,
    bounding_box_groups,
    metadata,
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    group_ratio,
    dispatch_group: custom_dispatch_group,
  } = options;
  const { class_names } = metadata;

  function* generate_sample_sequence() {
    for (const [group_type_str, image_label_dict] of Object.entries(
      bounding_box_groups
    )) {
      const group_type = group_type_str as GroupType;
      for (const [filename, boxes] of Object.entries(image_label_dict)) {
        for (const box of boxes) {
          const class_name = class_names
            ? class_names[box.class_idx]
            : `${box.class_idx}`;
          yield { group_type, class_name, filename, box };
        }
      }
    }
  }

  async function save_sample(args: { sample: Sample; group_type: GroupType }) {
    const { sample, group_type } = args;
    const { class_name, filename: image_filename } = sample;
    const label_filename =
      basename(image_filename, extname(image_filename)) + ".txt";

    const images_src_dir = join(
      import_dataset_dir,
      sample.group_type,
      "images"
    );
    const images_dest_dir = join(export_dataset_dir, group_type, "images");
    const labels_dest_dir = join(export_dataset_dir, group_type, "labels");

    await cachedMkdir(images_dest_dir);
    const image_src_path = join(images_src_dir, image_filename);
    const image_dest_path = join(images_dest_dir, image_filename);
    await copyFile(image_src_path, image_dest_path);

    await cachedMkdir(labels_dest_dir);
    const label_dest_path = join(labels_dest_dir, label_filename);
    let label_str = "";
    if (task === "detect") {
      label_str = toDetectLabelString({ ...sample.box, ...metadata });
    } else if (task === "pose") {
      const box = sample.box as BoundingBoxWithKeypoints;
      label_str = toPoseLabelString({ ...box, ...metadata });
    } else throw new Error(`Invalid task type: receive ${task}`);

    if (!existsSync(label_dest_path)) {
      //if txt file does not exist, create txt file
      await writeFile(label_dest_path, label_str);
    } else {
      //else add line to txt file
      await appendFile(label_dest_path, "\n" + label_str);
    }
  }

  let dispatch_group:
    | ((options: {
        sample: Sample;
        current: Record<GroupType, number>;
      }) => GroupType)
    | undefined;

  if (custom_dispatch_group) {
    dispatch_group = ({ sample }) => custom_dispatch_group(sample);
  } else if (group_ratio) {
    dispatch_group = ({ current }) =>
      dispatchGroup({ current, target: group_ratio });
  } else dispatch_group = undefined;

  async function create_dirs_and_data_yaml(group_types: string[]) {
    const { task, metadata } = options;
    const { class_names } = metadata;
    // to create empty dirs for empty group types/classes and create data.yaml

    //update paths for data.yaml
    options.metadata.train_path = "../train";
    options.metadata.test_path = "../test";
    options.metadata.val_path = "../val";

    //create dirs
    for (const group_type of group_types) {
      const export_dir_images_path = join(
        export_dataset_dir,
        group_type,
        "images"
      );
      const export_dir_labels_path = join(
        export_dataset_dir,
        group_type,
        "labels"
      );
      await cachedMkdir(export_dir_images_path);
      await cachedMkdir(export_dir_labels_path);
    }

    const new_content = toDataYamlString(task, metadata);
    const yaml_path = join(export_dataset_dir, "data.yaml");
    if (existsSync(yaml_path)) {
      const old_content = await readFile(yaml_path).toString();
      if (new_content === old_content) return "no change";
    }

    await writeFile(yaml_path, new_content);
    return "saved";
  }

  await exportDatasetHelper<Sample>({
    generate_sample_sequence,
    save_sample,
    dispatch_group, //to be implemented
    create_dirs_and_data_yaml,
  });

  console.log(`Exported dataset (task type: ${task})`);
}

export async function exportClassifyDataset(
  options: ExportClassifyDatasetOptions
) {
  const {
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    classify_groups,
    group_ratio,
    dispatch_group: custom_dispatch_group,
  } = options;

  type Sample = {
    group_type: GroupType | "";
    class_name: string;
    filename: string;
  };

  function* generate_sample_sequence(): Iterable<Sample> {
    for (const [group_type_str, class_map] of Object.entries(classify_groups)) {
      const group_type = group_type_str as GroupType | "";

      for (const [class_name, filenames] of Object.entries(class_map)) {
        for (const filename of filenames) {
          yield { group_type, class_name, filename };
        }
      }
    }
  }

  async function save_sample({
    sample,
    group_type,
  }: {
    sample: Sample;
    group_type: GroupType;
  }) {
    const { class_name, filename } = sample;

    const src_file = join(
      import_dataset_dir,
      sample.group_type,
      class_name,
      filename
    );
    const dest_file = join(
      export_dataset_dir,
      group_type,
      class_name,
      filename
    );

    await cachedMkdir(dirname(dest_file));
    await copyFile(src_file, dest_file);
  }

  let dispatch_group:
    | ((options: {
        sample: Sample;
        current: Record<GroupType, number>;
      }) => GroupType)
    | undefined;

  if (custom_dispatch_group) {
    dispatch_group = ({ sample }) => custom_dispatch_group(sample);
  } else if (group_ratio) {
    dispatch_group = ({ current }) =>
      dispatchGroup({ current, target: group_ratio });
  } else dispatch_group = undefined;

  async function create_dirs_and_data_yaml(group_types: string[]) {
    const class_names = options.metadata.class_names;
    // to create empty dirs for empty group types/classes and create data.yaml

    //update paths for data.yaml
    if (group_types.includes("")) {
      options.metadata.train_path = undefined;
      options.metadata.test_path = undefined;
      options.metadata.val_path = undefined;
    } else {
      options.metadata.train_path = group_types.includes("train")
        ? "../train"
        : undefined;
      options.metadata.test_path = group_types.includes("test")
        ? "../test"
        : undefined;
      options.metadata.val_path = group_types.includes("val")
        ? "../val"
        : undefined;
    }

    //create dirs
    for (const group_type of group_types) {
      for (const class_name of class_names) {
        const export_dir_path = join(
          export_dataset_dir,
          group_type,
          class_name
        );
        await cachedMkdir(export_dir_path);
      }
    }

    const new_content = toClassifyDataYamlString({ ...options.metadata });
    const yaml_path = join(export_dataset_dir, "data.yaml");
    if (existsSync(yaml_path)) {
      const old_content = await readFile(yaml_path).toString();
      if (new_content === old_content) return "no change";
    }

    await writeFile(yaml_path, new_content);
    return "saved";
  }

  await exportDatasetHelper<Sample>({
    generate_sample_sequence,
    save_sample,
    dispatch_group,
    create_dirs_and_data_yaml,
  });
  console.log(`Exported dataset (task type: classify)`);
}

export async function exportMultiLabelDataset(
  options: ExportMultiLabelDatasetOptions
) {
  type Sample = {
    group_type: "";
    class_name: string;
    filename: string;
    box: BoundingBox | BoundingBoxWithKeypoints;
  };

  const {
    task,
    import_dataset_dir,
    dataset_dir: export_dataset_dir,
    group_ratio,
    dispatch_group: custom_dispatch_group,
    //class_names
  } = options;

  const image_annotations_map =
    task === "pose"
      ? (options.image_annotations_map as ImageMultiLabelDict<MultiLabelBoundingBoxWithKeypoints>)
      : (options.image_annotations_map as ImageMultiLabelDict<BoundingBox>);

  function* generate_sample_sequence() {
    // for each image - annotations pair
    //    for each bounding box
    //      yield 

    // for detect, return 5 items
    // for pose, return 6 items
    for (const [image_id_str, filename_and_boxes] of Object.entries(
      image_annotations_map
    )) {
      const image_id: number = +image_id_str
      const { image_filename, bounding_boxes } = filename_and_boxes;
      for (const bounding_box of bounding_boxes){
        const category_id: number = +bounding_box.class_idx
        yield {image_id, image_filename, bounding_boxes, category_id}
      }
    }
  }

  async function save_sample(args: { sample: Sample; group_type: GroupType }) {
    const { sample, group_type } = args;
    const { class_name, filename: image_filename } = sample;
    const label_filename =
      basename(image_filename, extname(image_filename)) + ".txt";

    const images_src_dir = join(
      import_dataset_dir,
      sample.group_type,
      "images"
    );
    const images_dest_dir = join(export_dataset_dir, group_type, "images");
    const labels_dest_dir = join(export_dataset_dir, group_type, "labels");

    await cachedMkdir(images_dest_dir);
    const image_src_path = join(images_src_dir, image_filename);
    const image_dest_path = join(images_dest_dir, image_filename);
    await copyFile(image_src_path, image_dest_path);

    await cachedMkdir(labels_dest_dir);
    const label_dest_path = join(labels_dest_dir, label_filename);
    let label_str = "";
    if (task === "detect") {
      label_str = toDetectLabelString({ ...sample.box, ...metadata });
    } else if (task === "pose") {
      const box = sample.box as BoundingBoxWithKeypoints;
      label_str = toPoseLabelString({ ...box, ...metadata });
    } else throw new Error(`Invalid task type: receive ${task}`);

    if (!existsSync(label_dest_path)) {
      //if txt file does not exist, create txt file
      await writeFile(label_dest_path, label_str);
    } else {
      //else add line to txt file
      await appendFile(label_dest_path, "\n" + label_str);
    }
  }

  let dispatch_group:
    | ((options: {
        sample: Sample;
        current: Record<GroupType, number>;
      }) => GroupType)
    | undefined;

  if (custom_dispatch_group) {
    dispatch_group = ({ sample }) => custom_dispatch_group(sample);
  } else if (group_ratio) {
    dispatch_group = ({ current }) =>
      dispatchGroup({ current, target: group_ratio });
  } else dispatch_group = undefined;

  async function create_dirs_and_data_yaml(group_types: string[]) {
    const { task, metadata } = options;
    const { class_names } = metadata;
    // to create empty dirs for empty group types/classes and create data.yaml

    //update paths for data.yaml
    options.metadata.train_path = "../train";
    options.metadata.test_path = "../test";
    options.metadata.val_path = "../val";

    //create dirs
    for (const group_type of group_types) {
      const export_dir_images_path = join(
        export_dataset_dir,
        group_type,
        "images"
      );
      const export_dir_labels_path = join(
        export_dataset_dir,
        group_type,
        "labels"
      );
      await cachedMkdir(export_dir_images_path);
      await cachedMkdir(export_dir_labels_path);
    }

    const new_content = toDataYamlString(task, metadata);
    const yaml_path = join(export_dataset_dir, "data.yaml");
    if (existsSync(yaml_path)) {
      const old_content = await readFile(yaml_path).toString();
      if (new_content === old_content) return "no change";
    }

    await writeFile(yaml_path, new_content);
    return "saved";
  }

  await exportDatasetHelper<Sample>({
    generate_sample_sequence,
    save_sample,
    dispatch_group, //to be implemented
    create_dirs_and_data_yaml,
  });

  console.log(`Exported dataset (task type: ${task})`);
}
