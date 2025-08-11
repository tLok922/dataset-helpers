import { join, dirname, basename, extname } from "path";
import { copyFile, readFile, writeFile, appendFile } from "fs/promises";
import { existsSync } from "fs";

import {
  cachedMkdir,
  ExportClassifyDatasetOptions,
  ExportDatasetOptions,
} from "./fs";
import {GroupType} from "./group"
import { Category, ImageWithLabel, ImageWithBox, BoxAnnotation, KeypointsAnnotation } from "./co-r";
import { dispatchGroup } from "./split";
import {
  toClassifyDataYamlString,
  toDataYamlString,
} from "./yaml";
import {
  toDetectLabelString,
  toPoseLabelString,
} from "./label";

export async function exportDatasetHelper<Sample extends { group_type: string; class_name: string; filename: string }>(options: {
  generate_sample_sequence(): Iterable<Sample>;
  dispatch_group?: (options: {
    sample: Sample;
    current: Record<GroupType, number>;
  }) => GroupType;
  save_sample: (options: { sample: Sample; group_type: GroupType }) => Promise<void>;
  create_dirs_and_data_yaml?: (group_types: string[]) => Promise<"saved" | "no change" | undefined>;
}) {
  const sample_iterator = options.generate_sample_sequence();
  const { save_sample, dispatch_group, create_dirs_and_data_yaml } = options;

  const group_types: GroupType[] = [];
  const current_ratio_by_class: Record<string, Record<GroupType, number>> = Object.create(null);

  for (const sample of sample_iterator) {
    const class_name = sample.class_name;
    let group_type: GroupType;

    if (dispatch_group) {
      current_ratio_by_class[class_name] ??= { train: 0, val: 0, test: 0 };
      const current = current_ratio_by_class[class_name];
      group_type = dispatch_group({ sample, current });
      current[group_type]++;
    } else {
      group_type = sample.group_type as GroupType;
    }

    if (!group_types.includes(group_type)) {
      group_types.push(group_type);
    }

    await save_sample({ sample, group_type });
  }

  if (create_dirs_and_data_yaml) {
    await create_dirs_and_data_yaml(group_types);
  }
}

export async function exportYoloDataset(options: ExportDatasetOptions) {
  type Sample = {
    group_type: GroupType;
    class_name: string;
    filename: string;
    box: BoxAnnotation | KeypointsAnnotation;
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

  function* generate_sample_sequence(): Iterable<Sample> {
  }

  async function save_sample({ sample, group_type }: { sample: Sample; group_type: GroupType }) {
    
  }

  let dispatch_group: ((options: { sample: Sample; current: Record<GroupType, number> }) => GroupType) | undefined;

  if (custom_dispatch_group) {
    dispatch_group = ({ sample }) => custom_dispatch_group(sample);
  } else if (group_ratio) {
    dispatch_group = ({ current }) => dispatchGroup({ current, target: group_ratio });
  }

  async function create_dirs_and_data_yaml(group_types: string[]):Promise<"saved" | "no change" | undefined> {
    return 'saved'
  }

  await exportDatasetHelper<Sample>({
    generate_sample_sequence,
    save_sample,
    dispatch_group,
    create_dirs_and_data_yaml,
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
