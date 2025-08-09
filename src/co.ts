let images = [
  {
    id: 1,
    filename: "cat.jpeg",
    annotations: [
      {
        id: 0,
        boxes: {
          x: 0,
          y: 0,
          width: 1,
          height: 1,
          category_id: 0,
          keypoints: [
            {
              x: 0,
              y: 0,
              // 0: unannotated
              // 1: not visible
              // 2: visible
              visibility: "visible",
            },
          ],
        },
      },
    ],
  },
];

// (yolo detect, pose)
// train/images/apple.jpg
// images/apple.jpg
// data.yaml
//
// (yolo/mnist classify)
// train/apple/1.jpg
// apple/1.jpg
// data.yaml (optional)
//
// (coco)
// data/1.jpg
// labels.json
// labels_train.json
//
// (pascal voc)
// (https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
// (https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector)
// data/1.jpg
// [in folder name of xml]/1.jpg
// Annotations/1.xml
// annotations/1.xml

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
  import_dataset_path: string | string[];
};

type ExportDatasetOptions = {
  import_dataset_path: string | string[];
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
  export_dataset_path: string | string[];
  format: "yolo" | "coco" | "pascal_voc";
  // ratio
};

function importDataset(options: { task: "classify" }): ClassifyDataset;
function importDataset(options: { task: "detect" }): DetectDataset;
function importDataset(options: { task: "pose" }): PoseDataset;
function importDataset(options: {
  task: "classify" | "detect" | "pose";
}): UnionDataset {
  return "" as any;
}

function importClassifyDataset(): ClassifyDataset {
  return "" as any;
}
function importDetectDataset(): DetectDataset {
  return "" as any;
}
function importPoseDataset(): PoseDataset {
  return "" as any;
}

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
  class_name: string;
};
type ImageWithLabel = {
  id: number;
  filename: string;
  category_id: number;
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
  category_id: number;
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
