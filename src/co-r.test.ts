import { importDataset } from "./co-r";
import { exportCocoDataset } from "./export";
import { createPreviewImages } from "./preview";

async function test() {
  const task = "pose";
  const format = "coco";
  const group_ratio = { train: 7, test: 2, val: 1 };
  // const imageDirs = "coco-dataset/data";
  // const metadataPaths = "coco-dataset/internal.json";
  const imageDirs = {
    train: "coco-dataset-groups/train",
    val: "coco-dataset-groups/valid",
    test: "coco-dataset-groups/test",
  };
  const metadataPaths = {
    train: "coco-dataset-groups/train/_annotations.coco.json",
    val: "coco-dataset-groups/valid/_annotations.coco.json",
    test: "coco-dataset-groups/test/_annotations.coco.json",
  };
  // const exportImageDirs = "coco-dataset-groups2/data"
  // const exportMetadataPaths = "coco-dataset-groups2/label.json";
  const exportImageDirs = {
    train: "coco-dataset-groups2/train",
    val: "coco-dataset-groups2/valid",
    test: "coco-dataset-groups2/test",
  };
  const exportMetadataPaths = {
    train: "coco-dataset-groups2/train/_annotations.coco.json",
    val: "coco-dataset-groups2/valid/_annotations.coco.json",
    test: "coco-dataset-groups2/test/_annotations.coco.json",
  };
  const getCategoryId = (categoryName: string): number => {
    return categoryName.charCodeAt(0);
  };
  const getImageId = (imageFilename: string): number => {
    const match = imageFilename.match(/\d+/);
    return match ? parseInt(match[0], 10) : -1;
  };
  const dataset = await importDataset({
    task,
    format,
    metadataPaths,
    imageDirs,
    // getCategoryId,
    // getImageId,
  });

  await createPreviewImages({
    importDirs: imageDirs,
    images: dataset.images,
    categories: dataset.categories,
  });

  await exportCocoDataset({
    importMetadataPaths: metadataPaths,
    importImageDirs: imageDirs,
    exportImageDirs,
    exportMetadataPaths,
    dataset,
    format,
  });
  console.log("test complete");

  // await exportDataset({
  //   task,
  //   ...result,
  //   import_dataset_dir: import_dir,
  //   dataset_dir: export_dir,
  //   group_ratio,
  // })

  // // test if the exported dataset is complete (e.g. having both labels and images)
  // await importDataset({
  //   task,
  //   dataset_dir: export_dir,
  // })
  // debugger
  // // draw bounding box and keypoints in preview images
  // await createPreviewImages({
  //   task,
  //   dataset_dir: export_dir,
  //   ...result,
  // })
}
test();
