import {
  createPreviewImages,
  exportDataset,
  importDataset,
  importClassifyDataset,
  exportClassifyDataset,
} from "./fs";

async function test() {
  const task = "pose";
  // const task = 'detect'

  const import_dir = "cat-pose-dataset_kpt6";
  const export_dir = "dataset2";

  /**********************************************/
  /* copy dataset from import_dir to export_dir */
  /**********************************************/
  const path = "classify-dataset";
  const result = await importClassifyDataset({ dataset_dir: path });
  await exportClassifyDataset({
    ...result,
    dataset_dir: export_dir,
    import_dataset_dir: path,
  });
  // const result = await importDataset({
  //   task,
  //   dataset_dir: import_dir,
  // })

  // await exportDataset({
  //   task,
  //   ...result,
  //   import_dataset_dir: import_dir,
  //   dataset_dir: export_dir,
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
