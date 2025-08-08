import {
  createPreviewImages,
  exportDataset,
  importDataset,
  importClassifyDataset,
  exportClassifyDataset,
} from "./fs";

async function test() {
  const task = "pose";
  const group_ratio = {'train':7,'test':2,'val':1}

  const import_dir = "classify-dataset";
  const export_dir = "classify-dataset-2";

  const import_dir_2 = "classify-dataset-2";
  const export_dir_2 = "classify-dataset-3";
  

  /**********************************************/
  /* copy dataset from import_dir to export_dir */
  /**********************************************/
  const result = await importClassifyDataset({ dataset_dir: import_dir });
  await exportClassifyDataset({
    ...result,
    import_dataset_dir: import_dir,
    dataset_dir: export_dir,
    // group_ratio,
    // TODO: add train/test/val split
    // dispatch_group(options) {
    //     // if(options.filename.includes("user-1-")){
    //     //   return 'train'
    //     // }
    //     return Math.random()<1/3 ? 'train':'test'
    // },
  });

  const result2 = await importClassifyDataset({ dataset_dir: import_dir_2 });
  await exportClassifyDataset({
    ...result2,
    import_dataset_dir: import_dir_2,
    dataset_dir: export_dir_2,
    group_ratio,
    // TODO: add train/test/val split
    // dispatch_group(options) {
    //     // if(options.filename.includes("user-1-")){
    //     //   return 'train'
    //     // }
    //     return Math.random()<1/3 ? 'train':'test'
    // },
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
