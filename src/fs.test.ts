import { writeFileSync } from 'fs'
import { exportDataset, importDataset } from './fs'

import { createPreviewImages } from './fs'

async function test() {
  const task = 'pose'
  // const task = 'detect'

  const import_dir = 'res/datasets/tongue'
  const export_dir = 'res/datasets/tongue-export'

  /**********************************************/
  /* copy dataset from import_dir to export_dir */
  /**********************************************/

  const result = await importDataset({
    task,
    dataset_dir: import_dir,
  })
  writeFileSync('res/result.json', JSON.stringify(result, null, 2))

  await exportDataset({
    task,
    ...result,
    import_dataset_dir: import_dir,
    dataset_dir: export_dir,
  })

  // test if the exported dataset is complete (e.g. having both labels and images)
  await importDataset({
    task,
    dataset_dir: export_dir,
  })

  // draw bounding box and keypoints in preview images
  await createPreviewImages({
    task,
    dataset_dir: export_dir,
    ...result,
  })
}
test()
