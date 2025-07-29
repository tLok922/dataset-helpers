import { exportDataset, importDataset } from './fs'

import { createPreviewImages } from './fs'

async function test() {
  const task = 'pose'
  // const task = 'detect'

  const import_dir = 'res/datasets/tongue'
  const export_dir = 'res/datasets/tongue-export'

  const result = await importDataset({
    task,
    dir: import_dir,
  })

  await exportDataset({
    task,
    dir: export_dir,
    ...result,
  })

  await createPreviewImages({
    task,
    dir: export_dir,
    ...result,
  })
}
test()
