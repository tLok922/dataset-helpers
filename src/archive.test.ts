import { Archive } from './archive'

async function main() {
  let archive = new Archive()
  let promise = archive.pipeToFile('test.zip')
  archive.addFile({ src_file: 'package.json', zip_file: 'test/pkg.json' })
  archive.addFile({
    content: 'string content',
    zip_file: 'test/file-string.txt',
  })
  archive.addFile({
    content: Buffer.from('buffer content'),
    zip_file: 'test/file-buffer.txt',
  })
  await promise
}

main().catch(e => console.error(e))
