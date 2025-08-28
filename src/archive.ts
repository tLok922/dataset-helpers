import { ZipFile } from 'yazl'
import { createWriteStream } from 'fs'

/**
 * @example
 * ```ts
 * let archive = new Archive()
 *
 * // subsequence files will be streamed to the zip file
 * let promise = archive.pipeToFile('test.zip')
 *
 * // add file to zip by local file path
 * let file = 'package.json'
 * archive.addFile({ src_file: file, zip_file: 'package/pkg.json' })
 *
 * // add file to zip by content in buffer
 * let buffer = fs.readFileSync(file)
 * archive.addFile({ content: buffer, zip_file: 'dir1/file1.txt' })
 *
 * // add file to zip by content in string
 * let string = buffer.toString()
 * archive.addFile({ content: string, zip_file: 'dir1/file2.txt' })
 *
 * // signal the end of the archive stream
 * archive.end()
 *
 * // wait until the compression is completed
 * await promise
 * ```
 */
export class Archive {
  zipFile = new ZipFile()

  /**
   * @description
   * - Stream the content to a zip file.
   * - Can be called before or after adding files.
   * - Existing zip file will be overwritten if exists.
   *
   * @param file e.g. `"archive.zip"`
   */
  pipeToFile(file: string) {
    return new Promise<void>((resolve, reject) => {
      this.zipFile.outputStream
        .pipe(createWriteStream(file))
        .on('close', resolve)
        .on('error', reject)
    })
  }

  /**
   * @description
   * - Add file by local file path (string) or content (buffer).
   * - The content will be appended to the file in streaming mode.
   * - The directory of the zip file will be created if not exists.
   */
  addFile(args: { src_file: string; zip_file: string }): void
  addFile(args: { content: string | Buffer; zip_file: string }): void
  addFile(
    args:
      | { src_file: string; zip_file: string }
      | { content: string | Buffer; zip_file: string },
  ): void {
    if ('src_file' in args) {
      this.zipFile.addFile(args.src_file, args.zip_file, {})
      return
    }
    if ('content' in args) {
      let content = args.content
      if (typeof content == 'string') {
        content = Buffer.from(content)
      }
      this.zipFile.addBuffer(content, args.zip_file)
      return
    }
    throw new Error('Invalid arguments, expect src_file or content in args')
  }

  /**
   * @description
   * - Signal the end of the archive stream.
   * - Without calling this, the promise returned by `pipeToFile` will never be resolved.
   */
  end() {
    this.zipFile.end()
  }
}
