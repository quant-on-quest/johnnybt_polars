//! 分块的"读文件 → 解析"流水线。
//!
//! 一次性把几千个文件全读进内存再开始解析，峰值内存 = 全部原始字节 + 全部 batch，
//! 大批量下光是首次触碰这些内存的页错误就要好几秒（6080 个行情文件实测峰值 13.7 GB）。
//! 这里按块读：读第 N+1 块的同时解析第 N 块，峰值只剩一块的原始字节，
//! I/O 与 CPU 的重叠也还在。

use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::loader::batch_util::concat_aligned;

/// 单块的文件数上限
const IO_CHUNK_FILES: usize = 512;

/// 分块读取并解析，返回每块合并后的 RecordBatch（保持文件顺序）
///
/// `parse` 拿到 (文件字节, 文件路径)，返回该文件的 batch。
pub fn read_parse_chunked<F>(
    paths: &[String],
    io_threads: usize,
    parse: F,
) -> Result<Vec<RecordBatch>, String>
where
    F: Fn(&[u8], &str) -> Result<RecordBatch, String> + Sync,
{
    if paths.is_empty() {
        return Err("路径列表为空".to_string());
    }

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(io_threads)
        .enable_all()
        .build()
        .map_err(|e| format!("创建 tokio runtime 失败: {e}"))?;

    let chunks: Vec<&[String]> = paths.chunks(IO_CHUNK_FILES).collect();
    let mut pending = Some(spawn_chunk_read(&rt, chunks[0]));
    let mut out: Vec<RecordBatch> = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.iter().enumerate() {
        let handle = pending.take().expect("每轮循环开始时必然有待处理的读取");
        let raw_files = rt.block_on(handle).map_err(|e| format!("task 失败: {e}"))?;

        // 先把下一块的读取发出去，再解析当前块，让 I/O 和 CPU 重叠
        if i + 1 < chunks.len() {
            pending = Some(spawn_chunk_read(&rt, chunks[i + 1]));
        }

        let results: Vec<Result<RecordBatch, String>> = raw_files
            .into_par_iter()
            .zip(chunk.par_iter())
            .map(|(raw, path)| {
                let bytes = raw?;
                parse(&bytes, path)
            })
            .collect();

        let mut good: Vec<RecordBatch> = Vec::with_capacity(results.len());
        for r in results {
            let batch = r?;
            if batch.num_rows() > 0 {
                good.push(batch);
            }
        }
        // 块内先合并一次：限制单次拷贝的大小，也避免最终产出成千上万个碎片
        if !good.is_empty() {
            out.push(concat_aligned(&good)?);
        }
    }

    if out.is_empty() {
        return Err("无有效数据".to_string());
    }
    Ok(out)
}

/// 把一块文件的读取发到 tokio 上，返回等待句柄
fn spawn_chunk_read(
    rt: &tokio::runtime::Runtime,
    chunk: &[String],
) -> tokio::task::JoinHandle<Vec<Result<Vec<u8>, String>>> {
    let paths: Vec<String> = chunk.to_vec();
    rt.spawn(async move {
        let tasks: Vec<_> = paths
            .into_iter()
            .map(|p| {
                tokio::spawn(async move {
                    tokio::fs::read(&p)
                        .await
                        .map_err(|e| format!("读取文件失败 {p}: {e}"))
                })
            })
            .collect();

        futures::future::join_all(tasks)
            .await
            .into_iter()
            .map(|r| r.map_err(|e| format!("task 失败: {e}")).and_then(|v| v))
            .collect()
    })
}
