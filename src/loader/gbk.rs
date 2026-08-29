use encoding_rs::GBK;

/// 将 GBK 编码的字节解码为 UTF-8 字符串。
/// 使用 encoding_rs 的 SIMD 加速解码。
pub fn decode_gbk(bytes: &[u8]) -> String {
    let (cow, _, _) = GBK.decode(bytes);
    cow.into_owned()
}
