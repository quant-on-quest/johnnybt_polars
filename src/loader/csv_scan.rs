//! 引号感知的 CSV 记录扫描器。
//!
//! 用 memchr 找分隔符，普通字段零拷贝借用原文，只有含引号的字段才走缓冲区。
//! 引号规则（宽容模式，输入合规时与 polars 一致）：
//!   - `"a,b"`      → `a,b`（字段内逗号不切分）
//!   - `"a""b"`     → `a"b`（RFC4180 双写转义）
//!   - `"a"b"c"`    → `a"b"c`（闭合引号后不是分隔符 → 当作字面引号）
//!   - 引号内的换行不断行
//!
//! 第三条是数据源没按 RFC4180 转义时的兜底。polars 遇到这种输入会把这些引号直接丢掉
//! （`"a"b"c"` → `abc`），csv crate 会丢一半；这里选择一个字符都不丢。
//!
//! `quoting=false` 时完全跳过引号分支，引号成为普通字面字符（等价 0.1.x 的行为）。

use memchr::{memchr, memchr2};

/// 字段值的位置：要么借用原文的一段，要么在扫描器的缓冲区里（含转义处理）
enum FieldPos {
    Borrowed(usize, usize),
    Buffered,
}

pub struct Scanner<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    quoting: bool,
    /// 是否剥离字段首尾空白；false 时按字面值返回（与 polars read_csv 一致）
    trim: bool,
    buf: String,
}

impl<'a> Scanner<'a> {
    /// 跳过 `skip_rows` 个物理行后开始扫描
    pub fn new(text: &'a str, skip_rows: usize, quoting: bool, trim: bool) -> Self {
        let bytes = text.as_bytes();
        let mut pos = 0;
        for _ in 0..skip_rows {
            match memchr(b'\n', &bytes[pos..]) {
                Some(off) => pos += off + 1,
                None => {
                    pos = bytes.len();
                    break;
                }
            }
        }
        Scanner {
            text,
            bytes,
            pos,
            quoting,
            trim,
            buf: String::new(),
        }
    }

    /// 扫描下一条记录，逐字段调用 `f(字段序号, 字段值)`。
    ///
    /// `f` 返回 false 表示后面的字段不需要了，扫描器直接快进到行尾（投影时省掉尾部列的扫描）。
    /// 返回 false 表示已无记录。
    pub fn next_record<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(usize, &str) -> bool,
    {
        if !self.skip_blank_lines() {
            return false;
        }

        let mut field_idx = 0usize;
        loop {
            let (fp, next, eor) = self.scan_field(self.pos);
            let val = match fp {
                FieldPos::Borrowed(a, b) => &self.text[a..b],
                FieldPos::Buffered => self.buf.as_str(),
            };
            let want_more = f(field_idx, val);
            self.pos = next;
            if eor {
                break;
            }
            if !want_more {
                self.pos = self.skip_to_eol(self.pos);
                break;
            }
            field_idx += 1;
        }
        true
    }

    /// 跳过纯空白行；返回 false 表示已到末尾
    fn skip_blank_lines(&mut self) -> bool {
        let b = self.bytes;
        let len = b.len();
        while self.pos < len {
            let c = b[self.pos];
            // 绝大多数行首是数据字符，O(1) 判掉
            if c != b'\n' && c != b'\r' && c != b' ' && c != b'\t' {
                return true;
            }
            let eol = match memchr(b'\n', &b[self.pos..]) {
                Some(off) => self.pos + off + 1,
                None => len,
            };
            if self.text[self.pos..eol].trim().is_empty() {
                self.pos = eol;
            } else {
                return true;
            }
        }
        false
    }

    /// 引号感知地快进到下一行行首
    fn skip_to_eol(&self, mut pos: usize) -> usize {
        let b = self.bytes;
        let len = b.len();
        if !self.quoting {
            return match memchr(b'\n', &b[pos..]) {
                Some(off) => pos + off + 1,
                None => len,
            };
        }
        let mut in_quotes = false;
        while pos < len {
            match memchr2(b'"', b'\n', &b[pos..]) {
                None => return len,
                Some(off) => {
                    let p = pos + off;
                    if b[p] == b'"' {
                        in_quotes = !in_quotes;
                        pos = p + 1;
                    } else if in_quotes {
                        pos = p + 1;
                    } else {
                        return p + 1;
                    }
                }
            }
        }
        len
    }

    /// 扫描一个字段，返回 (值位置, 下一字段起点, 是否行末)
    fn scan_field(&mut self, start: usize) -> (FieldPos, usize, bool) {
        let b = self.bytes;
        let len = b.len();
        if start >= len {
            return (FieldPos::Borrowed(start, start), start, true);
        }

        if self.quoting && b[start] == b'"' {
            return self.scan_quoted_field(start);
        }

        match memchr2(b',', b'\n', &b[start..]) {
            None => (self.plain_field(start, len), len, true),
            Some(off) => {
                let p = start + off;
                let mut end = p;
                if b[p] == b'\n' && end > start && b[end - 1] == b'\r' {
                    end -= 1;
                }
                (self.plain_field(start, end), p + 1, b[p] == b'\n')
            }
        }
    }

    /// 引号字段：处理 `""` 转义与宽容的字面引号
    fn scan_quoted_field(&mut self, start: usize) -> (FieldPos, usize, bool) {
        let b = self.bytes;
        let len = b.len();
        self.buf.clear();
        let mut i = start + 1;
        let mut seg_start = i;

        loop {
            let q = match memchr(b'"', &b[i..]) {
                // 引号未闭合就到文件末尾：把剩下的都当字段内容
                None => {
                    self.buf.push_str(&self.text[seg_start..len]);
                    self.finish_quoted();
                    return (FieldPos::Buffered, len, true);
                }
                Some(off) => i + off,
            };

            // `""` → 一个字面引号
            if b.get(q + 1) == Some(&b'"') {
                self.buf.push_str(&self.text[seg_start..=q]);
                i = q + 2;
                seg_start = i;
                continue;
            }

            // 闭合引号后允许有空白，再跟分隔符/行尾/文件尾才算字段结束
            let mut j = q + 1;
            while j < len && (b[j] == b' ' || b[j] == b'\t') {
                j += 1;
            }
            if j >= len || b[j] == b',' || b[j] == b'\n' || b[j] == b'\r' {
                self.buf.push_str(&self.text[seg_start..q]);
                if b.get(j) == Some(&b'\r') {
                    j += 1;
                }
                let eor = j >= len || b[j] == b'\n';
                self.finish_quoted();
                return (FieldPos::Buffered, j + 1, eor);
            }

            // 后面还有别的字符 → 数据源没按 RFC4180 转义，当字面引号继续读
            self.buf.push_str(&self.text[seg_start..=q]);
            i = q + 1;
            seg_start = i;
        }
    }

    #[inline]
    fn plain_field(&self, start: usize, end: usize) -> FieldPos {
        if !self.trim {
            return FieldPos::Borrowed(start, end);
        }
        let s = &self.text[start..end];
        let a = start + (s.len() - s.trim_start().len());
        let trimmed = s.trim();
        FieldPos::Borrowed(a, a + trimmed.len())
    }

    /// 引号字段的内容也按 trim 选项处理，与非引号字段保持一致
    #[inline]
    fn finish_quoted(&mut self) {
        if self.trim {
            let t = self.buf.trim();
            if t.len() != self.buf.len() {
                self.buf = t.to_string();
            }
        }
    }

    /// 读取一条记录的全部字段（用于表头）
    pub fn read_row(&mut self) -> Option<Vec<String>> {
        let mut out = Vec::new();
        let got = self.next_record(|_, v| {
            out.push(v.trim().to_string());
            true
        });
        if got {
            Some(out)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(text: &str, quoting: bool) -> Vec<Vec<String>> {
        let mut sc = Scanner::new(text, 0, quoting, true);
        let mut out = Vec::new();
        while let Some(r) = sc.read_row() {
            out.push(r);
        }
        out
    }

    #[test]
    fn plain_fields() {
        assert_eq!(
            rows("a,b,c\n1,2,3\n", true),
            vec![vec!["a", "b", "c"], vec!["1", "2", "3"]]
        );
    }

    #[test]
    fn trims_whitespace_and_skips_blank_lines() {
        assert_eq!(rows(" a , b \n\n\n 1 , 2 \n", true), vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn quoted_comma_does_not_split() {
        // issue #1 案例 1b
        assert_eq!(
            rows("2024-05-08,\"关于变更公司名称,注册地址的公告\"\n", true),
            vec![vec!["2024-05-08", "关于变更公司名称,注册地址的公告"]]
        );
    }

    #[test]
    fn rfc4180_escaped_quote() {
        assert_eq!(rows("\"a\"\"b\",c\n", true), vec![vec!["a\"b", "c"]]);
    }

    #[test]
    fn lenient_unescaped_inner_quote() {
        // issue #1 案例 1：数据源没有把内部引号翻倍，这里一个字符都不丢
        // （polars 会把这两个引号丢掉，csv crate 会丢掉第一个）
        assert_eq!(
            rows("\"安徽凤凰:落实\"提质守信重回报\"行动公告\",x\n", true),
            vec![vec!["安徽凤凰:落实\"提质守信重回报\"行动公告", "x"]]
        );
    }

    #[test]
    fn whitespace_after_closing_quote() {
        assert_eq!(rows("\"a\" ,b\n", true), vec![vec!["a", "b"]]);
    }

    #[test]
    fn multiline_quoted_field() {
        // issue #1 案例 2：跨行引号字段算一条记录
        let out = rows("a,\"第一行\n第二行\"\nb,c\n", true);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec!["a", "第一行\n第二行"]);
        assert_eq!(out[1], vec!["b", "c"]);
    }

    #[test]
    fn quoting_disabled_keeps_quotes_literal() {
        assert_eq!(
            rows("\"a,b\",c\n", false),
            vec![vec!["\"a", "b\"", "c"]]
        );
    }

    #[test]
    fn crlf_line_endings() {
        assert_eq!(rows("a,b\r\n1,2\r\n", true), vec![vec!["a", "b"], vec!["1", "2"]]);
        assert_eq!(rows("\"a\",b\r\n1,2\r\n", true), vec![vec!["a", "b"], vec!["1", "2"]]);
    }

    #[test]
    fn unterminated_quote_at_eof() {
        assert_eq!(rows("a,\"未闭合\n还在里面", true), vec![vec!["a", "未闭合\n还在里面"]]);
    }

    #[test]
    fn early_exit_skips_rest_of_row() {
        let mut sc = Scanner::new("a,b,c,d\n1,2,3,4\n", 0, true, true);
        let mut first = Vec::new();
        sc.next_record(|i, v| {
            first.push(v.to_string());
            i < 1
        });
        assert_eq!(first, vec!["a", "b"]);
        // 下一条记录必须从第二行开头开始
        assert_eq!(sc.read_row().unwrap(), vec!["1", "2", "3", "4"]);
    }

    #[test]
    fn early_exit_is_quote_aware() {
        let mut sc = Scanner::new("a,b,\"含\n换行\"\n1,2,3\n", 0, true, true);
        let mut first = Vec::new();
        sc.next_record(|_, v| {
            first.push(v.to_string());
            false
        });
        assert_eq!(first, vec!["a"]);
        assert_eq!(sc.read_row().unwrap(), vec!["1", "2", "3"]);
    }
}
