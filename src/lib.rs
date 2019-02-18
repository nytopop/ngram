// Copyright 2019 Eric Izoita (nytopop)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to
// do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#![feature(specialization)]

use std::collections::VecDeque;

fn guard(x: bool) -> Option<()> {
    if x {
        Some(())
    } else {
        None
    }
}

/// An iterator adaptor for the production of [n-grams](https://en.wikipedia.org/wiki/N-gram) or
/// [k-skip-n-grams](https://en.wikipedia.org/wiki/N-gram#Skip-gram).
///
/// ```rust
/// use ngram::NGram;
///
/// let input = "the rain in spain falls mainly on the plain";
/// let bi_grams: Vec<_> = input
///     .split(' ')
///     .ngrams(2)
///     .map(|xs| format!("{} {}", xs[0], xs[1]))
///     .collect();
/// let skip_grams: Vec<_> = input
///     .split(' ')
///     .kskip_ngrams(1, 2)
///     .map(|xs| format!("{} {}", xs[0], xs[1]))
///     .collect();
///
/// vec![
///     "the rain", "rain in", "in spain", "spain falls",
///     "falls mainly", "mainly on", "on the", "the plain",
/// ]
/// .into_iter()
/// .map(str::to_owned)
/// .for_each(|s| {
///     assert!(bi_grams.contains(&s));
///     assert!(skip_grams.contains(&s));
/// });
///
/// vec![
///     "the in", "rain spain", "in falls", "spain mainly",
///     "falls on", "mainly the", "on plain",
/// ]
/// .into_iter()
/// .map(str::to_owned)
/// .for_each(|s| {
///     assert!(!bi_grams.contains(&s));
///     assert!(skip_grams.contains(&s));
/// });
/// ```
pub trait NGram<T, I: Iterator<Item = T>> {
    fn ngrams(self, n: usize) -> NGrams<T, I>;

    fn kskip_ngrams(self, k: usize, n: usize) -> KSkipNGrams<T, I>;
}

impl<T, I: Iterator<Item = T>> NGram<T, I> for I {
    fn ngrams(self, n: usize) -> NGrams<T, I> {
        NGrams {
            inner: self,
            buf: VecDeque::with_capacity(n),
            n,
        }
    }

    fn kskip_ngrams(self, k: usize, n: usize) -> KSkipNGrams<T, I> {
        KSkipNGrams {
            inner: self,
            peek_buf: VecDeque::with_capacity((n.saturating_sub(1) * k) + n),
            idx: (0..n).collect(),
            k,
            n,
        }
    }
}

/// An iterator over n-grams.
///
/// This struct is created by the `ngrams` method on `NGram`.
pub struct NGrams<T, I: Iterator<Item = T>> {
    inner: I,
    buf: VecDeque<T>,
    n: usize,
}

impl<T, I: Iterator<Item = T>> NGrams<T, I> {
    fn iter_next<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        if self.n == 0 {
            None
        } else if self.buf.len() < self.n {
            while self.buf.len() < self.n {
                self.buf.push_back(self.inner.next()?);
            }
            Some(self.buf.iter().map(copy).collect())
        } else {
            drop(self.buf.pop_front());
            self.buf.push_back(self.inner.next()?);

            Some(self.buf.iter().map(copy).collect())
        }
    }
}

impl<T: Clone, I: Iterator<Item = T>> Iterator for NGrams<T, I> {
    type Item = Vec<T>;

    default fn next(&mut self) -> Option<Self::Item> {
        self.iter_next(Clone::clone)
    }
}

impl<T: Copy, I: Iterator<Item = T>> Iterator for NGrams<T, I> {
    fn next(&mut self) -> Option<Self::Item> {
        self.iter_next(|&x| x)
    }
}

/// An iterator over k-skip-n-grams.
///
/// This struct is created by the `kskip_ngrams` method on `NGram`.
pub struct KSkipNGrams<T, I: Iterator<Item = T>> {
    inner: I,
    peek_buf: VecDeque<T>,
    idx: Vec<usize>,
    k: usize,
    n: usize,
}

impl<T, I: Iterator<Item = T>> KSkipNGrams<T, I> {
    fn fill(&mut self) -> Option<()> {
        while self.peek_buf.len() < self.n {
            let elt = self.inner.next()?;
            self.peek_buf.push_back(elt);
        }

        while self.peek_buf.len() < (self.n.saturating_sub(1) * self.k) + self.n {
            if let Some(elt) = self.inner.next() {
                self.peek_buf.push_back(elt);
            } else {
                break;
            }
        }

        Some(())
    }

    fn reset(&mut self) {
        for i in 0..self.n {
            self.idx[i] = i;
        }
    }

    fn find_skip_index(&mut self) -> Option<usize> {
        let mut i = self.n - 1;
        while self.idx[i] - self.idx[i - 1] > self.k {
            i -= 1;
            guard(i != 0)?;
        }
        Some(i)
    }

    fn next_skip(&mut self) -> bool {
        if let Some(mut i) = self.find_skip_index() {
            self.idx[i] += 1;
            while i < self.n - 1 {
                self.idx[i + 1] = self.idx[i] + 1;
                i += 1;
            }
            true
        } else {
            self.reset();
            false
        }
    }

    fn skip_is_valid(&self) -> bool {
        for &i in &self.idx {
            if i >= self.peek_buf.len() {
                return false;
            }
        }
        true
    }

    fn try_skip<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        if self.next_skip() {
            let mut is_valid = self.skip_is_valid();
            if is_valid {
                return self.elems(copy);
            } else {
                while !is_valid {
                    if !self.next_skip() {
                        return None;
                    }

                    is_valid = self.skip_is_valid();
                    if is_valid {
                        return self.elems(copy);
                    }
                }
            }
        }
        None
    }

    fn try_ngram<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        if let Some(elt) = self.inner.next() {
            self.peek_buf.pop_front();
            self.peek_buf.push_back(elt);
            self.ngram(copy)
        } else if self.peek_buf.len() > self.n {
            self.peek_buf.pop_front();
            self.ngram(copy)
        } else {
            None
        }
    }

    fn elems<F: Fn(&T) -> T>(&self, copy: F) -> Option<Vec<T>> {
        Some(self.idx.iter().map(|&i| copy(&self.peek_buf[i])).collect())
    }

    fn ngram<F: Fn(&T) -> T>(&self, copy: F) -> Option<Vec<T>> {
        Some(self.peek_buf.iter().take(self.n).map(copy).collect())
    }

    fn iter_next<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        if self.n == 0 {
            return None;
        }

        if self.n == 1 {
            return Some(vec![self.inner.next()?]);
        }

        if self.peek_buf.len() < self.n {
            self.fill()?;
            return self.ngram(copy);
        }

        self.try_skip(&copy).or_else(|| self.try_ngram(copy))
    }
}

impl<T: Clone, I: Iterator<Item = T>> Iterator for KSkipNGrams<T, I> {
    type Item = Vec<T>;

    default fn next(&mut self) -> Option<Self::Item> {
        self.iter_next(Clone::clone)
    }
}

impl<T: Copy, I: Iterator<Item = T>> Iterator for KSkipNGrams<T, I> {
    fn next(&mut self) -> Option<Self::Item> {
        self.iter_next(|&x| x)
    }
}

#[cfg(test)]
mod test_ngram {
    use super::*;

    fn str_ngrams(s: &str, n: usize) -> Vec<String> {
        s.chars().ngrams(n).map(|g| g.iter().collect()).collect()
    }

    #[test]
    fn ngrams() {
        let grams = str_ngrams("abcde", 0);
        assert_eq!(0, grams.len());

        let grams = str_ngrams("abcde", 1);
        assert_eq!(vec!["a", "b", "c", "d", "e"], grams);

        let grams = str_ngrams("abcde", 2);
        assert_eq!(vec!["ab", "bc", "cd", "de"], grams);

        let grams = str_ngrams("abcde", 3);
        assert_eq!(vec!["abc", "bcd", "cde"], grams);
    }

    fn str_kskip_ngrams(s: &str, k: usize, n: usize) -> Vec<String> {
        s.chars()
            .kskip_ngrams(k, n)
            .map(|g| g.iter().collect())
            .collect()
    }

    #[test]
    fn kskip_ngrams() {
        let grams = str_kskip_ngrams("abcde", 0, 0);
        assert_eq!(0, grams.len());

        let grams = str_kskip_ngrams("abcde", 1, 0);
        assert_eq!(0, grams.len());

        let grams = str_kskip_ngrams("abcde", 2, 0);
        assert_eq!(0, grams.len());

        let grams = str_kskip_ngrams("abcde", 0, 1);
        assert_eq!(vec!["a", "b", "c", "d", "e"], grams);

        let grams = str_kskip_ngrams("abcde", 0, 2);
        assert_eq!(vec!["ab", "bc", "cd", "de"], grams);

        let grams = str_kskip_ngrams("abcde", 1, 2);
        assert_eq!(vec!["ab", "ac", "bc", "bd", "cd", "ce", "de"], grams);

        let grams = str_kskip_ngrams("abcde", 2, 2);
        assert_eq!(
            vec!["ab", "ac", "ad", "bc", "bd", "be", "cd", "ce", "de"],
            grams,
        );
    }
}
