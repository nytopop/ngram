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

#![feature(test, specialization)]

use std::collections::VecDeque;

/// Ensure that a condition holds within a Fn(...) -> Option<T>.
///
/// Optionally, execute one or more statements if it doesn't.
macro_rules! guard {
    ($e:expr) => {
        if !$e {
            return None;
        }
    };

    ($e:expr; $($do:expr),+) => {
        if !$e {
            $(
                $do;
            )+
            return None;
        }
    };
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
        guard! { self.n > 0 };

        if self.buf.len() < self.n {
            while self.buf.len() < self.n {
                self.buf.push_back(self.inner.next()?);
            }
            Some(self.buf.iter().map(copy).collect())
        } else {
            self.buf.pop_front();
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

    default fn size_hint(&self) -> (usize, Option<usize>) {
        match self.n {
            0 => (0, Some(0)),
            1 => self.inner.size_hint(),
            n => {
                let (l, u) = self.inner.size_hint();
                (l.saturating_sub(n - 1), u.map(|x| x.saturating_sub(n - 1)))
            }
        }
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
    fn ngram<F: Fn(&T) -> T>(&self, copy: F) -> Vec<T> {
        self.idx.iter().map(|&i| copy(&self.peek_buf[i])).collect()
    }

    fn first_ngram<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        guard! { self.peek_buf.len() < self.n };

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

        Some(self.ngram(copy))
    }

    fn next_kskip_ngram<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        let mut i = self.n - 1;
        while self.idx[i] - self.idx[i - 1] > self.k {
            guard! {i > 1;
                (0..self.n).for_each(|j| self.idx[j] = j)
            };
            i -= 1;
        }

        self.idx[i] += 1;
        for j in i..self.n - 1 {
            self.idx[j + 1] = self.idx[j] + 1;
        }

        for &i in &self.idx[i..] {
            if i >= self.peek_buf.len() {
                return self.next_kskip_ngram(copy);
            }
        }

        Some(self.ngram(copy))
    }

    fn next_ngram<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        if let Some(elt) = self.inner.next() {
            self.peek_buf.pop_front();
            self.peek_buf.push_back(elt);
            Some(self.ngram(copy))
        } else if self.peek_buf.len() > self.n {
            self.peek_buf.pop_front();
            Some(self.ngram(copy))
        } else {
            None
        }
    }

    fn iter_next<F: Fn(&T) -> T>(&mut self, copy: F) -> Option<Vec<T>> {
        match self.n {
            0 => None,
            1 => self.inner.next().map(|e| vec![e]),
            _ => self
                .first_ngram(&copy)
                .or_else(|| self.next_kskip_ngram(&copy))
                .or_else(|| self.next_ngram(&copy)),
        }
    }
}

impl<T: Clone, I: Iterator<Item = T>> Iterator for KSkipNGrams<T, I> {
    type Item = Vec<T>;

    default fn next(&mut self) -> Option<Self::Item> {
        self.iter_next(Clone::clone)
    }

    default fn size_hint(&self) -> (usize, Option<usize>) {
        match self.n {
            0 => (0, Some(0)),
            1 => self.inner.size_hint(),
            n => {
                let (l, u) = self.inner.size_hint();
                let (lx, ux) = (l.saturating_sub(n - 1), u.map(|x| x.saturating_sub(n - 1)));
                (
                    // ∑ (l - (n - 1) - Ki)
                    (0..=self.k).map(|k| lx.saturating_sub(k)).sum(),
                    // ∑ (u - (n - 1) - Ki)
                    ux.map(|x| (0..=self.k).map(|k| x.saturating_sub(k)).sum()),
                )
            }
        }
    }
}

impl<T: Copy, I: Iterator<Item = T>> Iterator for KSkipNGrams<T, I> {
    fn next(&mut self) -> Option<Self::Item> {
        self.iter_next(|&x| x)
    }
}

#[cfg(test)]
mod test_ngram {
    extern crate test;
    use self::test::Bencher;
    use super::*;

    fn str_ngrams(s: &str, n: usize) -> (Vec<String>, (usize, Option<usize>)) {
        let grams = s.chars().ngrams(n);
        let sz = grams.size_hint();
        (grams.map(|g| g.iter().collect()).collect(), sz)
    }

    #[test]
    fn ngrams() {
        let (g, sz) = str_ngrams("abcde", 0);
        assert_eq!(0, g.len());
        assert_eq!((0, Some(g.len())), sz);

        let (g, sz) = str_ngrams("abcde", 1);
        assert_eq!(vec!["a", "b", "c", "d", "e"], g);
        assert_eq!((2, Some(g.len())), sz);

        let (g, sz) = str_ngrams("abcde", 2);
        assert_eq!(vec!["ab", "bc", "cd", "de"], g);
        assert_eq!((1, Some(g.len())), sz);

        let (g, sz) = str_ngrams("abcde", 3);
        assert_eq!(vec!["abc", "bcd", "cde"], g);
        assert_eq!((0, Some(g.len())), sz);
    }

    fn str_kskip_ngrams(s: &str, k: usize, n: usize) -> (Vec<String>, (usize, Option<usize>)) {
        let grams = s.chars().kskip_ngrams(k, n);
        let sz = grams.size_hint();
        (grams.map(|g| g.iter().collect()).collect(), sz)
    }

    #[test]
    fn kskip_ngrams() {
        let (g, sz) = str_kskip_ngrams("abcde", 0, 0);
        assert_eq!(0, g.len());
        assert_eq!((0, Some(g.len())), sz,);

        let (g, sz) = str_kskip_ngrams("abcde", 1, 0);
        assert_eq!(0, g.len());
        assert_eq!((0, Some(g.len())), sz,);

        let (g, sz) = str_kskip_ngrams("abcde", 2, 0);
        assert_eq!(0, g.len());
        assert_eq!((0, Some(g.len())), sz,);

        let (g, sz) = str_kskip_ngrams("abcde", 0, 1);
        assert_eq!(vec!["a", "b", "c", "d", "e"], g);
        assert_eq!((2, Some(g.len())), sz,);

        let (g, sz) = str_kskip_ngrams("abcde", 0, 2);
        assert_eq!(vec!["ab", "bc", "cd", "de"], g);
        assert_eq!((1, Some(g.len())), sz,);

        let (g, sz) = str_kskip_ngrams("abcde", 1, 2);
        assert_eq!(vec!["ab", "ac", "bc", "bd", "cd", "ce", "de"], g);
        assert_eq!((1, Some(g.len())), sz,);

        let (g, sz) = str_kskip_ngrams("abcde", 2, 2);
        assert_eq!(
            vec!["ab", "ac", "ad", "bc", "bd", "be", "cd", "ce", "de"],
            g,
        );
        assert_eq!((1, Some(g.len())), sz,);
    }

    #[bench]
    fn bench_ngrams(b: &mut Bencher) {
        b.iter(|| (0..12).for_each(|n| (0..48).ngrams(n).for_each(|_| {})))
    }

    #[bench]
    fn bench_kskip_ngrams(b: &mut Bencher) {
        b.iter(|| {
            (0..3).for_each(|k| (0..4).for_each(|n| (0..21).kskip_ngrams(k, n).for_each(|_| {})))
        })
    }
}
