# ngram [![crates.io](https://img.shields.io/crates/d/ngram.svg)](https://crates.io/crates/ngram) [![docs.rs](https://docs.rs/ngram/badge.svg)](https://docs.rs/ngram) [![travis-ci.org](https://api.travis-ci.org/nytopop/ngram.svg?branch=master)](https://travis-ci.org/nytopop/ngram)
Rust iterator adaptors for n-grams and k-skip-n-grams.

Requires nightly compiler due to the use of [trait specialization](https://github.com/rust-lang/rust/issues/31844) over `Copy + Clone` vs `Clone` types.

# References
- [N-Gram](https://en.wikipedia.org/wiki/N-gram)

# License
MIT
