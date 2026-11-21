//! Extension point for "don't freeze the PC" scheduling.
//!
//! v0 does nothing here: the PC is protected only by fewer threads, below-normal priority and
//! one generation at a time. Every generation calls `before_generation` (after it has the
//! slot and the model is loaded) and `after_generation` (with its timing).
//!
//! Planned uses (not built yet): pause or slow down while the user is typing or another app
//! needs memory bandwidth, change the thread count per request from measured lag, and split
//! long generations into chunks with short breaks.

use crate::stats::RequestTiming;

#[derive(Default)]
pub struct Pacer;

impl Pacer {
    pub async fn before_generation(&self) {}

    pub fn after_generation(&self, _timing: &RequestTiming) {}
}
