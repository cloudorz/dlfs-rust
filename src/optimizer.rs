use crate::types::{NNArrayD, NNFloat};
use std::collections::HashMap;

pub trait Optimizer {
    fn update(&mut self, k: NNArrayD, z: NNArrayD) -> NNArrayD;
    fn set_current_layer(&mut self, current_layer: usize) {}
}

pub struct SGD {
    learning_rate: NNFloat,
}

impl SGD {
    pub fn new(learning_rate: NNFloat) -> Self {
        Self { learning_rate }
    }
}

impl Default for SGD {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Optimizer for SGD {
    fn update(&mut self, k: NNArrayD, z: NNArrayD) -> NNArrayD {
        let mut k = k;
        k.zip_mut_with(&z, |x, x1| {
            *x -= x1 * self.learning_rate;
        });

        k
    }
}


pub struct Momentum {
    learning_rate: NNFloat,
    momentum: NNFloat,
    v: HashMap<usize, NNArrayD>,
    current_layer: usize,
}

impl Momentum {
    pub fn new(learning_rate: NNFloat, momentum: NNFloat) -> Self {
        Self {
            learning_rate,
            momentum,
            v: HashMap::new(),
            current_layer: 0,
        }
    }

    pub fn default() -> Self {
        Self::new(0.01, 0.9)
    }

    pub fn set_current_layer(&mut self, current_layer: usize) {
        self.current_layer = current_layer;
    }
}

impl Optimizer for Momentum {
    fn update(&mut self, k: NNArrayD, z: NNArrayD) -> NNArrayD {
        let mut k = k;
        let v_current_layer = self.v.entry(self.current_layer).or_insert_with(|| NNArrayD::zeros(k.raw_dim()));

        v_current_layer.zip_mut_with(&z, |x, x1| {
            *x = self.momentum * *x - self.learning_rate * *x1;
        });

        k.zip_mut_with(v_current_layer, |x, x1| {
            *x += *x1;
        });

        k
    }
}