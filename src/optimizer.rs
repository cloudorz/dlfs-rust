use crate::types::{NNArrayD, NNFloat};
pub trait Optimizer {
    fn update(&mut self, k: NNArrayD, z: NNArrayD) -> NNArrayD;
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
