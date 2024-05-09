use crate::layers::Parameter;
use crate::types::{NNFloat, NNMatrix};
use std::iter::zip;

pub trait Optimizer {
    fn update(&mut self, parameters: Vec<Parameter>);
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
    fn update(&mut self, parameters: Vec<Parameter>) {
        for (weight, grad) in parameters.into_iter() {
            weight.zip_mut_with(grad, |x, x1| {
                *x -= x1 * self.learning_rate;
            });
        }
    }
}

pub struct Momentum {
    learning_rate: NNFloat,
    momentum: NNFloat,
    v: Vec<NNMatrix>,
}

impl Momentum {
    pub fn new(learning_rate: NNFloat, momentum: NNFloat) -> Self {
        Self {
            learning_rate,
            momentum,
            v: vec![],
        }
    }

    pub fn default() -> Self {
        Self::new(0.01, 0.9)
    }
}

impl Optimizer for Momentum {
    fn update(&mut self, parameters: Vec<Parameter>) {
        if self.v.is_empty() {
            for (weight, _) in parameters.iter() {
                self.v.push(NNMatrix::zeros(weight.raw_dim()));
            }
        }

        for ((weight, grad), v_items) in zip(parameters, &mut self.v) {
            v_items.zip_mut_with(grad, |v_value, grad_value| {
                *v_value = self.momentum * *v_value - self.learning_rate * grad_value;
            });
            weight.zip_mut_with(v_items, |w_value, grad_value| {
                *w_value += grad_value;
            });
        }
    }
}
