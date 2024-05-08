use crate::layers::Parameter;
use crate::types::NNFloat;
use ndarray::{Array, Dimension};

pub trait Optimizer {
    fn update<T: Dimension>(&self, weight: &mut Array<NNFloat, T>, grad: &Array<NNFloat, T>);
    fn update_all(&self, parameters: Vec<Parameter>) {
        for item in parameters.into_iter() {
            match item {
                Parameter::Matrix(weight, grad) => {
                    self.update(weight, grad);
                }
                Parameter::Bias(bias, grad) => {
                    self.update(bias, grad);
                }
            }
        }
    }
}

pub struct SGD {
    learning_rate: NNFloat,
}

impl SGD {
    pub fn new(learning_rate: NNFloat) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update<T: Dimension>(&self, weight: &mut Array<NNFloat, T>, grad: &Array<NNFloat, T>) {
        weight.zip_mut_with(grad, |x, x1| *x -= x1 * self.learning_rate);
    }
}
