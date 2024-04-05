use ndarray::prelude::*;

pub fn sigmoid<D: Dimension>(x: &Array<f32, D>) -> Array<f32, D> {
    1.0 / (1.0 + (-x).mapv_into(|value| {value.exp()}))
}