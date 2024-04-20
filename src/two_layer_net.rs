use std::iter::zip;
use ndarray::{Array, Array1, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use crate::layers::{Affine, Relu, SoftmaxWithLoss};
use crate::types::{NNFloat, NNMatrix};

pub struct TwoLayerNet {
    affine1: Affine,
    relu: Relu,
    affine2: Affine,
    last_layer: SoftmaxWithLoss,
}

impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, weight_init_std: NNFloat) -> Self {

        let w1 = weight_init_std * Array::random(
            (input_size, hidden_size),
            Uniform::new(-2.0, 2.0));
        let w2 = weight_init_std * Array::random(
            (hidden_size, output_size),
            Uniform::new(-2.0, 2.0));
        let b1 = Array1::zeros(hidden_size);
        let b2 = Array1::zeros(output_size);
        Self {
            affine1: Affine::new(w1, b1),
            relu: Relu::new(),
            affine2: Affine::new(w2, b2),
            last_layer: SoftmaxWithLoss::new(),
        }
    }
}


impl TwoLayerNet {
    fn predict(&mut self, x: &NNMatrix) -> NNMatrix {
        let x = self.affine1.forward(x);
        let x = self.relu.forward(&x);
        let x = self.affine2.forward(&x);

        x
    }

    pub fn loss(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        let y = self.predict(x);

        self.last_layer.forward(&y, t)
    }

    pub fn accuary(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        let y = self.predict(x);
        let t = t.map_axis(Axis(1),
                           |arr| {
                               if arr.len() == 1 {
                                   arr[0] as usize
                               } else {
                                   arr.iter().position(|value| {
                                       *value == 1.0
                                   }).unwrap()
                               }
                           });
        let y = y.map_axis(Axis(1), |arr| {
            let max_v = arr.iter().max_by(|x1, x2| { x1.partial_cmp(x2).unwrap() }).unwrap();
            arr.iter().position(|v| { *v == *max_v }).unwrap()
        });
        let match_num = zip(y, t).fold(0.0, |acc, (y1, t1)| {
            if y1 == t1 {
                acc + 1.0
            } else {
                acc
            } });

        match_num / x.shape()[0] as NNFloat
    }

    pub fn update_params_with_gradient(&mut self, x: &NNMatrix, t: &NNMatrix, learning_rate: NNFloat) {
        // forward
        self.loss(x, t);
        let d_out = self.last_layer.backward();
        let d_out = self.affine2.backward(&d_out);
        let d_out = self.relu.backward(&d_out);
        self.affine1.backward(&d_out);

        self.affine1.update(learning_rate);
        self.affine2.update(learning_rate);
    }
}
