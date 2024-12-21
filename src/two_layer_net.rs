use crate::functions::accuary;
use crate::layers::{Affine, Relu, SoftmaxWithLoss};
use crate::optimizer::Optimizer;
use crate::types::{NNFloat, NNMatrix};
use ndarray::{Array1, Ix2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct TwoLayerNet {
    affine_layer1: Affine,
    relu_layer: Relu,
    affine_layer2: Affine,
    last_layer: SoftmaxWithLoss,
}

impl TwoLayerNet {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight_init_std: NNFloat,
    ) -> Self {
        let w1 =
            weight_init_std * NNMatrix::random((input_size, hidden_size), Uniform::new(-2.0, 2.0));
        let w2 =
            weight_init_std * NNMatrix::random((hidden_size, output_size), Uniform::new(-2.0, 2.0));
        let b1 = Array1::<NNFloat>::zeros(hidden_size);
        let b2 = Array1::<NNFloat>::zeros(output_size);
        Self {
            affine_layer1: Affine::new(w1, b1),
            relu_layer: Relu::new(),
            affine_layer2: Affine::new(w2, b2),
            last_layer: SoftmaxWithLoss::new(),
        }
    }
}

impl TwoLayerNet {
    fn predict(&mut self, x: &NNMatrix) -> NNMatrix {
        let affine_layer1_output = self.affine_layer1.forward(&x.clone().into_dyn());
        let relu_output = self.relu_layer.forward(&affine_layer1_output);
        let affine_layer2_output = self.affine_layer2.forward(&relu_output);

        affine_layer2_output.into_dimensionality::<Ix2>().unwrap()
    }

    pub fn loss(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        let y = self.predict(x);

        self.last_layer.forward(&y, t)
    }

    pub fn accuary(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        accuary(&self.predict(x), t)
    }

    pub fn train<T: Optimizer>(&mut self, x: &NNMatrix, t: &NNMatrix, optimizer: &mut T) {
        let _ = self.loss(x, t);
        let d_out = self.last_layer.backward();
        let d_out = self.affine_layer2.backward(&d_out);
        let d_out = self.relu_layer.backward(&d_out);
        let _ = self.affine_layer1.backward(&d_out);
        optimizer.set_current_layer(1);
        self.affine_layer1.update(optimizer);
        optimizer.set_current_layer(2);
        self.affine_layer2.update(optimizer);
    }
}
