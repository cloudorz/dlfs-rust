use crate::functions::accuary;
use crate::layers::{Affine, Layer, Relu, Sequence, SoftmaxWithLoss};
use crate::optimizer::Optimizer;
use crate::types::{NNFloat, NNMatrix};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct TwoLayerNet {
    model: Sequence,
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
        let b1 = NNMatrix::zeros([1, hidden_size]);
        let b2 = NNMatrix::zeros([1, output_size]);
        let mut model = Sequence::new();
        model.add(Box::new(Affine::new(w1, b1)));
        model.add(Box::new(Relu::new()));
        model.add(Box::new(Affine::new(w2, b2)));
        Self {
            model,
            last_layer: SoftmaxWithLoss::new(),
        }
    }
}

impl TwoLayerNet {
    fn predict(&mut self, x: &NNMatrix) -> NNMatrix {
        self.model.forward(x)
    }

    pub fn loss(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        let y = self.predict(x);

        self.last_layer.forward(&y, t)
    }

    pub fn accuary(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        accuary(&self.predict(x), t)
    }

    pub fn train<T: Optimizer>(&mut self, x: &NNMatrix, t: &NNMatrix, optimizer: &mut T) {
        // forward
        self.loss(x, t);
        let d_out = self.last_layer.backward();
        self.model.backward(&d_out);

        optimizer.update(self.model.parameters());
    }
}
