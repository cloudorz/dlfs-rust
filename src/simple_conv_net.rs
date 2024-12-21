
use ndarray::{Array4, Array1, Ix2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::functions::accuary;
use crate::layers::{Affine, Relu, SoftmaxWithLoss, Convolution, Pooling};
use crate::optimizer::Optimizer;
use crate::types::{NNFloat, NNMatrix};

pub struct SimpleConvNet {
    conv1: Convolution,
    relu1: Relu,
    pool1: Pooling,
    affine1: Affine,
    relu2: Relu,
    affine2: Affine,
    last_layer: SoftmaxWithLoss,
}

impl SimpleConvNet {
    pub fn new(input_dim: [usize; 3], conv_param: [usize; 4], hidden_size: usize, output_size: usize, weight_init_std: NNFloat) -> Self {
        let input_size = input_dim[1];
        let filter_num = conv_param[0];
        let filter_size = conv_param[1];
        let filter_pad = conv_param[2];
        let filter_stride = conv_param[3];
        let conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1;
        let pool_output_size = filter_num * (conv_output_size/2) * (conv_output_size/2);

        let w1 = weight_init_std * Array4::<NNFloat>::random((filter_num, input_dim[0], filter_size, filter_size), Uniform::new(-2.0, 2.0));
        let b1 = Array1::zeros(filter_num);
        let w2 = weight_init_std * NNMatrix::random((pool_output_size, hidden_size), Uniform::new(-2.0, 2.0));
        let b2 = Array1::zeros(hidden_size);
        let w3 = weight_init_std * NNMatrix::random((hidden_size, output_size), Uniform::new(-2.0, 2.0));
        let b3 = Array1::zeros(output_size);

        Self {
            conv1: Convolution::new(w1, b1, filter_stride, filter_pad),
            relu1: Relu::new(),
            pool1: Pooling::new(2, 2, 1, 0),
            affine1: Affine::new(w2, b2),
            relu2: Relu::new(),
            affine2: Affine::new(w3, b3),
            last_layer: SoftmaxWithLoss::new(),
        }
    }

    pub fn predict(&mut self, x: &Array4<NNFloat>) -> NNMatrix {
        let conv1_output = self.conv1.forward(x.clone().into_dyn());
        let relu1_output = self.relu1.forward(&conv1_output);
        let pool1_output = self.pool1.forward(relu1_output);
        let affine1_output = self.affine1.forward(&pool1_output);
        let relu2_output = self.relu2.forward(&affine1_output);
        let affine2_output = self.affine2.forward(&relu2_output);

        affine2_output.into_dimensionality::<Ix2>().unwrap()
    }

    pub fn loss(&mut self, x: &Array4<NNFloat>, t: &NNMatrix) -> NNFloat {
        let y = self.predict(x);

        self.last_layer.forward(&y, t)
    }

    pub fn accuary(&mut self, x: &Array4<NNFloat>, t: &NNMatrix) -> NNFloat {
        accuary(&self.predict(x), t)
    }

    pub fn train<T: Optimizer>(&mut self, x: &Array4<NNFloat>, t: &NNMatrix, optimizer: &mut T) {
        let _ = self.loss(x, t);
        let d_out = self.last_layer.backward();
        let d_out = self.affine2.backward(&d_out);
        let d_out = self.relu2.backward(&d_out);
        let d_out = self.affine1.backward(&d_out);
        let d_out = self.pool1.backward(d_out);
        let d_out = self.relu1.backward(&d_out);
        let _ = self.conv1.backward(d_out);

        optimizer.set_current_layer(1);
        self.conv1.update(optimizer);
        optimizer.set_current_layer(2);
        self.affine1.update(optimizer);
        optimizer.set_current_layer(3);
        self.affine2.update(optimizer);
    }
}